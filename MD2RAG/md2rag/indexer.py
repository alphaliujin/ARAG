"""索引器 - 借鉴 bisheng 的端到端入库流程.

完整流程：
1. 扫描 MD 目录下的 .parents.json / .children.json / .chunks.json
2. 加载切片
3. （可选）调用 LLM 重新生成摘要
4. 关联 bbox 元数据
5. 嵌入（bge-m3 1024 维）
6. 父子块双 collection 入库
7. 检索时：child 搜 → parent 回查

输出：向量集合（每密级 2 个 collection：parent + child）+ 图片集合
"""

from __future__ import annotations

import gc
import hashlib
import signal
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from md2rag.bbox_extractor import get_bboxes_for_record
from md2rag.config import MD2RAGConfig, load_config
from md2rag.embedder import Embedder, create_embedder
from md2rag.image_processor import ImageProcessor, ViTImageEmbedder
from md2rag.llm_summary import LLMClient, extract_abstract
from md2rag.loader import (
    CLASSIFICATION_DIR_MAP,
    VALID_CLASSIFICATIONS,
    ChunkLoader,
    ChunkRecord,
)
from md2rag.logger import get_logger, log_step, log_timing, log_memory, setup_exception_logging
from md2rag.parent_child_retriever import ParentChildRetriever
from md2rag.vector_store import VectorStore

# 设置全局异常捕获
setup_exception_logging()

logger = get_logger("md2rag.indexer")


# ------------------------------------------------------------------------
# 信号处理：确保收到终止信号时记录日志
# ------------------------------------------------------------------------

_received_signal = False


def _signal_handler(signum, frame):
    """处理终止信号，记录日志后退出."""
    global _received_signal
    if _received_signal:
        return  # 避免重复处理
    _received_signal = True

    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.critical(f"[SIGNAL] 收到信号 {sig_name}，正在安全退出...")
    logger.critical(f"[SIGNAL] 退出时堆栈:\n{''.join(traceback.format_stack(frame))}")
    sys.exit(128 + signum)


# 注册信号处理器
for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass  # Windows 上可能没有 SIGHUP


# ------------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------------

@dataclass
class IndexResult:
    """索引结果."""
    status: str = "pending"  # pending | success | error | empty
    message: str = ""
    documents_processed: int = 0
    parents_added: int = 0
    children_added: int = 0
    images_added: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def chunks_added(self) -> int:
        """总切片数 = parents + children (向后兼容 api_server.py 引用)."""
        return self.parents_added + self.children_added


# ------------------------------------------------------------------------
# 密级映射统一在 md2rag.loader.CLASSIFICATION_DIR_MAP 维护,本文件不再持有副本。
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# 主索引器
# ------------------------------------------------------------------------

class Indexer:
    """MD 切片索引器 - 借鉴 bisheng 完整入库流程.

    责任链:
      X2MD .parents/.children.json  →  ChunkLoader  →  LLM摘要  →
      bbox 关联  →  Embedder  →  ParentChildRetriever (双 collection)  →
      VectorStore (ChromaDB)
    """

    def __init__(self, config: Optional[MD2RAGConfig] = None):
        log_step(logger, "INIT", "Initializing Indexer...")
        init_start = time.time()

        self.config = config or MD2RAGConfig()
        logger.info(f"[CONFIG] md_dir: {self.config.md_dir}")
        logger.info(f"[CONFIG] vector_db_dir: {self.config.vector_db_dir}")
        logger.info(f"[CONFIG] embedding_model: {self.config.embedding_model}")
        logger.info(f"[CONFIG] collection_prefix: {self.config.collection_prefix}")

        self.loader = ChunkLoader(self.config.md_dir)
        self.embedder = self._create_embedder()
        self._current_embedder_name = self._resolve_embedder_name()
        logger.info(f"[EMBEDDER] dimension: {self.embedder.dimension}, name: {self._current_embedder_name}")

        self.store = VectorStore(
            db_dir=self.config.vector_db_dir,
            embedder=self.embedder,
            collection_prefix=self.config.collection_prefix,
        )

        # Per-classification 父子检索器缓存
        self._retrievers: Dict[str, ParentChildRetriever] = {}

        # LLM 客户端（可选，延迟初始化）
        self._llm_client: Optional[LLMClient] = None

        elapsed_ms = (time.time() - init_start) * 1000
        log_timing(logger, "Indexer initialization", elapsed_ms)
        log_step(logger, "INIT_COMPLETE", "Indexer ready")

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    def _create_embedder(self) -> Embedder:
        """根据配置创建嵌入器."""
        # 优先使用 MPS（如果配置为 mps 设备）
        if self.config.device == "mps":
            logger.info("[EMBEDDER] Using MPS embedder (Apple Silicon GPU)")
            return create_embedder(
                model_type="mps",
                device="mps",
            )
        if self.config.ollama_enabled:
            return create_embedder(
                model_type="ollama",
                ollama_url=self.config.ollama_base_url,
                ollama_model=self.config.ollama_model,
            )
        if self.config.st_enabled:
            return create_embedder(
                model_type="sentence-transformers",
                st_model=self.config.st_model_name,
                device=self.config.device,
            )
        return create_embedder(model_type="chromadb-default")

    _EMBEDDER_MODEL_MAP = {
        "ollama-bge-m3": ("ollama", {"ollama_model": "bge-m3:latest"}),
        "chromadb-default": ("chromadb-default", {}),
        "sentence-transformers": ("sentence-transformers", {}),
        "mps": ("mps", {}),
    }

    def _resolve_embedder_name(self) -> str:
        if self.config.device == "mps":
            return "mps"
        if self.config.ollama_enabled:
            return "ollama-bge-m3" if self.config.ollama_model == "bge-m3:latest" else "ollama"
        if self.config.st_enabled:
            return "sentence-transformers"
        return "chromadb-default"

    def switch_embedder(self, embedding_model: str) -> None:
        """切换嵌入模型（维度变化时清空 collection）."""
        if embedding_model == self._current_embedder_name:
            logger.info(f"[SWITCH] Embedder already {embedding_model}, skipping")
            return

        entry = self._EMBEDDER_MODEL_MAP.get(embedding_model)
        if entry is None:
            logger.warning(f"[SWITCH] Unknown embedder '{embedding_model}', fallback to chromadb-default")
            model_type, kwargs = "chromadb-default", {}
        else:
            model_type, kwargs = entry

        old_dim = self.embedder.dimension
        new_embedder = create_embedder(
            model_type=model_type,
            ollama_url=self.config.ollama_base_url,
            device=self.config.device,
            **kwargs,
        )
        new_dim = new_embedder.dimension

        if old_dim != new_dim:
            logger.info(f"[SWITCH] Dimension changed ({old_dim} -> {new_dim}), clearing collections")
            self.store.reset_all()

        self.embedder = new_embedder
        self.store.embedder = new_embedder
        self._current_embedder_name = embedding_model
        self._retrievers.clear()  # 重建
        logger.info(f"[SWITCH] Embedder switched to {embedding_model}, dim={new_dim}")

    def _get_retriever(self, classification: str) -> ParentChildRetriever:
        if classification not in self._retrievers:
            self._retrievers[classification] = ParentChildRetriever(
                vector_store=self.store,
                embedder=self.embedder,
                classification=classification,
                collection_prefix=self.config.collection_prefix,
            )
        return self._retrievers[classification]

    def set_llm_client(self, client: Optional[LLMClient]) -> None:
        """设置/清除 LLM 客户端."""
        self._llm_client = client
        if client is not None:
            logger.info("[LLM] LLM client configured")
        else:
            logger.info("[LLM] LLM client cleared")

    # ------------------------------------------------------------------
    # 核心：索引目录
    # ------------------------------------------------------------------

    def index_directory(
        self,
        classification: Optional[str] = None,
        strategy: str = "auto",
        regenerate_abstract: bool = False,
        include_images: bool = True,
    ) -> IndexResult:
        """索引 MD 目录下所有切片文件.

        Args:
            classification: 指定密级，None 时扫描所有
            strategy: 切片策略 auto/chunk/parent-child
            regenerate_abstract: 是否用 LLM 重新生成 abstract
            include_images: 是否同时处理图片（ViT-Large）

        Returns:
            IndexResult
        """
        logger.info("=" * 80)
        log_step(logger, "INDEX_DIRECTORY", f"classification={classification}, strategy={strategy}")
        logger.info("=" * 80)
        start = time.time()

        result = IndexResult()

        try:
            # 1. 发现文件
            chunk_files = self.loader.discover_files(classification)
            if not chunk_files:
                result.status = "empty"
                result.message = "No chunk files found"
                logger.warning(result.message)
                return result

            logger.info(f"[FILES] Discovered {len(chunk_files)} chunk files")

            # 2. 加载所有文件（串行加载，避免 ThreadPoolExecutor 嵌套冲突）
            all_records_by_file: Dict[Path, List[ChunkRecord]] = {}
            load_errors: List[str] = []
            for i, fp in enumerate(chunk_files):
                try:
                    records = self.loader.load_file(fp)
                    if records:
                        all_records_by_file[fp] = records
                    if (i + 1) % 10 == 0:
                        logger.info(f"[PROGRESS] Loaded {i + 1}/{len(chunk_files)} files")
                except Exception as e:
                    logger.error(f"[LOAD_ERROR] Failed to load {fp}: {e}")
                    load_errors.append(f"{fp.name}: {e}")

            result.errors.extend(load_errors)
            total_records = sum(len(r) for r in all_records_by_file.values())
            log_memory(logger, "Records loaded", total_records)

            if not all_records_by_file:
                result.status = "empty"
                result.message = "No records loaded"
                logger.warning(result.message)
                return result

            # 3. 按密级分组
            by_class: Dict[str, Dict[str, List[ChunkRecord]]] = {}  # class -> {file_path: [records]}
            for fp, records in all_records_by_file.items():
                cls = classification or self.loader.get_classification_from_path(fp)
                by_class.setdefault(cls, {}).setdefault(str(fp), []).extend(records)

            # 4. 入库每个密级
            for cls, file_records_map in by_class.items():
                logger.info(f"[INDEX] Processing classification: {cls} ({len(file_records_map)} files)")
                try:
                    parents, children = self._index_classification(cls, file_records_map, regenerate_abstract)
                    result.parents_added += parents
                    result.children_added += children
                    result.documents_processed += len(file_records_map)
                    logger.info(f"[INDEX] {cls}: +{parents} parents, +{children} children")
                except Exception as e:
                    logger.critical(f"[CRITICAL] Failed to index classification {cls}: {e}", exc_info=True)
                    result.errors.append(f"classification {cls}: {e}")

            # 5. 处理图片
            if include_images and self.config.vit_enabled:
                try:
                    logger.info("[IMAGES] Starting image indexing...")
                    images_count = self._index_images(classification)
                    result.images_added = images_count
                    logger.info(f"[IMAGES] Indexed {images_count} images")
                except Exception as e:
                    logger.error(f"[IMAGES] Failed: {e}", exc_info=True)
                    result.errors.append(f"images: {e}")

        except Exception as e:
            logger.critical(f"[CRITICAL] index_directory failed: {e}", exc_info=True)
            result.status = "error"
            result.message = f"Critical error: {e}"
            result.errors.append(str(e))
            return result

        elapsed_ms = (time.time() - start) * 1000
        log_timing(logger, "Total index_directory", elapsed_ms)

        result.status = "success"
        result.message = (
            f"Indexed {result.documents_processed} files, "
            f"{result.parents_added} parents, "
            f"{result.children_added} children, "
            f"{result.images_added} images"
        )
        log_step(logger, "INDEX_COMPLETE", result.message)
        return result

    def index_md_file(
        self,
        md_file: str | Path,
        classification: Optional[str] = None,
        strategy: str = "auto",
        regenerate_abstract: bool = False,
    ) -> IndexResult:
        """索引单个 MD 切片文件.

        与 index_directory 的区别: 仅加载并入库指定的单个 chunk 文件,
        不扫描整个 md_dir, 也不处理图片。供 CLI `index-file` 命令使用
        (此前 CLI 调用了不存在的方法, 导致该命令必抛 AttributeError)。

        Args:
            md_file: 切片文件路径 (*.parents.json / *.children.json / *.chunks.json)
            classification: 指定密级, None 时按路径自动推断
            strategy: 切片策略 (仅用于日志; 实际 parent/child 由文件记录决定)
            regenerate_abstract: 是否用 LLM 重新生成 abstract
        """
        logger.info("=" * 80)
        log_step(logger, "INDEX_MD_FILE", f"file={md_file}, classification={classification}, strategy={strategy}")
        logger.info("=" * 80)
        start = time.time()

        result = IndexResult()

        try:
            file_path = Path(md_file)
            try:
                records = self.loader.load_file(file_path)
            except Exception as e:
                logger.error(f"[LOAD_ERROR] Failed to load {file_path}: {e}")
                result.status = "error"
                result.message = f"Failed to load {file_path.name}: {e}"
                result.errors.append(f"{file_path.name}: {e}")
                return result

            if not records:
                result.status = "empty"
                result.message = f"No records in {file_path.name}"
                logger.warning(result.message)
                return result

            cls = classification or self.loader.get_classification_from_path(file_path)
            logger.info(f"[INDEX] Processing classification: {cls} (1 file)")

            try:
                parents, children = self._index_classification(
                    cls, {str(file_path): records}, regenerate_abstract
                )
                result.parents_added += parents
                result.children_added += children
                result.documents_processed = 1
                logger.info(f"[INDEX] {cls}: +{parents} parents, +{children} children")
            except Exception as e:
                logger.critical(f"[CRITICAL] Failed to index {file_path}: {e}", exc_info=True)
                result.status = "error"
                result.message = f"Critical error: {e}"
                result.errors.append(f"{file_path.name}: {e}")
                return result

        except Exception as e:
            logger.critical(f"[CRITICAL] index_md_file failed: {e}", exc_info=True)
            result.status = "error"
            result.message = f"Critical error: {e}"
            result.errors.append(str(e))
            return result

        elapsed_ms = (time.time() - start) * 1000
        log_timing(logger, "Total index_md_file", elapsed_ms)

        result.status = "success"
        result.message = (
            f"Indexed 1 file, {result.parents_added} parents, {result.children_added} children"
        )
        log_step(logger, "INDEX_COMPLETE", result.message)
        return result

    def _index_classification(
        self,
        classification: str,
        file_records_map: Dict[str, List[ChunkRecord]],
        regenerate_abstract: bool = False,
    ) -> Tuple[int, int]:
        """入库单个密级下的所有切片（优化版：批量嵌入）."""
        log_step(logger, "INDEX_CLASS", f"class={classification}, files={len(file_records_map)}")
        retriever = self._get_retriever(classification)

        parents_total = 0
        children_total = 0
        processed_files = 0
        total_files = len(file_records_map)

        # 收集所有需要嵌入的数据，按类型分组
        # 结构: [(record, meta_dict, id_str, type_str), ...]
        # type_str: "parent" | "child" | "chunk"
        embed_batch = []  # 批量嵌入的文本列表
        embed_items = []  # 对应的元数据

        # 4.1 收集所有 records，按 doc_id 关联 parent 和 children
        for file_path, records in file_records_map.items():
            processed_files += 1
            if processed_files % 10 == 0:
                logger.info(f"[PROGRESS] {classification}: {processed_files}/{total_files} files processed")

            # 分类：parent / child / 普通 chunk
            parents = [r for r in records if r.is_parent]
            children = [r for r in records if r.is_child]
            chunks = [r for r in records if not r.is_parent and not r.is_child]

            # 读取 MD 原文用于 bbox 合成
            md_path = self._md_path_from_chunk_path(Path(file_path))
            source_text = ""
            if md_path and md_path.exists():
                try:
                    source_text = md_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"[INDEX] Failed to read MD {md_path}: {e}")

            # 4.2 处理 parents
            for p in parents:
                if regenerate_abstract and self._llm_client:
                    new_abs = extract_abstract(self._llm_client, p.text)
                    if new_abs:
                        p.abstract = new_abs

                # bbox 关联
                if not p.bbox and source_text:
                    bboxes = get_bboxes_for_record(p.text, source_text)
                    if bboxes:
                        p.bbox = str(bboxes)

                meta = p.to_metadata(classification, file_path)
                doc_id = p.doc_id or str(uuid.uuid4())
                meta["doc_id"] = doc_id
                meta["parent_doc_id"] = doc_id

                embed_batch.append(p.embedding_text)
                embed_items.append({
                    "record": p,
                    "meta": meta,
                    "id": doc_id,
                    "type": "parent",
                    "file_path": file_path,
                })

            # 4.3 处理 children
            for c in children:
                if regenerate_abstract and self._llm_client and not c.abstract:
                    new_abs = extract_abstract(self._llm_client, c.text)
                    if new_abs:
                        c.abstract = new_abs

                if not c.bbox and source_text:
                    bboxes = get_bboxes_for_record(c.text, source_text)
                    if bboxes:
                        c.bbox = str(bboxes)

                meta = c.to_metadata(classification, file_path)
                meta["parent_doc_id"] = c.parent_doc_id or c.doc_id
                meta["is_child"] = True

                child_id = f"{meta['parent_doc_id']}_child_{c.chunk_index}"
                embed_batch.append(c.embedding_text)
                embed_items.append({
                    "record": c,
                    "meta": meta,
                    "id": child_id,
                    "type": "child",
                    "file_path": file_path,
                })

            # 4.4 处理普通 chunks（无 parent/child 关系的）
            for ch in chunks:
                meta = ch.to_metadata(classification, file_path)
                doc_id = ch.doc_id or str(uuid.uuid4())
                meta["doc_id"] = doc_id
                meta["parent_doc_id"] = doc_id
                meta["is_parent"] = True
                meta["is_child"] = False

                embed_batch.append(ch.embedding_text)
                embed_items.append({
                    "record": ch,
                    "meta": meta,
                    "id": doc_id,
                    "type": "chunk",
                    "file_path": file_path,
                })

        # 4.5 批量嵌入所有文本
        if not embed_batch:
            logger.info(f"[INDEX_CLASS] {classification}: No items to embed")
            return 0, 0

        logger.info(f"[INDEX_CLASS] {classification}: Embedding {len(embed_batch)} items in batch...")
        try:
            all_embeddings = self.embedder.embed(embed_batch)
            if not all_embeddings or len(all_embeddings) != len(embed_batch):
                logger.error(f"[INDEX] Batch embedding failed: expected {len(embed_batch)}, got {len(all_embeddings) if all_embeddings else 0}")
                return parents_total, children_total
        except Exception as e:
            logger.error(f"[INDEX] Batch embedding failed: {e}")
            return parents_total, children_total

        # 4.6 将嵌入结果入库
        parent_chunks = []
        child_chunks = []
        parent_embeddings = []
        child_embeddings = []
        parent_ids = []
        child_ids = []

        for i, item in enumerate(embed_items):
            emb = all_embeddings[i]
            if not emb:
                logger.error(f"[INDEX] Empty embedding for {item['type']} {item['record'].source}#{item['record'].chunk_index}")
                continue

            if item["type"] == "parent" or item["type"] == "chunk":
                parent_chunks.append((item["record"].embedding_text, item["meta"]))
                parent_embeddings.append(emb)
                parent_ids.append(item["id"])
                parents_total += 1
            elif item["type"] == "child":
                child_chunks.append((item["record"].embedding_text, item["meta"]))
                child_embeddings.append(emb)
                child_ids.append(item["id"])
                children_total += 1

        # 批量入库
        if parent_chunks:
            try:
                retriever.add_parent_chunks(parent_chunks, embeddings=parent_embeddings, ids=parent_ids)
                logger.info(f"[INDEX] Added {len(parent_chunks)} parent/chunk items")
            except Exception as e:
                logger.error(f"[INDEX] Failed to add parent chunks: {e}")

        if child_chunks:
            try:
                retriever.add_child_chunks(child_chunks, embeddings=child_embeddings, ids=child_ids)
                logger.info(f"[INDEX] Added {len(child_chunks)} child items")
            except Exception as e:
                logger.error(f"[INDEX] Failed to add child chunks: {e}")

        logger.info(f"[INDEX_CLASS_COMPLETE] {classification}: {parents_total} parents, {children_total} children")
        return parents_total, children_total

    def _md_path_from_chunk_path(self, chunk_path: Path) -> Optional[Path]:
        """从 .parents.json / .children.json / .chunks.json 反推 MD 原文路径."""
        # 路径形如: MD/0Public/银渐层.md/银渐层.parents.json
        # MD 原文:   MD/0Public/银渐层.md/银渐层.md
        if chunk_path.suffixes[-2] in (".parents", ".children", ".chunks"):
            stem = chunk_path.name.split(".")[0]  # "银渐层"
            return chunk_path.parent / f"{stem}.md"
        return None

    # ------------------------------------------------------------------
    # 图片索引
    # ------------------------------------------------------------------

    def _index_images(self, classification: Optional[str] = None) -> int:
        """用 ViT-Large 处理 MD 目录下的图片."""
        log_step(logger, "INDEX_IMAGES", f"Processing images (classification={classification})")
        start = time.time()

        vit_embedder = None
        processor = None
        records = None
        try:
            # 先构造正确的 ViT embedder，再传给 ImageProcessor（避免使用默认参数下载远程模型）
            vit_embedder = ViTImageEmbedder(
                model_name=self.config.vit_model,
                device=self.config.vit_device,
            )
            processor = ImageProcessor(self.config.md_dir, embedder=vit_embedder)
            records = processor.process_images(classification)
            if not records:
                logger.info("[IMAGES] No images found")
                return 0

            total_images = 0
            for rec in records:
                if rec.embedding is None:
                    continue
                meta = {
                    "source": rec.image_path.name,
                    "document_name": rec.source_md,
                    "chunk_index": rec.image_index,
                    "chunk_type": "image",
                    "classification": rec.classification,
                    "content_type": "image",
                    "image_path": str(rec.image_path),
                    "original_md_path": str(rec.image_path),
                    "original_file_name": rec.source_md,
                    "is_parent": False,
                    "is_child": False,
                    "doc_id": rec.image_id,
                    "parent_doc_id": rec.image_id,
                }
                # store.get_or_create_collection() 内部会自动加 collection_prefix,
                # 所以这里只传 "{cls}_images" 即可。
                # 历史 bug(已根治): 之前在这里又拼了 prefix,导致 collection 实际名为
                # md2rag_md2rag_{cls}_images (双前缀)。
                image_collection_key = f"{rec.classification}_images"
                try:
                    coll = self.store.get_or_create_collection(image_collection_key)
                    coll.add(
                        documents=[rec.embedding_text],
                        embeddings=[rec.embedding],
                        metadatas=[meta],
                        ids=[rec.image_id],
                    )
                    total_images += 1
                except Exception as e:
                    logger.error(f"[IMAGES] Failed {rec.image_path.name}: {e}")

            elapsed_ms = (time.time() - start) * 1000
            log_timing(logger, "Index images", elapsed_ms)
            return total_images
        finally:
            if vit_embedder is not None:
                vit_embedder.release()
            records = None
            processor = None

    # ------------------------------------------------------------------
    # 检索（便捷方法）
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        classification: Optional[str] = None,
        n_results: int = 5,
        return_parents: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """搜索接口（向后兼容 api_server.search_all 风格）."""
        if classification:
            classifications = [classification]
        else:
            classifications = list(VALID_CLASSIFICATIONS)

        all_results: Dict[str, List[Dict[str, Any]]] = {}
        for cls in classifications:
            retriever = self._get_retriever(cls)
            hits = retriever.search(query, n_results=n_results, return_parents=return_parents)
            all_results[cls] = [
                {
                    "content": h.text,
                    "score": h.score,
                    "metadata": h.metadata,
                    "parent_text": h.parent_text,
                }
                for h in hits
            ]
        return all_results

    # ------------------------------------------------------------------
    # 统计与清理
    # ------------------------------------------------------------------

    def release(self) -> None:
        """释放入库期间持有的模型、ChromaDB client 和 PyTorch 缓存。"""
        try:
            self._retrievers.clear()
            self._llm_client = None
            if hasattr(self.embedder, "release"):
                self.embedder.release()
            if hasattr(self.store, "close"):
                self.store.close()
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
            except Exception:
                pass

    def get_stats(self) -> Dict[str, int]:
        """获取所有 collection 的统计.

        三档密级: public / restricted / confidential.
        """
        stats: Dict[str, int] = {}
        for cls in CLASSIFICATION_DIR_MAP:
            try:
                retriever = self._get_retriever(cls)
                rs = retriever.get_stats()
                stats[cls] = rs["parent_count"] + rs["child_count"]
                stats[f"{cls}_parents"] = rs["parent_count"]
                stats[f"{cls}_children"] = rs["child_count"]
            except Exception as e:
                logger.warning(f"[STATS] {cls}: {e}")
                stats[cls] = 0
                stats[f"{cls}_parents"] = 0
                stats[f"{cls}_children"] = 0
        return stats

    def clear(self, classification: Optional[str] = None) -> None:
        """清空 collection."""
        if classification:
            retriever = self._get_retriever(classification)
            retriever.clear()
            # 主动清空 vector_store 缓存和 ChromaDB 底层 collection
            # 用 _get_client() 触发初始化, 不能直接访问 _client (会是 None)
            try:
                client = self.store._get_client()
            except Exception as e:
                logger.error(f"[CLEAR] Failed to init ChromaDB client: {e}")
                return
            for name in [retriever.parent_collection, retriever.child_collection]:
                try:
                    client.delete_collection(name=name)
                    logger.info(f"[CLEAR] Deleted collection: {name}")
                except Exception as e:
                    logger.warning(f"[CLEAR] Failed to delete {name}: {e}")
                self.store._collections.pop(name, None)
            # 原代码 pop(name, ...) 在循环外, name 残留最后一次值; 上面已在循环内 pop, 这里清空 retriever 缓存即可
            self._retrievers.pop(classification, None)
        else:
            # 全部清空: 枚举 ChromaDB 实际存在的全部 collection 后逐一删除
            # 不再用硬编码名字列表 (此前的列表会漏掉 md2rag_md2rag_*_images 这类
            # 前缀加倍的名字, 也漏掉外部写入的 legacy collection)
            # store._client 是懒加载的, 必须用 _get_client() 触发初始化, 否则
            # 直接访问 store._client 在用户"启动→直接清空"的路径上是 None,
            # 后续 None.delete_collection() 会抛 AttributeError 被静默吞掉
            try:
                client = self.store._get_client()
            except Exception as e:
                logger.error(f"[CLEAR] Failed to init ChromaDB client: {e}")
                self._retrievers.clear()
                self.store._collections.clear()
                return

            try:
                existing = client.list_collections()
            except Exception as e:
                logger.error(f"[CLEAR] Failed to list collections: {e}")
                existing = []

            deleted, failed = 0, 0
            for coll in existing:
                name = coll.name if hasattr(coll, "name") else str(coll)
                try:
                    client.delete_collection(name=name)
                    deleted += 1
                    logger.info(f"[CLEAR] Deleted collection: {name}")
                except Exception as e:
                    failed += 1
                    # 不再静默: 失败要让用户能从日志里看到
                    logger.warning(f"[CLEAR] Failed to delete collection {name}: {e}")
                self.store._collections.pop(name, None)
            logger.info(f"[CLEAR] All-clear done: {deleted} deleted, {failed} failed")
            self._retrievers.clear()
            self.store._collections.clear()
