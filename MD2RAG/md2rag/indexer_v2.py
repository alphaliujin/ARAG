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

V2 优化：
- 分批入库，避免内存溢出
- 流式处理，及时释放内存
- 细粒度进度回调
"""

from __future__ import annotations

import gc
import hashlib
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from md2rag.bbox_extractor import build_line_index, get_bboxes_for_record
from md2rag.config import MD2RAGConfig, load_config
from md2rag.embedder import Embedder, create_embedder
from md2rag.embedding_cache import IndexManifest
from md2rag.image_processor import ImageProcessor, ViTImageEmbedder
from md2rag.llm_summary import LLMClient, extract_abstract
from md2rag.loader import (
    CLASSIFICATION_DIR_MAP,
    VALID_CLASSIFICATIONS,
    ChunkLoader,
    ChunkRecord,
)
from md2rag.logger import get_logger, log_step, log_timing, log_memory, setup_exception_logging
from md2rag.memory_monitor import MemoryMonitor, memory_checkpoint, force_gc
from md2rag.parent_child_retriever import ParentChildRetriever
from md2rag.vector_store import VectorStore

# 设置全局异常捕获
setup_exception_logging()

logger = get_logger("md2rag.indexer")


# ------------------------------------------------------------------------
# 配置常量
# ------------------------------------------------------------------------

# 分批入库配置
# BATCH_SIZE_FILES 控制外层文件批 — 单文件已经是流式入库, 这里只决定多久打一次
# RSS 日志/清理. 设小一点有助于在大文件密集时及时回收.
BATCH_SIZE_FILES = 5
BATCH_SIZE_EMBED = 128         # 每批嵌入文本数 - 2026-07-23 实测大批更快 (见 embedder._call_batch 注释)
BATCH_SIZE_IMAGES = 8          # 图片单批数 — ViT-Large peak 内存大, 不宜过大
MEMORY_CLEANUP_INTERVAL = 1    # 每批文件都清一次 (旧值 5, 大文件下太稀疏)


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
# 主索引器
# ------------------------------------------------------------------------

class Indexer:
    """MD 切片索引器 - 借鉴 bisheng 完整入库流程.

    责任链:
      X2MD .parents/.children.json  →  ChunkLoader  →  LLM摘要  →
      bbox 关联  →  Embedder  →  ParentChildRetriever (双 collection)  →
      VectorStore (ChromaDB)

    V2 优化:
      - 分批入库，避免内存溢出
      - 流式处理，及时释放内存
      - 细粒度进度回调
    """

    def __init__(self, config: Optional[MD2RAGConfig] = None, enable_memory_monitor: bool = False):
        log_step(logger, "INIT", "Initializing Indexer...")
        init_start = time.time()

        self.config = config or MD2RAGConfig()
        self.enable_memory_monitor = enable_memory_monitor
        self._memory_monitor: Optional[MemoryMonitor] = None
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

        # 增量索引清单: 记录已成功 embed+upsert 的 chunk 文件 (path+content_hash)。
        # 重跑时跳过未变文件, 省 embed + ChromaDB upsert; 清库时联动 clear。
        self._manifest = IndexManifest(self.config.vector_db_dir / "md2rag_index_manifest.sqlite")

        # Per-classification 父子检索器缓存
        self._retrievers: Dict[str, ParentChildRetriever] = {}

        # LLM 客户端（可选，延迟初始化）
        self._llm_client: Optional[LLMClient] = None

        elapsed_ms = (time.time() - init_start) * 1000
        log_timing(logger, "Indexer initialization", elapsed_ms)
        log_step(logger, "INIT_COMPLETE", "Indexer ready")

        # 启动内存监控
        if self.enable_memory_monitor:
            self._memory_monitor = MemoryMonitor(sample_interval=2.0)
            self._memory_monitor.start()
            logger.info("[MEMORY] Memory monitoring enabled")

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
                ollama_cache_path=str(self.config.vector_db_dir / "md2rag_embedding_cache.sqlite"),
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
            ollama_cache_path=str(self.config.vector_db_dir / "md2rag_embedding_cache.sqlite") if model_type == "ollama" else None,
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
        self._manifest.clear()  # 换了 embedder, "已索引"记录失效, 下次重嵌 (同维度换模型也必须重嵌, 否则留旧模型向量)
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
    # 内存管理工具
    # ------------------------------------------------------------------

    def _cleanup_memory(self) -> None:
        """清理内存缓存."""
        # 记录检查点（如果监控启用）
        if self._memory_monitor:
            self._memory_monitor.checkpoint("before_cleanup")

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

        if self._memory_monitor:
            self._memory_monitor.checkpoint("after_cleanup")

    def _drop_classification_caches(self, classification: str) -> None:
        """单密级入库完成后, 把该密级相关的 collection/retriever 句柄从缓存里丢掉.

        VectorStore._collections / Indexer._retrievers 缓存的 ChromaDB Collection
        对象内部带 HNSW writer 引用 (~MB 级常驻); ingest_all_levels 跨 3 个密级
        共享同一 indexer 时, 不清理会让 9 个 collection (3 parent + 3 child + 3 images)
        + 3 个 retriever 全程驻留, 显著抬高 RSS。
        """
        try:
            self._retrievers.pop(classification, None)
            # 与 ParentChildRetriever / vector_store 命名约定保持一致
            prefix = self.config.collection_prefix
            keys_to_drop = [
                classification,
                f"{prefix}_{classification}",
                f"{prefix}_{classification}_parent",
                f"{prefix}_{classification}_child",
                f"{classification}_images",
            ]
            for k in keys_to_drop:
                try:
                    self.store._collections.pop(k, None)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[CLEANUP] drop classification caches failed: {e}")

    @staticmethod
    def _log_rss(tag: str) -> None:
        """轻量 RSS 打印 — 用于排查内存累积. psutil 不在时 silent skip."""
        try:
            import psutil, os
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info(f"[RSS] {tag}: {rss_mb:.1f} MB")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 核心：索引目录 (V2 分批优化版)
    # ------------------------------------------------------------------

    def index_directory(
        self,
        classification: Optional[str] = None,
        strategy: str = "auto",
        regenerate_abstract: bool = False,
        include_images: bool = True,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        cancel_event=None,
    ) -> IndexResult:
        """索引 MD 目录下所有切片文件 (V2 分批优化版).

        Args:
            classification: 指定密级，None 时扫描所有
            strategy: 切片策略 auto/chunk/parent-child
            regenerate_abstract: 是否用 LLM 重新生成 abstract
            include_images: 是否同时处理图片（ViT-Large）
            progress_callback: 进度回调函数，接收 {"phase": ..., "progress": ..., "message": ...}

        Returns:
            IndexResult
        """
        logger.info("=" * 80)
        log_step(logger, "INDEX_DIRECTORY", f"classification={classification}, strategy={strategy}")
        logger.info("=" * 80)
        start = time.time()

        result = IndexResult()

        def report_progress(phase: str, progress: float, message: str):
            """统一的进度报告函数."""
            if progress_callback:
                progress_callback({
                    "phase": phase,
                    "progress": progress,
                    "message": message,
                })
            logger.info(f"[PROGRESS] {phase}: {progress*100:.1f}% - {message}")

        try:
            # 1. 发现文件
            report_progress("discovering", 0.0, "正在扫描文件...")
            chunk_files = self.loader.discover_files(classification)
            if not chunk_files:
                result.status = "empty"
                result.message = "No chunk files found"
                logger.warning(result.message)
                return result

            total_files = len(chunk_files)
            logger.info(f"[FILES] Discovered {total_files} chunk files")
            report_progress("discovering", 1.0, f"发现 {total_files} 个文件")

            # 2. 按密级分组（只做路径分组，不加载内容）
            by_class: Dict[str, List[Path]] = {}
            for fp in chunk_files:
                cls = classification or self.loader.get_classification_from_path(fp)
                by_class.setdefault(cls, []).append(fp)

            # 3. 按密级入库
            total_classifications = len(by_class)
            for cls_idx, (cls, files) in enumerate(by_class.items()):
                # 协作式取消检查: 在每个密级开始前检查
                if cancel_event and cancel_event.is_set():
                    result.status = "cancelled"
                    result.message = "索引操作被用户取消"
                    logger.info("[CANCEL] Indexing cancelled by user")
                    return result

                cls_progress_base = cls_idx / total_classifications
                cls_progress_range = 1.0 / total_classifications

                logger.info(f"[INDEX] Processing classification: {cls} ({len(files)} files)")
                report_progress(
                    "indexing",
                    cls_progress_base,
                    f"开始处理 {cls} 密级 ({len(files)} 个文件)"
                )

                try:
                    # ★ 分批入库
                    parents, children = self._index_classification_batched(
                        cls,
                        files,
                        regenerate_abstract,
                        progress_callback=lambda p: report_progress(
                            "indexing",
                            cls_progress_base + p["progress"] * cls_progress_range,
                            p["message"]
                        ),
                    )
                    result.parents_added += parents
                    result.children_added += children
                    result.documents_processed += len(files)
                    logger.info(f"[INDEX] {cls}: +{parents} parents, +{children} children")

                    # 每个密级处理完后清理内存:
                    # 1) 丢弃本密级的 retriever/collection 句柄 (防 3 密级累积)
                    # 2) 清 Ollama embedder LRU cache (避免单调增长)
                    # 3) gc + torch 缓存
                    self._drop_classification_caches(cls)
                    try:
                        if hasattr(self.embedder, "_cache"):
                            self.embedder._cache.clear()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._cleanup_memory()

                except Exception as e:
                    logger.critical(f"[CRITICAL] Failed to index classification {cls}: {e}", exc_info=True)
                    result.errors.append(f"classification {cls}: {e}")

            # 4. 处理图片
            if include_images and self.config.vit_enabled:
                report_progress("images", 0.9, "开始处理图片...")
                try:
                    images_count = self._index_images_batched(
                        classification,
                        progress_callback=lambda p: report_progress("images", 0.9 + p["progress"] * 0.1, p["message"])
                    )
                    result.images_added = images_count
                    logger.info(f"[IMAGES] Indexed {images_count} images")
                except Exception as e:
                    logger.error(f"[IMAGES] Failed: {e}", exc_info=True)
                    result.errors.append(f"images: {e}")

                # 图片处理完后清理内存
                self._cleanup_memory()

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
        report_progress("complete", 1.0, result.message)
        return result

    def _index_classification_batched(
        self,
        classification: str,
        files: List[Path],
        regenerate_abstract: bool = False,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Tuple[int, int]:
        """分批入库单个密级下的所有切片 (V2 内存优化版).

        关键优化:
        1. 按文件分批处理，每批处理完后立即入库并释放内存
        2. 嵌入也分批进行，避免一次性生成大量向量
        3. 定期执行内存清理
        """
        log_step(logger, "INDEX_CLASS", f"class={classification}, files={len(files)}")
        retriever = self._get_retriever(classification)

        parents_total = 0
        children_total = 0
        total_files = len(files)

        def report_progress(progress: float, message: str):
            if progress_callback:
                progress_callback({"progress": progress, "message": message})

        # ★ 分批处理文件
        for batch_start in range(0, total_files, BATCH_SIZE_FILES):
            batch_end = min(batch_start + BATCH_SIZE_FILES, total_files)
            batch_files = files[batch_start:batch_end]
            batch_idx = batch_start // BATCH_SIZE_FILES + 1
            total_batches = (total_files + BATCH_SIZE_FILES - 1) // BATCH_SIZE_FILES

            report_progress(
                batch_start / total_files,
                f"处理第 {batch_idx}/{total_batches} 批文件 ({batch_start+1}-{batch_end}/{total_files})"
            )

            # ★ 加载当前批次的文件（而非全部文件）
            batch_records: List[Tuple[Path, List[ChunkRecord]]] = []
            skipped_in_batch = 0
            for fp in batch_files:
                # 增量跳过: 内容未变且非重生成摘要 -> 直接跳过 (省 load + embed + upsert)
                if not regenerate_abstract:
                    try:
                        fp_hash = IndexManifest.file_hash(fp)
                        if self._manifest.is_done(str(fp), fp_hash):
                            skipped_in_batch += 1
                            continue
                    except Exception as e:
                        logger.warning(f"[SKIP_CHECK] hash/manifest check failed for {fp}: {e}")
                try:
                    records = self.loader.load_file(fp)
                    if records:
                        batch_records.append((fp, records))
                except Exception as e:
                    logger.error(f"[LOAD_ERROR] Failed to load {fp}: {e}")
            if skipped_in_batch:
                logger.info(f"[INDEX] batch{batch_idx}/{total_batches}: skipped {skipped_in_batch} unchanged file(s)")

            if not batch_records:
                continue

            # 处理当前批次
            parents, children = self._process_batch(
                classification,
                batch_records,
                retriever,
                regenerate_abstract,
            )
            parents_total += parents
            children_total += children

            # ★ 每批处理完后立即清理内存
            del batch_records
            if (batch_idx % MEMORY_CLEANUP_INTERVAL == 0) or (batch_end == total_files):
                self._cleanup_memory()
                self._log_rss(f"{classification} batch{batch_idx}/{total_batches} done "
                              f"(+{parents_total}p +{children_total}c)")

        report_progress(1.0, f"完成 {classification} 密级入库")
        logger.info(f"[INDEX_CLASS_COMPLETE] {classification}: {parents_total} parents, {children_total} children")
        return parents_total, children_total

    def _process_batch(
        self,
        classification: str,
        batch_records: List[Tuple[Path, List[ChunkRecord]]],
        retriever: ParentChildRetriever,
        regenerate_abstract: bool = False,
    ) -> Tuple[int, int]:
        """处理单个文件批次 — 按文件级别立即流式入库, 不再跨文件累积.

        ★ 关键修复 (内存): 单文件 sub-batch 化
        - 历史上本方法把 BATCH_SIZE_FILES=20 个文件的所有 chunk 全部塞进 embed_batch
          后再分 sub-batch 嵌入。当其中一个文件是大文件 (例: 6MB children.json,
          ~5000-10000 chunks), 整批 chunk 文本会全部驻留到本批结束, peak 内存
          可达数百 MB; 长跑入库到几千条就 OOM。
        - 现在: 每个文件独立调用 _ingest_file_chunks, 解析 → sub-batch 嵌入 → 入库
          → 释放, 全程占用恒定于"单 sub-batch 的 embedding 大小" (BATCH_SIZE_EMBED
          × 1024 dim ≈ 200KB), 与文件大小完全无关。
        """
        parents_total = 0
        children_total = 0

        for file_path, records in batch_records:
            try:
                p, c = self._ingest_file_chunks(
                    classification, file_path, records, retriever, regenerate_abstract
                )
                parents_total += p
                children_total += c
                # 成功 (embed+upsert 全过) 才记 manifest -> 下次跳过; 异常不记 -> 重试
                try:
                    self._manifest.mark_done(str(file_path), IndexManifest.file_hash(file_path), classification)
                except Exception as me:
                    logger.warning(f"[MANIFEST] mark_done failed for {file_path}: {me}")
            except Exception as e:
                logger.error(f"[INDEX] Failed processing file {file_path}: {e}", exc_info=True)
            finally:
                # 文件级释放: ChunkRecord 列表 (含 text/abstract/raw_metadata 整片) 立即丢弃
                records.clear() if isinstance(records, list) else None

        return parents_total, children_total

    def _ingest_file_chunks(
        self,
        classification: str,
        file_path: Path,
        records: List[ChunkRecord],
        retriever: ParentChildRetriever,
        regenerate_abstract: bool,
    ) -> Tuple[int, int]:
        """单文件: 解析 → sub-batch 嵌入 → 入库 → 释放. 全程恒定内存."""
        # 分类：parent / child / 普通 chunk
        parents = [r for r in records if r.is_parent]
        children = [r for r in records if r.is_child]
        chunks = [r for r in records if not r.is_parent and not r.is_child]

        # 读取 MD 原文用于 bbox 合成 — 仅本文件作用域, 出函数立即释放
        md_path = self._md_path_from_chunk_path(Path(file_path))
        source_text = ""
        if md_path and md_path.exists():
            try:
                source_text = md_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"[INDEX] Failed to read MD {md_path}: {e}")

        # 行级索引每文件构建一次, 复用给本文件所有 parent/child,
        # 避免旧实现每条 chunk 都 make_synthetic_bboxes + 线性扫描 (大文件 CPU 瓶颈).
        line_index = build_line_index(source_text) if source_text else None

        # 构造轻量 embed items (不持 ChunkRecord 引用)
        embed_items: List[Dict[str, Any]] = []

        for p in parents:
            if regenerate_abstract and self._llm_client:
                new_abs = extract_abstract(self._llm_client, p.text)
                if new_abs:
                    p.abstract = new_abs
            if not p.bbox and source_text:
                bboxes = get_bboxes_for_record(p.text, source_text, line_index=line_index)
                if bboxes:
                    p.bbox = str(bboxes)
            meta = p.to_metadata(classification, str(file_path))
            doc_id = p.doc_id or str(uuid.uuid4())
            meta["doc_id"] = doc_id
            meta["parent_doc_id"] = doc_id
            embed_items.append({
                "embedding_text": p.embedding_text,
                "meta": meta,
                "id": doc_id,
                "type": "parent",
            })

        for c in children:
            if regenerate_abstract and self._llm_client and not c.abstract:
                new_abs = extract_abstract(self._llm_client, c.text)
                if new_abs:
                    c.abstract = new_abs
            if not c.bbox and source_text:
                bboxes = get_bboxes_for_record(c.text, source_text, line_index=line_index)
                if bboxes:
                    c.bbox = str(bboxes)
            meta = c.to_metadata(classification, str(file_path))
            meta["parent_doc_id"] = c.parent_doc_id or c.doc_id
            meta["is_child"] = True
            child_id = f"{meta['parent_doc_id']}_child_{c.chunk_index}"
            embed_items.append({
                "embedding_text": c.embedding_text,
                "meta": meta,
                "id": child_id,
                "type": "child",
            })

        for ch in chunks:
            meta = ch.to_metadata(classification, str(file_path))
            doc_id = ch.doc_id or str(uuid.uuid4())
            meta["doc_id"] = doc_id
            meta["parent_doc_id"] = doc_id
            meta["is_parent"] = True
            meta["is_child"] = False
            embed_items.append({
                "embedding_text": ch.embedding_text,
                "meta": meta,
                "id": doc_id,
                "type": "chunk",
            })

        # 释放 ChunkRecord 列表 + source_text — bbox/abstract 已经物化到 meta 里
        parents.clear()
        children.clear()
        chunks.clear()
        source_text = ""

        if not embed_items:
            return 0, 0

        parents_total = 0
        children_total = 0
        total = len(embed_items)

        # ★ sub-batch 流式: 即使本文件有 5000 chunk, peak 也只是单 sub-batch
        for embed_start in range(0, total, BATCH_SIZE_EMBED):
            embed_end = min(embed_start + BATCH_SIZE_EMBED, total)
            batch_items = embed_items[embed_start:embed_end]
            batch_texts = [it["embedding_text"] for it in batch_items]

            try:
                embeddings = self.embedder.embed(batch_texts)
                if not embeddings or len(embeddings) != len(batch_texts):
                    logger.error(
                        f"[INDEX] Embed mismatch in {Path(file_path).name} "
                        f"[{embed_start}-{embed_end}]: expected {len(batch_texts)}, "
                        f"got {len(embeddings) if embeddings else 0}"
                    )
                    # 已处理槽位置 None, 进入下一批
                    for j in range(embed_start, embed_end):
                        embed_items[j] = None  # type: ignore
                    batch_texts = None
                    continue
            except Exception as e:
                logger.error(f"[INDEX] Embed failed in {Path(file_path).name} "
                             f"[{embed_start}-{embed_end}]: {e}")
                for j in range(embed_start, embed_end):
                    embed_items[j] = None  # type: ignore
                batch_texts = None
                continue

            # 入库 — 全部局部, 出循环即释放
            parent_chunks: List[Tuple[str, Dict[str, Any]]] = []
            child_chunks: List[Tuple[str, Dict[str, Any]]] = []
            parent_embeddings: List[List[float]] = []
            child_embeddings: List[List[float]] = []
            parent_ids: List[str] = []
            child_ids: List[str] = []

            for i, item in enumerate(batch_items):
                emb = embeddings[i]
                if not emb:
                    continue
                if item["type"] == "parent" or item["type"] == "chunk":
                    parent_chunks.append((item["embedding_text"], item["meta"]))
                    parent_embeddings.append(emb)
                    parent_ids.append(item["id"])
                    parents_total += 1
                elif item["type"] == "child":
                    child_chunks.append((item["embedding_text"], item["meta"]))
                    child_embeddings.append(emb)
                    child_ids.append(item["id"])
                    children_total += 1

            if parent_chunks:
                try:
                    retriever.add_parent_chunks(parent_chunks, embeddings=parent_embeddings, ids=parent_ids)
                except Exception as e:
                    logger.error(f"[INDEX] add_parent_chunks failed: {e}")
            if child_chunks:
                try:
                    retriever.add_child_chunks(child_chunks, embeddings=child_embeddings, ids=child_ids)
                except Exception as e:
                    logger.error(f"[INDEX] add_child_chunks failed: {e}")

            # 本 sub-batch 的所有引用归零, 加 del 帮 CPython 即时 refcount=0
            for j in range(embed_start, embed_end):
                embed_items[j] = None  # type: ignore
            del parent_chunks, child_chunks, parent_embeddings, child_embeddings
            del parent_ids, child_ids, embeddings, batch_texts, batch_items

        # 文件结束: embed_items 已全 None, 显式清掉 list 本身
        embed_items.clear()
        return parents_total, children_total

    def _md_path_from_chunk_path(self, chunk_path: Path) -> Optional[Path]:
        """从 .parents.json / .children.json / .chunks.json 反推 MD 原文路径."""
        if chunk_path.suffixes[-2] in (".parents", ".children", ".chunks"):
            stem = chunk_path.name.split(".")[0]  # "银渐层"
            return chunk_path.parent / f"{stem}.md"
        return None

    # ------------------------------------------------------------------
    # 图片索引 (V2 分批优化版)
    # ------------------------------------------------------------------

    def _index_images_batched(
        self,
        classification: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> int:
        """用 ViT-Large 处理 MD 目录下的图片 (V2 分批优化版).

        ★ 内存优化: discover_images 是 generator, 用 islice 分批拉取,
        避免 list(...) 把所有 ImageRecord 一次性物化到内存
        (大目录下可能几千上万条, metadata + path 累积可观)。
        """
        from itertools import islice

        log_step(logger, "INDEX_IMAGES", f"Processing images (classification={classification})")
        start = time.time()

        def report_progress(progress: float, message: str):
            if progress_callback:
                progress_callback({"progress": progress, "message": message})

        vit_embedder = None
        processor = None
        try:
            # 创建 ViT embedder
            vit_embedder = ViTImageEmbedder(
                model_name=self.config.vit_model,
                device=self.config.vit_device,
            )
            processor = ImageProcessor(self.config.md_dir, embedder=vit_embedder)

            # 发现图片 — 保持 generator, 不要 list()
            report_progress(0.0, "扫描图片文件...")
            image_iter = processor.discover_images(classification)

            processed = 0
            batch_idx = 0
            report_progress(0.1, "开始分批处理图片")

            while True:
                # islice 流式取下一批, 处理完即被 GC
                batch = list(islice(image_iter, BATCH_SIZE_IMAGES))
                if not batch:
                    break
                batch_idx += 1

                image_paths = [r.image_path for r in batch]
                try:
                    embeddings, valid_paths = vit_embedder.embed_images(image_paths)
                except Exception as e:
                    logger.error(f"[IMAGES] Failed to embed batch {batch_idx}: {e}")
                    continue

                # 入库当前批次
                # embed_images 跳过加载失败的图片, 返回 (embeddings, valid_paths) 1:1 对齐;
                # 按路径匹配回 rec, 避免按位置错位赋值/写入错误向量。
                emb_by_path = dict(zip(valid_paths, embeddings))
                for rec in batch:
                    rec.embedding = emb_by_path.get(rec.image_path)
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
                    image_collection_key = f"{rec.classification}_images"
                    try:
                        coll = self.store.get_or_create_collection(image_collection_key)
                        coll.add(
                            documents=[rec.embedding_text],
                            embeddings=[rec.embedding],
                            metadatas=[meta],
                            ids=[rec.image_id],
                        )
                        processed += 1
                    except Exception as e:
                        logger.error(f"[IMAGES] Failed {rec.image_path.name}: {e}")

                # 清理当前批次的嵌入数据 + batch 本身
                del embeddings
                del batch

                report_progress(min(0.99, 0.1 + 0.9 * processed / max(processed + BATCH_SIZE_IMAGES, 1)),
                                f"已处理 {processed} 张图片")

                # 定期内存清理
                if batch_idx % MEMORY_CLEANUP_INTERVAL == 0:
                    self._cleanup_memory()

            elapsed_ms = (time.time() - start) * 1000
            log_timing(logger, "Index images", elapsed_ms)
            report_progress(1.0, f"完成图片入库: {processed} 张")
            return processed

        finally:
            # ★ 确保释放 ViT 模型
            if vit_embedder is not None:
                vit_embedder.release()
            processor = None
            self._cleanup_memory()

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
        """释放入库期间持有的模型、ChromaDB client 和 PyTorch 缓存."""
        try:
            self._retrievers.clear()
            self._llm_client = None
            if hasattr(self.embedder, "release"):
                self.embedder.release()
            if hasattr(self.store, "close"):
                self.store.close()
        finally:
            self._cleanup_memory()

            # 停止内存监控并生成报告
            if self._memory_monitor:
                self._memory_monitor.stop()
                self._memory_monitor.report()
                self._memory_monitor = None

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
            self._retrievers.pop(classification, None)
            self._manifest.clear(classification)
        else:
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
                    logger.warning(f"[CLEAR] Failed to delete collection {name}: {e}")
                self.store._collections.pop(name, None)
            logger.info(f"[CLEAR] All-clear done: {deleted} deleted, {failed} failed")
            self._retrievers.clear()
            self.store._collections.clear()
            self._manifest.clear()
