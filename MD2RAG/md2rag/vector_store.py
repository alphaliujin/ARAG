"""向量数据库存储模块 - 基于 ChromaDB."""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from md2rag.embedder import Embedder, create_embedder
from md2rag.loader import VALID_CLASSIFICATIONS
from md2rag.logger import get_logger, log_step, log_timing, log_memory

logger = get_logger("md2rag.vector_store")


class VectorStore:
    """向量数据库封装，管理 ChromaDB 集合."""

    def __init__(
        self,
        db_dir: str | Path,
        embedder: Optional[Embedder] = None,
        collection_prefix: str = "md2rag",
    ):
        log_step(logger, "INIT", "Initializing VectorStore...")
        self.db_dir = Path(db_dir)
        self.db_dir.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or create_embedder()
        self.collection_prefix = collection_prefix

        self._client: Optional[chromadb.Client] = None
        self._collections: dict[str, Any] = {}
        self._collection_lock = threading.Lock()  # 保护 _collections dict 的并发读写
        # 进程退出时清理 ChromaDB 客户端引用，避免与 backend 进程锁竞争。
        # 注意: close() 时反注册, 防止 release_after=True 反复重建 VectorStore 后,
        # 旧实例被 atexit 全局列表永久持有, GC 永远收不掉 (内存泄漏)。
        atexit.register(self._close)

        logger.info(f"[CONFIG] db_dir: {self.db_dir}")
        logger.info(f"[CONFIG] collection_prefix: {self.collection_prefix}")
        logger.info(f"[CONFIG] embedder_dimension: {self.embedder.dimension}")

    def _close(self):
        try:
            with self._collection_lock:
                self._collections.clear()
                self._client = None
        except Exception:
            pass

    def close(self):
        """显式释放 ChromaDB client/collection 引用 + 反注册 atexit."""
        self._close()
        try:
            atexit.unregister(self._close)
        except Exception:
            pass

    def _get_client(self) -> chromadb.Client:
        if self._client is None:
            log_step(logger, "CHROMADB", "Creating PersistentClient...")
            os.makedirs(self.db_dir, exist_ok=True)
            client_start = time.time()
            self._client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            client_elapsed = (time.time() - client_start) * 1000
            log_timing(logger, "ChromaDB client creation", client_elapsed)
            logger.info(f"[CHROMADB] Client ready, db_path: {self.db_dir}")
        return self._client

    def _collection_name(self, classification: str) -> str:
        """生成分类对应的集合名称."""
        return f"{self.collection_prefix}_{classification}"

    def get_or_create_collection(self, classification: str):
        """获取或创建分类集合（按密级，自动加 prefix）."""
        with self._collection_lock:
            if classification in self._collections:
                logger.debug(f"[CACHE] Returning cached collection: {classification}")
                return self._collections[classification]

            client = self._get_client()
            name = self._collection_name(classification)
            log_step(logger, "GET_COLLECTION", f"Getting or creating collection: {name}")

            collection_start = time.time()
            # 记录 embedder 信息到 collection metadata,防止跨模型切换后
            # ChromaDB 返回旧的 collection 且维度不匹配。
            embed_dim = self.embedder.dimension
            embed_model_name = getattr(self.embedder, "model", type(self.embedder).__name__)
            collection = client.get_or_create_collection(
                name=name,
                metadata={
                    "description": f"MD2RAG {classification} collection",
                    "classification": classification,
                    "embedder_model": embed_model_name,
                    "embedder_dimension": str(embed_dim),
                },
            )
            # 校验: 若已存在 collection 且维度与当前 embedder 不一致,发出严重警告
            existing_meta = getattr(collection, "metadata", None) or {}
            existing_dim = existing_meta.get("embedder_dimension")
            if existing_dim and str(existing_dim) != str(embed_dim):
                logger.error(
                    f"[COLLECTION MISMATCH] Collection '{name}' was created with "
                    f"dimension {existing_dim} but current embedder produces {embed_dim}. "
                    f"Please reset the collection or switch back to the original embedder. "
                    f"Continuing will cause ChromaDB add failures."
                )
            collection_elapsed = (time.time() - collection_start) * 1000
            log_timing(logger, f"Get/create collection {name}", collection_elapsed)

            self._collections[classification] = collection
            logger.info(f"[COLLECTION] Ready: {name}")
            return collection

    def _get_or_create_collection_by_name(self, name: str):
        """按完整名称获取/创建 collection（不自动加 prefix）.

        parent/child/images 等真实数据集合都走这里。此前不记录 embedder 元数据
        也不校验维度 -> 跨模型复用同库无任何告警, 冲突只在单文件报错, 根因不可见。
        现与 get_or_create_collection 对齐: 写入 embedder_model/dimension 并校验。
        """
        with self._collection_lock:
            if name in self._collections:
                return self._collections[name]

            client = self._get_client()
            log_step(logger, "GET_COLLECTION", f"Getting or creating collection: {name}")

            collection_start = time.time()
            embed_dim = self.embedder.dimension
            embed_model_name = getattr(self.embedder, "model", type(self.embedder).__name__)
            collection = client.get_or_create_collection(
                name=name,
                metadata={
                    "description": f"MD2RAG {name} collection",
                    "classification": name,
                    "embedder_model": embed_model_name,
                    "embedder_dimension": str(embed_dim),
                },
            )
            # 校验: 若已存在 collection 且维度与当前 embedder 不一致, 发出严重警告
            existing_meta = getattr(collection, "metadata", None) or {}
            existing_dim = existing_meta.get("embedder_dimension")
            if existing_dim and str(existing_dim) != str(embed_dim):
                logger.error(
                    f"[COLLECTION MISMATCH] Collection '{name}' was created with "
                    f"dimension {existing_dim} but current embedder produces {embed_dim}. "
                    f"Please reset the collection or switch back to the original embedder. "
                    f"Continuing will cause ChromaDB add failures."
                )
            collection_elapsed = (time.time() - collection_start) * 1000
            log_timing(logger, f"Get/create collection {name}", collection_elapsed)

            self._collections[name] = collection
            logger.info(f"[COLLECTION] Ready: {name}")
            return collection

    def add_chunks(
        self,
        classification: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        embeddings: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """向指定分类集合添加文档.

        Args:
            classification: 密级分类 (public/restricted/confidential)
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            embeddings: 预计算的嵌入向量列表，为 None 时自动生成

        Returns:
            操作结果
        """
        if not texts:
            logger.warning("[SKIP] No texts to add")
            return {"status": "empty", "added": 0}

        log_step(logger, "ADD_CHUNKS", f"Adding {len(texts)} chunks to {classification}")
        add_start = time.time()

        collection = self.get_or_create_collection(classification)

        # 生成嵌入向量（如果未提供预计算的 embeddings）
        if embeddings is None:
            log_step(logger, "EMBED", f"Generating embeddings for {len(texts)} texts...")
            embed_start = time.time()
            embeddings = self.embedder.embed(texts)
            embed_elapsed = (time.time() - embed_start) * 1000
            log_timing(logger, f"Embed {len(texts)} texts", embed_elapsed)
            logger.info(f"[EMBED] Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}")
        else:
            logger.info(f"[EMBED] Using pre-computed embeddings ({len(embeddings)} vectors, dim={len(embeddings[0]) if embeddings else 0})")

        # 添加到集合
        log_step(logger, "COLLECTION_ADD", f"Adding to ChromaDB collection...")
        chroma_start = time.time()
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        chroma_elapsed = (time.time() - chroma_start) * 1000
        log_timing(logger, f"ChromaDB add {len(texts)} documents", chroma_elapsed)

        add_elapsed = (time.time() - add_start) * 1000
        log_timing(logger, f"Total add_chunks for {classification}", add_elapsed)
        log_step(logger, "ADD_CHUNKS_SUCCESS", f"Added {len(texts)} chunks to {classification}")

        return {
            "status": "success",
            "added": len(texts),
            "collection": self._collection_name(classification),
        }

    def search(
        self,
        classification: str,
        query_texts: list[str],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict[str, Any]:
        """在指定分类中搜索相似文档.

        Args:
            classification: 密级分类
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 过滤条件

        Returns:
            搜索结果
        """
        log_step(logger, "SEARCH", f"Searching in {classification}, n_results={n_results}")
        search_start = time.time()

        # 防御空查询: 否则 Ollama/SBERT 会被请求嵌入 "" 并返回零/垃圾向量,
        # 检索结果毫无意义。直接返回空。
        non_empty_queries = [q for q in query_texts if q and q.strip()]
        if not non_empty_queries:
            logger.warning("[SEARCH] All query texts are empty after stripping; returning no results")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        query_texts = non_empty_queries

        collection = self.get_or_create_collection(classification)

        # 生成查询向量
        log_step(logger, "EMBED_QUERY", f"Embedding {len(query_texts)} query texts...")
        embed_start = time.time()
        query_embeddings = self.embedder.embed(query_texts)
        embed_elapsed = (time.time() - embed_start) * 1000
        log_timing(logger, f"Embed {len(query_texts)} queries", embed_elapsed)

        count = collection.count()
        logger.info(f"[SEARCH] Collection {classification} has {count} documents")

        if count == 0:
            logger.warning(f"[SEARCH] Collection {classification} is empty")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        actual_n = min(n_results, count)
        logger.info(f"[SEARCH] Querying for top {actual_n} results...")

        query_start = time.time()
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        query_elapsed = (time.time() - query_start) * 1000
        log_timing(logger, f"ChromaDB query", query_elapsed)

        search_elapsed = (time.time() - search_start) * 1000
        log_timing(logger, f"Total search in {classification}", search_elapsed)

        return results

    def _search_with_embeddings(
        self,
        classification: str,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict[str, Any]:
        """用预计算的 embeddings 搜索（避免重复嵌入同一查询）"""
        if not query_embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        collection = self.get_or_create_collection(classification)
        count = collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_n = min(n_results, count)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def search_all(
        self,
        query_text: str,
        classifications: Optional[list[str]] = None,
        n_results: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """在多个分类中搜索.

        Args:
            query_text: 查询文本
            classifications: 要搜索的分类列表，None 表示全部
            n_results: 每个分类返回结果数

        Returns:
            按分类组织的搜索结果
        """
        if classifications is None:
            classifications = list(VALID_CLASSIFICATIONS)

        # 空查询保护: 空白串经 embed 得零向量, ChromaDB query 会报错或返回
        # 无意义的"距零向量最近"结果。与 search() / ParentChildRetriever.search 一致。
        if not query_text or not query_text.strip():
            log_step(logger, "SEARCH_ALL", "Empty query, returning empty results")
            return {cls: [] for cls in classifications}

        log_step(logger, "SEARCH_ALL", f"Query='{query_text}', classifications={classifications}")
        search_start = time.time()

        # 一次嵌入，所有分类共用 query embedding
        log_step(logger, "EMBED_QUERY", f"Embedding 1 query for {len(classifications)} collections...")
        embed_start = time.time()
        query_embeddings = self.embedder.embed([query_text])
        query_emb = query_embeddings[0] if query_embeddings else []
        embed_elapsed = (time.time() - embed_start) * 1000
        log_timing(logger, f"Embed 1 query (shared across {len(classifications)} collections)", embed_elapsed)

        all_results = {}
        for classification in classifications:
            results = self._search_with_embeddings(classification, [query_emb], n_results)
            docs = results.get("documents", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or []
            dists = results.get("distances", [[]])[0] or []

            items = []
            for i, doc in enumerate(docs):
                distance = dists[i] if i < len(dists) else 0
                # ChromaDB 默认使用 L2 squared distance; 归一化向量下 L2² = 2(1 - cos_sim),
                # 因此 similarity = 1 - distance/2.0 (与 parent_child_retriever.py 一致)
                similarity = max(0.0, min(1.0, 1.0 - distance / 2.0))
                items.append({
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "similarity": similarity,
                })

            all_results[classification] = items
            logger.info(f"[SEARCH_ALL] {classification}: {len(items)} results")

        search_elapsed = (time.time() - search_start) * 1000
        log_timing(logger, f"Total search_all", search_elapsed)

        return all_results

    def get_collection_stats(self) -> dict[str, int]:
        """获取各集合的文档数量."""
        logger.debug("[STATS] Getting collection stats...")
        stats = {}
        for classification in VALID_CLASSIFICATIONS:
            try:
                collection = self.get_or_create_collection(classification)
                count = collection.count()
                stats[classification] = count
                logger.debug(f"[STATS] {classification}: {count} documents")
            except Exception as e:
                logger.warning(f"[STATS] Failed to get count for {classification}: {e}")
                stats[classification] = 0
        return stats

    def clear_collection(self, classification: str):
        """清空指定分类的集合 — 通过列出现有 collection 精确匹配,
        而不是盲目尝试两种名称然后吞掉异常(那样 caller 无从得知是否真的删了)。
        """
        log_step(logger, "CLEAR_COLLECTION", f"Clearing {classification}...")
        with self._collection_lock:
            self._collections.pop(classification, None)
            # 再尝试用完整 prefix 名清缓存
            name_with_prefix = self._collection_name(classification)
            self._collections.pop(name_with_prefix, None)

            client = self._get_client()
            # 列出真实存在的 collection,匹配两种可能命名
            try:
                existing_names = {c.name for c in client.list_collections()}
            except Exception as e:
                logger.warning(f"[CLEAR] list_collections failed: {e}")
                existing_names = set()

            candidates = [n for n in (classification, name_with_prefix) if n in existing_names]
            if not candidates:
                logger.warning(
                    f"[CLEAR] No matching collection for '{classification}' "
                    f"(tried '{classification}' and '{name_with_prefix}'); nothing to delete"
                )
                return

            for name in candidates:
                try:
                    client.delete_collection(name=name)
                    logger.info(f"[CLEAR] Deleted collection: {name}")
                except Exception as e:
                    logger.error(f"[CLEAR] Failed to delete '{name}': {e}")

    def reset_all(self):
        """重置所有集合.

        枚举 ChromaDB 实际存在的全部 collection 并逐一删除, 而不是只删 3 个
        基础 collection (md2rag_<cls>)。真实数据存放在 md2rag_<cls>_parent /
        _child / _images 中, 旧实现漏删它们会导致 switch_embedder 切换模型后
        旧向量残留 -> 维度冲突或异模型向量混入同一 collection, 检索全部错乱。
        与 Indexer.clear() 的全清枚举逻辑保持一致。
        """
        log_step(logger, "RESET_ALL", "Resetting all collections...")
        try:
            client = self._get_client()
        except Exception as e:
            logger.error(f"[RESET] Failed to init ChromaDB client: {e}")
            with self._collection_lock:
                self._collections.clear()
            return

        try:
            existing = client.list_collections()
        except Exception as e:
            logger.error(f"[RESET] Failed to list collections: {e}")
            existing = []

        deleted, failed = 0, 0
        for coll in existing:
            name = coll.name if hasattr(coll, "name") else str(coll)
            try:
                client.delete_collection(name=name)
                deleted += 1
                logger.info(f"[RESET] Deleted collection: {name}")
            except Exception as e:
                failed += 1
                logger.warning(f"[RESET] Failed to delete collection {name}: {e}")

        with self._collection_lock:
            self._collections.clear()
        log_step(logger, "RESET_ALL_COMPLETE", f"All collections reset: {deleted} deleted, {failed} failed")
