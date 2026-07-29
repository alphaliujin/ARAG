"""父子块检索器 - 参考 bisheng SmallerChunksVectorRetriever.

核心设计：
- parent chunks 和 child chunks 分别存储到不同 collection
  - <classification>_parent  - 父块（提供上下文）
  - <classification>_child   - 子块（精准匹配）
- 通过共享 doc_id 关联（X2MD 已经为每个 parent 生成唯一 UUID，children 共享这个 doc_id）
- 检索流程：先用 child 做相似度搜索 → 命中后用 doc_id 回查 parent

这与 bisheng 的实现思想一致，但适配我们自己的 vector_store / embedder 接口。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from md2rag.embedder import Embedder
from md2rag.logger import get_logger, log_step, log_timing
from md2rag.vector_store import VectorStore

logger = get_logger("md2rag.parent_child_retriever")


@dataclass
class RetrievalHit:
    """检索命中."""
    chunk_id: str
    text: str
    score: float  # 相似度
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_text: Optional[str] = None  # 父块文本（若有）


class ParentChildRetriever:
    """父子块双 collection 检索器.

    Args:
        vector_store: 向量存储实例
        embedder: 嵌入模型实例
        classification: 密级分类 (public/restricted/confidential)
        parent_collection_suffix: 父块 collection 后缀（默认 "parent"）
        child_collection_suffix: 子块 collection 后缀（默认 "child"）
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        classification: str,
        parent_collection_suffix: str = "parent",
        child_collection_suffix: str = "child",
        collection_prefix: str = "",
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.classification = classification
        self.parent_suffix = parent_collection_suffix
        self.child_suffix = child_collection_suffix
        self.prefix = collection_prefix

        # 完整 collection 名（处理空 prefix 的边界情况）
        if self.prefix:
            base = f"{self.prefix}_{self.classification}"
        else:
            base = self.classification
        self.parent_collection = f"{base}_{self.parent_suffix}"
        self.child_collection = f"{base}_{self.child_suffix}"

    def _parent_coll(self):
        # 直接用完整的 collection 名称，不走 vector_store 的 prefix 处理
        return self.vector_store._get_or_create_collection_by_name(self.parent_collection)

    def _child_coll(self):
        return self.vector_store._get_or_create_collection_by_name(self.child_collection)

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------

    def add_parent_chunks(
        self,
        chunks: List[Tuple[str, Dict[str, Any]]],  # (text, metadata)
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """入库父块."""
        if not chunks:
            return 0
        coll = self._parent_coll()
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        doc_ids = [m.get("doc_id") or str(uuid.uuid4()) for m in metas]
        # 把 doc_id 写回 metadata (深拷贝,避免原地修改 caller 的 dict)
        metas = [{**m, "doc_id": d, "is_parent": True} for m, d in zip(metas, doc_ids)]
        ids = ids or [d for d in doc_ids]
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        coll.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)
        return len(chunks)

    def add_child_chunks(
        self,
        chunks: List[Tuple[str, Dict[str, Any]]],  # (text, metadata) - parent_doc_id 必填
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """入库子块（每条 metadata 必须有 parent_doc_id 指向父块）."""
        if not chunks:
            return 0
        coll = self._child_coll()
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        # 确保子块有独立 id
        # 旧实现: f"{parent_doc_id or 'unknown'}_child_{chunk_index or i}"
        # 当多个文件都缺 parent_doc_id 且 chunk_index 重复时,ID 会碰撞,
        # ChromaDB upsert 会静默覆盖已存在的 child。
        # 修复: 缺关键字段时,追加 uuid4 短串保证全局唯一。
        if ids is None:
            import uuid as _uuid
            ids = []
            for i, m in enumerate(metas):
                parent_id = m.get("parent_doc_id")
                chunk_idx = m.get("chunk_index")
                if parent_id and chunk_idx is not None:
                    ids.append(f"{parent_id}_child_{chunk_idx}")
                else:
                    # 字段不全,附加随机后缀保证唯一
                    fallback_parent = parent_id or "unknown"
                    fallback_idx = chunk_idx if chunk_idx is not None else i
                    ids.append(f"{fallback_parent}_child_{fallback_idx}_{_uuid.uuid4().hex[:8]}")
        # 深拷贝 metadata,避免原地修改 caller 的 dict
        metas = [{**m, "is_child": True, "parent_doc_id": m.get("parent_doc_id") or m.get("doc_id", "")} for m in metas]
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        coll.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)
        return len(chunks)

    def add_parent_child_pair(
        self,
        parent: Tuple[str, Dict[str, Any]],
        children: List[Tuple[str, Dict[str, Any]]],
        parent_embedding: Optional[List[float]] = None,
        children_embeddings: Optional[List[List[float]]] = None,
    ) -> Tuple[int, int]:
        """原子性入库一对 parent + 多个 children.

        Returns:
            (parent_count, children_count)
        """
        parent_text, parent_meta = parent
        # 分配 doc_id (深拷贝 parent_meta,避免原地修改)
        doc_id = parent_meta.get("doc_id") or str(uuid.uuid4())
        parent_meta_copy = {**parent_meta, "doc_id": doc_id, "is_parent": True, "parent_doc_id": doc_id}

        # 写 parent
        if parent_embedding is None:
            parent_embedding = self.embedder.embed([parent_text])[0]
        self.add_parent_chunks([(parent_text, parent_meta_copy)], embeddings=[parent_embedding], ids=[doc_id])

        # 写 children（共享 doc_id,深拷贝 child_meta 避免原地修改）
        if not children:
            return 1, 0
        children_copy = [
            (c_text, {**c_meta, "doc_id": doc_id, "parent_doc_id": doc_id, "is_child": True})
            for c_text, c_meta in children
        ]
        child_texts = [c[0] for c in children_copy]
        if children_embeddings is None:
            children_embeddings = self.embedder.embed(child_texts)
        self.add_child_chunks(children_copy, embeddings=children_embeddings)
        return 1, len(children)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        return_parents: bool = True,
    ) -> List[RetrievalHit]:
        """搜索：先在 child 搜，再回查 parent.

        Args:
            query: 查询文本
            n_results: 返回结果数
            return_parents: 是否在结果中包含父块文本

        Returns:
            RetrievalHit 列表
        """
        log_step(logger, "PARENT_CHILD_SEARCH", f"query='{query[:50]}...', n={n_results}")
        start = time.time()

        # 空 query 防护: embed_query("") 会得到零向量,等价于随机命中。
        if not query or not query.strip():
            logger.warning("[SEARCH] Empty query, returning no results")
            return []

        child_coll = self._child_coll()
        count = child_coll.count()
        if count == 0:
            logger.info("[SEARCH] Child collection is empty")
            return []

        # 1. 嵌入 query
        query_embedding = self.embedder.embed_query(query)

        # 2. 在 child collection 搜
        actual_n = min(n_results, count)
        results = child_coll.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        if not results.get("documents") or not results["documents"][0]:
            return []

        # 3. 收集命中，按 doc_id 去重（每个 parent 只返回一次）
        hits: List[RetrievalHit] = []
        seen_doc_ids = set()

        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 0
            # chromadb L2 squared distance + 归一化向量:
            #   L2² = 2(1 - cos), 所以 similarity = cos = 1 - L2²/2
            #   旧公式 score = max(0, 1 - L2²) 对低相似匹配截断到 0 (当 L2²>1)
            score = max(0.0, min(1.0, 1.0 - distance / 2.0))
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            doc_id = meta.get("doc_id") or meta.get("parent_doc_id", "")

            # 旧实现: 当多条 child 都缺 doc_id (空字符串),seen_doc_ids 会把它们当作"同一个 parent"
            # 全部去掉,只保留第一条。改为: doc_id 为空时不参与 dedup,用 chunk_id 兜底。
            dedup_key = doc_id or results["ids"][0][i] if results.get("ids") else doc_id
            if dedup_key and dedup_key in seen_doc_ids:
                continue
            if dedup_key:
                seen_doc_ids.add(dedup_key)

            hit = RetrievalHit(
                chunk_id=results["ids"][0][i] if results.get("ids") else "",
                text=doc_text,
                score=score,
                metadata=meta,
            )

            # 4. 回查 parent
            if return_parents and doc_id:
                parent_text = self._get_parent_text(doc_id)
                hit.parent_text = parent_text

            hits.append(hit)

        elapsed_ms = (time.time() - start) * 1000
        log_timing(logger, "Parent-child search", elapsed_ms)
        return hits

    def _get_parent_text(self, doc_id: str) -> Optional[str]:
        """从 parent collection 取出 doc_id 对应的父块文本."""
        try:
            parent_coll = self._parent_coll()
            result = parent_coll.get(ids=[doc_id], include=["documents"])
            if result.get("documents") and result["documents"]:
                return result["documents"][0]
            # parent 缺失: 子块入库但 parent 未对应入库(典型于重 ingest 漂移)。
            # 旧实现静默返回 None,使排障无线索。
            logger.warning(
                f"[SEARCH] Child references parent_doc_id={doc_id[:12]}... but no parent found "
                f"in collection '{self.parent_collection}'. Possible orphan child or index drift."
            )
        except Exception as e:
            logger.warning(f"[SEARCH] Failed to fetch parent {doc_id[:8]}: {e}")
        return None

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """获取父子 collection 的统计."""
        return {
            "parent_count": self._parent_coll().count(),
            "child_count": self._child_coll().count(),
        }

    def clear(self):
        """清空父子 collection."""
        self.vector_store.clear_collection(self.parent_collection)
        self.vector_store.clear_collection(self.child_collection)
