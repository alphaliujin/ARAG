"""Backend 向量库服务

直接复用 MD2RAG 入库的 `md2rag_{cls}_child` collection,不再维护独立的
`public_documents` / `confidential_documents` / `restricted_documents`。

历史背景(2026-06-09 整改):
- 旧代码维护了一对平行的 `*_documents` collection,但 ingestion 链路从未
  写入,导致 sensitive scan 永远查不到命中。

向量归一化(2026-06-14 修复):
- Ollama bge-m3 返回的原始向量模长≈26(未归一化),之前误认为已归一化,
  导致 ChromaDB 的 1-d²/2 公式不等于 cosine。
- MD2RAG OllamaEmbedder.embed() 已加入归一化步骤,与 MPS 版一致。
- 归一化后: ||v||=1, 1-d²/2 = cosine, L2 距离与 cosine 等价。
- bge-m3 归一化后的不相关基线 ≈ 0.37 (实测)。
"""
import atexit
import json
import numpy as np
import os
import threading
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import Any, List, Dict, Optional, Tuple
from app.core.config import settings

# MD2RAG path is configured in app/main.py at startup.
# Lazy import to avoid ModuleNotFoundError when this module is loaded before main.py sets sys.path.
_CLASSIFICATION_DIR_MAP = None

def _get_classification_dir_map():
    global _CLASSIFICATION_DIR_MAP
    if _CLASSIFICATION_DIR_MAP is None:
        from md2rag.loader import CLASSIFICATION_DIR_MAP
        _CLASSIFICATION_DIR_MAP = CLASSIFICATION_DIR_MAP
    return _CLASSIFICATION_DIR_MAP


# 复用 MD2RAG 入库的子块 collection
# MD2RAG indexer 写入名为 "md2rag_{cls}_child" 的 collection
def _md2rag_child_collection_name(classification: str) -> str:
    return f"md2rag_{classification}_child"


class _OllamaEmbeddingFn:
    """Ollama 嵌入函数 - 通过本地 Ollama HTTP API 生成向量.

    ★ 返回 L2 归一化向量, 与 MD2RAG OllamaEmbedder.embed() 行为一致。
    Ollama bge-m3 返回的原始向量模长≈26 (未归一化);
    归一化后 cosine = dot product, 且 ChromaDB 的 1-d²/2 公式才成立。
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3:latest", max_concurrent: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_concurrent = max_concurrent
        # urllib.request.OpenerDirector 不是线程安全的;旧实现共享一个 opener,
        # ThreadPoolExecutor.map 中可能损坏内部 handler 状态。
        # 改为每次调用本地 urlopen,无共享状态。

    def _normalize(self, embedding: List[float]) -> List[float]:
        """L2 归一化: 与 MD2RAG OllamaEmbedder.embed() 的归一化逻辑一致."""
        emb_np = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            return (emb_np / norm).tolist()
        return embedding

    def _embed_single(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    emb = data.get("embedding", [])
                    if not emb:
                        raise RuntimeError(f"Ollama returned empty embedding for text (len={len(text)})")
                    # 归一化: Ollama bge-m3 返回的原始向量模长≈26
                    return self._normalize(emb)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[OLLAMA] Embed error (attempt {attempt+1}/{max_retries}): {e}")
                    continue
                # 所有重试失败 → 抛异常而不是返回空列表 (空列表会损坏 ChromaDB)
                raise RuntimeError(f"Ollama embedding failed after {max_retries} attempts: {e}")

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 1:
            return [self._embed_single(texts[0])]
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            return list(ex.map(self._embed_single, texts))


class VectorDBService:
    def __init__(self):
        self.embedding_fn = None
        self.client = None
        # collection 缓存: classification -> Collection
        self._collections: Dict[str, object] = {}
        self._initialized = False
        self._init_lock = threading.Lock()  # 保护 initialize() 的并发安全
        self._collection_lock = threading.Lock()  # 保护 _collections dict 的并发读写
        # 进程退出时显式关闭 chromadb 客户端,释放 SQLite 锁
        atexit.register(self._close)

    def _close(self):
        # 进程退出时清理 chromadb 客户端,避免与 MD2RAG 进程锁竞争
        try:
            if self.client is not None:
                with self._collection_lock:
                    self._collections.clear()
                self.client = None
        except Exception:
            pass

    def initialize(self):
        with self._init_lock:
            if self._initialized:
                return

            os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)

            # 根据配置选择嵌入模型
            if settings.EMBEDDING_MODEL == "ollama-bge-m3":
                print(f"[EMBED] Using Ollama bge-m3 (1024维) @ {settings.OLLAMA_BASE_URL}")
                self.embedding_fn = _OllamaEmbeddingFn(
                    base_url=getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
                    model="bge-m3:latest",
                )
            elif settings.EMBEDDING_MODEL == "mps-bge-m3":
                # MPS 本地嵌入: 复用 MD2RAG MPSEmbedder,通过适配器包装成 ChromaDB EmbeddingFunction
                # 否则 backend 这一侧会回落到 384 维 ChromaDB 默认 embedder,与 MD2RAG 1024 维冲突。
                print("[EMBED] Using MPS bge-m3 (1024维) — local model from bge-m3-local/")
                from md2rag.embedder import MPSEmbedder

                class _MPSEmbeddingFn:
                    def __init__(self):
                        self._embedder = MPSEmbedder(model_name="BAAI/bge-m3", device="mps")
                    def __call__(self, texts):
                        if isinstance(texts, str):
                            texts = [texts]
                        return self._embedder.embed(texts)

                self.embedding_fn = _MPSEmbeddingFn()
            else:
                # ChromaDB 默认或 sentence-transformers 都走 DefaultEmbeddingFunction (384维)
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            self.client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # 复用 MD2RAG 已建好的 md2rag_{cls}_child collection
            # 注意: 不传 metadata={"hnsw:space":...},避免与 MD2RAG 既有 collection 的 l2 配置冲突
            # MD2RAG 默认 l2 + 归一化向量,与 cosine 公式等价(见模块 docstring)
            for cls in _get_classification_dir_map():
                name = _md2rag_child_collection_name(cls)
                try:
                    coll = self.client.get_or_create_collection(name=name)
                    with self._collection_lock:
                        self._collections[cls] = coll
                except Exception as e:
                    print(f"[VECTOR_DB] failed to open collection {name}: {e}")

            self._initialized = True

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的嵌入向量 (已归一化).

        ★ _OllamaEmbeddingFn 现在在内部完成归一化, 与 MD2RAG OllamaEmbedder 行为一致。
        DefaultEmbeddingFunction (all-MiniLM-L6-v2) 输出近似归一化向量 (norm≈0.99),
        为保证严格性, 也显式归一化。
        """
        if not texts:
            return []

        embeddings = self.embedding_fn(texts)
        # 兼容 numpy.ndarray / list[list[float]] / list[ndarray]
        if hasattr(embeddings, 'tolist'):
            result = embeddings.tolist()
        else:
            result = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

        # ★ 显式 L2 归一化: 保证所有 embedder 输出的向量严格 norm=1
        # ChromaDB 默认 L2 squared distance + 公式 similarity = 1 - d²/2 要求向量严格归一化
        arr = np.array(result, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0, norms, 1.0)
        normalized = arr / safe_norms
        return normalized.tolist()

    def add_documents(
        self,
        classification: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """⚠️ DEPRECATED: 主入库路径是 MD2RAG indexer 或 ImageIngestService.
        此方法仅供测试脚本使用, 生产代码不应调用。
        如需旁路写入, 请使用对应 service 的 ingest 方法。

        注: get_embeddings() 已统一归一化, 此方法写入的向量也是 norm=1 的归一化向量。
        """
        warnings.warn(
            "vector_db_service.add_documents() is deprecated; use MD2RAG indexer or "
            "ImageIngestService for production ingestion paths.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._collection_lock:
            collection = self._collections.get(classification)
        if not collection:
            return

        embeddings = self.get_embeddings(documents)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search_similar(
        self,
        classification: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """⚠️ DEPRECATED: 此方法未被任何生产代码调用。

        存在历史缺陷: 原版 get_embeddings() 未归一化 Ollama 向量,
        导致 query (norm≈26) 与入库向量 (norm=1) 不在同一空间,
        similarity = 1-d²/2 公式完全错误。

        现已修复: get_embeddings() 统一归一化, 此方法可正常工作,
        但生产代码应使用 search_combined() (支持多密级 + MD2RAG indexer 优先路径)。
        """
        warnings.warn(
            "vector_db_service.search_similar() is deprecated; use search_combined() "
            "for multi-classification queries with MD2RAG indexer-priority routing.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._collection_lock:
            collection = self._collections.get(classification)
        if not collection:
            return {"documents": [], "metadatas": [], "distances": []}

        query_embeddings = self.get_embeddings(query_texts)

        count = collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_n = min(n_results, count)

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def search_combined(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, List[Dict]]:
        """对 confidential + restricted 两个密级各检索 n_results 条候选.

        优先使用 MD2RAG Indexer.search() 执行搜索(确保 embedder 与入库一致);
        如果 indexer 不可用或超时,回退到 Ollama embedder + 归一化修正。
        """
        # 方式1: 通过 MD2RAG Indexer (embedder 与入库一致)
        from app.services.ingestion import data_ingestion_service
        try:
            indexer = data_ingestion_service._get_indexer()
            results = indexer.search(
                query=query_text,
                classification=None,  # 搜索所有密级
                n_results=n_results,
                return_parents=False,  # scanner 只需要子块文本
            )
            all_results = {"confidential": [], "restricted": []}
            for cls, hits in results.items():
                if cls not in all_results:
                    continue
                for h in hits:
                    all_results[cls].append({
                        "content": h["content"],
                        "metadata": h["metadata"],
                        "similarity": h["score"],
                    })
            return all_results
        except Exception as e:
            print(f"[VECTOR_DB] indexer.search failed ({e}), falling back to Ollama+normalize")

        # 方式2: Ollama 回退 (embedder 已内置归一化, 与入库向量空间一致)
        # _OllamaEmbeddingFn.__call__() 现在返回已归一化向量 (norm=1),
        # 不再需要手动归一化修正。仍做防御性校验: 若 norm 偏离 1.0 则修正。
        if not isinstance(self.embedding_fn, _OllamaEmbeddingFn):
            # 非 ollama 配置下回退: 临时创建,仅在 fallback 时使用
            ollama_fn = _OllamaEmbeddingFn(
                base_url=getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
                model="bge-m3:latest",
            )
        else:
            ollama_fn = self.embedding_fn
        raw_embeddings = ollama_fn([query_text])
        query_emb = raw_embeddings[0] if raw_embeddings else []
        if not query_emb:
            return {"confidential": [], "restricted": []}

        # 防御性校验: embedder 应返回 norm=1, 但若因浮点误差偏离则修正
        emb_np = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(emb_np)
        if norm > 0 and abs(norm - 1.0) > 1e-6:
            query_emb = (emb_np / norm).tolist()

        all_results = {"confidential": [], "restricted": []}

        def _to_similarity(distance: float) -> float:
            # 归一化向量 + chromadb L2 squared distance: sim = 1 - d/2 = cos
            return max(0.0, min(1.0, 1.0 - distance / 2.0))

        for level in ("confidential", "restricted"):
            raw = self._search_with_embeddings(level, [query_emb], n_results)
            if raw.get("documents") and raw["documents"] and raw["documents"][0]:
                for i, doc in enumerate(raw["documents"][0]):
                    distance = raw["distances"][0][i] if raw.get("distances") else 2.0
                    all_results[level].append({
                        "content": doc,
                        "metadata": raw["metadatas"][0][i] if raw.get("metadatas") else {},
                        "similarity": _to_similarity(distance)
                    })

        return all_results

    def _search_with_embeddings(
        self,
        classification: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        with self._collection_lock:
            collection = self._collections.get(classification)
        if not collection:
            return {"documents": [], "metadatas": [], "distances": []}

        count = collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_n = min(n_results, count)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

    def get_client(self):
        """获取共享的 ChromaDB PersistentClient (供其他服务复用,避免多客户端锁竞争)."""
        self.initialize()  # 确保 client 已初始化
        return self.client

    def get_collection(self, collection_name: str):
        """按 collection 名称获取或创建 collection (供 dedup/docscan 等服务复用).

        与 _get_collection(classification) 不同, 此方法接受完整的 collection 名称
        (如 "md2rag_public_child"), 而不是 classification key.
        """
        self.initialize()
        try:
            return self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"[VECTOR_DB] Error getting collection {collection_name}: {e}")
            return None

    def _get_collection(self, classification: str):
        """按 classification key (public/restricted/confidential) 返回 collection."""
        with self._collection_lock:
            return self._collections.get(classification)

    def get_collection_count(self) -> Dict[str, int]:
        with self._collection_lock:
            snapshot = dict(self._collections)
        return {
            cls: (coll.count() if coll else 0)
            for cls, coll in snapshot.items()
        }

    def clear_collection(self, classification: str):
        """清空指定密级的 collection.

        注意: 这会同时清掉 MD2RAG 的入库数据(因为 collection 是复用的),
        想重建 MD2RAG 数据请走 MD2RAG indexer 的 reset 接口。
        """
        with self._collection_lock:
            collection = self._collections.get(classification)
        if not collection:
            return
        name = _md2rag_child_collection_name(classification)
        try:
            self.client.delete_collection(name=name)
            # 重建空 collection,继承 MD2RAG 的默认 hnsw 配置
            new_coll = self.client.get_or_create_collection(name=name)
            with self._collection_lock:
                self._collections[classification] = new_coll
        except Exception as e:
            print(f"Error clearing collection {name}: {e}")

    def reset_all(self):
        if not self.client:
            return
        try:
            existing = self.client.list_collections()
            for coll in existing:
                name = coll.name if hasattr(coll, "name") else str(coll)
                try:
                    self.client.delete_collection(name=name)
                except Exception as e:
                    print(f"Error deleting collection {name}: {e}")
            with self._collection_lock:
                self._collections.clear()
            self._initialized = False
            self.initialize()
        except Exception as e:
            print(f"Error resetting database: {e}")


vector_db_service = VectorDBService()
