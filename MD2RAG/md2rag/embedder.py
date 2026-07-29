"""嵌入模块 - 支持多种嵌入模型."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

from md2rag.logger import get_logger, log_step, log_timing
from md2rag.embedding_cache import EmbeddingCache

logger = get_logger("md2rag.embedder")


def _l2_normalize_list(embeddings: list[list[float]]) -> list[list[float]]:
    """对向量列表做 L2 归一化, 确保每个向量严格 norm=1.

    ChromaDB 默认 L2 squared distance, 公式 similarity = 1 - d²/2 = cosine
    仅在向量严格归一化 (||v||=1) 时成立。部分模型 (SentenceTransformer)
    输出近似归一化向量 (norm≈0.99), 需显式归一化消除误差。

    使用 numpy 批量归一化 (100x faster than pure Python for 1024-dim vectors).
    """
    if not embeddings:
        return []
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero: keep zero vectors unchanged
    safe_norms = np.where(norms > 0, norms, 1.0)
    normalized = arr / safe_norms
    return normalized.tolist()


def _release_torch_cache() -> None:
    """Best-effort 释放 PyTorch/MPS/CUDA 缓存。"""
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


class Embedder(ABC):
    """嵌入模型抽象基类."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """将单个查询文本转换为向量."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入向量的维度."""
        ...

    def release(self) -> None:
        """释放模型/缓存资源。默认无状态嵌入器无需处理。"""
        return None


class ChromaDefaultEmbedder(Embedder):
    """ChromaDB 默认嵌入 (all-MiniLM-L6-v2)."""

    def __init__(self):
        log_step(logger, "INIT", "Initializing ChromaDefaultEmbedder...")
        try:
            from chromadb.utils import embedding_functions
            self._ef = embedding_functions.DefaultEmbeddingFunction()
            logger.info("[INIT] ChromaDB DefaultEmbeddingFunction loaded")
        except ImportError:
            raise ImportError(
                "ChromaDB embedding function not available. "
                "Install with: pip install chromadb"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.debug(f"[EMBED] Embedding {len(texts)} texts with ChromaDB default...")
        embed_start = time.time()

        embeddings = self._ef(texts)

        # 转换为 list[list[float]]
        if hasattr(embeddings, "tolist"):
            result = embeddings.tolist()
        else:
            result = [
                emb.tolist() if hasattr(emb, "tolist") else list(emb)
                for emb in embeddings
            ]

        # 显式 L2 归一化: SentenceTransformer 模型输出近似归一化向量 (norm≈0.99),
        # 但 ChromaDB L2 squared distance + 公式 1-d²/2 = cosine 要求严格 norm=1
        result = _l2_normalize_list(result)

        embed_elapsed = (time.time() - embed_start) * 1000
        log_timing(logger, f"ChromaDB embed {len(texts)} texts", embed_elapsed)
        logger.debug(f"[EMBED] Generated {len(result)} embeddings, dimension: {len(result[0]) if result else 0}")

        return result

    def embed_query(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        return 384

    def release(self) -> None:
        # ChromaDB 默认 EF 内部持 SentenceTransformer 模型 (~80MB),
        # release 时丢弃引用 + 触发 torch 缓存回收, 下次 embed 重新构造。
        self._ef = None
        _release_torch_cache()


class OllamaEmbedder(Embedder):
    """Ollama 嵌入模型 (默认使用 bge-m3, 1024维)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3:latest",
        timeout: int = 120,
        max_concurrent: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_path: Optional[str | Path] = None,
    ):
        log_step(logger, "INIT", f"Initializing OllamaEmbedder (model={model})...")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._dimension: Optional[int] = None
        # 内存 LRU cache: sha256(model_name + text) → embedding
        # 同一文档反复 re-index 时省去 Ollama 调用;入库结束后随进程退出。
        # 容量上限防止长跑 OOM。
        self._cache: dict[str, list[float]] = {}
        self._cache_max = 5000
        # 磁盘持久缓存 (SQLite): 进程重启后仍命中, 省 Ollama 调用 (生成向量是入库瓶颈)。
        # None 时退化为纯内存 (旧行为)。key 含 model 名 -> 换模型自动 miss, 无需手动失效。
        self._disk_cache: Optional[EmbeddingCache] = None
        if cache_path is not None:
            try:
                self._disk_cache = EmbeddingCache(cache_path)
            except Exception as e:
                logger.warning(f"[INIT] disk cache init failed (fallback to memory-only): {e}")
                self._disk_cache = None
        # urllib.request.OpenerDirector 不是线程安全的;旧实现共享一个 opener,
        # 在 ThreadPoolExecutor.map 中可能损坏内部 handler 状态。
        # 改为每次 HTTP 调用本地构造,这样 urlopen 路径完全无共享状态。
        logger.info(f"[INIT] Ollama embedder: base_url={base_url}, model={model}, max_concurrent={max_concurrent}, timeout={timeout}s")

    def _call_single(self, text: str) -> list[float]:
        import urllib.error
        import urllib.request
        import socket

        payload = json.dumps({
            "model": self.model,
            "prompt": text,
            # keep_alive: 入库几十分钟内不让 bge-m3 卸载, 避免重载时的设备重评估
            # (GB10 统一内存上重载可能触发 CPU 计算路径, 导致 embed 变慢)
            "keep_alive": "30m",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    embedding = data.get("embedding", [])
                    if not embedding:
                        logger.warning(f"[OLLAMA] Empty embedding returned for text: {text[:50]}...")
                    return embedding
            except urllib.error.HTTPError as e:
                logger.error(f"[OLLAMA] HTTP error (attempt {attempt + 1}/{self.max_retries}): {e.code} - {e.reason}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama embedding failed after {self.max_retries} retries: "
                    f"HTTP {e.code} {e.reason} for text: {text[:50]}..."
                )
            except (socket.timeout, TimeoutError) as e:
                logger.error(f"[OLLAMA] Timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama embedding failed after {self.max_retries} retries: "
                    f"timeout for text: {text[:50]}..."
                )
            except Exception as e:
                logger.error(f"[OLLAMA] Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama embedding failed after {self.max_retries} retries: "
                    f"{type(e).__name__}: {e} for text: {text[:50]}..."
                )

        raise RuntimeError(
            f"Ollama embedding failed after {self.max_retries} retries "
            f"for text: {text[:50]}..."
        )

    def _call_batch(self, texts: list[str]) -> list[list[float]]:
        """使用 Ollama /api/embed 批量嵌入接口（分批串行处理）.

        串行而非并发: Metal 上 Ollama 是 GPU 密集, 多个并发 /api/embed 请求会在
        Ollama 侧排队(GPU 一次只算一个), 排在后面的请求 120s 内等不到数据 ->
        "Batch timeout ... failed after 3 retries" -> 整个 embed 500(曾让《鄧小平時代》
        港版.DocScan 生成向量失败)。串行下每批 ~25s 即处理即返回, 不超时;
        长文档总耗时靠前端 30 分钟超时兜底完成(~3453 chunk ≈ 14 分钟)。
        若 Ollama 侧设置了 OLLAMA_NUM_PARALLEL>1 真正并行, 再考虑恢复并发。
        """
        # 128 (2026-07-23 实测调回大批): GB10+CUDA+当前 Ollama 上, 长/短文本大批均更快
        # (短 71->108 t/s, 长 16->24 t/s, 256 文本基准), 未复现早年 Metal 上 batch=100
        # 长文本触发 CPU 计算路径 (GPU-Util=0) 的问题 - 疑似旧 Ollama/Metal 后端 bug。
        # 若日志再现 "embed 1-5s vs 正常 600ms" 的 CPU 路径, 再降回 32。
        batch_size = 128
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            batch_embeddings = self._call_batch_single(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _call_batch_single(self, texts: list[str]) -> list[list[float]]:
        """发送单批嵌入请求到 Ollama."""
        import urllib.error
        import urllib.request
        import socket

        payload = json.dumps({
            "model": self.model,
            "input": texts,
            "keep_alive": "30m",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    embeddings = data.get("embeddings", [])
                    if not embeddings:
                        raise RuntimeError(
                            f"Ollama returned empty embeddings for batch of {len(texts)} texts"
                        )
                    if len(embeddings) != len(texts):
                        # 数量不符 (服务端截断/部分失败) 必须报错, 不能返回短列表:
                        # embed() 里 zip(missing_idx, normalized) 会静默截断, 留下的 None
                        # 变成空向量 [], 混入 add_chunks 会让整批 ChromaDB add 维度报错。
                        raise RuntimeError(
                            f"Ollama returned {len(embeddings)} embeddings for "
                            f"{len(texts)} texts (count mismatch)"
                        )
                    return embeddings
            except urllib.error.HTTPError as e:
                logger.error(f"[OLLAMA] Batch HTTP error (attempt {attempt + 1}/{self.max_retries}): {e.code} - {e.reason}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama batch embedding failed after {self.max_retries} retries: "
                    f"HTTP {e.code} {e.reason} for batch of {len(texts)} texts"
                )
            except (socket.timeout, TimeoutError) as e:
                logger.error(f"[OLLAMA] Batch timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama batch embedding failed after {self.max_retries} retries: "
                    f"timeout for batch of {len(texts)} texts"
                )
            except Exception as e:
                logger.error(f"[OLLAMA] Batch error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama batch embedding failed after {self.max_retries} retries: "
                    f"{type(e).__name__}: {e} for batch of {len(texts)} texts"
                )

        raise RuntimeError(
            f"Ollama batch embedding failed after {self.max_retries} retries "
            f"for batch of {len(texts)} texts"
        )

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        logger.debug(f"[OLLAMA] Calling API for {len(texts)} texts...")
        call_start = time.time()
        if len(texts) == 1:
            return [self._call_single(texts[0])]
        # 使用批量嵌入接口
        embeddings = self._call_batch(texts)
        call_elapsed = (time.time() - call_start) * 1000
        log_timing(logger, f"Ollama embed {len(texts)} texts", call_elapsed)
        logger.debug(f"[OLLAMA] Generated {len(embeddings)} embeddings")
        return embeddings

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> Optional[list[float]]:
        return self._cache.get(self._cache_key(text))

    def _cache_put(self, text: str, embedding: list[float]) -> None:
        if not embedding:
            return
        if len(self._cache) >= self._cache_max:
            # FIFO 简单逐出: 丢前一半,避免精确 LRU 的额外开销
            for k in list(self._cache.keys())[: self._cache_max // 2]:
                del self._cache[k]
        self._cache[self._cache_key(text)] = embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # 缓存路径: 命中的直接复用,未命中的发到 Ollama
        result: list[Optional[list[float]]] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                result[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            # 三级缓存第二级: 磁盘 SQLite (进程重启后仍命中, 省 Ollama 调用)
            still_idx: list[int] = []
            still_texts: list[str] = []
            if self._disk_cache is not None:
                keys = [self._cache_key(t) for t in missing_texts]
                disk_hits = self._disk_cache.get_many(keys)
                for slot, t in zip(missing_idx, missing_texts):
                    hit = disk_hits.get(self._cache_key(t))
                    if hit is not None:
                        result[slot] = hit
                        self._cache_put(t, hit)  # 回填内存 LRU
                    else:
                        still_idx.append(slot)
                        still_texts.append(t)
            else:
                still_idx, still_texts = missing_idx, missing_texts

            if still_texts:
                embeddings = self._call_ollama(still_texts)
                # 归一化: Ollama bge-m3 返回原始向量模长≈26;
                # ChromaDB 1-d²/2=cosine 公式仅在 norm=1 时成立
                normalized = _l2_normalize_list(embeddings)
                new_items: list[tuple[str, str, int, list[float]]] = []
                for slot, emb in zip(still_idx, normalized):
                    result[slot] = emb
                    self._cache_put(texts[slot], emb)
                    if self._disk_cache is not None and emb:
                        new_items.append((self._cache_key(texts[slot]), self.model, len(emb), emb))
                if self._disk_cache is not None and new_items:
                    try:
                        self._disk_cache.put_many(new_items)
                    except Exception as e:
                        logger.warning(f"[OLLAMA] disk cache write failed (non-fatal): {e}")

        return [r if r is not None else [] for r in result]

    def embed_query(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            sample = self.embed_query("test")
            if not sample:
                # 旧实现猜测 1024。问题: 用户配置了非 bge-m3 模型时(如 768d),
                # 这会导致后续 ChromaDB add 时维度错误,且 1024 与 768 collection 之间
                # 互不兼容,数据损坏后难以恢复。改为 raise,让上层处理 Ollama 不可用。
                raise RuntimeError(
                    f"Failed to probe embedding dimension: Ollama server returned empty embedding. "
                    f"Check that the model '{self.model}' is pulled and Ollama is reachable at {self.base_url}."
                )
            self._dimension = len(sample)
            logger.info(f"[DIMENSION] Ollama embedder dimension: {self._dimension}")
        return self._dimension

    def release(self) -> None:
        # Ollama embedder 本身无模型权重 (走 HTTP), 但维护一个 in-memory LRU
        # cache (sha256 → embedding, 上限 5000 条). 长跑入库会让该 cache 单调
        # 增长到上限 (5000 × 1024 × 4B ≈ 20MB), 多个密级累计常驻; release 时清空。
        self._cache.clear()
        if self._disk_cache is not None:
            self._disk_cache.close()
            self._disk_cache = None


class SentenceTransformerEmbedder(Embedder):
    """Sentence-Transformers 嵌入模型."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        log_step(logger, "INIT", f"Initializing SentenceTransformerEmbedder (model={model_name})...")
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension_value: Optional[int] = None
        logger.info(f"[INIT] ST embedder: model={model_name}, device={device}")

    def _get_model(self):
        if self._model is None:
            log_step(logger, "LOAD_MODEL", f"Loading SentenceTransformer model: {self.model_name}...")
            load_start = time.time()
            try:
                from sentence_transformers import SentenceTransformer
                # local_files_only: 防止入库卡死（联网下载超时）
                self._model = SentenceTransformer(self.model_name, device=self.device, cache_folder=None)
                load_elapsed = (time.time() - load_start) * 1000
                log_timing(logger, f"Load ST model {self.model_name}", load_elapsed)
                logger.info(f"[LOAD_MODEL] Model loaded on {self.device}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.debug(f"[ST] Embedding {len(texts)} texts...")
        embed_start = time.time()

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        result = embeddings.tolist()

        # 显式 L2 归一化: SentenceTransformer 输出近似归一化向量 (norm≈0.99),
        # ChromaDB 公式 1-d²/2 = cosine 要求严格 norm=1
        result = _l2_normalize_list(result)

        embed_elapsed = (time.time() - embed_start) * 1000
        log_timing(logger, f"ST embed {len(texts)} texts", embed_elapsed)
        logger.debug(f"[ST] Generated {len(result)} embeddings, dimension: {len(result[0]) if result else 0}")

        return result

    def embed_query(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        if self._dimension_value is None:
            model = self._get_model()
            self._dimension_value = model.get_sentence_embedding_dimension()
            logger.info(f"[DIMENSION] ST embedder dimension: {self._dimension_value}")
        return self._dimension_value

    def release(self) -> None:
        self._model = None
        _release_torch_cache()


class MPSEmbedder(Embedder):
    """使用 MPS (Metal Performance Shaders) 本地嵌入模型.

    适用于 Apple Silicon Mac，通过 PyTorch MPS 后端利用 GPU 加速.
    默认使用 BAAI/bge-m3 模型 (1024维).

    模型路径查找优先级:
    1. 项目本地目录 bge-m3-local（与 MD2RAG 同级）
    2. HuggingFace 缓存 ~/.cache/huggingface/hub/
    3. 找不到则报错（不联网下载，防止入库卡死）
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "mps", batch_size: int = 8):
        log_step(logger, "INIT", f"Initializing MPSEmbedder (model={model_name})...")
        self.model_name = model_name
        self.device_str = device
        # batch_size 默认 8 (旧值 16): MPS 上 padding 到 max_length 时,
        # 中间 tensor 大小 ∝ batch × seq_len × hidden. 16 在大 chunk 时易触 OOM。
        self.batch_size = batch_size
        # 单条 chunk 的最大 token 数。旧值 8192 是 bge-m3 模型上限, 但 X2MD 切片
        # 一般 ≤ 500 字符 (≈ 500 token), padding 到 8192 会让单批 MPS 内存
        # 暴涨 16×, 长跑入库时反复申请未复用的 MPS 块, 累计到几千条就 OOM。
        # 512 是与 reranker / 普通中文 chunk 一致的合理上界。
        self.max_length = 512
        self._model = None
        self._tokenizer = None
        self._dimension_value: Optional[int] = None
        # 记录已处理批次, 定期 empty_cache 防止 MPS 碎片化累积
        self._batches_since_cleanup = 0
        self._cleanup_every_n_batches = 20
        logger.info(f"[INIT] MPS embedder: model={model_name}, device={device}, batch_size={batch_size}, max_length={self.max_length}")

    def _resolve_local_model_path(self) -> Optional[Path]:
        """解析本地模型路径（纯离线，不联网）.

        优先级:
        1. 绝对路径（如果 model_name 是路径形式）
        2. 项目根目录下的 bge-m3-local 目录
           — 同时检查 MD2RAG 包根 (md2rag/embedder.py 上溯 2 层) 与其再上一层的 workspace,
             因为 MD2RAG 既可能独立部署,也可能作为 ARAG_V0.2 子目录被 backend 调用,
             两种布局下 bge-m3-local 的位置不同。
        3. HuggingFace 缓存中的 BAAI/bge-m3
        """
        from pathlib import Path

        # 1. 如果 model_name 本身就是本地路径
        direct = Path(self.model_name)
        if direct.exists() and direct.is_dir() and (direct / "config.json").exists():
            logger.info(f"[RESOLVE] Model found at direct path: {direct}")
            return direct

        # 2. 在 MD2RAG 包根 + 其父目录(workspace 根) 查找 bge-m3-local
        md2rag_pkg_root = Path(__file__).resolve().parent.parent
        candidate_roots = [
            md2rag_pkg_root,                        # MD2RAG 独立部署
            md2rag_pkg_root.parent,                 # 嵌入 ARAG_V0.2 这种 monorepo
        ]
        for root in candidate_roots:
            local_dir = root / "bge-m3-local"
            if local_dir.exists() and local_dir.is_dir() and (local_dir / "config.json").exists():
                logger.info(f"[RESOLVE] Model found at: {local_dir}")
                return local_dir

        # 3. HuggingFace 缓存
        import os
        hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        hf_model_dir = hf_cache / "models--BAAI--bge-m3"
        if hf_model_dir.exists():
            # 找到包含 config.json 的 snapshot
            snapshots = hf_model_dir / "snapshots"
            if snapshots.exists():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if snap.is_dir() and (snap / "config.json").exists():
                        # 检查该 snapshot 或其他 snapshot 中是否有模型权重
                        has_weights = (
                            (snap / "pytorch_model.bin").exists()
                            or (snap / "model.safetensors").exists()
                        )
                        if not has_weights:
                            # 检查其他 snapshot 是否有权重文件
                            for other_snap in snapshots.iterdir():
                                if other_snap.is_dir() and (
                                    (other_snap / "pytorch_model.bin").exists()
                                    or (other_snap / "model.safetensors").exists()
                                ):
                                    # 将权重文件链接/复制到 config 所在的 snapshot
                                    # 或直接合并：返回 config snapshot（AutoModel 会跨 snapshot 查找权重）
                                    logger.info(f"[RESOLVE] Model found in HF cache (config at {snap.name}, weights at {other_snap.name})")
                                    return hf_model_dir  # 返回 repo 级路径，让 from_pretrained 自己处理
                        logger.info(f"[RESOLVE] Model found in HF cache snapshot: {snap}")
                        return snap
            logger.info(f"[RESOLVE] Model found in HF cache repo: {hf_model_dir}")
            return hf_model_dir

        return None

    def _get_model(self):
        """延迟加载模型（纯本地，不联网下载）."""
        if self._model is None:
            log_step(logger, "LOAD_MODEL", f"Loading MPS model: {self.model_name}...")
            load_start = time.time()
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                # 检查设备可用性
                if self.device_str == "mps" and not torch.backends.mps.is_available():
                    logger.warning("[MPS] MPS not available, falling back to CPU")
                    self.device_str = "cpu"
                elif self.device_str == "cuda" and not torch.cuda.is_available():
                    logger.warning("[MPS] CUDA not available, falling back to CPU")
                    self.device_str = "cpu"

                # 查找本地模型路径
                local_path = self._resolve_local_model_path()
                if local_path:
                    # 本地路径加载（local_files_only=True 确保不联网）
                    logger.info(f"[LOAD_MODEL] Loading from local path: {local_path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
                    self._model = AutoModel.from_pretrained(str(local_path), local_files_only=True)
                else:
                    # 降级到 HF 缓存（仍不联网）
                    logger.info(f"[LOAD_MODEL] No local path found, trying HF cache for {self.model_name}")
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)

                self._model.to(self.device_str)
                self._model.eval()

                load_elapsed = (time.time() - load_start) * 1000
                log_timing(logger, f"Load MPS model {self.model_name}", load_elapsed)
                logger.info(f"[LOAD_MODEL] Model loaded on {self.device_str}")
            except ImportError as e:
                raise ImportError(
                    f"transformers or torch not installed. "
                    f"Install with: pip install transformers torch"
                ) from e
            except (OSError, FileNotFoundError) as e:
                raise FileNotFoundError(
                    f"Model '{self.model_name}' not found locally. "
                    f"Place model files in <project_root>/bge-m3-local/ "
                    f"or pre-download to ~/.cache/huggingface/hub/ "
                    f"(python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{self.model_name}')\")"
                ) from e
        return self._model

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - 取注意力加权的平均值."""
        import torch
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        import torch

        logger.debug(f"[MPS] Embedding {len(texts)} texts...")
        embed_start = time.time()

        model = self._get_model()
        tokenizer = self._tokenizer

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                # padding="longest" 而非默认 True (=longest in batch): 显式选最长,
                # 同时 max_length=512 截断, 防止超长文本把整批 padding 拉到 8192。
                encoded = tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {k: v.to(self.device_str) for k, v in encoded.items()}

                model_output = model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # cpu().numpy() 拷贝出来后, 立即丢弃 GPU tensor
                cpu_arr = embeddings.detach().cpu().numpy()
                all_embeddings.extend(cpu_arr.tolist())

                # ★ 关键: 显式释放本批的 GPU tensor 引用, 否则 PyTorch
                # 会持有 model_output / embeddings / encoded 直到 Python GC,
                # MPS allocator 无法回收, 累计几十批就吃光内存。
                del encoded, model_output, embeddings, cpu_arr

                self._batches_since_cleanup += 1
                if self._batches_since_cleanup >= self._cleanup_every_n_batches:
                    self._batches_since_cleanup = 0
                    # 定期清 MPS allocator 内部空闲块, 防止碎片化累积
                    if self.device_str == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    elif self.device_str == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

        embed_elapsed = (time.time() - embed_start) * 1000
        log_timing(logger, f"MPS embed {len(texts)} texts", embed_elapsed)
        logger.debug(f"[MPS] Generated {len(all_embeddings)} embeddings, dimension: {len(all_embeddings[0]) if all_embeddings else 0}")

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        if self._dimension_value is None:
            # bge-m3 的维度是 1024
            self._dimension_value = 1024
            logger.info(f"[DIMENSION] MPS embedder dimension: {self._dimension_value}")
        return self._dimension_value

    def release(self) -> None:
        self._model = None
        if hasattr(self, "_tokenizer"):
            self._tokenizer = None
        _release_torch_cache()


def create_embedder(
    model_type: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "bge-m3:latest",
    ollama_concurrent: int = 4,
    st_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    ollama_cache_path: Optional[str | Path] = None,
) -> Embedder:
    """工厂函数：根据类型创建嵌入模型.

    Args:
        model_type: 模型类型
            - "chromadb-default": ChromaDB 默认模型 (all-MiniLM-L6-v2)
            - "ollama": Ollama 嵌入
            - "sentence-transformers": Sentence-Transformers
            - "mps": MPS 本地嵌入 (Apple Silicon GPU 加速)
        ollama_url: Ollama 服务地址
        ollama_model: Ollama 模型名称
        ollama_concurrent: Ollama 并发请求数
        st_model: Sentence-Transformers 模型名称
        device: 计算设备 (cpu/cuda/mps)

    Returns:
        Embedder 实例
    """
    logger.info(f"[FACTORY] Creating embedder: type={model_type}")
    if model_type == "ollama":
        return OllamaEmbedder(
            base_url=ollama_url,
            model=ollama_model,
            max_concurrent=ollama_concurrent,
            cache_path=ollama_cache_path,
        )
    elif model_type in ("sentence-transformer", "sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model_name=st_model, device=device)
    elif model_type == "mps":
        return MPSEmbedder(model_name="BAAI/bge-m3", device=device)
    else:
        return ChromaDefaultEmbedder()
