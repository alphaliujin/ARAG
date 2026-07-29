"""Cross-encoder reranker — 对 docscan top-N 候选做精排.

设计原则:
1. **优雅降级**: 模型文件不存在或加载失败时, rerank() 返回 None,
   compare 流程退回到原 combined_similarity, 不阻塞主流程。
2. **零网络依赖**: 用 transformers 直接加载本地目录, 不联网。
3. **延迟加载**: 第一次 rerank() 调用时才载入模型, 启动开销不影响 cold path。
4. **线程安全单例**: load 用 lock 保护, 多个并发 compare 共享一份模型。

模型文件路径优先级:
- settings.RERANKER_MODEL_PATH (env / settings) — 可指向任意本地目录
- 默认: <project_root>/bge-reranker-v2-m3-local/

期望模型: `BAAI/bge-reranker-v2-m3` (568MB, ~568M params, 中英双语).
用 transformers AutoModelForSequenceClassification 加载, num_labels=1, sigmoid 后得到 0~1 score。

不存在时 docscan 走传统 combined_similarity, 系统功能完整。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.config import settings


# 默认模型目录 (与项目里的 bge-m3-local / clip-vit-large-patch14-local 同级)
_DEFAULT_RERANKER_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
) / "bge-reranker-v2-m3-local"


class _Reranker:
    """单例: 持有模型 + tokenizer, 第一次调用时延迟加载."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded_path: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._load_failed = False
        self._fail_reason = ""

    def _resolve_model_path(self) -> Path:
        # 1) 显式 env / settings
        explicit = getattr(settings, "RERANKER_MODEL_PATH", "") or os.getenv("RERANKER_MODEL_PATH", "")
        if explicit:
            return Path(explicit)
        return _DEFAULT_RERANKER_DIR

    def _try_load(self) -> bool:
        """尝试加载; 成功返回 True, 失败返回 False (并标记 _load_failed 避免重试)."""
        if self._load_failed:
            return False
        if self._model is not None:
            return True

        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False

            model_path = self._resolve_model_path()
            if not model_path.exists() or not model_path.is_dir():
                self._load_failed = True
                self._fail_reason = f"reranker model dir not found: {model_path}"
                print(f"[RERANKER] disabled — {self._fail_reason}")
                return False

            try:
                # 内网环境: 强制本地, 不联网
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                import torch  # noqa: F401  (确认 torch 可用)
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                # 选设备: settings.DEVICE auto/mps/cuda/cpu
                device_pref = (getattr(settings, "DEVICE", "auto") or "auto").lower()
                if device_pref == "auto":
                    if torch.backends.mps.is_available():
                        self._device = "mps"
                    elif torch.cuda.is_available():
                        self._device = "cuda"
                    else:
                        self._device = "cpu"
                elif device_pref in ("mps", "cuda", "cpu"):
                    self._device = device_pref
                else:
                    self._device = "cpu"

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path), local_files_only=True
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path), local_files_only=True
                )
                self._model.eval()
                self._model.to(self._device)
                self._loaded_path = str(model_path)
                print(f"[RERANKER] loaded {model_path} on {self._device}")
                return True
            except Exception as e:
                self._load_failed = True
                self._fail_reason = f"load failed: {type(e).__name__}: {e}"
                print(f"[RERANKER] disabled — {self._fail_reason}")
                self._tokenizer = None
                self._model = None
                return False

    def is_available(self) -> bool:
        """Reranker 是否可用 (尝试加载并缓存结果)."""
        return self._try_load()

    def rerank(
        self,
        query: str,
        candidates: List[str],
        max_length: int = 512,
    ) -> Optional[List[float]]:
        """对 (query, cand_i) pair 列表打分, 返回 sigmoid 后的 0~1 分数列表.

        Args:
            query: 待检文本
            candidates: 候选文本列表
            max_length: tokenizer 最大长度 (bge-reranker-v2-m3 推荐 ≤ 8192, 取 512 控开销)

        Returns:
            与 candidates 等长的 float 列表; 模型不可用时返回 None。
        """
        if not candidates:
            return []
        if not self._try_load():
            return None

        import torch
        try:
            pairs = [[query, c or ""] for c in candidates]
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores = torch.sigmoid(logits).cpu().tolist()
            return scores
        except Exception as e:
            print(f"[RERANKER] inference failed: {e}")
            return None

    def release(self) -> None:
        """卸载 reranker 模型, 释放 ~1GB+ 常驻内存.

        bge-reranker-v2-m3 (~568M params) 一旦在 DocScan compare 中加载, 此前
        没有任何释放路径 → 进程内常驻直到退出。提供显式 release 入口让
        DocScan compare / ingest 完成时主动卸载; 下次需要再用时由 _try_load 重载。
        """
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded_path = None
            # 复位 _load_failed, 让下次调用可以重试加载 (例如用户更新了模型路径)
            self._load_failed = False
            self._fail_reason = ""
        try:
            import gc
            gc.collect()
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


# 模块级单例
reranker = _Reranker()
