"""图像处理模块 - 使用 ViT-Large 提取图像特征向量."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from md2rag.loader import CLASSIFICATION_DIR_MAP, DIR_TO_CLASSIFICATION
from md2rag.logger import get_logger, log_step, log_timing

logger = get_logger("md2rag.image_processor")


@dataclass
class ImageRecord:
    """图像记录."""

    image_path: Path
    source_md: str  # 原始 MD 文件名
    classification: str  # 密级
    embedding: Optional[list[float]] = None
    image_index: int = 0
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_id(self) -> str:
        """生成唯一图像 ID."""
        content = f"{self.classification}:{self.source_md}:img_{self.image_index}:{self.image_path.name}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @property
    def embedding_text(self) -> str:
        """用于嵌入的文本描述."""
        parts = []
        parts.append(f"[图像: {self.image_path.name}]")
        parts.append(f"[来源文档: {self.source_md}]")
        if self.raw_metadata.get("description"):
            parts.append(f"[描述: {self.raw_metadata['description']}]")
        return "\n".join(parts)


class ViTImageEmbedder:
    """ViT-Large 图像嵌入器 - 提取视觉主干 1024 维特征."""

    def __init__(self, model_name: str = "clip-vit-large-patch14-local", device: str = "auto"):
        log_step(logger, "INIT", f"Initializing ViTImageEmbedder (model={model_name})...")
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._dimension: Optional[int] = None
        logger.info(f"[INIT] ViT embedder: model={model_name}, device={device}")

    def _resolve_local_model_path(self) -> Optional[Path]:
        """解析本地模型路径.

        优先级：
        1. 绝对路径或当前 cwd 下能找到
        2. MD2RAG 目录同级（项目根 ARAG_V0.2/clip-vit-large-patch14-local）
        3. 都找不到则返回 None（走 HF 远程）
        """
        # 1. 直接路径
        direct = Path(self.model_name)
        if direct.exists() and direct.is_dir():
            return direct

        # 2. ARAG_V0.2 项目根（md2rag/image_processor.py 上溯 3 层）
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / self.model_name
        if candidate.exists() and candidate.is_dir():
            return candidate

        return None

    def _load_model(self):
        """延迟加载模型."""
        if self._model is not None:
            return

        log_step(logger, "LOAD_MODEL", f"Loading ViT model: {self.model_name}...")
        load_start = time.time()

        try:
            from transformers import CLIPModel, CLIPProcessor

            local_path = self._resolve_local_model_path()
            if local_path:
                # 本地路径加载
                logger.info(f"[LOAD_MODEL] Loading from local path: {local_path}")
                self._model = CLIPModel.from_pretrained(str(local_path), local_files_only=True)
                self._processor = CLIPProcessor.from_pretrained(str(local_path), local_files_only=True)
            else:
                # 强制走本地，不准联网下载 — 防止入库卡死
                logger.error(f"[LOAD_MODEL] Local model not found: {self.model_name}")
                logger.error(f"[LOAD_MODEL] Expected at: <project_root>/{self.model_name}/")
                raise FileNotFoundError(
                    f"ViT model '{self.model_name}' not found locally. "
                    f"Place it at the project root or disable image indexing."
                )

            if self.device != "cpu":
                import torch
                if self.device == "auto":
                    if torch.backends.mps.is_available():
                        self.device = "mps"
                    elif torch.cuda.is_available():
                        self.device = "cuda"
                    else:
                        self.device = "cpu"
                self._model = self._model.to(self.device)

            load_elapsed = (time.time() - load_start) * 1000
            log_timing(logger, f"Load ViT model {self.model_name}", load_elapsed)
            logger.info(f"[LOAD_MODEL] ViT model loaded on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers torch Pillow"
            )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
            # ViT-Large 视觉主干输出 1024 维（绕过 CLIP 768 维投影层）
            self._dimension = self._model.config.vision_config.hidden_size
            logger.info(f"[DIMENSION] ViT embedder dimension: {self._dimension}")
        return self._dimension

    def embed_image(self, image_path: Path) -> list[float]:
        """提取单张图片的视觉主干特征向量 (1024维)."""
        self._load_model()

        from PIL import Image

        embed_start = time.time()
        # ★ with-context: 防止 PIL 把原始文件 fp 持到下一次 GC 才释放
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # 使用视觉模型提取特征（绕过投影层，获取 1024 维）
                vision_outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
                # vision_outputs.last_hidden_state: [batch, seq_len, hidden_size]
                # 取 [CLS] token (第一个位置) 作为全局特征
                image_features = vision_outputs.last_hidden_state[:, 0, :]

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0].tolist()

            embed_elapsed = (time.time() - embed_start) * 1000
            log_timing(logger, f"Embed image {image_path.name}", embed_elapsed)

            return embedding
        finally:
            try:
                image.close()
            except Exception:
                pass

    def release(self) -> None:
        """释放 ViT 模型和 PyTorch 缓存。"""
        self._model = None
        self._processor = None
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

    def embed_images(self, image_paths: list[Path]) -> tuple[list[list[float]], list[Path]]:
        """批量提取图片视觉主干特征向量 (1024维).

        Returns:
            (embeddings, valid_paths): 两者按相同顺序 1:1 对齐。
            加载失败的图片会被跳过, 因此 valid_paths 是 image_paths 的子集,
            调用方必须按 valid_paths 映射回记录, 不能按原下标取用
            (旧实现按位置取用会把后一张图的向量错位赋给前一张, 并静默丢弃末尾)。
        """
        self._load_model()

        from PIL import Image

        embed_start = time.time()
        images = []
        valid_paths = []

        for path in image_paths:
            try:
                # with-context 关闭原始 fp; convert("RGB") 返回的是新对象,
                # 后面统一在 finally 里 close。
                with Image.open(path) as raw:
                    img = raw.convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"[SKIP] Failed to load image {path}: {e}")

        if not images:
            return [], []

        try:
            inputs = self._processor(images=images, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # 使用视觉模型提取特征（绕过投影层，获取 1024 维）
                vision_outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
                image_features = vision_outputs.last_hidden_state[:, 0, :]

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings = image_features.cpu().numpy().tolist()

            embed_elapsed = (time.time() - embed_start) * 1000
            log_timing(logger, f"Embed {len(images)} images", embed_elapsed)

            return embeddings, valid_paths
        finally:
            # 批 16 张高分辨率图 = 数百 MB pixel buffer, 必须显式 close
            for im in images:
                try:
                    im.close()
                except Exception:
                    pass
            images.clear()


class ImageProcessor:
    """图像处理器 - 扫描、嵌入并入库图片."""

    def __init__(self, md_dir: Path, embedder: Optional[ViTImageEmbedder] = None):
        self.md_dir = Path(md_dir)
        self.embedder = embedder or ViTImageEmbedder()
        logger.info(f"[INIT] ImageProcessor: md_dir={md_dir}")

    def discover_images(self, classification: Optional[str] = None) -> Iterator[ImageRecord]:
        """发现 MD 目录下的所有图片.

        图片目录结构: MD/<classification>/<md_file_name>/images/
        """
        log_step(logger, "DISCOVER_IMAGES", f"Scanning for images (classification={classification})...")
        discover_start = time.time()

        # 密级目录映射统一从 loader 复用,避免再分叉。
        # 历史 bug(已根治): 旧版本曾用 confidential→1Restricted / secret→2Confidential 的错位体系,
        # 导致 ingest confidential 时图片去 1Restricted 找,ingest restricted 时图片完全不入库。
        if classification:
            mapped = CLASSIFICATION_DIR_MAP.get(classification)
            if mapped is None:
                logger.warning(
                    f"[DISCOVER_IMAGES] Unknown classification '{classification}', skipping"
                )
                return
            dir_names = [mapped]
        else:
            dir_names = list(CLASSIFICATION_DIR_MAP.values())

        total_images = 0
        for dir_name in dir_names:
            dir_path = self.md_dir / dir_name
            if not dir_path.exists():
                continue

            # 遍历每个 MD 文件目录下的 images 子目录
            for md_dir in dir_path.iterdir():
                if not md_dir.is_dir():
                    continue

                images_dir = md_dir / "images"
                if not images_dir.exists():
                    continue

                cls = DIR_TO_CLASSIFICATION.get(dir_name, "public")
                img_index = 0

                for img_path in images_dir.iterdir():
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                        continue

                    record = ImageRecord(
                        image_path=img_path,
                        source_md=md_dir.name,
                        classification=cls,
                        image_index=img_index,
                    )
                    img_index += 1
                    yield record
                    total_images += 1

        discover_elapsed = (time.time() - discover_start) * 1000
        log_timing(logger, "Discover images", discover_elapsed)
        logger.info(f"[DISCOVER_IMAGES] Found {total_images} images")

    def process_images(
        self,
        classification: Optional[str] = None,
        batch_size: int = 16,
    ) -> list[ImageRecord]:
        """处理图片：发现、嵌入并返回记录.

        Returns:
            处理后的 ImageRecord 列表
        """
        log_step(logger, "PROCESS_IMAGES", f"Processing images (classification={classification})...")
        process_start = time.time()

        records = list(self.discover_images(classification))
        if not records:
            logger.info("[PROCESS_IMAGES] No images found")
            return []

        logger.info(f"[PROCESS_IMAGES] Processing {len(records)} images in batches of {batch_size}")

        # 分批嵌入
        for batch_start in range(0, len(records), batch_size):
            batch_end = min(batch_start + batch_size, len(records))
            batch = records[batch_start:batch_end]

            image_paths = [r.image_path for r in batch]
            embeddings, valid_paths = self.embedder.embed_images(image_paths)

            # embed_images 跳过加载失败的图片, 返回的 embeddings 与 valid_paths 1:1 对齐
            # (而非与 image_paths 对齐)。按路径匹配回 record, 避免按位置错位赋值。
            emb_by_path = dict(zip(valid_paths, embeddings))
            for record in batch:
                emb = emb_by_path.get(record.image_path)
                if emb is not None:
                    record.embedding = emb

        process_elapsed = (time.time() - process_start) * 1000
        log_timing(logger, "Process all images", process_elapsed)
        logger.info(f"[PROCESS_IMAGES] Completed: {len(records)} images processed")

        return records


# 导入 torch 用于类型提示
try:
    import torch
except ImportError:
    torch = None  # type: ignore
