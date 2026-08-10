from __future__ import annotations

import os
import json
from pathlib import Path

# 必须在导入 transformers 之前设置，防止 tokenizers 的 fork 安全问题
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image

_DEFAULT_LABELS = [
    "a photo of a document",
    "a photo of a chart",
    "a photo of a diagram",
    "a photo of a table",
    "a photo of a screenshot",
    "a photo of a logo",
    "a photo of a person",
    "a photo of a building",
    "a photo of a product",
    "a photo of a slide",
    "a photo of text",
    "a photo of a flowchart",
    "a photo of a graph",
    "a photo of an icon",
    "a photo of a map",
]


class VitExtractor:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        local_files_only: bool = False,
        mirror: str = "",
        labels: list[str] | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.mirror = mirror
        self.labels = labels or _DEFAULT_LABELS
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.mirror:
            os.environ["HF_ENDPOINT"] = self.mirror

        is_local = Path(self.model_name).is_dir()

        kwargs = {}
        if is_local or self.local_files_only:
            kwargs["local_files_only"] = True

        self._processor = CLIPProcessor.from_pretrained(
            self.model_name, **kwargs
        )
        self._model = CLIPModel.from_pretrained(
            self.model_name, **kwargs
        )
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def extract_features(self, image_path: Path) -> dict:
        self._load_model()

        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        logits = outputs.logits_per_image.softmax(dim=1)
        probs = logits[0].tolist()

        label_scores = sorted(
            zip(self.labels, probs),
            key=lambda x: x[1],
            reverse=True,
        )

        image_embeds = outputs.image_embeds[0].tolist()

        return {
            "labels": label_scores,
            "image_embeds": image_embeds,
        }

    def extract_features_batch(self, image_paths: list[Path]) -> list[dict]:
        """批量处理多张图像。CLIP batch 推理比逐张快 5-10×。

        单张图片大小限制为 50MB(解压前);超过则跳过避免 OOM。
        无法打开的文件(如 EMF/损坏图)在该批次中静默跳过,返回结果对齐输入列表。
        """
        self._load_model()
        if not image_paths:
            return []

        # 加载阶段过滤 + 用 context manager 避免 fd 泄漏
        loaded: list[tuple[int, "Image.Image"]] = []  # (orig_idx, image)
        MAX_BYTES = 50 * 1024 * 1024
        for i, p in enumerate(image_paths):
            try:
                if p.stat().st_size > MAX_BYTES:
                    print(f"Warning: skip oversized image {p} ({p.stat().st_size} bytes > {MAX_BYTES})")
                    continue
            except OSError:
                continue
            try:
                # PIL 是 lazy load,convert("RGB") 后 close 原 fp
                im = Image.open(p)
                rgb = im.convert("RGB")
                im.close()
                loaded.append((i, rgb))
            except Exception as e:
                print(f"Warning: skip unreadable image {p}: {e}")
                continue

        # 准备结果占位 (跳过的图片返回空 dict,调用方可识别)
        results: list[dict] = [{} for _ in image_paths]
        if not loaded:
            return results

        orig_indices = [i for i, _ in loaded]
        images = [img for _, img in loaded]

        inputs = self._processor(
            text=self.labels,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        # logits_per_image shape: [batch, num_labels]
        probs_batch = outputs.logits_per_image.softmax(dim=1)
        # image_embeds shape: [batch, embed_dim]
        embeds_batch = outputs.image_embeds

        for batch_idx, orig_i in enumerate(orig_indices):
            probs = probs_batch[batch_idx].tolist()
            label_scores = sorted(
                zip(self.labels, probs),
                key=lambda x: x[1],
                reverse=True,
            )
            image_embeds = embeds_batch[batch_idx].tolist()
            results[orig_i] = {
                "labels": label_scores,
                "image_embeds": image_embeds,
            }
        # 关闭 PIL 句柄
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        return results

    def process_directory(
        self,
        images_dir: Path,
        output_dir: Path | None = None,
        batch_size: int = 8,
    ) -> list[Path]:
        if not images_dir.exists():
            return []

        out_dir = output_dir or images_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
        }
        image_files = sorted(
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )

        if not image_files:
            return []

        self._load_model()

        output_files: list[Path] = []

        # 批处理：每 batch_size 张一次推理
        for batch_start in range(0, len(image_files), batch_size):
            batch = image_files[batch_start:batch_start + batch_size]
            try:
                results = self.extract_features_batch(batch)
                for image_path, features in zip(batch, results):
                    # extract_features_batch 对跳过的图片返回 {} ;不写 md
                    if not features:
                        continue
                    result = self._write_md(image_path, features, out_dir)
                    if result:
                        output_files.append(result)
            except Exception as e:
                print(f"Warning: Failed to process batch {batch}: {e}")

        return sorted(output_files)

    @staticmethod
    def _write_md(
        image_path: Path, features: dict, output_dir: Path
    ) -> Path:
        md_name = image_path.stem + ".md"
        md_path = output_dir / md_name

        lines = [
            f"# {image_path.name}",
            "",
        ]

        label_scores = features.get("labels", [])
        if label_scores:
            lines.append("## Zero-shot Classification")
            lines.append("")
            for label, score in label_scores[:5]:
                lines.append(f"- {label}: {score:.4f}")
            lines.append("")

        # 注: 不再写入 image_embeds 到 .md。X2MD 用 clip-vit-base-patch32 (512d),
        # 而 MD2RAG 用 clip-vit-large-patch14 (1024d) 会重新嵌入图像, 两者模型/维度
        # 不同无法直接复用; 截断为前 20 维写入既不可用于检索又有误导性, 纯属浪费。
        # zero-shot 分类标签已写入上方, 足够供检索元数据使用。

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path


def process_images(
    images_dir: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    local_files_only: bool = False,
    mirror: str = "",
    labels: list[str] | None = None,
) -> list[Path]:
    extractor = VitExtractor(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        mirror=mirror,
        labels=labels,
    )
    return extractor.process_directory(
        Path(images_dir),
        Path(output_dir) if output_dir else None,
    )
