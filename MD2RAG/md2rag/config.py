"""MD2RAG 配置模块."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MD2RAGConfig:
    """MD2RAG 配置类."""

    # 路径配置
    md_dir: Path = field(default_factory=lambda: Path("../MD"))
    vector_db_dir: Path = field(default_factory=lambda: Path("../backend/vector_db"))

    # 向量数据库配置
    collection_prefix: str = "md2rag"
    embedding_model: str = "ollama-bge-m3"   # 默认使用 bge-m3，1024维
    embedding_batch_size: int = 32

    # Ollama 嵌入配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "bge-m3:latest"        # 默认模型: bge-m3 (1024维)
    ollama_enabled: bool = True

    # 计算设备配置 (auto/mps/cuda/cpu)
    device: str = "auto"  # auto 时自动检测: mps > cuda > cpu

    # Sentence-Transformers 配置
    st_model_name: str = "all-MiniLM-L6-v2"
    st_enabled: bool = False

    # ViT 图像处理配置
    vit_enabled: bool = True
    vit_model: str = "clip-vit-large-patch14-local"  # 本地 ViT-Large 模型路径
    vit_device: str = "auto"
    vit_local_files_only: bool = False
    vit_mirror: str = ""
    vit_labels: list[str] = field(default_factory=list)
    skip_existing: bool = True
    default_chunk_strategy: str = "auto"

    def __post_init__(self):
        if isinstance(self.md_dir, str):
            self.md_dir = Path(self.md_dir)
        if isinstance(self.vector_db_dir, str):
            self.vector_db_dir = Path(self.vector_db_dir)


_DEFAULTS: dict[str, dict[str, Any]] = {
    "md2rag": {
        "md_dir": "../MD",
        "vector_db_dir": "../backend/vector_db",
        "collection_prefix": "md2rag",
        "default_chunk_strategy": "auto",
        "skip_existing": "true",
    },
    "embedding": {
        "model": "ollama-bge-m3",
        "batch_size": "32",
    },
    "ollama": {
        "enabled": "true",
        "base_url": "http://localhost:11434",
        "model": "bge-m3:latest",
    },
    "device": {
        "type": "auto",  # auto/mps/cuda/cpu
    },
    "vit": {
        "enabled": "true",
        "model": "clip-vit-large-patch14-local",
        "device": "auto",
        "local_files_only": "true",
        "mirror": "",
        "labels": "",
    },
    "sentence_transformers": {
        "enabled": "false",
        "model_name": "all-MiniLM-L6-v2",
    },
}


def _coerce_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def _coerce_value(value: str) -> bool | str | int | float:
    """将字符串值转换为合适的类型.

    优先尝试数值(int/float),避免将 "0" 误转为 False.
    仅对明确的布尔关键字(true/false/yes/no/on/off)返回 bool.
    """
    # 布尔关键字: 只接受明确的布尔词, 不将 "1"/"0" 转为 bool (避免破坏数值零)
    if value.lower() in ("true", "false", "yes", "no", "on", "off"):
        return _coerce_bool(value)
    # 数值优先: 先试 int, 再试 float
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _detect_device() -> str:
    """自动检测可用的计算设备: mps > cuda > cpu.

    If torch is not installed, returns "cpu" as the default device.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(config_path: str | Path | None = None) -> MD2RAGConfig:
    """加载 MD2RAG 配置.

    优先从 md2rag.conf 配置文件加载，若不存在则使用默认配置.
    """
    parser = configparser.ConfigParser()

    # 加载默认值
    for section, options in _DEFAULTS.items():
        parser.add_section(section)
        for key, value in options.items():
            parser.set(section, key, str(value))

    # 尝试加载配置文件
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            parser.read(str(path), encoding="utf-8")
    else:
        # 从当前目录向上查找 md2rag.conf
        current = Path.cwd()
        while True:
            candidate = current / "md2rag.conf"
            if candidate.exists():
                parser.read(str(candidate), encoding="utf-8")
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

    # 构建配置对象
    cfg = MD2RAGConfig()

    # 路径配置
    cfg.md_dir = Path(parser.get("md2rag", "md_dir", fallback="../MD"))
    if not cfg.md_dir.is_absolute():
        cfg.md_dir = Path(__file__).resolve().parent.parent / cfg.md_dir

    cfg.vector_db_dir = Path(parser.get("md2rag", "vector_db_dir", fallback="../backend/vector_db"))
    if not cfg.vector_db_dir.is_absolute():
        cfg.vector_db_dir = Path(__file__).resolve().parent.parent / cfg.vector_db_dir

    cfg.collection_prefix = parser.get("md2rag", "collection_prefix", fallback="md2rag")
    cfg.default_chunk_strategy = parser.get("md2rag", "default_chunk_strategy", fallback="auto")
    cfg.skip_existing = _coerce_bool(parser.get("md2rag", "skip_existing", fallback="true"))

    # 嵌入配置
    cfg.embedding_model = parser.get("embedding", "model", fallback="ollama-bge-m3")
    cfg.embedding_batch_size = int(parser.get("embedding", "batch_size", fallback="32"))

    # Ollama 配置
    cfg.ollama_enabled = _coerce_bool(parser.get("ollama", "enabled", fallback="true"))
    cfg.ollama_base_url = parser.get("ollama", "base_url", fallback="http://localhost:11434")
    cfg.ollama_model = parser.get("ollama", "model", fallback="bge-m3:latest")

    # 计算设备配置
    cfg.device = parser.get("device", "type", fallback="auto")
    if cfg.device == "auto":
        cfg.device = _detect_device()

    # Sentence-Transformers 配置
    cfg.st_enabled = _coerce_bool(
        parser.get("sentence_transformers", "enabled", fallback="false")
    )
    cfg.st_model_name = parser.get(
        "sentence_transformers", "model_name", fallback="all-MiniLM-L6-v2"
    )

    # ViT 图像处理配置
    cfg.vit_enabled = _coerce_bool(
        parser.get("vit", "enabled", fallback="true")
    )
    # 支持本地路径或 HuggingFace 模型名
    vit_model_path = parser.get("vit", "model", fallback="clip-vit-large-patch14-local")
    cfg.vit_model = vit_model_path
    cfg.vit_device = parser.get("vit", "device", fallback="auto")
    if cfg.vit_device == "auto":
        cfg.vit_device = cfg.device
    cfg.vit_local_files_only = _coerce_bool(
        parser.get("vit", "local_files_only", fallback="true")
    )
    cfg.vit_mirror = parser.get("vit", "mirror", fallback="")
    labels_str = parser.get("vit", "labels", fallback="")
    if labels_str:
        cfg.vit_labels = [l.strip() for l in labels_str.split(",") if l.strip()]

    return cfg
