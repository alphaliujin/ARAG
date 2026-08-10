"""X2MD 切片加载器 - 加载 .parents.json / .children.json / .chunks.json.

X2MD 已把 DOC/PDF 等源文档转换为 MD + 三种切片 JSON：
  - .parents.json  - 父块 (大段，~4x chunk_size)
  - .children.json - 子块 (小段，chunk_size 级别)
  - .chunks.json   - 普通切片

父子块通过共享 doc_id 关联：每个 parent 有唯一 doc_id，其 children 共享这个 doc_id。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from md2rag.logger import get_logger, log_step, log_timing, log_memory

logger = get_logger("md2rag.loader")


class ChunkFileType(str, Enum):
    """X2MD 切片文件类型."""
    CHUNKS = "chunks"        # .chunks.json
    PARENTS = "parents"      # .parents.json
    CHILDREN = "children"    # .children.json


@dataclass
class ChunkRecord:
    """统一化的 Chunk 记录.

    字段说明:
      - text: 切片文本
      - chunk_index: 在所属文件中的索引
      - source: 源文件名（不含扩展名）
      - page: 页码（可空）
      - bbox: 边界框坐标 [x0, y0, x1, y1]（可空字符串）
      - chunk_type: 切片类型 text/title/table/image
      - abstract: LLM 生成的摘要（children 通常有）
      - document_name: 文档名
      - doc_id: 父块 ID（X2MD 为每个 parent 生成唯一 UUID）
      - parent_doc_id: 父块 doc_id（children 等于 doc_id）
      - is_parent / is_child: 类型标记
      - raw_metadata: 原始 metadata 字典
    """
    text: str
    chunk_index: int
    source: str
    page: Optional[int] = None
    bbox: str = ""
    chunk_type: str = "text"
    abstract: str = ""
    document_name: str = ""
    doc_id: Optional[str] = None
    parent_doc_id: Optional[str] = None
    is_parent: bool = False
    is_child: bool = False
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """用于嵌入的文本 — 只用纯正文.

        之前版本会把 [文档:xxx]、[摘要:xxx]、[父块ID:xxx]、[类型:xxx] 等元数据前缀
        拼到正文前面再嵌入, 导致不同文件名的相同内容向量不一致, 也让不相关内容
        因前缀贡献约 0.5 的基础相似度噪声。

        现改为只用纯正文 (self.text) 做嵌入:
        - 相同内容 → 相同向量, 不受文件名/摘要等影响
        - 不同内容 → 低相似度, 不受前缀噪声干扰
        - 元数据信息仍保留在 ChromaDB metadata 中, 检索结果回显不受影响
        """
        return self.text

    def to_metadata(self, classification: str, file_path: str = "") -> Dict[str, Any]:
        """导出为 ChromaDB 元数据字典."""
        # 反推原始文件路径
        md_base = self.source
        if md_base.endswith(".md"):
            md_base = md_base[:-3]

        return {
            # 基础信息
            "source": self.source,
            "document_name": self.document_name or self.source,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "page": self.page or 0,
            "classification": classification,
            "is_parent": self.is_parent,
            "is_child": self.is_child,
            "doc_id": self.doc_id or "",
            "parent_doc_id": self.parent_doc_id or "",
            "abstract": self.abstract,
            "bbox": self.bbox,
            # 文件追踪（用于回溯）
            "original_md_path": str(file_path) if file_path else "",
            "original_file_name": md_base,
        }


def stable_doc_id(source: str, chunk_index: int, kind: str = "") -> str:
    """生成确定性兜底 doc_id (替代 uuid4 随机 ID), 保证重入库可复现.

    X2MD 正常会为 parent/child 生成 UUID 并写入切片 metadata; 本函数仅在 doc_id
    缺失时兜底。旧实现用 uuid4(): 每次重入库都生成新随机 ID, 使 collection.upsert(
    ids=[doc_id]) 无法覆盖旧记录而是追加, 同一 chunk 在库中堆积多份过期向量,
    污染 HNSW 索引并让检索可能返回过期结果。

    kind 用于区分 parent/chunk 等, 防止同 source+chunk_index 但不同类型的块
    生成相同兜底 ID (二者可能落入同一 collection)。
    """
    raw = f"{kind}::{source}::{chunk_index}".encode("utf-8")
    return "auto_" + hashlib.sha256(raw).hexdigest()[:16]


# classification key -> 实际目录名映射
# 唯一权威定义：全项目所有模块（backend / api_server / image_processor 等）都
# 必须从本文件 import 这两个常量，不要在别处复制副本。
# 三档密级：public / restricted / confidential，对应 0Public / 1Restricted / 2Confidential。
# 历史遗留的 "secret" 别名已于 2026-06-09 彻底废除（统一为 confidential）。
CLASSIFICATION_DIR_MAP = {
    "public": "0Public",
    "restricted": "1Restricted",
    "confidential": "2Confidential",
}

# 密级目录名 -> classification key（反向）
DIR_TO_CLASSIFICATION = {v: k for k, v in CLASSIFICATION_DIR_MAP.items()}

# 合法的 classification key 集合，便于参数校验
VALID_CLASSIFICATIONS = frozenset(CLASSIFICATION_DIR_MAP.keys())

# 兼容旧名（仅本模块内部历史调用使用，新代码请用去前缀的公开名）
_CLASSIFICATION_DIR_MAP = CLASSIFICATION_DIR_MAP
_DIR_TO_CLASSIFICATION = DIR_TO_CLASSIFICATION


class ChunkLoader:
    """加载 X2MD 生成的切片文件.

    支持切片模式：
    - 普通切片: .chunks.json
    - 父子块切片: .parents.json + .children.json
    """

    def __init__(self, md_dir: str | Path):
        self.md_dir = Path(md_dir)
        logger.info(f"[INIT] ChunkLoader initialized with md_dir: {self.md_dir}")

    # ------------------------------------------------------------------
    # 文件发现
    # ------------------------------------------------------------------

    def discover_files(self, classification: Optional[str] = None) -> List[Path]:
        """发现 MD 目录下的所有切片文件."""
        log_step(logger, "DISCOVER", f"Scanning chunk files (classification={classification})")
        start = time.time()

        if classification:
            dir_name = _CLASSIFICATION_DIR_MAP.get(classification, classification)
            search_dirs = [self.md_dir / dir_name]
        else:
            search_dirs = [d for d in self.md_dir.iterdir() if d.is_dir()]

        results: List[Path] = []
        for dir_path in search_dirs:
            if not dir_path.exists():
                logger.warning(f"[DISCOVER] Directory not found: {dir_path}")
                continue
            for file_path in dir_path.rglob("*.json"):
                if file_path.suffixes in (
                    [".chunks", ".json"],
                    [".parents", ".json"],
                    [".children", ".json"],
                ):
                    results.append(file_path)

        elapsed_ms = (time.time() - start) * 1000
        log_timing(logger, "Discover files", elapsed_ms)
        log_memory(logger, "Chunk files discovered", len(results))
        return results

    def discover_md_dirs(self, classification: Optional[str] = None) -> List[Path]:
        """发现 MD 目录下含切片的 MD 文档目录（MD/0Public/xxx.md/）."""
        if classification:
            dir_name = _CLASSIFICATION_DIR_MAP.get(classification, classification)
            search_dirs = [self.md_dir / dir_name]
        else:
            search_dirs = [d for d in self.md_dir.iterdir() if d.is_dir()]

        results: List[Path] = []
        for d in search_dirs:
            if not d.exists():
                continue
            for sub in d.iterdir():
                if sub.is_dir() and ((sub / "images").exists() or any(sub.glob("*.parents.json"))):
                    results.append(sub)
        return results

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------

    def load_file(self, file_path: str | Path) -> List[ChunkRecord]:
        """加载单个切片文件."""
        path = Path(file_path)
        logger.debug(f"[LOAD] Loading file: {path}")
        start = time.time()

        if not path.exists():
            logger.warning(f"[LOAD] File not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"[LOAD] Failed to parse {path}: {e}")
            return []

        if not isinstance(data, list):
            logger.warning(f"[LOAD] Invalid data in {path}: expected list")
            return []

        # 判断文件类型
        suffixes = path.suffixes
        if suffixes == [".chunks", ".json"]:
            file_type = ChunkFileType.CHUNKS
        elif suffixes == [".parents", ".json"]:
            file_type = ChunkFileType.PARENTS
        elif suffixes == [".children", ".json"]:
            file_type = ChunkFileType.CHILDREN
        else:
            logger.warning(f"[LOAD] Unknown file type: {path}")
            return []

        records = self._parse_records(data, file_type)
        elapsed_ms = (time.time() - start) * 1000
        log_timing(logger, f"Load file {path.name}", elapsed_ms)
        logger.debug(f"[LOAD] {len(records)} records from {path.name}")
        return records

    def _parse_records(self, data: List[Dict], file_type: ChunkFileType) -> List[ChunkRecord]:
        """解析 JSON 数据为 ChunkRecord 列表."""
        records: List[ChunkRecord] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")
            metadata = item.get("metadata", {})
            if isinstance(metadata, list):
                metadata = metadata[0] if metadata else {}

            record = ChunkRecord(
                text=text,
                chunk_index=metadata.get("chunk_index", i),
                source=metadata.get("source", ""),
                page=metadata.get("page"),
                bbox=metadata.get("bbox", ""),
                chunk_type=metadata.get("chunk_type", "text"),
                abstract=metadata.get("abstract", ""),
                document_name=metadata.get("document_name", ""),
                doc_id=metadata.get("doc_id"),
                is_parent=(file_type == ChunkFileType.PARENTS),
                is_child=(file_type == ChunkFileType.CHILDREN),
                raw_metadata=metadata,
            )
            records.append(record)

        # children 文件：parent_doc_id = doc_id
        if file_type == ChunkFileType.CHILDREN:
            for record in records:
                record.parent_doc_id = record.doc_id

        return records

    # ------------------------------------------------------------------
    # 按文档加载（按 strategy 决定返回哪些 chunk）
    # ------------------------------------------------------------------

    def load_document_chunks(
        self,
        md_file_path: str | Path,
        strategy: str = "auto",
    ) -> List[ChunkRecord]:
        """根据策略加载单个 MD 文件对应的切片.

        Args:
            md_file_path: MD 文件路径，如 MD/0Public/银渐层.md/银渐层.md
            strategy: 切片策略
                - "auto": 自动检测存在的切片文件
                - "chunk": 只加载 .chunks.json
                - "parent-child": 加载 .parents.json + .children.json
                - "parents-only": 只加载 .parents.json
                - "children-only": 只加载 .children.json
        """
        md_path = Path(md_file_path)
        log_step(logger, "LOAD_DOC_CHUNKS", f"Loading chunks for {md_path.name} (strategy={strategy})")

        base_path = md_path.with_suffix("")
        records: List[ChunkRecord] = []

        # 只加载 chunks（parents-only 和 children-only 策略不加载普通切片）
        if strategy in ("auto", "chunk"):
            chunks_path = Path(str(base_path) + ".chunks.json")
            if chunks_path.exists():
                records.extend(self.load_file(chunks_path))

        # 加载 parents（parents-only 策略只加载父块，不加载普通切片）
        if strategy in ("auto", "parent-child", "parents-only"):
            parents_path = Path(str(base_path) + ".parents.json")
            if parents_path.exists():
                records.extend(self.load_file(parents_path))

        if strategy in ("auto", "parent-child", "children-only"):
            children_path = Path(str(base_path) + ".children.json")
            if children_path.exists():
                records.extend(self.load_file(children_path))

        return records

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_classification_from_path(self, file_path: str | Path) -> str:
        """从文件路径推断密级分类.

        - 0Public        -> public
        - 1Restricted    -> restricted
        - 2Confidential  -> confidential
        路径中找不到密级目录时,默认返回 public.
        """
        path = Path(file_path)
        for part in path.parts:
            if part in DIR_TO_CLASSIFICATION:
                return DIR_TO_CLASSIFICATION[part]
        return "public"
