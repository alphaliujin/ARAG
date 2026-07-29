"""bbox 元数据提取工具 - 从 MD 内容推断位置信息.

X2MD 当前生成的 .parents.json/.children.json 中 bbox 字段为空字符串。
本模块提供：
1. 行级 bbox：从 MD 文本中按行号分配 bbox
2. 段落级 bbox：按空行分段的段落分配 bbox
3. 当 X2MD 升级能提供真实 bbox 时，自动透传
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from md2rag.logger import get_logger

logger = get_logger("md2rag.bbox_extractor")


@dataclass
class LineInfo:
    """一行 MD 文本的位置信息."""
    line_no: int          # 行号（0-based）
    char_start: int       # 在全文中的字符起始位置
    char_end: int         # 在全文中的字符结束位置
    text: str             # 行内容
    bbox: Tuple[float, float, float, float]  # [x0, y0, x1, y1] 伪坐标


def make_synthetic_bboxes(
    text: str,
    page_width: float = 595.0,
    page_height: float = 842.0,
    line_height: float = 14.0,
) -> List[LineInfo]:
    """为 MD 文本的每一行生成合成 bbox.

    当原始 PDF/DOCX 的真实 bbox 不可得时，使用基于行号的合成坐标：
    - x0 = 0, x1 = page_width
    - 按 lines_per_page = floor(page_height/line_height) 分页,
      页内行号决定 y 坐标(线性,不回卷重叠)

    旧实现: y0 = (line_no * line_height) % page_height 会让跨页坐标在 line_no=60 处
    回卷重叠,使不同"页"的行获得完全相同 bbox;同时 line_no//50 与该模数不一致。
    现在显式以 lines_per_page 派生 y 与 page,保持自洽。

    Args:
        text: MD 文本
        page_width: 虚拟页面宽度（PDF A4 = 595 pt）
        page_height: 虚拟页面高度
        line_height: 行高 (默认 14pt)

    Returns:
        LineInfo 列表（每个非空行一项）
    """
    lines_per_page = max(1, int(page_height // line_height))
    lines: List[LineInfo] = []
    char_pos = 0

    for line_no, line in enumerate(text.splitlines(keepends=True)):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            char_pos += len(line)
            continue
        char_start = char_pos
        char_end = char_pos + len(line)
        line_in_page = line_no % lines_per_page
        y0 = line_in_page * line_height
        y1 = y0 + line_height
        lines.append(LineInfo(
            line_no=line_no,
            char_start=char_start,
            char_end=char_end,
            text=stripped,
            bbox=(0.0, y0, page_width, y1),
        ))
        char_pos = char_end

    return lines


def _line_to_page(
    line_no: int,
    page_height: float = 842.0,
    line_height: float = 14.0,
) -> int:
    """统一的"line → page"换算,与 make_synthetic_bboxes 同源."""
    lines_per_page = max(1, int(page_height // line_height))
    return line_no // lines_per_page


def assign_bboxes_to_chunk(
    chunk_text: str,
    chunk_char_start: int,
    line_infos: List[LineInfo],
) -> List[Dict[str, Any]]:
    """根据 chunk 在原文中的字符位置，关联 bbox 信息.

    Args:
        chunk_text: chunk 文本
        chunk_char_start: chunk 在原文中的字符起始位置
        line_infos: 全文的行级 bbox 信息

    Returns:
        [{"page": int, "bbox": [x0,y0,x1,y1], "line_no": int}, ...]
    """
    chunk_char_end = chunk_char_start + len(chunk_text)
    related: List[Dict[str, Any]] = []

    for line in line_infos:
        # 行与 chunk 有重叠
        if line.char_end > chunk_char_start and line.char_start < chunk_char_end:
            related.append({
                "page": _line_to_page(line.line_no),  # 与 make_synthetic_bboxes 用同一 lines_per_page
                "bbox": list(line.bbox),
                "line_no": line.line_no,
            })

    if not related:
        # 兜底：找不到时返回空列表
        return []

    return related


def get_bboxes_for_record(
    record_text: str,
    source_text: str,
    record_char_start: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """为一条 chunk 记录生成 bbox 列表.

    Args:
        record_text: chunk 文本
        source_text: 完整 MD 原文
        record_char_start: chunk 在 source_text 中的起始位置（None 时自动搜索）

    Returns:
        bbox 列表
    """
    if not source_text:
        return []

    # 自动查找位置
    if record_char_start is None:
        pos = source_text.find(record_text)
        if pos == -1:
            return []
        record_char_start = pos

    line_infos = make_synthetic_bboxes(source_text)
    return assign_bboxes_to_chunk(record_text, record_char_start, line_infos)
