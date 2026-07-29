"""bbox 元数据提取工具 - 从 MD 内容推断位置信息.

X2MD 当前生成的 .parents.json/.children.json 中 bbox 字段为空字符串。
本模块提供：
1. 行级 bbox：从 MD 文本中按行号分配 bbox
2. 段落级 bbox：按空行分段的段落分配 bbox
3. 当 X2MD 升级能提供真实 bbox 时，自动透传
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
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


@dataclass
class LineIndex:
    """一个文件全文的行级索引, 用于加速 bbox 关联.

    make_synthetic_bboxes 产出的 line_infos 本身就按 char_start 升序
    (char_pos 单调累加), char_end 非降。本结构额外缓存排序好的
    char_starts / char_ends 数组, 使 assign_bboxes_to_chunk 可用二分
    在 O(log 行数 + 命中数) 内完成, 而非每个 chunk 都 O(行数) 扫一遍。

    一个文件的 MD 原文可能被上万 chunk 共享: 旧实现每条 chunk 都重新
    make_synthetic_bboxes + 线性扫描, 遇到大文件 (数万行 × 数万 chunk)
    会成为 CPU 瓶颈并长期占用 GIL, 拖垮 asyncio 端点。本索引每文件算
    一次, 通过 get_bboxes_for_record(line_index=...) 传入。
    """
    line_infos: List[LineInfo]
    char_starts: List[int] = field(default_factory=list)  # 升序
    char_ends: List[int] = field(default_factory=list)    # 非降


def build_line_index(source_text: str) -> LineIndex:
    """为一份 MD 原文构建行级索引 (每文件调用一次, 复用给该文件所有 chunk).

    等价于 make_synthetic_bboxes + 预取 char_starts/char_ends, 供
    assign_bboxes_to_chunk 二分查找使用。

    Args:
        source_text: 完整 MD 原文

    Returns:
        LineIndex: line_infos + 排序好的 char_starts/char_ends
    """
    line_infos = make_synthetic_bboxes(source_text)
    return LineIndex(
        line_infos=line_infos,
        char_starts=[li.char_start for li in line_infos],
        char_ends=[li.char_end for li in line_infos],
    )


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
    char_starts: Optional[List[int]] = None,
    char_ends: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """根据 chunk 在原文中的字符位置，关联 bbox 信息.

    Args:
        chunk_text: chunk 文本
        chunk_char_start: chunk 在原文中的字符起始位置
        line_infos: 全文的行级 bbox 信息 (按 char_start 升序)
        char_starts: 可选, line_infos 的 char_start 升序数组 (来自 build_line_index).
            传入时启用 O(log 行数) 二分路径; 不传则回退 O(行数) 线性扫描。
        char_ends: 可选, 对应 char_end 非降数组, 与 char_starts 配套使用。

    Returns:
        [{"page": int, "bbox": [x0,y0,x1,y1], "line_no": int}, ...]
    """
    chunk_char_end = chunk_char_start + len(chunk_text)
    related: List[Dict[str, Any]] = []

    if char_starts is not None and char_ends is not None:
        # 二分路径: 行与 chunk 重叠的充要条件是
        #   line.char_end > chunk_char_start  AND  line.char_start < chunk_char_end
        # -> lo = 第一个 char_end > chunk_char_start 的行
        # -> hi = 第一个 char_start >= chunk_char_end 的行
        # 命中区即 [lo, hi)。char_starts 严格升序 / char_ends 非降, bisect 合法。
        lo = bisect.bisect_right(char_ends, chunk_char_start)
        hi = bisect.bisect_left(char_starts, chunk_char_end)
        for i in range(lo, hi):
            line = line_infos[i]
            related.append({
                "page": _line_to_page(line.line_no),
                "bbox": list(line.bbox),
                "line_no": line.line_no,
            })
    else:
        # 兼容路径: 旧调用方 / 测试直接传 line_infos
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
    line_index: Optional[LineIndex] = None,
) -> List[Dict[str, Any]]:
    """为一条 chunk 记录生成 bbox 列表.

    Args:
        record_text: chunk 文本
        source_text: 完整 MD 原文
        record_char_start: chunk 在 source_text 中的起始位置（None 时自动搜索）
        line_index: 可选, 预构建的全文行级索引 (build_line_index), 复用给同文件
            的所有 chunk。传入时跳过 make_synthetic_bboxes 重建并走二分路径,
            避免每个 chunk 都 O(行数) 扫描。不传则按旧行为每次重建。

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

    if line_index is not None:
        return assign_bboxes_to_chunk(
            record_text,
            record_char_start,
            line_index.line_infos,
            char_starts=line_index.char_starts,
            char_ends=line_index.char_ends,
        )

    line_infos = make_synthetic_bboxes(source_text)
    return assign_bboxes_to_chunk(record_text, record_char_start, line_infos)
