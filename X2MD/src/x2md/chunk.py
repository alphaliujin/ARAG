"""
高级文档切片模块
提供结构化 chunk 元数据、重叠切片、bbox 追踪、分隔符规则控制、父子块策略等功能
"""

from __future__ import annotations

import json
import bisect
import re
from enum import Enum
from typing import Any, Optional, List, Dict, Union

from pydantic import BaseModel


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    CODE = "code"
    LIST = "list"
    OTHER = "other"


class ChunkMetadata(BaseModel):
    """每个 chunk 的结构化元数据"""
    chunk_index: int = 0
    source: str = ""
    page: Optional[int] = None
    bbox: str = ""  # JSON string of chunk_bboxes
    chunk_type: ChunkType = ChunkType.TEXT
    abstract: str = ""
    document_name: str = ""
    doc_id: Optional[str] = None  # 父子块关联 ID


class Chunk(BaseModel):
    """带元数据的切片单元"""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.model_dump(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        meta = d.get("metadata", {})
        if isinstance(meta, dict):
            meta = ChunkMetadata(**meta)
        return cls(text=d["text"], metadata=meta)


class IntervalSearch:
    """区间搜索：将字符位置范围映射回原始元素索引.

    使用扫描线算法正确处理重叠区间:
    对每个排序后的区间 [s_i, e_i]，与查询范围 [start, end] 比较交集条件
    s_i <= end AND e_i >= start.  扫描线在 start 时激活、end 时关闭，
    可正确处理 ends 数组不单调（重叠区间）的情况。
    """

    def __init__(self, intervals: list[list[int]]):
        if not intervals:
            self._starts: list[int] = []
            self._ends: list[int] = []
            self._sorted: list[tuple[int, int]] = []
        else:
            # 按 start 排序,保持 start-end 对齐
            self._sorted = sorted(intervals, key=lambda ie: ie[0])
            self._starts = [s for s, _ in self._sorted]
            self._ends = [e for _, e in self._sorted]

    def find(self, char_range: list[int]) -> list[int]:
        """给定字符范围 [start, end]，返回覆盖的元素索引列表（升序）。

        元素 i 满足 s_i <= end AND e_i >= start 时被视为覆盖（即与查询区间有交集）。
        对重叠区间也能正确工作，因为使用扫描线而非纯 bisect。
        """
        start, end = char_range
        if not self._sorted:
            return []

        # 找 s_i <= end 的范围上限 (bisect_right 在 starts 中定位)
        hi = bisect.bisect_right(self._starts, end)
        # 在 [0, hi) 中筛选 e_i >= start 的元素
        result = []
        for i in range(hi):
            if self._ends[i] >= start:
                result.append(i)
        return result


class ChunkSplitter:
    """
    高级文本切片引擎
    支持重叠切片、分隔符规则、bbox 映射、表格独立处理
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[list[str]] = None,
        separator_rule: Optional[list[str]] = None,
        # 分隔符均为字面量 (\n\n / 。 / ！ / . 等), 非正则。默认 True 会让 "." 被当作
        # "任意字符" 正则: 在无 \n/。/！/？/； 的英文段落上 _split_with_separator 会
        # 把文本逐字符拆分 (re.finditer(".") 命中每个字符), 再靠 merge 逻辑重新拼回,
        # 退化为字符级切分而非按句号断句。改为 False 后所有分隔符经 re.escape 视作
        # 字面量, "." 正确匹配英文句号 (CJK 标点 。！？； 本就不是正则元字符, 不受影响)。
        is_separator_regex: bool = False,
        keep_separator: bool = True,
        max_chunk_limit: int = 10000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]
        self.is_separator_regex = is_separator_regex
        self.keep_separator = keep_separator
        self.max_chunk_limit = max_chunk_limit

        # separator_rule 映射每个分隔符到 "after" 或 "before"
        if separator_rule is None:
            self.separator_rule = {
                sep: "after" for sep in self.separators
            }
        else:
            if len(separator_rule) != len(self.separators):
                print(
                    f"Warning: separator_rule has {len(separator_rule)} entries "
                    f"but separators has {len(self.separators)} entries. "
                    f"Missing rules will default to 'after'."
                )
            self.separator_rule = {}
            for i, sep in enumerate(self.separators):
                rule = separator_rule[i] if i < len(separator_rule) else "after"
                self.separator_rule[sep] = rule

    def split_text(self, text: str) -> list[str]:
        """递归分隔符层级切片"""
        return self._split_text_recursive(text, self.separators)

    def split_text_with_positions(self, text: str) -> list[tuple[str, int, int]]:
        """切片并返回每个 chunk 的 (文本, 原文起始位置, 原文结束位置).

        位置追踪使用核心文本（不含 overlap）在原文中的精确范围,
        overlap 扩展仅用于 IntervalSearch 覆盖范围,不改变 chunk_text 本身的位置锚定。
        """
        base_chunks = self._split_text_recursive(text, self.separators)
        if not base_chunks:
            return []

        # 在原文中顺序 find 每个 chunk 的真实区间。
        # 不能用 pos += len(chunk) 累加: base_chunks 已含 overlap 前缀 (前缀来自上一
        # chunk 的尾部), len(chunk) 比核心正文多出 overlap, 累加会使后续 chunk 的
        # start/end 随序号线性漂移 (chunk_overlap=100 时第 5 个 chunk 端点偏移约
        # 500 字符), 让 IntervalSearch 把 bbox/page 归因到错误的元素。
        # find 不受前缀影响: 前缀是上一核心的尾部, 与本核心在原文中连续, 故整个
        # chunk (前缀+核心) 在原文中是一段连续区间, find 可精确定位。
        positions: list[tuple[str, int, int]] = []
        search_offset = 0
        for chunk in base_chunks:
            start = text.find(chunk, search_offset)
            if start == -1:
                # 退化: 多层 overlap 拼接或原文被清洗后无法精确匹配, 用偏移估算
                start = search_offset
            end = min(start + len(chunk), len(text))
            positions.append((chunk, start, end))
            # 只推进 1 字符: 相邻 chunk 因 overlap 在原文中区间重叠, 不能跳过整段
            search_offset = start + 1

        if self.chunk_overlap <= 0 or len(positions) <= 1:
            return positions

        # 对含 overlap 的 chunk：核心范围是 chunk_text 的精确锚定,
        # range_start/range_end 扩展覆盖邻接 overlap 区域用于 IntervalSearch
        overlapped_positions: list[tuple[str, int, int]] = []
        for i, (chunk_text, core_start, core_end) in enumerate(positions):
            # 核心范围不变——chunk_text 锚定在原文 [core_start, core_end]
            range_start = core_start
            range_end = core_end

            # 仅在搜索覆盖时扩展范围(overlap 是前后 chunk 的共享区)
            if i > 0 and positions[i - 1][0]:
                range_start = max(0, core_start - self.chunk_overlap)
            if i < len(positions) - 1 and positions[i + 1][0]:
                range_end = min(len(text), core_end + self.chunk_overlap)

            overlapped_positions.append((chunk_text, range_start, range_end))

        return overlapped_positions

    def _split_text_recursive(
        self, text: str, separators: list[str]
    ) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # 找到第一个能匹配当前文本的分隔符
        for i, sep in enumerate(separators):
            if not sep:
                # 最后的分隔符：逐字符切
                continue

            if self.is_separator_regex:
                pattern = sep
            else:
                pattern = re.escape(sep)

            try:
                splits = self._split_with_separator(text, pattern, i)
            except re.error:
                continue

            if len(splits) > 1:
                # 用更深层分隔符递归切过长的片段，合并短的片段
                new_seps = separators[i + 1:]
                result: list[str] = []
                current: list[str] = []
                current_len = 0

                for split in splits:
                    split_len = len(split)
                    if current_len + split_len > self.chunk_size and current:
                        merged = "".join(current)
                        if len(merged) > self.chunk_size and new_seps:
                            result.extend(
                                self._split_text_recursive(merged, new_seps)
                            )
                        else:
                            result.append(merged)
                        current = [split]
                        current_len = split_len
                    else:
                        current.append(split)
                        current_len += split_len

                if current:
                    merged = "".join(current)
                    if len(merged) > self.chunk_size and new_seps:
                        result.extend(
                            self._split_text_recursive(merged, new_seps)
                        )
                    else:
                        result.append(merged)

                return self._add_overlap(result)

        # 没有分隔符能切，逐字符切
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks

    def _split_with_separator(
        self, text: str, pattern: str, sep_index: int
    ) -> list[str]:
        """根据 separator_rule 切分并保留分隔符"""
        sep = self.separators[sep_index]
        rule = self.separator_rule.get(sep, "after")

        if not self.keep_separator:
            return re.split(pattern, text)

        # 使用 finditer 精确重建文本片段
        matches = list(re.finditer(pattern, text))
        if not matches:
            return [text]

        result: list[str] = []
        pos = 0
        pending_sep = ""

        for match in matches:
            before = text[pos:match.start()]
            sep_text = match.group()

            if rule == "after":
                result.append(pending_sep + before + sep_text)
                pending_sep = ""
            else:
                # before: 分隔符附在下一段开头，暂存
                result.append(pending_sep + before)
                pending_sep = sep_text

            pos = match.end()

        # 最后一段
        remaining = text[pos:]
        if remaining:
            result.append(pending_sep + remaining)
        elif pending_sep:
            # 末尾的分隔符单独保留
            result.append(pending_sep)

        # 清理空片段
        return [p for p in result if p]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """为相邻 chunk 添加单向重叠: 每个 chunk 前置上一 chunk 的尾部.

        仅取前一个 chunk 的尾部作前缀 (单向), 不再同时追加后一个 chunk 的头部。
        原因:
        - 双向叠加会使一个满 chunk 实际长度膨胀到 chunk_size + 2*chunk_overlap
          (chunk_size=500, overlap=100 时达 700), 既稀释该 chunk 的语义焦点,
          又使 split_text_with_positions 用 len(chunk) 累加计算位置时持续偏移,
          导致 bbox/page 溯源错位。
        - 单向 overlap 已足以让相邻 chunk 在边界处共享上下文, 避免敏感信息被切到
          两个 chunk 两侧都无法完整匹配 (与 md2rag.text_splitter._merge_splits
          仅取 current[-chunk_overlap:] 的做法一致)。
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = []
        for i, chunk in enumerate(chunks):
            prefix = ""
            if i > 0 and chunks[i - 1]:
                # 从前一个 chunk 尾部取 overlap
                prev = chunks[i - 1]
                prefix = prev[-self.chunk_overlap:]
            overlapped.append(prefix + chunk)

        return overlapped

    def split_documents(
        self,
        text: str,
        source: str = "",
        page: Optional[int] = None,
        abstract: str = "",
        document_name: str = "",
        element_indexes: Optional[list[list[int]]] = None,
        element_pages: Optional[list[int]] = None,
        element_bboxes: Optional[list[list[float]]] = None,
        element_types: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """
        切片文档并生成带元数据的 Chunk 列表

        element_indexes/pages/bboxes/types: 原始文档中每个元素的字符范围、页码、坐标和类型
        """
        chunks_with_pos = self.split_text_with_positions(text)

        # 构建区间搜索器（如果有元素级数据）
        searcher = None
        if element_indexes:
            searcher = IntervalSearch(element_indexes)

        result: list[Chunk] = []

        for i, (chunk_text, chunk_start, chunk_end) in enumerate(chunks_with_pos):
            # 检查硬限制
            if len(chunk_text) > self.max_chunk_limit:
                raise ChunkMaxLimitError(
                    f"Chunk {i} exceeds max_chunk_limit "
                    f"({len(chunk_text)} > {self.max_chunk_limit}). "
                    "Try using more separators (e.g. \\n, 。, .) for finer splitting."
                )

            # 确定 bbox 和 page
            chunk_bboxes: list[dict] = []
            chunk_page: Optional[int] = page
            chunk_type = ChunkType.TEXT

            if searcher and element_pages and element_bboxes and element_types:
                indices = searcher.find([chunk_start, chunk_end])
                if indices:
                    # 合并覆盖元素的 bbox 和 page
                    for idx in indices:
                        bbox = element_bboxes[idx] if idx < len(element_bboxes) else []
                        pg = element_pages[idx] if idx < len(element_pages) else 0
                        chunk_bboxes.append({"page": pg, "bbox": bbox})

                    # 取第一个元素的页码作为 chunk 页码
                    chunk_page = element_pages[indices[0]] if indices[0] < len(element_pages) else page

                    # 取最常见的类型作为 chunk 类型
                    type_counts: dict[str, int] = {}
                    for idx in indices:
                        t = element_types[idx] if idx < len(element_types) else "text"
                        type_counts[t] = type_counts.get(t, 0) + 1
                    dominant_type = max(type_counts, key=type_counts.get)
                    chunk_type = _map_element_type(dominant_type)

            bbox_str = json.dumps({"chunk_bboxes": chunk_bboxes}) if chunk_bboxes else ""

            meta = ChunkMetadata(
                chunk_index=i,
                source=source,
                page=chunk_page,
                bbox=bbox_str,
                chunk_type=chunk_type,
                abstract=abstract,
                document_name=document_name,
            )

            result.append(Chunk(text=chunk_text, metadata=meta))

        return result


class SmallerChunksStrategy:
    """
    父子块策略：用细粒度 child chunk 检索，返回粗粒度 parent chunk 提供上下文
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        child_overlap: int = 50,
        parent_overlap: int = 100,
        separators: Optional[list[str]] = None,
        separator_rule: Optional[list[str]] = None,
    ):
        self.parent_splitter = ChunkSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=separators,
            separator_rule=separator_rule,
        )
        self.child_splitter = ChunkSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=separators,
            separator_rule=separator_rule,
        )

    def split(
        self,
        text: str,
        source: str = "",
        page: Optional[int] = None,
        abstract: str = "",
        document_name: str = "",
        element_indexes: Optional[list[list[int]]] = None,
        element_pages: Optional[list[int]] = None,
        element_bboxes: Optional[list[list[float]]] = None,
        element_types: Optional[list[str]] = None,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        返回 (parent_chunks, child_chunks)
        每个 child chunk 的 doc_id 指向对应 parent chunk 的 chunk_index
        """
        import uuid

        parent_chunks = self.parent_splitter.split_documents(
            text=text,
            source=source,
            page=page,
            abstract=abstract,
            document_name=document_name,
            element_indexes=element_indexes,
            element_pages=element_pages,
            element_bboxes=element_bboxes,
            element_types=element_types,
        )

        # 为每个 parent 分配唯一 doc_id
        for parent in parent_chunks:
            parent.metadata.doc_id = str(uuid.uuid4())

        child_chunks: list[Chunk] = []
        child_index = 0
        search_offset = 0  # 在原文中逐个定位 parent 文本的起始字符位置，避免 find 重复命中

        for parent in parent_chunks:
            # 子分块切分 parent.text (而非原始全文),所以元素索引需要
            # 重新映射为相对于 parent.text 的偏移量,而非文档级偏移。
            child_element_indexes = None
            child_element_pages = None
            child_element_bboxes = None
            child_element_types = None

            if element_indexes is not None and element_pages is not None:
                # parent 在原文中的实际字符偏移（而非顺序索引 chunk_index）
                # chunk_index 是顺序编号(0,1,2...)，不是字符位置，不能用于 IntervalSearch
                # 使用 text.find + 递增 search_offset 精确定位每个 parent 在原文中的起始位置
                p_start = text.find(parent.text, search_offset)
                if p_start == -1:
                    # fallback: 无法精确定位（overlap 导致文本不精确匹配），
                    # 用 search_offset 作为近似值（它是上一个 parent 结束位置的估算）
                    p_start = search_offset
                # 推进 search_offset，保证下一个 parent 搜索不会重复命中当前位置
                search_offset = max(search_offset, p_start + 1)
                # 通过 parent 的 IntervalSearch 找覆盖的元素
                parent_searcher = IntervalSearch(element_indexes)
                parent_el_indices = parent_searcher.find([p_start, p_start + len(parent.text)])
                if parent_el_indices:
                    # 将文档级索引转换为 parent-relative 累积偏移
                    child_element_indexes = []
                    child_element_pages = []
                    child_element_bboxes = []
                    child_element_types = []
                    offset = 0
                    for idx in parent_el_indices:
                        # 相对偏移: 每个元素在 parent.text 中占 len(原文片段) 字符
                        orig_len = element_indexes[idx][1] - element_indexes[idx][0]
                        child_element_indexes.append([offset, offset + orig_len])
                        offset += orig_len
                        child_element_pages.append(element_pages[idx])
                        child_element_bboxes.append(element_bboxes[idx] if idx < len(element_bboxes) else [])
                        child_element_types.append(element_types[idx] if idx < len(element_types) else "text")

            parent_children = self.child_splitter.split_documents(
                text=parent.text,
                source=source,
                page=parent.metadata.page,
                abstract=abstract,
                document_name=document_name,
                element_indexes=child_element_indexes,
                element_pages=child_element_pages,
                element_bboxes=child_element_bboxes,
                element_types=child_element_types,
            )

            for child in parent_children:
                child.metadata.doc_id = parent.metadata.doc_id
                child.metadata.chunk_index = child_index
                child_index += 1
                child_chunks.append(child)

        return parent_chunks, child_chunks


class ChunkMaxLimitError(Exception):
    """单个 chunk 超过最大长度限制"""
    pass


def _map_element_type(elem_type: str) -> ChunkType:
    """将 bisheng 风格的元素类型映射到 ChunkType"""
    type_map = {
        "Title": ChunkType.HEADING,
        "Header": ChunkType.HEADING,
        "heading": ChunkType.HEADING,
        "Table": ChunkType.TABLE,
        "table": ChunkType.TABLE,
        "Image": ChunkType.IMAGE,
        "image": ChunkType.IMAGE,
        "Figure": ChunkType.IMAGE,
        "figure": ChunkType.IMAGE,
        "Code": ChunkType.CODE,
        "code": ChunkType.CODE,
        "List": ChunkType.LIST,
        "list": ChunkType.LIST,
        "text": ChunkType.TEXT,
        "Text": ChunkType.TEXT,
        "paragraph": ChunkType.TEXT,
        "Paragraph": ChunkType.TEXT,
    }
    return type_map.get(elem_type, ChunkType.OTHER)


def aggregate_chunk_text(chunk: Chunk) -> str:
    """
    将 chunk 文本和元数据聚合为带标签的文本
    参考 bisheng 的 <file_title>/<file_abstract>/<paragraph_content> 包裹方式
    """
    parts: list[str] = []

    if chunk.metadata.document_name:
        parts.append(f"<file_title>{chunk.metadata.document_name}</file_title>")

    if chunk.metadata.abstract:
        parts.append(f"<file_abstract>{chunk.metadata.abstract}</file_abstract>")

    parts.append(f"<paragraph_content>{chunk.text}</paragraph_content>")

    return "\n".join(parts)


def split_chunk_text(aggregated: str) -> str:
    """从聚合文本中提取原始 chunk 内容"""
    match = re.search(r"<paragraph_content>(.*?)</paragraph_content>", aggregated, re.DOTALL)
    if match:
        return match.group(1)
    return aggregated