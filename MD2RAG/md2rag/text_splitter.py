"""带 bbox 关联的文本切片器 - 借鉴 bisheng ElemCharacterTextSplitter.

核心算法（来自 bisheng text_splitter.py:159-216）:
1. 解析时为每个 element 记录 (char_start, char_end) + bbox + page + type
2. 切片时: text.find(chunk) 找到 chunk 在原文中的字符区间
3. IntervalSearch 找到该 chunk 跨越了哪些 element
4. 把这些 element 的 bbox 合并到 chunk_bboxes
"""

from __future__ import annotations

import copy
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from md2rag.interval_search import IntervalSearch
from md2rag.logger import get_logger, log_step, log_timing

logger = get_logger("md2rag.text_splitter")


@dataclass
class Document:
    """轻量级 Document 替代 langchain.Document."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """带 bbox 关联的切片结果."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_content(self) -> str:
        return self.text

    @page_content.setter
    def page_content(self, value: str):
        self.text = value


# ========================================================================
# 间隔辅助
# ========================================================================

def _split_text_with_regex(text: str, separator: str, keep_separator: bool, rule: str) -> List[str]:
    """按正则切分文本，rule 控制分隔符保留位置: 'after' | 'before'."""
    if rule == "after":
        # 分隔符留在每段末尾
        splits = re.split(f"({separator})", text)
    elif rule == "before":
        # 分隔符留在每段开头
        splits = re.split(f"(?={separator})", text)
    else:
        splits = re.split(f"({separator})", text)

    if not keep_separator:
        # 移除空字符串和分隔符 token。
        # re.split(f"({separator})", text) 把捕获组分隔符放在奇数下标位;
        # 用位置判断而非字符串相等, 因为 is_separator_regex=True 时 separator 是
        # 正则模式 (如 \d+), 实际匹配文本 ("123") 与模式串不相等, 旧 s != separator
        # 无法过滤, 导致分隔符 token 残留进切片结果。
        if rule == "before":
            # (?=...) 零宽前瞻切分, splits 全是 content, 无分隔符 token
            return [s for s in splits if s]
        # rule == "after" / 默认: 奇数下标是分隔符捕获组, 仅取偶数下标
        return [s for i, s in enumerate(splits) if i % 2 == 0 and s]
    else:
        # 保留分隔符
        result = []
        i = 0
        while i < len(splits):
            s = splits[i]
            if not s:
                i += 1
                continue
            if i + 1 < len(splits):
                # 紧跟一个 separator（被 re.split 的捕获组）
                next_s = splits[i + 1]
                result.append(s + next_s)
                i += 2
            else:
                result.append(s)
                i += 1
        return result


# ========================================================================
# 主切片器
# ========================================================================

class ElemCharacterTextSplitter:
    """带 bbox 关联的字符级文本切片器.

    与普通 CharacterTextSplitter 的核心区别：在切片时，根据每个 chunk 的
    字符区间，从原始 element 列表中查找对应的 bbox / page / type 关联。
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        separator_rule: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        is_separator_regex: bool = False,
        keep_separator: Union[bool, str] = False,
        length_function: callable = len,
    ):
        self._separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
        # 保证 separator 列表以 "" 结尾,使 _split_text 在所有真实分隔符耗尽后
        # 仍能以单字符兜底切分(避免单段超长导致死循环)。
        # 注意只追加一次:旧实现的两个 if 在空列表场景会重复追加。
        if not self._separators or self._separators[-1] != "":
            self._separators.append("")
        # 输入校验: chunk_overlap >= chunk_size 会让 _merge_splits 不断把前缀
        # 追加为下一段的 overlap_text,current 长度只增不减 → 死循环 / 内存爆炸。
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size}); "
                f"otherwise overlap_text contains the whole previous chunk and merging never progresses"
            )
        self._separator_rule = separator_rule or ["after"] * len(self._separators)
        # 补齐 rule
        while len(self._separator_rule) < len(self._separators):
            self._separator_rule.append("after")
        self._is_separator_regex = is_separator_regex
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator
        self._length_function = length_function
        # 构建 separator → rule 映射
        self.separator_rule_map = {
            sep: self._separator_rule[i] for i, sep in enumerate(self._separators)
        }

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """对纯文本做切片（不含 bbox 关联）."""
        return self._split_text(text, list(self._separators))

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """对 Document 列表做切片，保留 metadata 并关联 bbox."""
        texts = [doc.page_content for doc in documents]
        metadatas = [copy.deepcopy(doc.metadata) for doc in documents]
        return self.create_documents(texts, metadatas=metadatas)

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """核心方法：把多段文本切片，并为每片关联 bbox/page/type.

        要求 metadata 中存在 bboxes/pages/indexes/types 四个并行数组
        （由 parser.py 生成）。
        """
        documents: List[Document] = []
        metadatas = metadatas or [{} for _ in texts]

        for i, text in enumerate(texts):
            base_meta = copy.deepcopy(metadatas[i])

            # 弹出 bbox 元数据字段（消费后从 base_meta 中移除，避免重复继承）
            indexes = base_meta.pop("indexes", [])
            pages_list = base_meta.pop("pages", [])
            types = base_meta.pop("types", [])
            bboxes = base_meta.pop("bboxes", [])

            if not indexes or not bboxes:
                # 无 bbox 元数据时按普通文本切片
                for chunk_text in self.split_text(text):
                    new_meta = copy.deepcopy(base_meta)
                    new_meta["chunk_index"] = len(documents)
                    documents.append(Document(page_content=chunk_text, metadata=new_meta))
                continue

            # 有 bbox 关联
            searcher = IntervalSearch(indexes)
            split_texts = self.split_text(text)
            # search_pos 指向下次 find 的起点。
            # 旧实现用 char_search_pos+1 推进,当 overlap>0 时相邻 chunk 共享前后缀,
            # +1 可能匹配到错误的副本位置(或返回 -1)。改为按 "已消费长度 - overlap" 推进。
            search_pos = 0

            for chunk_text in split_texts:
                # 找到 chunk 在原 text 中的位置
                char_search_pos = text.find(chunk_text, search_pos)
                if char_search_pos == -1:
                    # 极端情况：找不到，赋予基础元数据
                    new_meta = copy.deepcopy(base_meta)
                    new_meta["chunk_index"] = len(documents)
                    new_meta["chunk_bboxes"] = []
                    new_meta["chunk_type"] = "text"
                    new_meta["page"] = new_meta.get("page", 0)
                    new_meta["char_start"] = -1
                    new_meta["char_end"] = -1
                    documents.append(Document(page_content=chunk_text, metadata=new_meta))
                    continue

                # char_start 闭/char_end 开 (Python 半开约定): text[char_start:char_end] = chunk_text
                # IntervalSearch.find() 需要闭合区间,所以下面 inter0 用 end-1。
                inter0 = [char_search_pos, char_search_pos + len(chunk_text) - 1]
                matched_indices = searcher.find(inter0)
                # 推进搜索光标: 按已匹配位置 + chunk大小 - overlap,
                # 确保相邻 chunk 的 search_pos 正确越过当前 chunk。
                search_pos = char_search_pos + len(chunk_text) - self._chunk_overlap
                if search_pos < char_search_pos:
                    search_pos = char_search_pos + len(chunk_text)
                if not matched_indices:
                    # 找不到关联 element
                    new_meta = copy.deepcopy(base_meta)
                    new_meta["chunk_index"] = len(documents)
                    new_meta["chunk_bboxes"] = []
                    new_meta["chunk_type"] = "text"
                    new_meta["page"] = new_meta.get("page", 0)
                    new_meta["char_start"] = char_search_pos
                    new_meta["char_end"] = char_search_pos + len(chunk_text)
                    documents.append(Document(page_content=chunk_text, metadata=new_meta))
                    continue

                # 收集 chunk 跨越的所有 element 的 bbox / page
                chunk_bboxes = []
                for j in matched_indices:
                    chunk_bboxes.append({
                        "page": pages_list[j] if j < len(pages_list) else 0,
                        "bbox": bboxes[j] if j < len(bboxes) else None,
                    })

                # 主导 type
                if types:
                    type_counter = Counter(types[j] for j in matched_indices)
                    chunk_type = type_counter.most_common(1)[0][0]
                else:
                    chunk_type = "text"

                new_meta = copy.deepcopy(base_meta)
                new_meta["chunk_index"] = len(documents)
                new_meta["chunk_bboxes"] = chunk_bboxes
                new_meta["chunk_type"] = chunk_type
                new_meta["char_start"] = char_search_pos
                new_meta["char_end"] = char_search_pos + len(chunk_text)
                if chunk_bboxes:
                    new_meta["page"] = chunk_bboxes[0]["page"]
                documents.append(Document(page_content=chunk_text, metadata=new_meta))

        log_step(logger, "SPLIT_DOCUMENTS", f"{len(texts)} docs → {len(documents)} chunks")
        return documents

    # ------------------------------------------------------------------
    # 内部：递归切分
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        separator = separators[-1] if separators else ""
        new_separators: List[str] = []
        rule = "after"

        for i, _s in enumerate(separators):
            sep_pat = _s if self._is_separator_regex else re.escape(_s)
            rule = self.separator_rule_map.get(_s, "after")
            if _s == "":
                separator = _s
                break
            if re.search(sep_pat, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        sep_pat = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, sep_pat, bool(self._keep_separator), rule)

        final_chunks: List[str] = []
        good_splits: List[str] = []
        sep_str = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, sep_str)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # 已耗尽所有分隔符,但 s 仍超长。
                    # 旧实现直接 append(s),违反 chunk_size 上限。
                    # 改为按 chunk_size 硬切(带 overlap),保证产物均不超长。
                    final_chunks.extend(self._hard_split(s))
                else:
                    final_chunks.extend(self._split_text(s, new_separators))

        if good_splits:
            merged = self._merge_splits(good_splits, sep_str)
            final_chunks.extend(merged)

        return final_chunks

    def _hard_split(self, text: str) -> List[str]:
        """按 chunk_size 强制硬切,作为 separators 全耗尽时的兜底.

        步长 = chunk_size - chunk_overlap;chunk_overlap < chunk_size 由 __init__ 保证。
        """
        if self._length_function(text) <= self._chunk_size:
            return [text]
        step = self._chunk_size - self._chunk_overlap
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            chunks.append(text[i:i + self._chunk_size])
            i += step
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并小段到接近 chunk_size，但保留 overlap."""
        chunks: List[str] = []
        current = ""
        for s in splits:
            if not s:
                continue
            if not current:
                current = s
                continue
            if self._length_function(current) + self._length_function(s) <= self._chunk_size:
                current = current + separator + s if separator else current + s
            else:
                chunks.append(current)
                # overlap: 取 current 末尾 chunk_overlap 字符
                if self._chunk_overlap > 0 and self._length_function(current) > self._chunk_overlap:
                    overlap_text = current[-self._chunk_overlap:]
                    current = (overlap_text + separator + s) if separator else (overlap_text + s)
                else:
                    current = s
        if current:
            chunks.append(current)
        return chunks
