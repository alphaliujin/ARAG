"""区间搜索工具 - 用于 chunk 关联到原文档 element 的 bbox.

源自 bisheng 的 IntervalSearch 思路：每个 element（text/table/image）在原文中有
字符区间 [start, end]；chunk 在原文中有字符区间，需要找出该区间跨越了哪些 element，
从而关联到对应的 bbox / page / type。
"""

from __future__ import annotations

import bisect
from typing import List, Sequence


class IntervalSearch:
    """区间搜索器.

    接受一个 [[start, end], ...] 区间列表（无需按 start 升序），内部排序后支持：
    - find([s, e]) → 命中区间的原始下标列表（所有与 target 有"实质性重叠"的区间）
    - find_strict([s, e]) → 严格包含 target 的原始下标列表

    返回的下标对应原始输入 indexes 的位置，而非内部排序后的位置，
    因此可直接用于索引原始的 pages_list / bboxes / types 等并行数组。

    实质性重叠定义：两个区间 [a,b] 与 [c,d] 有重叠当且仅当 a < d 且 c < b。
    这样 [0,10] 与 [10,25] 不算重叠（仅边界点接触），避免把无关 element 关联进 chunk。
    """

    def __init__(self, indexes: Sequence[Sequence[int]]):
        # 按 start 升序排列，同时保留原始下标映射
        sorted_with_orig = sorted(enumerate(indexes), key=lambda x: x[1][0])
        self._orig_indices: List[int] = [orig_idx for orig_idx, iv in sorted_with_orig]
        sorted_indexes = [iv for _, iv in sorted_with_orig]
        self._starts: List[int] = [iv[0] for iv in sorted_indexes]
        self._ends: List[int] = [iv[1] for iv in sorted_indexes]
        self._indexes: List[List[int]] = [list(iv) for iv in sorted_indexes]

    def find(self, target: Sequence[int]) -> List[int]:
        """找与 [target[0], target[1]] 有"实质性重叠"的所有区间原始下标.

        实质性重叠: a < d 且 c < b（排除仅边界接触的情况）

        Returns:
            匹配的原始下标列表（升序排列）。
            空列表表示无命中。
            下标对应原始 indexes 输入的位置，可直接索引 pages_list / bboxes / types。
        """
        if not self._indexes:
            return []
        t_start, t_end = target[0], target[1]
        # hi: 最后一个 start < t_end。_starts 升序, 用 bisect 降为 O(log n)
        # (原线性倒扫对大文档数万 element 时每个 chunk 都 O(n))
        hi = bisect.bisect_left(self._starts, t_end) - 1
        if hi < 0:
            return []
        # lo: 第一个 end > t_start。_ends 未排序 (按 start 排序后 ends 可能乱序),
        # 无法 bisect; 但范围已缩到 [0, hi], 线性扫描成本上限为 hi。
        lo = -1
        for i in range(hi + 1):
            if self._ends[i] > t_start:  # 严格大于, 排除边界接触
                lo = i
                break
        if lo == -1:
            return []
        # 映射回原始下标列表，但需验证实际重叠
        # 注意：ends 未排序（按 start 排序后 ends 可能乱序），
        # 因此 [lo,hi] 范围可能包含不实际重叠的区间，需逐个验证
        results = []
        for j in range(lo, hi + 1):
            a, b = self._indexes[j]
            # 实质性重叠: a < t_end AND t_start < b
            if a < t_end and t_start < b:
                results.append(self._orig_indices[j])
        return sorted(results)

    def find_strict(self, target: Sequence[int]) -> List[int]:
        """找"严格包含 target"的所有 element 原始下标.

        严格包含: idx.start <= target.start AND idx.end >= target.end.
        返回原始下标列表（升序排列），仅包含满足条件的 element。
        空列表表示无匹配。
        """
        if not self._indexes:
            return []
        t_start, t_end = target[0], target[1]

        # 找所有满足 idx.start <= t_start AND idx.end >= t_end 的 sorted 下标
        matched_sorted = []
        for i in range(len(self._starts)):
            if self._starts[i] <= t_start and self._ends[i] >= t_end:
                matched_sorted.append(i)

        if not matched_sorted:
            return []

        # 映射回原始下标列表
        orig_indices = sorted(self._orig_indices[j] for j in matched_sorted)
        return orig_indices

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, idx: int) -> List[int]:
        return self._indexes[idx]
