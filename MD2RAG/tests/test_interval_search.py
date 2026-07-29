"""测试 IntervalSearch 类."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.interval_search import IntervalSearch


class TestIntervalSearch(unittest.TestCase):
    def setUp(self):
        # 模拟 5 个 element 的字符区间
        self.indexes = [
            [0, 10],
            [10, 25],
            [25, 40],
            [40, 60],
            [60, 80],
        ]

    def test_basic_construction(self):
        s = IntervalSearch(self.indexes)
        self.assertEqual(len(s), 5)

    def test_find_full_overlap(self):
        """完全包含一个 element."""
        s = IntervalSearch(self.indexes)
        result = s.find([0, 10])
        self.assertEqual(result, [0])

    def test_find_span_two(self):
        """跨 2 个 element."""
        s = IntervalSearch(self.indexes)
        result = s.find([20, 30])  # 跨 idx 1 (10-25) 和 idx 2 (25-40)
        self.assertEqual(result, [1, 2])

    def test_find_span_three(self):
        """跨 3 个 element."""
        s = IntervalSearch(self.indexes)
        result = s.find([15, 50])  # 跨 idx 1,2,3
        self.assertEqual(result, [1, 2, 3])

    def test_find_no_overlap_before(self):
        s = IntervalSearch(self.indexes)
        result = s.find([-10, -1])
        self.assertEqual(result, [])

    def test_find_no_overlap_after(self):
        s = IntervalSearch(self.indexes)
        result = s.find([100, 200])
        self.assertEqual(result, [])

    def test_find_touching_boundary(self):
        """刚好接触边界不算重叠（排除边界点）."""
        s = IntervalSearch(self.indexes)
        # target end = 10，idx 0 (0-10) 与 target [5,10] 重叠（5<10 且 0<10）
        # idx 1 (10-25) 与 target [5,10] 不重叠（仅边界接触：10 < 10 不成立）
        result = s.find([5, 10])
        self.assertEqual(result, [0])

    def test_find_strict(self):
        """find_strict: 严格包含（每个 element 都独立包含 target）."""
        s = IntervalSearch(self.indexes)
        # [15, 50] 没有单个 element 包含它（idx 1 [10-25] 不够大，idx 3 [40-60] 不够早）
        result = s.find_strict([15, 50])
        self.assertEqual(result, [])

        # 找一个能被完全包含的 target: [0, 8] 应被 idx 0 [0, 10] 包含
        result = s.find_strict([0, 8])
        self.assertEqual(result, [0])

        # [0, 80] 没有单个 element 包含它
        result = s.find_strict([0, 80])
        self.assertEqual(result, [])

    def test_find_strict_skips_non_containing(self):
        """find_strict 应跳过不包含 target 的 element，继续检查后续 element.

        旧 bug: 遇到 end < t_end 的 element 时立即返回 (0, -1)，
        即使后面还有满足条件的 element。修复后应继续检查。
        """
        # 场景: element [0,5] 不包含 [3,12]，但 element [2,8] 和 [3,15] 都包含
        indexes = [[0, 5], [2, 8], [3, 15]]
        s = IntervalSearch(indexes)
        # target [3, 12]: 检查条件 start<=3 AND end>=12
        # idx 0 [0,5]: start=0<=3 OK, end=5<12 FAIL → 应跳过
        # idx 1 [2,8]: start=2<=3 OK, end=8<12 FAIL → 应跳过
        # idx 2 [3,15]: start=3<=3 OK, end=15>=12 OK → 应命中
        result = s.find_strict([3, 12])
        self.assertEqual(result, [2])

    def test_unsorted_input(self):
        """乱序输入，返回下标应对应原始位置."""
        unsorted = [[60, 80], [10, 25], [0, 10], [40, 60], [25, 40]]
        s = IntervalSearch(unsorted)
        self.assertEqual(s[0], [0, 10])
        self.assertEqual(s[4], [60, 80])

        # find 应返回原始下标，而非排序后的下标
        # target [0, 80] 跨越所有 5 个 element，原始下标为 [0,1,2,3,4]
        result = s.find([0, 80])
        self.assertEqual(result, [0, 1, 2, 3, 4])

        # target [5, 22] 在排序后跨 [0,10], [10,25]，
        # 原始下标分别是 2 和 1
        result = s.find([5, 22])
        self.assertEqual(result, [1, 2])

    def test_empty(self):
        s = IntervalSearch([])
        result = s.find([0, 5])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
