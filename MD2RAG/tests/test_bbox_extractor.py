"""测试 bbox_extractor 模块."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.bbox_extractor import (
    assign_bboxes_to_chunk,
    get_bboxes_for_record,
    make_synthetic_bboxes,
)


class TestMakeSyntheticBboxes(unittest.TestCase):
    def test_empty_text(self):
        result = make_synthetic_bboxes("")
        self.assertEqual(result, [])

    def test_single_line(self):
        result = make_synthetic_bboxes("hello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "hello world")
        self.assertEqual(result[0].line_no, 0)
        self.assertEqual(result[0].char_start, 0)
        self.assertEqual(result[0].char_end, 11)

    def test_multiline(self):
        text = "第一行\n第二行\n\n第四行\n"
        result = make_synthetic_bboxes(text)
        # 跳过空行，应有 3 个 LineInfo
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].line_no, 0)
        self.assertEqual(result[1].line_no, 1)
        self.assertEqual(result[2].line_no, 3)  # 跳过空行

    def test_bbox_dimensions(self):
        result = make_synthetic_bboxes("hello", page_width=595, page_height=842)
        bbox = result[0].bbox
        self.assertEqual(len(bbox), 4)
        self.assertEqual(bbox[0], 0.0)        # x0
        self.assertEqual(bbox[2], 595.0)      # x1
        self.assertEqual(bbox[1], 0.0)        # y0
        self.assertEqual(bbox[3], 14.0)       # y1 = line_height


class TestGetBboxesForRecord(unittest.TestCase):
    def test_basic_chunk(self):
        source = "第一行\n第二行\n第三行\n"
        chunk_text = "第二行"
        result = get_bboxes_for_record(chunk_text, source)
        # 应至少命中一行
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["line_no"], 1)
        self.assertIn("bbox", result[0])

    def test_multi_line_chunk(self):
        source = "line1\nline2\nline3\nline4\n"
        # chunk 横跨 line1, line2
        chunk_text = "line1\nline2"
        result = get_bboxes_for_record(chunk_text, source)
        # 应命中 2 行
        line_nos = [r["line_no"] for r in result]
        self.assertIn(0, line_nos)
        self.assertIn(1, line_nos)

    def test_explicit_start(self):
        source = "abcdef\nline2\nline3\n"
        result = get_bboxes_for_record("line2", source, record_char_start=7)
        self.assertEqual(result[0]["line_no"], 1)

    def test_not_found(self):
        result = get_bboxes_for_record("nonexistent_text_xyz", "short source")
        self.assertEqual(result, [])

    def test_empty_source(self):
        result = get_bboxes_for_record("anything", "")
        self.assertEqual(result, [])


class TestAssignBboxesToChunk(unittest.TestCase):
    def test_assign(self):
        text = "alpha\nbeta\ngamma\n"
        from md2rag.bbox_extractor import make_synthetic_bboxes
        line_infos = make_synthetic_bboxes(text)
        # chunk 包含 "beta" 字符在第 6-10 位
        result = assign_bboxes_to_chunk("beta", 6, line_infos)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["line_no"], 1)


if __name__ == "__main__":
    unittest.main()
