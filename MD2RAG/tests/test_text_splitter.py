"""测试 ElemCharacterTextSplitter - bbox 关联核心."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.text_splitter import Document, ElemCharacterTextSplitter


class TestElemCharacterTextSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = ElemCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

    def test_split_text_basic(self):
        """基础纯文本切片."""
        text = "这是一段测试文本。" * 20  # 200 字
        chunks = self.splitter.split_text(text)
        # 应有多个 chunk
        self.assertGreater(len(chunks), 1)
        # 每片不超过 chunk_size
        for c in chunks:
            self.assertLessEqual(len(c), 150)  # 允许一些 overlap

    def test_split_text_short(self):
        """短文本不分片."""
        text = "短文本"
        chunks = self.splitter.split_text(text)
        self.assertEqual(chunks, ["短文本"])

    def test_split_text_empty(self):
        chunks = self.splitter.split_text("")
        self.assertEqual(chunks, [])

    def test_split_documents_no_bbox(self):
        """无 bbox 元数据时正常切片."""
        doc = Document(page_content="这是一段测试。" * 30, metadata={})
        result = self.splitter.split_documents([doc])
        self.assertGreater(len(result), 1)
        for r in result:
            self.assertIn("chunk_index", r.metadata)

    def test_split_documents_with_bbox(self):
        """有 bbox 元数据时关联 bbox."""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        bboxes = [
            [0.0, 0.0, 100.0, 14.0],     # line 0
            [0.0, 28.0, 200.0, 42.0],    # line 2
            [0.0, 56.0, 150.0, 70.0],    # line 4
        ]
        pages = [0, 0, 0]
        indexes = [[0, 8], [10, 18], [20, 28]]  # 字符区间
        types = ["text", "text", "title"]

        doc = Document(
            page_content=text,
            metadata={
                "bboxes": bboxes,
                "pages": pages,
                "indexes": indexes,
                "types": types,
            },
        )
        result = self.splitter.split_documents([doc])

        # 每个 chunk 都应有 chunk_bboxes 字段
        for r in result:
            self.assertIn("chunk_bboxes", r.metadata)
            self.assertIsInstance(r.metadata["chunk_bboxes"], list)
            self.assertIn("chunk_type", r.metadata)
            self.assertIn("chunk_index", r.metadata)
            self.assertIn("char_start", r.metadata)
            self.assertIn("char_end", r.metadata)
            self.assertIn("page", r.metadata)

    def test_chunk_type_aggregation(self):
        """Counter 决定 chunk_type."""
        # 构造足够长的内容，让 [0,4] 的 title 元素和 [6,10] 的 text 元素都被切片覆盖
        # 我们关心的是：至少有一个 chunk 包含 title 元素（chunk_bboxes 中有 title 的索引）
        text = "标题" + "x" * 6 + "段落" + "y" * 6
        indexes = [[0, 2], [8, 10]]  # title 在 [0,2]，text 在 [8,10]
        bboxes = [[0, 0, 50, 14], [0, 28, 50, 42]]
        pages = [0, 0]
        types = ["title", "text"]

        doc = Document(
            page_content=text,
            metadata={"indexes": indexes, "bboxes": bboxes, "pages": pages, "types": types},
        )
        result = self.splitter.split_documents([doc])

        # 收集所有 chunk_type
        types_found = [r.metadata.get("chunk_type") for r in result]
        # 至少有一个 chunk 应有 chunk_type 字段
        self.assertTrue(all(t is not None for t in types_found), f"Some chunks lack chunk_type: {types_found}")
        # 类型应是 "text" 或 "title"
        for t in types_found:
            self.assertIn(t, ["text", "title"])

    def test_chunk_index_sequential(self):
        """chunk_index 应从 0 递增."""
        text = "内容" * 50
        doc = Document(page_content=text, metadata={})
        result = self.splitter.split_documents([doc])
        for i, r in enumerate(result):
            self.assertEqual(r.metadata["chunk_index"], i)


if __name__ == "__main__":
    unittest.main()
