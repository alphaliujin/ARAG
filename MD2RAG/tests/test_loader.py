"""测试 ChunkLoader - 加载 X2MD 切片 JSON."""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.loader import ChunkLoader, ChunkFileType, ChunkRecord


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class TestChunkRecord(unittest.TestCase):
    def test_construction(self):
        r = ChunkRecord(
            text="测试文本",
            chunk_index=0,
            source="test.md",
            doc_id="abc-123",
            is_parent=True,
        )
        self.assertEqual(r.text, "测试文本")
        self.assertTrue(r.is_parent)
        self.assertFalse(r.is_child)

    def test_embedding_text_pure_content_only(self):
        """embedding_text 应只返回纯正文, 不含元数据前缀.

        旧版会把 [文档:xxx]、[摘要:xxx] 等前缀拼到正文前面再嵌入,
        导致不同文件名的相同内容向量不一致, 也让不相关内容因前缀贡献
        约 0.5 的基础相似度噪声。现已修复为只用纯正文做嵌入。
        """
        r = ChunkRecord(
            text="正文",
            chunk_index=0,
            source="test.md",
            document_name="测试文档",
            abstract="这是摘要",
            doc_id="abc-123",
            is_child=True,
            parent_doc_id="abc-123",
        )
        emb = r.embedding_text
        # ★ 核心断言: embedding_text 只返回纯正文, 不含元数据前缀
        self.assertEqual(emb, "正文")
        # 确保元数据前缀不再出现 (旧版行为, 会引入噪声)
        self.assertNotIn("[文档:", emb)
        self.assertNotIn("[摘要:", emb)
        self.assertNotIn("[父块ID:", emb)
        self.assertNotIn("[类型:", emb)
        # 元数据信息仍可通过 to_metadata() 获取, 不丢失
        meta = r.to_metadata("public", "/path/to/test.md")
        self.assertEqual(meta["document_name"], "测试文档")
        self.assertEqual(meta["abstract"], "这是摘要")
        self.assertEqual(meta["doc_id"], "abc-123")

    def test_to_metadata(self):
        r = ChunkRecord(
            text="x",
            chunk_index=0,
            source="test",
            doc_id="d1",
            is_parent=True,
        )
        meta = r.to_metadata("public", "/path/to/test.parents.json")
        self.assertEqual(meta["classification"], "public")
        self.assertEqual(meta["doc_id"], "d1")
        self.assertEqual(meta["is_parent"], True)
        self.assertIn("original_md_path", meta)


class TestChunkLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.loader = ChunkLoader(self.tmpdir)

    def _make_parents(self, dir_name="0Public", file_name="test", n=2):
        path = self.tmpdir / dir_name / f"{file_name}.md" / f"{file_name}.parents.json"
        data = [
            {
                "text": f"parent {i}",
                "metadata": {
                    "chunk_index": i,
                    "source": file_name,
                    "doc_id": f"parent-{i}",
                    "abstract": f"摘要 {i}",
                },
            }
            for i in range(n)
        ]
        _write_json(path, data)
        return path

    def _make_children(self, dir_name="0Public", file_name="test", n=4, parent_idx=0):
        path = self.tmpdir / dir_name / f"{file_name}.md" / f"{file_name}.children.json"
        data = [
            {
                "text": f"child {i}",
                "metadata": {
                    "chunk_index": i,
                    "source": file_name,
                    "doc_id": f"parent-{parent_idx}",  # 共享 doc_id
                    "abstract": f"child {i} 摘要",
                },
            }
            for i in range(n)
        ]
        _write_json(path, data)
        return path

    def test_discover_files_no_classification(self):
        self._make_parents("0Public", "doc1", 2)
        self._make_children("0Public", "doc1", 4)
        self._make_parents("1Restricted", "doc2", 1)

        files = self.loader.discover_files()
        self.assertEqual(len(files), 3)

    def test_discover_files_with_classification(self):
        self._make_parents("0Public", "doc1", 1)
        self._make_parents("1Restricted", "doc2", 1)
        files = self.loader.discover_files("public")
        self.assertEqual(len(files), 1)
        self.assertIn("0Public", str(files[0]))

    def test_load_parents(self):
        path = self._make_parents("0Public", "doc1", 2)
        records = self.loader.load_file(path)
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertTrue(r.is_parent)
            self.assertEqual(r.source, "doc1")
            self.assertTrue(r.doc_id.startswith("parent-"))

    def test_load_children_links_to_parent(self):
        """children 加载时 parent_doc_id 应等于 doc_id."""
        path = self._make_children("0Public", "doc1", 3, parent_idx=0)
        records = self.loader.load_file(path)
        for r in records:
            self.assertTrue(r.is_child)
            self.assertEqual(r.parent_doc_id, r.doc_id)
            self.assertEqual(r.doc_id, "parent-0")

    def test_classification_from_path(self):
        self.assertEqual(self.loader.get_classification_from_path("MD/0Public/x"), "public")
        self.assertEqual(self.loader.get_classification_from_path("MD/1Restricted/x"), "restricted")
        self.assertEqual(self.loader.get_classification_from_path("MD/2Confidential/x"), "confidential")
        # 默认 public
        self.assertEqual(self.loader.get_classification_from_path("MD/other/x"), "public")

    def test_load_document_chunks_parent_child(self):
        self._make_parents("0Public", "doc1", 2)
        self._make_children("0Public", "doc1", 3, parent_idx=0)

        md_path = self.tmpdir / "0Public" / "doc1.md" / "doc1.md"
        records = self.loader.load_document_chunks(md_path, strategy="parent-child")
        # 应同时有 parent 和 child
        parents = [r for r in records if r.is_parent]
        children = [r for r in records if r.is_child]
        self.assertEqual(len(parents), 2)
        self.assertEqual(len(children), 3)

    def test_load_document_chunks_parents_only(self):
        self._make_parents("0Public", "doc1", 2)
        self._make_children("0Public", "doc1", 3, parent_idx=0)

        md_path = self.tmpdir / "0Public" / "doc1.md" / "doc1.md"
        records = self.loader.load_document_chunks(md_path, strategy="parents-only")
        for r in records:
            self.assertTrue(r.is_parent)

    def test_nonexistent_file(self):
        records = self.loader.load_file("/nonexistent/file.json")
        self.assertEqual(records, [])

    def test_invalid_json(self):
        path = self.tmpdir / "bad.json"
        path.write_text("not valid json {{{")
        records = self.loader.load_file(path)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
