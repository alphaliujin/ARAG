"""测试 ParentChildRetriever - 父子块双 collection 检索."""

import sys
import tempfile
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.embedder import ChromaDefaultEmbedder
from md2rag.parent_child_retriever import ParentChildRetriever
from md2rag.vector_store import VectorStore


class TestParentChildRetriever(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.embedder = ChromaDefaultEmbedder()
        self.store = VectorStore(db_dir=self.tmpdir, embedder=self.embedder)
        self.retriever = ParentChildRetriever(
            vector_store=self.store,
            embedder=self.embedder,
            classification="public",
            collection_prefix="",
        )

    def test_collection_names(self):
        self.assertEqual(self.retriever.parent_collection, "public_parent")
        self.assertEqual(self.retriever.child_collection, "public_child")

    def test_add_parent_child_pair(self):
        parent_text = "银渐层是英国短毛猫的常见色系，圆脸蛋银白渐变毛绿眼睛。"
        children_texts = [
            "一、起源：英国短毛猫的历史可追溯至古罗马时期。",
            "二、外观：圆脸、浓密被毛、性格温顺。",
            "三、护理：定期梳毛、检查耳朵、清理泪痕。",
        ]
        doc_id = "test-doc-001"

        parent = (parent_text, {"doc_id": doc_id, "source": "银渐层"})
        children = [
            (ct, {"parent_doc_id": doc_id, "chunk_index": i, "source": "银渐层"})
            for i, ct in enumerate(children_texts)
        ]

        p_count, c_count = self.retriever.add_parent_child_pair(parent, children)
        self.assertEqual(p_count, 1)
        self.assertEqual(c_count, 3)

        # 验证 collection 状态
        stats = self.retriever.get_stats()
        self.assertEqual(stats["parent_count"], 1)
        self.assertEqual(stats["child_count"], 3)

    def test_search_returns_parents(self):
        parent_text = "银渐层是英国短毛猫的常见色系，圆脸蛋银白渐变毛绿眼睛。"
        children_texts = [
            "一、起源：英国短毛猫的历史可追溯至古罗马时期。",
            "二、外观：圆脸、浓密被毛、性格温顺。",
        ]
        doc_id = "doc-001"
        parent = (parent_text, {"doc_id": doc_id, "source": "银渐层"})
        children = [
            (ct, {"parent_doc_id": doc_id, "chunk_index": i})
            for i, ct in enumerate(children_texts)
        ]
        self.retriever.add_parent_child_pair(parent, children)

        # 搜一个与子块相关的问题
        hits = self.retriever.search("英国短毛猫历史起源", n_results=2, return_parents=True)
        self.assertGreater(len(hits), 0)

        # 至少有一个 hit 应包含 parent_text
        any_with_parent = any(h.parent_text and "银渐层" in h.parent_text for h in hits)
        self.assertTrue(any_with_parent, "Expected at least one hit to include parent text")

    def test_search_empty_collection(self):
        hits = self.retriever.search("anything", n_results=5)
        self.assertEqual(hits, [])

    def test_clear(self):
        parent = ("text", {"source": "x"})
        children = [("c1", {"parent_doc_id": "ignored"})]
        self.retriever.add_parent_child_pair(parent, children)
        self.assertGreater(self.retriever.get_stats()["parent_count"], 0)

        self.retriever.clear()
        self.assertEqual(self.retriever.get_stats()["parent_count"], 0)
        self.assertEqual(self.retriever.get_stats()["child_count"], 0)

    def test_multiple_docs_dedup(self):
        """检索时同一 doc_id 的多个 children 应只回查一次 parent."""
        doc_id = "shared-doc"
        parent = ("PARENT TEXT", {"doc_id": doc_id})
        children = [
            (f"child {i}", {"parent_doc_id": doc_id, "chunk_index": i})
            for i in range(5)
        ]
        self.retriever.add_parent_child_pair(parent, children)

        # 用子块文本中的内容查询，确保多个 child 命中同一 parent
        hits = self.retriever.search("child", n_results=5)
        # 验证去重：5 个 child 都应归属于同一 doc_id
        doc_ids = [h.metadata.get("doc_id") or h.metadata.get("parent_doc_id") for h in hits]
        # 可能结果数量 <= 5（去重后），但每个 doc_id 应只出现一次
        self.assertEqual(len(doc_ids), len(set(doc_ids)), "Expected deduped doc_ids")


if __name__ == "__main__":
    unittest.main()
