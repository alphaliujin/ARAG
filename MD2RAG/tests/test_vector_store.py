"""测试 vector_store 基础操作."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.embedder import ChromaDefaultEmbedder
from md2rag.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """用临时目录隔离测试."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.embedder = ChromaDefaultEmbedder()  # 384 维，免依赖 Ollama
        self.store = VectorStore(
            db_dir=self.tmpdir,
            embedder=self.embedder,
            collection_prefix="test",
        )

    def test_create_and_get_collection(self):
        coll = self.store.get_or_create_collection("public")
        self.assertIsNotNone(coll)
        self.assertEqual(coll.count(), 0)

    def test_add_chunks(self):
        texts = ["hello world", "goodbye world"]
        metas = [{"source": "a"}, {"source": "b"}]
        ids = ["1", "2"]
        result = self.store.add_chunks("public", texts, metas, ids)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["added"], 2)

        coll = self.store.get_or_create_collection("public")
        self.assertEqual(coll.count(), 2)

    def test_search(self):
        self.store.add_chunks(
            "public",
            ["苹果是一种水果", "香蕉是黄色的", "汽车是交通工具"],
            [{"source": f"f{i}"} for i in range(3)],
            ["1", "2", "3"],
        )
        results = self.store.search("public", ["水果"], n_results=2)
        self.assertIn("documents", results)
        # 第一个命中应是"苹果"
        self.assertIn("苹果", results["documents"][0][0])

    def test_clear_collection(self):
        self.store.add_chunks("public", ["x"], [{"source": "test"}], ["1"])
        self.assertEqual(self.store.get_or_create_collection("public").count(), 1)
        self.store.clear_collection("public")
        # clear 后 collection 会被重建为空
        self.assertEqual(self.store.get_or_create_collection("public").count(), 0)

    def test_reset_all(self):
        self.store.add_chunks("public", ["a"], [{"source": "test"}], ["1"])
        self.store.add_chunks("confidential", ["b"], [{"source": "test"}], ["1"])
        self.store.reset_all()
        self.assertEqual(self.store.get_or_create_collection("public").count(), 0)
        self.assertEqual(self.store.get_or_create_collection("confidential").count(), 0)


if __name__ == "__main__":
    unittest.main()
