"""测试 embedder 基础功能."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.embedder import (
    ChromaDefaultEmbedder,
    Embedder,
    create_embedder,
)


class TestChromaDefaultEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = ChromaDefaultEmbedder()

    def test_dimension(self):
        self.assertEqual(self.embedder.dimension, 384)

    def test_embed_single(self):
        result = self.embedder.embed(["hello world"])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 384)

    def test_embed_multiple(self):
        result = self.embedder.embed(["hello", "world", "test"])
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertEqual(len(r), 384)

    def test_embed_empty(self):
        result = self.embedder.embed([])
        self.assertEqual(result, [])

    def test_embed_query(self):
        result = self.embedder.embed_query("test query")
        self.assertEqual(len(result), 384)


class TestCreateEmbedder(unittest.TestCase):
    def test_chromadb_default(self):
        e = create_embedder(model_type="chromadb-default")
        self.assertEqual(e.dimension, 384)

    def test_unknown_falls_back_to_default(self):
        e = create_embedder(model_type="unknown-model")
        self.assertEqual(e.dimension, 384)


if __name__ == "__main__":
    unittest.main()
