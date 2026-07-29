"""测试 config 模块."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.config import MD2RAGConfig, load_config


class TestMD2RAGConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = MD2RAGConfig()
        self.assertEqual(cfg.embedding_model, "ollama-bge-m3")
        self.assertEqual(cfg.ollama_model, "bge-m3:latest")
        self.assertTrue(cfg.ollama_enabled)
        self.assertEqual(cfg.collection_prefix, "md2rag")

    def test_default_paths(self):
        cfg = MD2RAGConfig()
        from pathlib import Path as P
        self.assertIsInstance(cfg.md_dir, P)
        self.assertIsInstance(cfg.vector_db_dir, P)


class TestLoadConfig(unittest.TestCase):
    def test_load_default(self):
        cfg = load_config()
        self.assertIsInstance(cfg, MD2RAGConfig)

    def test_load_with_md_dir_resolution(self):
        cfg = load_config()
        # 默认 md_dir 应是相对路径被解析为绝对路径
        self.assertTrue(cfg.md_dir.is_absolute())


if __name__ == "__main__":
    unittest.main()
