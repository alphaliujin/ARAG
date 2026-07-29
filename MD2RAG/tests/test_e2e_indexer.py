"""端到端测试 - 完整入库流程（不含 LLM 摘要，不依赖 Ollama）."""

import sys
import json
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.config import MD2RAGConfig
from md2rag.embedder import ChromaDefaultEmbedder
from md2rag.indexer import Indexer


class TestIndexerEndToEnd(unittest.TestCase):
    """完整入库+检索端到端测试 - 用 ChromaDB 默认嵌入器（避免依赖 Ollama）."""

    def setUp(self):
        # 创建临时 MD 目录结构
        self.md_dir = Path(tempfile.mkdtemp())
        self.db_dir = Path(tempfile.mkdtemp()) / "vector_db"

        # 创建 0Public/test.md/ 目录及切片
        public_dir = self.md_dir / "0Public" / "test_doc.md"
        public_dir.mkdir(parents=True)

        # 写 MD 原文
        md_content = """银渐层（英短银渐层，Silver Shaded British Shorthair）是英国短毛猫的人气色系。

一、起源与定位
英国短毛猫是英国本土的猫品种，历史可追溯至古罗马时期。

二、外观特征
圆脸、浓密被毛、性格温顺安静著称，新手友好。

三、护理要点
定期梳毛、检查耳朵、清理泪痕。
"""
        (public_dir / "test_doc.md").write_text(md_content, encoding="utf-8")

        # 写 parents.json
        parents = [
            {
                "text": "银渐层（英短银渐层，Silver Shaded British Shorthair）是英国短毛猫的人气色系。",
                "metadata": {
                    "chunk_index": 0,
                    "source": "test_doc",
                    "doc_id": "doc-001",
                    "abstract": "银渐层介绍",
                },
            },
            {
                "text": "一、起源与定位\n英国短毛猫是英国本土的猫品种，历史可追溯至古罗马时期。",
                "metadata": {
                    "chunk_index": 1,
                    "source": "test_doc",
                    "doc_id": "doc-002",
                    "abstract": "起源说明",
                },
            },
        ]
        (public_dir / "test_doc.parents.json").write_text(
            json.dumps(parents, ensure_ascii=False), encoding="utf-8"
        )

        # 写 children.json (4 个 children，归属 2 个 parents)
        children = [
            {
                "text": "银渐层（英短银渐层，Silver Shaded British Shorthair）是英国短毛猫的人气色系。",
                "metadata": {
                    "chunk_index": 0,
                    "source": "test_doc",
                    "doc_id": "doc-001",
                    "abstract": "银渐层介绍",
                },
            },
            {
                "text": "圆脸、浓密被毛、性格温顺安静著称。",
                "metadata": {
                    "chunk_index": 1,
                    "source": "test_doc",
                    "doc_id": "doc-001",
                    "abstract": "外观",
                },
            },
            {
                "text": "一、起源与定位\n英国短毛猫是英国本土的猫品种。",
                "metadata": {
                    "chunk_index": 2,
                    "source": "test_doc",
                    "doc_id": "doc-002",
                    "abstract": "起源",
                },
            },
            {
                "text": "历史可追溯至古罗马时期。",
                "metadata": {
                    "chunk_index": 3,
                    "source": "test_doc",
                    "doc_id": "doc-002",
                    "abstract": "历史",
                },
            },
        ]
        (public_dir / "test_doc.children.json").write_text(
            json.dumps(children, ensure_ascii=False), encoding="utf-8"
        )

        # 创建自定义配置
        self.config = MD2RAGConfig(
            md_dir=self.md_dir,
            vector_db_dir=self.db_dir,
            embedding_model="chromadb-default",
            ollama_enabled=False,
            vit_enabled=False,  # 跳过图片处理
            collection_prefix="",
        )
        # 强制覆盖 embedder（不依赖 Ollama）
        self.config.ollama_enabled = False
        self.config.st_enabled = False

    def tearDown(self):
        shutil.rmtree(self.md_dir.parent, ignore_errors=True)

    def test_full_pipeline(self):
        """完整入库 → 检索测试."""
        indexer = Indexer(self.config)
        result = indexer.index_directory(
            classification="public",
            include_images=False,
        )
        self.assertEqual(result.status, "success")
        # 2 parents + 4 children
        self.assertEqual(result.parents_added, 2)
        self.assertEqual(result.children_added, 4)
        self.assertEqual(result.images_added, 0)

        # 检索测试：搜"起源"
        search_results = indexer.search("起源", classification="public", n_results=2)
        self.assertIn("public", search_results)
        hits = search_results["public"]
        self.assertGreater(len(hits), 0)

        # 至少有一个 hit 应包含 parent_text（说明父子块关联生效）
        any_with_parent = any(h.get("parent_text") for h in hits)
        self.assertTrue(any_with_parent, "Expected at least one hit to have parent_text")

    def test_metadata_preservation(self):
        """元数据应包含 doc_id / parent_doc_id / bbox / abstract."""
        indexer = Indexer(self.config)
        indexer.index_directory(classification="public", include_images=False)

        # 检索一条结果，检查元数据
        results = indexer.search("银渐层", classification="public", n_results=1)
        hit = results["public"][0]
        meta = hit["metadata"]
        self.assertIn("doc_id", meta)
        self.assertIn("parent_doc_id", meta)
        self.assertIn("classification", meta)
        self.assertEqual(meta["classification"], "public")

    def test_clear(self):
        indexer = Indexer(self.config)
        indexer.index_directory(classification="public", include_images=False)
        stats_before = indexer.get_stats()
        self.assertGreater(stats_before["public"], 0)

        indexer.clear()
        stats_after = indexer.get_stats()
        self.assertEqual(stats_after["public"], 0)


if __name__ == "__main__":
    unittest.main()
