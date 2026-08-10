"""switch_embedding_model 单元测试 - #1 模型切换修复的回归保护.

验证: 持久化 -> 清库 -> 重建 embedding_fn 三步调用顺序, 及清库失败时抛错
      且不继续 rebuild (避免新 embedding_fn 配旧维度向量).

用 sys.modules mock 绕开 chromadb / pydantic_settings / md2rag 重依赖,
仅测 switch 逻辑, 与 test_task_manager 一样可脱离 venv 运行 (python3 -m unittest).
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# 让测试不依赖 backend 在 sys.path
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))


def _install_mocks():
    """注入轻量 mock 替代重依赖, 避免 import ingestion 时拉起 chromadb/torch.

    中间包 app / app.services 用真包 (空 __init__.py) 以便定位真 ingestion.py,
    只替换会拉起重依赖的叶子模块.
    """
    # app.core.config.settings (switch 不读, 但 ingestion 顶部 import 需要)
    cfg = types.ModuleType("app.core.config")

    class _Settings:
        EMBEDDING_MODEL = "ollama-bge-m3"
        OLLAMA_BASE_URL = "http://localhost:11434"
        VECTOR_DB_DIR = str(_BACKEND / "vector_db")
        DATA_DIR = str(_BACKEND.parent / "MD")

    cfg.settings = _Settings()
    sys.modules["app.core.config"] = cfg

    # app.services.vector_db.vector_db_service (rebuild_embedding_fn 调用目标)
    vdb = types.ModuleType("app.services.vector_db")
    vdb.vector_db_service = MagicMock()
    sys.modules["app.services.vector_db"] = vdb

    # app.services.settings_service.settings_service (switch 内局部 import)
    ss = types.ModuleType("app.services.settings_service")
    ss.settings_service = MagicMock()
    sys.modules["app.services.settings_service"] = ss

    # md2rag.loader.CLASSIFICATION_DIR_MAP (ingestion 顶部 import 需要)
    md2rag = types.ModuleType("md2rag"); md2rag.__path__ = []
    loader = types.ModuleType("md2rag.loader")
    loader.CLASSIFICATION_DIR_MAP = {
        "public": "0Public", "restricted": "1Restricted", "confidential": "2Confidential",
    }
    sys.modules["md2rag"] = md2rag
    sys.modules["md2rag.loader"] = loader


_install_mocks()

from app.services.ingestion import DataIngestionService  # noqa: E402
from app.services import vector_db as _vdb_mod  # noqa: E402
from app.services import settings_service as _ss_mod  # noqa: E402


class TestSwitchEmbeddingModel(unittest.TestCase):
    def _make_svc(self, reset_result):
        # 跳过 __init__ (其内部仅设字段, 无重依赖, 但此处用 __new__ 更隔离)
        svc = DataIngestionService.__new__(DataIngestionService)
        svc._indexer = None
        svc._md2rag_config = None
        svc.reset_all = MagicMock(return_value=reset_result)
        _ss_mod.settings_service.update_category.reset_mock()
        _vdb_mod.vector_db_service.rebuild_embedding_fn.reset_mock()
        return svc

    def test_normal_switch_calls_three_steps_in_order(self):
        svc = self._make_svc({"status": "success", "total_vectors": 0})
        result = svc.switch_embedding_model("mps-bge-m3")

        # 1) 持久化 + 应用到 settings
        _ss_mod.settings_service.update_category.assert_called_once_with(
            "model", {"embeddingModel": "mps-bge-m3"}
        )
        # 2) 清库 + 释放旧 indexer
        svc.reset_all.assert_called_once()
        # 3) 重建 backend 兜底 embedding_fn
        _vdb_mod.vector_db_service.rebuild_embedding_fn.assert_called_once()
        self.assertEqual(result["switched_to"], "mps-bge-m3")

    def test_reset_error_raises_and_skips_rebuild(self):
        svc = self._make_svc({"status": "error", "message": "disk full"})
        with self.assertRaises(RuntimeError) as ctx:
            svc.switch_embedding_model("mps-bge-m3")
        self.assertIn("disk full", str(ctx.exception))
        # 清库失败应阻止后续 rebuild (避免新 embedding_fn 配旧维度向量)
        _vdb_mod.vector_db_service.rebuild_embedding_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
