"""测试 LLM 摘要模块 - 用 mock LLM 客户端."""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from unittest.mock import MagicMock

from md2rag.llm_summary import (
    LLMClient,
    extract_title,
    extract_abstract,
    build_title_prompt,
)


class TestLLMSummary(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.generate.return_value = "测试标题"

    def test_extract_title_sync(self):
        text = "这是一篇关于敏感信息识别的技术文档" * 100
        result = extract_title(self.mock_llm, text)
        self.assertEqual(result, "测试标题")
        self.mock_llm.generate.assert_called_once()

    def test_extract_title_truncates_text(self):
        long_text = "x" * 10000
        extract_title(self.mock_llm, long_text, max_length=100)
        # 传给 generate 的 prompt 应被截断
        call_args = self.mock_llm.generate.call_args
        self.assertLess(len(call_args[0][0]), 200)  # 100 字 + 提示模板

    def test_extract_title_custom_prompt(self):
        extract_title(self.mock_llm, "text", abstract_prompt="自定义 system prompt")
        call_args = self.mock_llm.generate.call_args
        self.assertEqual(call_args[1]["system"], "自定义 system prompt")

    def test_extract_abstract(self):
        self.mock_llm.generate.return_value = "本文讨论了敏感信息的识别方法。"
        text = "敏感信息识别是..." * 50
        result = extract_abstract(self.mock_llm, text)
        self.assertIn("敏感信息", result)
        self.mock_llm.generate.assert_called_once()

    def test_build_title_prompt(self):
        sys_msg, user_msg = build_title_prompt("测试文本")
        self.assertIn("标题", sys_msg)
        self.assertIn("测试文本", user_msg)


class TestAsyncExtractTitle(unittest.TestCase):
    def test_async_extract_title(self):
        # 异步函数不能直接 unittest，用 asyncio.run
        from md2rag.llm_summary import async_extract_title

        class AsyncMockLLM:
            async def agenerate(self, prompt, system="", temperature=0.01):
                return "异步标题"

        result = asyncio.run(async_extract_title(AsyncMockLLM(), "测试"))
        self.assertEqual(result, "异步标题")


if __name__ == "__main__":
    unittest.main()
