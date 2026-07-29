"""测试 logger 模块."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from md2rag.logger import get_logger, log_step, log_timing, log_memory


class TestLogger(unittest.TestCase):
    def test_get_logger(self):
        logger = get_logger("test")
        self.assertIsNotNone(logger)

    def test_get_logger_same_name(self):
        a = get_logger("same_name")
        b = get_logger("same_name")
        self.assertIs(a, b)

    def test_log_functions_no_error(self):
        """日志函数不应抛异常."""
        logger = get_logger("test_log")
        log_step(logger, "TEST_STEP", "Test message")
        log_timing(logger, "Test operation", 123.45)
        log_memory(logger, "Test memory", 100)
        # 无异常即通过
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
