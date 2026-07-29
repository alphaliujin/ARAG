#!/usr/bin/env python3
"""运行所有测试."""

import sys
import unittest
from pathlib import Path

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    loader = unittest.TestLoader()
    tests_dir = Path(__file__).parent
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py", top_level_dir=str(tests_dir))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
