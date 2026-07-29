"""X2MD CLI 安全调用包装器.

供 backend preprocess.py 通过 subprocess 安全调用 X2MD CLI,
避免将路径拼接到 python3 -c 字符串中(命令注入风险).

此文件被 _invoke.py 自身通过 sys.path 动态定位,
不依赖外部路径参数——只需 python3 _invoke.py [x2md CLI 参数...] 即可。
"""
import sys
from pathlib import Path

# 自动定位 x2md 包: _invoke.py 位于 x2md/src/x2md/_invoke.py
_x2md_src = str(Path(__file__).resolve().parent.parent)
if _x2md_src not in sys.path:
    sys.path.insert(0, _x2md_src)

from x2md.cli import main

if __name__ == "__main__":
    main(prog_name="x2md")
