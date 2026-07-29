#!/bin/bash
# backend/run.sh — 开发期单独启动 backend 的辅助脚本。
# 推荐使用项目根目录的 start.sh,它带有 PID 文件、健康等待、清理逻辑。
# 本脚本仅作开发 quick-iteration 用,行为与 start.sh 对齐:
#   - 绑定 127.0.0.1 (不对外暴露)
#   - 依赖未变时跳过 pip install
#   - cd 失败立即退出
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || { echo "Failed to resolve script dir" >&2; exit 1; }
cd "$SCRIPT_DIR" || { echo "Failed to cd $SCRIPT_DIR" >&2; exit 1; }

PROJECT_ROOT="$(cd .. && pwd)"

echo "Starting 敏感信息识别系统 Backend..."
echo "Project: 敏感信息识别系统"
echo ""

# 用绝对路径,避免脚本被从其他目录调用时 ../MD 解析到错误位置
if [ ! -d "$PROJECT_ROOT/MD/0Public" ] || [ ! -d "$PROJECT_ROOT/MD/1Restricted" ] || [ ! -d "$PROJECT_ROOT/MD/2Confidential" ]; then
    echo "Creating data directories..."
    mkdir -p "$PROJECT_ROOT/MD/0Public" "$PROJECT_ROOT/MD/1Restricted" "$PROJECT_ROOT/MD/2Confidential"
fi

if [ ! -d "$PROJECT_ROOT/vector_db" ]; then
    echo "Creating vector_db directory..."
    mkdir -p "$PROJECT_ROOT/vector_db"
fi

# 仅在 venv 不存在或被显式标记为需重装时跑 pip install
# (start.sh 用 .deps_installed 标记同样做了这件事)
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv || { echo "venv creation failed" >&2; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -q -e . && touch .venv/.deps_installed
fi

echo ""
echo "Starting FastAPI server (127.0.0.1:8000)..."
# 与 start.sh 一致: 绑定 127.0.0.1, 由前端代理或 ssh 隧道暴露
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
