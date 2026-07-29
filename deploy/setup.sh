#!/bin/bash
# =============================================================================
# ARAG_V0.2 部署安装脚本 - 在目标 Linux 机器上执行一次
# 用法: ./setup.sh
# 做的事: 检查依赖 -> 建 venv -> 装 Python 依赖 -> 拉 Ollama 模型 -> 配置路径/数据目录
# =============================================================================
set -uo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$INSTALL_DIR"

color() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
info()  { echo "$(color 36 [STEP]) $1"; }
ok()    { echo "$(color 32 [OK]) $1"; }
warn()  { echo "$(color 33 [WARN]) $1"; }
die()   { echo "$(color 31 [ERROR]) $1" >&2; exit 1; }

echo "$(color 36)═══════════════════════════════════════════════════$(color 0)"
echo "  ARAG_V0.2 部署安装  (install dir: $INSTALL_DIR)"
echo "$(color 36)═══════════════════════════════════════════════════$(color 0)"

# ---------- 1. 依赖检查 ----------
info "1/5 检查系统依赖..."
miss=()
for cmd in python3 pip3 ollama curl; do
  command -v "$cmd" >/dev/null 2>&1 || miss+=("$cmd")
done
[ ${#miss[@]} -gt 0 ] && die "缺少命令: ${miss[*]}。请先安装 (macOS: brew; Debian/Ubuntu: apt-get install -y python3 python3-pip curl && curl -fsSL https://ollama.com/install.sh | sh)"
python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' || die "Python 版本需 >= 3.9 (当前 $(python3 --version))"
# antiword/soffice 仅 .doc/.ppt 转换需要; 缺失只警告
for cmd in antiword soffice libreoffice; do command -v "$cmd" >/dev/null 2>&1 && break; done || warn "未找到 antiword/libreoffice: .doc/.ppt 转换会受限 (其它格式不受影响)"
ok "依赖检查通过"

# ---------- 2. 虚拟环境 ----------
info "2/5 创建 Python 虚拟环境..."
if [ ! -f backend/.venv/bin/activate ]; then
  python3 -m venv backend/.venv || die "venv 创建失败"
fi
# shellcheck disable=SC1091
source backend/.venv/bin/activate
python -m pip install --upgrade pip -q
ok "venv 就绪"

# ---------- 3. Python 依赖 ----------
info "3/5 安装 Python 依赖 (首次较慢, 含 chromadb/sentence-transformers/torch)..."
pip install -e backend -q 2>&1 | tail -3 || die "backend 依赖安装失败"
# MD2RAG/X2MD 经 sys.path(_invoke.py + main.py)导入, 无需 pip install; 其核心依赖已在 backend pyproject
ok "Python 依赖安装完成"

# ---------- 4. Ollama 模型 ----------
info "4/5 拉取 Ollama 嵌入模型 bge-m3:latest (~1.2GB)..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  ollama pull bge-m3:latest && ok "bge-m3 已就绪" || warn "ollama pull 失败, 稍后手动: ollama pull bge-m3:latest"
else
  warn "Ollama 服务未运行 (localhost:11434)。请先启动 ollama 服务, 再 'ollama pull bge-m3:latest'"
fi

# ---------- 5. 配置路径 + 数据目录 ----------
info "5/5 配置安装路径 + 创建数据目录..."
# 把配置文件里的 __INSTALL_DIR__ 占位符替换为实际安装路径
for f in X2MD/x2md.conf MD2RAG/md2rag.conf runtime_settings.json; do
  if [ -f "$f" ]; then
    sed -i "s#__INSTALL_DIR__#$INSTALL_DIR#g" "$f"
  fi
done
# .env (默认开发免认证; 生产请设 API_KEY 并去掉 AUTH_DISABLED)
if [ ! -f backend/.env ]; then
  printf 'AUTH_DISABLED=True\n' > backend/.env
fi
# 数据目录骨架
mkdir -p MD/0Public MD/1Restricted MD/2Confidential \
         DOC/0Public DOC/1Restricted DOC/2Confidential \
         vector_db DocScan dedup_results
ok "配置与数据目录就绪"

echo ""
ok "安装完成!"
echo ""
echo "  启动:   ./start.sh                 (单进程, 同时服务 UI + API)"
echo "  访问:   http://<本机IP>:8000       (本机访问 http://localhost:8000)"
echo "  停止:   ./stop.sh"
echo "  日志:   tail -f server.log"
echo "  API文档: http://<本机IP>:8000/docs"
echo ""
echo "  注: 首次入库/比对前, 请把待处理文档放入 DOC/<密级>/ 后通过前端预处理。"
