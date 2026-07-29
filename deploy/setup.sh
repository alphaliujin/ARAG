#!/bin/bash
# =============================================================================
# ARAG_V0.2 部署安装脚本 - 在目标 Linux 机器上执行一次
# 用法: ./setup.sh
# 做的事: 检测并自动安装系统依赖 -> 建 venv -> 装 Python 依赖 -> 拉 Ollama 模型 -> 配置路径/数据目录
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

# ---------- 1. 系统依赖 (检测 + 自动安装) ----------
info "1/5 检测系统依赖 (缺失自动安装)..."
# sudo 前缀: 非 root 且有 sudo 时使用 (apt/systemctl 需要)
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# 1a. 检测缺失的必需包: python3 / pip3 / curl / python3-venv
need_pkgs=()
command -v python3 >/dev/null 2>&1 || need_pkgs+=(python3)
command -v pip3    >/dev/null 2>&1 || need_pkgs+=(python3-pip)
command -v curl    >/dev/null 2>&1 || need_pkgs+=(curl)
# Debian/Ubuntu 上 `python3 -m venv` 由独立包 python3-venv 提供, 常不随 python3 安装
if ! python3 -m venv --help >/dev/null 2>&1; then need_pkgs+=(python3-venv); fi

if [ ${#need_pkgs[@]} -gt 0 ]; then
  # 识别包管理器 (DGX Spark/Ubuntu 用 apt-get)
  pm=""
  for p in apt-get apt dnf yum; do command -v "$p" >/dev/null 2>&1 && { pm="$p"; break; }; done
  if [ -z "$pm" ]; then
    die "缺少: ${need_pkgs[*]}。未识别到包管理器 (apt/dnf/yum), 请手动安装后重试。"
  fi
  # dnf/yum 上 venv 随 python3 自带, 没有 python3-venv 包 -> 替换成 python3
  if [ "$pm" = "dnf" ] || [ "$pm" = "yum" ]; then
    need_pkgs=("${need_pkgs[@]/python3-venv/python3}")
  fi
  info "通过 $pm 自动安装: ${need_pkgs[*]}"
  if [ "$pm" = "apt-get" ] || [ "$pm" = "apt" ]; then
    $SUDO "$pm" update -qq 2>/dev/null || true
  fi
  $SUDO "$pm" install -y "${need_pkgs[@]}" || die "系统依赖安装失败, 请手动安装: ${need_pkgs[*]}"
fi
# 复检 (apt 装完仍可能因 PATH/版本不符失败, 这里兜底)
command -v python3 >/dev/null 2>&1 || die "python3 仍不可用"
command -v curl    >/dev/null 2>&1 || die "curl 仍不可用"
python3 -m venv --help >/dev/null 2>&1 || die "python3-venv 仍不可用 (python3 -m venv 失败)"
python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' || die "Python 版本需 >= 3.9 (当前 $(python3 --version))"
ok "系统依赖就绪 (python3 / pip3 / venv / curl)"

# 1b. Ollama (缺失则用官方脚本自动装; 支持 aarch64/x86_64, 脚本内部自理 sudo)
if ! command -v ollama >/dev/null 2>&1; then
  info "未检测到 Ollama, 通过官方脚本安装 (支持 aarch64/x86_64)..."
  curl -fsSL https://ollama.com/install.sh | sh || die "Ollama 安装失败。可手动: curl -fsSL https://ollama.com/install.sh | sh"
  ok "Ollama 安装完成"
else
  ok "Ollama 已安装"
fi
# 1c. Ollama 服务未运行则自动启动 (systemd 优先, 回退 nohup ollama serve)
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  info "Ollama 服务未运行, 尝试启动..."
  if command -v systemctl >/dev/null 2>&1 && $SUDO systemctl start ollama 2>/dev/null; then
    : # systemd 已拉起
  else
    nohup ollama serve > "$INSTALL_DIR/.ollama.log" 2>&1 &
  fi
  # 等待就绪 (最多 ~20s)
  for i in $(seq 1 20); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
  curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && ok "Ollama 服务已就绪" || warn "Ollama 未就绪, 稍后手动: ollama serve (日志: .ollama.log)"
fi

# 1d. 可选: antiword/libreoffice (.doc/.ppt 转换); 缺失只警告
for cmd in antiword soffice libreoffice; do command -v "$cmd" >/dev/null 2>&1 && break; done || warn "未找到 antiword/libreoffice: .doc/.ppt 转换会受限 (其它格式不受影响)"
ok "依赖检查通过"

# ---------- GPU / 架构检测 (DGX Spark 等带 NVIDIA GPU 的机器) ----------
arch="$(uname -m)"
case "$arch" in
  aarch64|arm64) ok "架构 $arch (ARM64, 如 DGX Spark Grace Hopper - 源码部署包原生支持)" ;;
  x86_64)        ok "架构 $arch (x86_64)" ;;
  *)             warn "架构 $arch 未测试, 若依赖装不上请反馈" ;;
esac
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ -n "$gpu" ] && ok "NVIDIA GPU: $gpu (Ollama bge-m3 嵌入将走 GPU 加速)" || warn "nvidia-smi 在但读不到 GPU, Ollama 将退回 CPU 嵌入"
else
  warn "未检测到 nvidia-smi: Ollama 将用 CPU 嵌入 (慢)。DGX Spark 应有 GPU, 请确认 NVIDIA 驱动已装"
fi

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
