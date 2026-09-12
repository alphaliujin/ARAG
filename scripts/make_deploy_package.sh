#!/bin/bash
# =============================================================================
# 构建 ARAG_V0.2-linux.tar.gz 部署包 (自包含, 排除重型数据/模型/venv/node_modules)
# 产物: ARAG_V0.2-linux.tar.gz (顶层目录 ARAG_V0.2/)
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE_ROOT="$ROOT/.deploy-stage"
STAGE="$STAGE_ROOT/ARAG_V0.2"
OUT="$ROOT/ARAG_V0.2-linux.tar.gz"

rm -rf "$STAGE_ROOT"; mkdir -p "$STAGE"
v() { echo "  $1"; }

echo "[1/6] 后端代码..."
mkdir -p "$STAGE/backend"
cp -R "$ROOT/backend/app" "$STAGE/backend/app"
cp "$ROOT/backend/pyproject.toml" "$STAGE/backend/"
v "backend/app + pyproject.toml"

echo "[2/6] MD2RAG / X2MD 源码 + 配置..."
mkdir -p "$STAGE/MD2RAG" "$STAGE/X2MD"
cp -R "$ROOT/MD2RAG/md2rag" "$STAGE/MD2RAG/md2rag"
cp "$ROOT/MD2RAG/md2rag.conf" "$STAGE/MD2RAG/"
cp -R "$ROOT/X2MD/src" "$STAGE/X2MD/src"
cp "$ROOT/X2MD/x2md.conf" "$STAGE/X2MD/"
v "MD2RAG/md2rag + md2rag.conf, X2MD/src + x2md.conf"

echo "[3/6] 前端构建产物 (相对 /api/v1, 同源服务)..."
mkdir -p "$STAGE/UI"
cp -R "$ROOT/UI/build" "$STAGE/UI/build"
v "UI/build"

echo "[4/6] 运行时配置 + 部署脚本..."
cp "$ROOT/runtime_settings.json" "$STAGE/runtime_settings.json"
# ★ 生产启动/停止脚本【唯一真相】在 deploy/: 根目录的 start.sh/stop.sh 是 macOS
#   开发双进程版, 与本打包无关。改动生产部署行为只改 deploy/, 此处始终从 deploy/ 拷。
cp "$ROOT/deploy/setup.sh" "$STAGE/setup.sh"
cp "$ROOT/deploy/start.sh" "$STAGE/start.sh"
cp "$ROOT/deploy/stop.sh" "$STAGE/stop.sh"
cp "$ROOT/deploy/README.md" "$STAGE/README.md"
chmod +x "$STAGE/setup.sh" "$STAGE/start.sh" "$STAGE/stop.sh"
v "runtime_settings.json + setup/start/stop + README"

echo "[5/6] 清理缓存 + 配置路径占位符化 + 数据目录骨架..."
find "$STAGE" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find "$STAGE" -name "*.pyc" -delete 2>/dev/null
find "$STAGE" -name ".DS_Store" -delete 2>/dev/null
# 配置里的本机绝对路径 -> __INSTALL_DIR__ (setup.sh 在目标机替换回实际路径)
for f in "$STAGE/X2MD/x2md.conf" "$STAGE/MD2RAG/md2rag.conf" "$STAGE/runtime_settings.json"; do
  sed "s#$ROOT#__INSTALL_DIR__#g" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
# 数据目录骨架 (内容不打包)
for d in MD/0Public MD/1Restricted MD/2Confidential \
         DOC/0Public DOC/1Restricted DOC/2Confidential \
         vector_db DocScan dedup_results; do
  mkdir -p "$STAGE/$d"; touch "$STAGE/$d/.gitkeep"
done
v "占位符 + 骨架就绪"

echo "[6/6] 打包..."
cd "$STAGE_ROOT"
tar czf "$OUT" ARAG_V0.2
cd "$ROOT"
rm -rf "$STAGE_ROOT"

echo ""
echo "✅ 完成: $OUT  ($(du -h "$OUT" | cut -f1))"
echo ""
echo "部署步骤:"
echo "  scp ARAG_V0.2-linux.tar.gz <user>@<linux-host>:~/"
echo "  ssh <user>@<linux-host>"
echo "  tar xzf ARAG_V0.2-linux.tar.gz && cd ARAG_V0.2"
echo "  ./setup.sh && ./start.sh"
