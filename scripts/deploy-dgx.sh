#!/bin/bash
# =============================================================================
# DGX 远程更新脚本: 把新部署包替换进现有安装, 保留数据/设置, 然后重启。
# 用法 (在 DGX 上):
#   cd /root
#   scp ARAG_V0.2-linux.tar.gz 和本脚本 到 /root
#   bash deploy-dgx.sh
#
# 数据目录 (MD DOC vector_db DocScan dedup_results) 与运行时设置
# (runtime_settings.json .env) 一律保留, 只替换代码/配置/前端产物。
# =============================================================================
set -uo pipefail

PKG="ARAG_V0.2-linux.tar.gz"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 1. 定位现有安装目录 (排除 .new/.bak 和本包本身)
OLD=""
for cand in "$HERE/ARAG_V0.2" "$HERE/ARAG" "$(ls -d "$HERE"/ARAG* 2>/dev/null | grep -vE '\.new$|\.bak$|tar\.gz$' | head -1)"; do
  if [ -n "$cand" ] && [ -d "$cand" ] && [ -f "$cand/backend/app/main.py" ]; then
    OLD="$cand"
    break
  fi
done

if [ -z "$OLD" ]; then
  echo "❌ 未找到现有 ARAG 安装目录 (需含 backend/app/main.py)。"
  echo "   若是首次安装, 请改用: tar xzf $PKG && cd ARAG_V0.2 && ./setup.sh && ./start.sh"
  exit 1
fi
echo "✅ 现有安装: $OLD"

if [ ! -f "$HERE/$PKG" ]; then
  echo "❌ 找不到包: $HERE/$PKG (请先 scp 上传)"
  exit 1
fi

# 2. 解压新包到临时目录
STAGE="$HERE/.deploy-dgx-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar xzf "$HERE/$PKG" -C "$STAGE"
NEW="$STAGE/ARAG_V0.2"
echo "✅ 解压新包: $NEW"

# 3. 把旧实例的数据/设置拷进新包 (存在才拷, 不覆盖新包自带的骨架)
for d in MD DOC vector_db DocScan dedup_results; do
  if [ -d "$OLD/$d" ]; then
    # 保留旧目录内容; 新包骨架 (.gitkeep) 无冲突
    cp -a "$OLD/$d/." "$NEW/$d/" 2>/dev/null || true
    echo "  - 保留数据: $d"
  fi
done
for f in runtime_settings.json backend/.env; do
  if [ -f "$OLD/$f" ]; then
    cp -a "$OLD/$f" "$NEW/$f" && echo "  - 保留设置: $f"
  fi
done

# 4. 停旧服务 (若在跑)
if [ -f "$OLD/start.sh" ]; then
  ( cd "$OLD" && ./start.sh stop ) >/dev/null 2>&1 && echo "✅ 旧服务已停止"
  # 兜底: 端口还占着就再清一次
  sleep 1
fi

# 5. 原子切换: 旧 → .bak, 新 → 原位
STAMP="$(date +%Y%m%d_%H%M%S)"
BAK="${OLD%/}.bak.$STAMP"
mv "$OLD" "$BAK" && echo "✅ 旧安装已备份: $BAK"
mv "$NEW" "$OLD" && echo "✅ 新代码已就位: $OLD"

# 6. 重启 (不重跑 setup.sh: 依赖已装, 只重启; 若首次/依赖变了可手动 ./setup.sh)
cd "$OLD"
if ./start.sh; then
  echo ""
  echo "✅ 更新完成并已启动"
  echo "   http://$(hostname -I 2>/dev/null | awk '{print $1}'):8000"
  echo "   日志: tail -f $OLD/server.log"
  echo "   如需回滚: stop 后 mv $BAK $OLD 再 start"
else
  echo "⚠️  启动失败, 回滚..."
  ./start.sh stop >/dev/null 2>&1 || true
  mv "$OLD" "$STAGE.rollback" 2>/dev/null || true
  mv "$BAK" "$OLD"
  ./start.sh
  echo "已回滚到旧版本: $OLD"
fi
rm -rf "$STAGE"
