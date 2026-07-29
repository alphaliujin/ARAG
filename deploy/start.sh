#!/bin/bash
# =============================================================================
# ARAG_V0.2 部署版启动脚本 - 单进程 uvicorn 同时服务 前端(/) + API(/api/v1)
# 用法: ./start.sh [stop|status]
#   ./start.sh         启动 (默认 0.0.0.0:8000)
#   ./start.sh stop    停止
#   ./start.sh status  查看状态
# 环境变量: ARAG_HOST (默认 0.0.0.0)  ARAG_PORT (默认 8000)
# =============================================================================
set -uo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$INSTALL_DIR"
HOST="${ARAG_HOST:-0.0.0.0}"
PORT="${ARAG_PORT:-8000}"
PID_FILE="$INSTALL_DIR/.server.pid"
LOG="$INSTALL_DIR/server.log"

cmd=${1:-start}
check_port() { lsof -nP -i :"$1" -sTCP:LISTEN 2>/dev/null | grep -v COMMAND | awk '{print $2}' | head -1; }

case "$cmd" in
  stop)
    if [ -f "$PID_FILE" ]; then
      pid=$(cat "$PID_FILE" 2>/dev/null)
      [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "已停止 (PID $pid)"
      rm -f "$PID_FILE"
    else
      pid=$(check_port "$PORT"); [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "已停止端口进程 (PID $pid)"
    fi
    ;;
  status)
    pid=$(check_port "$PORT")
    if [ -n "$pid" ]; then echo "运行中 (PID $pid)  http://$HOST:$PORT  (PID file: $(cat "$PID_FILE" 2>/dev/null))"; else echo "已停止"; fi
    ;;
  start|"")
    # 清理可能残留的旧进程
    old=$(check_port "$PORT")
    [ -n "$old" ] && kill "$old" 2>/dev/null && echo "清理旧进程 (PID $old)"
    # venv
    if [ ! -f backend/.venv/bin/activate ]; then echo "未安装: 请先运行 ./setup.sh"; exit 1; fi
    # shellcheck disable=SC1091
    source backend/.venv/bin/activate
    # 数据目录
    mkdir -p MD/0Public MD/1Restricted MD/2Confidential vector_db DocScan dedup_results
    # 启动 (日志轮转)
    [ -f "$LOG" ] && mv "$LOG" "$LOG.prev" 2>/dev/null
    nohup python -m uvicorn app.main:app --app-dir backend --host "$HOST" --port "$PORT" >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    # 等待就绪
    for i in $(seq 1 30); do
      curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1
    done
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "✅ 启动成功  http://$HOST:$PORT  (PID $(cat "$PID_FILE"))"
      echo "   前端: http://$HOST:$PORT   API 文档: http://$HOST:$PORT/docs"
      echo "   日志: tail -f $LOG"
    else
      echo "❌ 启动超时, 请查日志: $LOG"; tail -20 "$LOG" 2>/dev/null; exit 1
    fi
    ;;
  *) echo "用法: $0 [start|stop|status]"; exit 1 ;;
esac
