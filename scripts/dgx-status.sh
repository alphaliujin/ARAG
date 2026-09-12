#!/bin/bash
# DGX 部署状态检查 (远程执行, 避免 ssh 命令转义问题)
echo "==stage=="; ls -d /home/alpha/.deploy-dgx-stage* 2>/dev/null
echo "==bak=="; ls -d /home/alpha/ARAG_V0.2.bak* 2>/dev/null
echo "==venv=="; [ -d /home/alpha/ARAG_V0.2/backend/.venv ] && echo VENV_OK || echo NO_VENV
echo "==ui-build=="; ls /home/alpha/ARAG_V0.2/UI/build/static/js 2>/dev/null | head -3
echo "==port=="; ss -ltn 2>/dev/null | grep ':8000' || echo PORT_8000_DOWN
echo "==procs=="; ps aux | grep -E 'uvicorn|ollama' | grep -v grep | awk '{print $11, $12}' | head
echo "==log=="; tail -8 /home/alpha/ARAG_V0.2/server.log 2>/dev/null || tail -8 /home/alpha/server.log 2>/dev/null
