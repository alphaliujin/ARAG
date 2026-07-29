#!/bin/bash
# 停止 ARAG_V0.2 (转发到 start.sh stop)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/start.sh" stop
