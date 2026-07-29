#!/bin/bash
# ARAG Ollama 监控 v2 - 加 GPU-Util + 瞬时CPU, 抓 "GPU闲+CPU满+embed慢" 时刻
# 用法: bash monitor_ollama.sh [log路径]   停止: Ctrl+C
# 在 DGX Spark 上运行: bash monitor_ollama.sh  (journalctl 权限不够则 sudo)
LOG="${1:-$(cd "$(dirname "$0")" && pwd)/ollama_monitor.log}"
log() { printf '%s %s\n' "$(date '+%T')" "$*" >> "$LOG"; }
log "================ 监控启动 $(date '+%F %T') (v2: GPU-Util+瞬时CPU) ================"
systemctl cat ollama 2>/dev/null | grep -iE 'Environment' >> "$LOG"

# 后台: journalctl /api/embed 耗时 + offload/evict 事件
journalctl -u ollama -f -o cat --since now 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *offload*|*load_tensors*|*llama_prepare_model_devices*|*evict*|*unloaded*|*model_loader*|*/api/embed*|*/api/embeddings*|*fit_params*|*CPU_Mapped*|*kv*buffer*|*device*)
            log "[jrn] $line" ;;
    esac
done &
JPID=$!
cleanup() { kill "$JPID" 2>/dev/null; pkill -P $$ 2>/dev/null; log "================ 监控停止 ================"; }
trap cleanup INT TERM

OPID=$(pgrep -f 'ollama serve' | head -1)
log "ollama serve PID=$OPID"

while true; do
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    # 瞬时 ollama CPU%: top 两次快照, 第二次是 1s 内瞬时值
    ocpu=$(top -bn2 -d1 -p "${OPID:-0}" 2>/dev/null | awk -v p="${OPID:-0}" '$1==p{c=$9} END{print c}')
    ps_out=$(ollama ps 2>/dev/null)
    bge_proc=$(printf '%s\n' "$ps_out" | grep 'bge-m3' | grep -oE '[0-9]+%[^ ]* (GPU|CPU)(/(GPU|CPU))?' | head -1)
    [ -z "$bge_proc" ] && bge_proc="(未加载)"
    log "[采样] GPU_util=${gpu_util:-NA} | ollama_cpu=${ocpu:-NA}% | bge=$bge_proc"
    sleep 1
done
