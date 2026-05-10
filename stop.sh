#!/usr/bin/env bash
# stop.sh — Gracefully stop all face-api instances started by start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "[stop] No PID file found — killing by process name instead"
    pkill -f "uvicorn app:app" && echo "[stop] Done" || echo "[stop] No uvicorn processes found"
    exit 0
fi

mapfile -t PIDS < "$PID_FILE"
echo "[stop] Stopping PIDs: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null && echo "[stop] Sent SIGTERM" || echo "[stop] Some PIDs already gone"
rm -f "$PID_FILE"
