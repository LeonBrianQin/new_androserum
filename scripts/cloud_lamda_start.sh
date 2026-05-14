#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Start the lamda240 cloud pipeline in the background and detach from SSH.
#
# Usage:
#   bash scripts/cloud_lamda_start.sh [eval_dir]
#
# Writes:
#   * <eval_dir>/logs/lamda_run.log
#   * <eval_dir>/lamda_run.pid
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
mkdir -p "$EVAL_DIR/logs"

LOG_FILE="$EVAL_DIR/logs/lamda_run.log"
PID_FILE="$EVAL_DIR/lamda_run.pid"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "[lamda-start] already running: pid=$old_pid"
        exit 0
    fi
fi

nohup bash scripts/cloud_lamda_run.sh "$EVAL_DIR" >"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "[lamda-start] started in background"
echo "[lamda-start] pid: $pid"
echo "[lamda-start] log: $LOG_FILE"
