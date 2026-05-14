#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Start the lamda240 cloud train-only pipeline in the background.
#
# Usage:
#   bash scripts/cloud_lamda_train_start.sh [eval_dir]
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
mkdir -p "$EVAL_DIR/logs"

LOG_FILE="$EVAL_DIR/logs/lamda_train_only.log"
PID_FILE="$EVAL_DIR/lamda_train_only.pid"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "[lamda-train-start] already running: pid=$old_pid"
        exit 0
    fi
fi

nohup bash scripts/cloud_lamda_train_only.sh "$EVAL_DIR" >"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "[lamda-train-start] started in background"
echo "[lamda-train-start] pid: $pid"
echo "[lamda-train-start] log: $LOG_FILE"
