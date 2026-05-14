#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Pull lamda cloud outputs back to local.
#
# Usage:
#   bash scripts/cloud_lamda_pull.sh <user@host> <remote_repo_path> [eval_dir]
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash scripts/cloud_lamda_pull.sh <user@host> <remote_repo_path> [eval_dir]"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
EVAL_DIR="${3:-data_lamda_eval}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$REPO_ROOT/$EVAL_DIR"

echo "[lamda-pull] syncing lamda eval artifacts"
rsync -avzP \
  --include='*/' \
  --include='*.json' \
  --include='*.txt' \
  --include='*.parquet' \
  --include='*.npz' \
  --include='*.pt' \
  --exclude='*' \
  "$REMOTE:$REMOTE_REPO/$EVAL_DIR/" \
  "$REPO_ROOT/$EVAL_DIR/"

echo "[lamda-pull] done"
