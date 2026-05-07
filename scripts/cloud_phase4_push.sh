#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Push the *minimum* Phase 4 inputs from local -> remote GPU box.
#
# Usage:
#   bash scripts/cloud_phase4_push.sh <user@host> <remote_repo_path>
#
# Example:
#   bash scripts/cloud_phase4_push.sh root@1.2.3.4 /root/new_androserum
#
# What it syncs:
#   * assets/        (DexBERT ckpt + vocab + baksmali)
#   * data/methods/  (Phase 2/2b output; enough to run Phase 4)
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: bash scripts/cloud_phase4_push.sh <user@host> <remote_repo_path>"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "[ERR] missing required path: $path"
        exit 1
    fi
}

need "$REPO_ROOT/assets/model_steps_604364.pt"
need "$REPO_ROOT/assets/vocab.txt"
need "$REPO_ROOT/assets/baksmali-2.5.2.jar"
need "$REPO_ROOT/data/methods"

echo "[phase4-push] ensuring remote directories exist"
ssh "$REMOTE" "mkdir -p '$REMOTE_REPO/assets' '$REMOTE_REPO/data/methods'"

echo "[phase4-push] syncing assets/ (~1.8 GB)"
rsync -avzP "$REPO_ROOT/assets/" "$REMOTE:$REMOTE_REPO/assets/"

echo "[phase4-push] syncing data/methods/ (~170 MB in current local snapshot)"
rsync -avzP "$REPO_ROOT/data/methods/" "$REMOTE:$REMOTE_REPO/data/methods/"

echo "[phase4-push] done"
