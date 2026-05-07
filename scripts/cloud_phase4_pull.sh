#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Pull Phase 4 outputs from remote GPU box -> local machine.
#
# Usage:
#   bash scripts/cloud_phase4_pull.sh <user@host> <remote_repo_path> [RUN_NAME]
#
# Example:
#   bash scripts/cloud_phase4_pull.sh root@1.2.3.4 /root/new_androserum p4_dev200_run1
#
# What it pulls:
#   * data/checkpoints/<RUN_NAME>/
#   * data/embeddings/finetuned/<RUN_NAME>/
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash scripts/cloud_phase4_pull.sh <user@host> <remote_repo_path> [run_name]"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
RUN_NAME="${3:-p4_dev200_run1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
    "$REPO_ROOT/data/checkpoints/$RUN_NAME" \
    "$REPO_ROOT/data/embeddings/finetuned/$RUN_NAME"

echo "[phase4-pull] syncing checkpoints/$RUN_NAME"
rsync -avzP \
    "$REMOTE:$REMOTE_REPO/data/checkpoints/$RUN_NAME/" \
    "$REPO_ROOT/data/checkpoints/$RUN_NAME/"

echo "[phase4-pull] syncing finetuned embeddings/$RUN_NAME"
rsync -avzP \
    "$REMOTE:$REMOTE_REPO/data/embeddings/finetuned/$RUN_NAME/" \
    "$REPO_ROOT/data/embeddings/finetuned/$RUN_NAME/"

echo "[phase4-pull] done"
