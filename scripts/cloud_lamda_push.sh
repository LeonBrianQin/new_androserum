#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Push the inputs needed for the lamda240 cloud pipeline.
#
# Usage:
#   bash scripts/cloud_lamda_push.sh <user@host> <remote_repo_path> [local_eval_dir]
#
# Example:
#   bash scripts/cloud_lamda_push.sh root@1.2.3.4 /root/new_androserum data_lamda_eval
#
# Syncs:
#   * assets/
#   * configs/lamda_*.csv + lamda summaries
#   * metadata.csv (for reference / reproducibility)
#   * target eval dir skeleton (apks/processed/methods/overrides/fcg/embeddings/checkpoints)
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash scripts/cloud_lamda_push.sh <user@host> <remote_repo_path> [local_eval_dir]"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
LOCAL_EVAL_DIR="${3:-data_lamda_eval}"
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
need "$REPO_ROOT/configs/lamda_train_160.csv"
need "$REPO_ROOT/configs/lamda_val_40.csv"
need "$REPO_ROOT/configs/lamda_test_40.csv"

echo "[lamda-push] ensuring remote directories exist"
ssh "$REMOTE" "mkdir -p \
  '$REMOTE_REPO/assets' \
  '$REMOTE_REPO/configs' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/apks' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/processed' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/methods' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/overrides' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/fcg' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/embeddings/baseline' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/embeddings/finetuned/train' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/embeddings/finetuned/val' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/embeddings/finetuned/test' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/embeddings/finetuned/full240' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/gnn_embeddings/relay_package/train' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/gnn_embeddings/relay_package/val' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/gnn_embeddings/relay_package/test' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/gnn_embeddings/relay_package/full240' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/checkpoints/p4_lamda_abe' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/checkpoints/p6_lamda_relay_package' \
  '$REMOTE_REPO/$LOCAL_EVAL_DIR/reports'"

echo "[lamda-push] syncing assets/"
rsync -avzP "$REPO_ROOT/assets/" "$REMOTE:$REMOTE_REPO/assets/"

echo "[lamda-push] syncing lamda split configs"
rsync -avzP \
  "$REPO_ROOT/configs/lamda_binary_eval_240_candidates.csv" \
  "$REPO_ROOT/configs/lamda_binary_eval_240_candidates.summary.json" \
  "$REPO_ROOT/configs/lamda_train_160.csv" \
  "$REPO_ROOT/configs/lamda_val_40.csv" \
  "$REPO_ROOT/configs/lamda_test_40.csv" \
  "$REPO_ROOT/configs/lamda_train_val_test.summary.json" \
  "$REMOTE:$REMOTE_REPO/configs/"

if [[ -f "$REPO_ROOT/metadata.csv" ]]; then
    echo "[lamda-push] syncing metadata.csv"
    rsync -avzP "$REPO_ROOT/metadata.csv" "$REMOTE:$REMOTE_REPO/"
fi

if [[ -d "$REPO_ROOT/$LOCAL_EVAL_DIR" ]]; then
    echo "[lamda-push] syncing any existing local eval artifacts under $LOCAL_EVAL_DIR/"
    rsync -avzP \
      --include='*/' \
      --include='*.csv' \
      --include='*.txt' \
      --include='*.json' \
      --include='*.parquet' \
      --include='*.npz' \
      --exclude='*' \
      "$REPO_ROOT/$LOCAL_EVAL_DIR/" \
      "$REMOTE:$REMOTE_REPO/$LOCAL_EVAL_DIR/"
fi

echo "[lamda-push] done"
