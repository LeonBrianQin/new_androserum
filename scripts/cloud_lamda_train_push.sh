#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Push lamda240 training inputs after local preprocessing is complete.
#
# Uploads only what the cloud box needs for Phase 4 / Phase 6:
#   * assets/
#   * lamda split configs
#   * metadata.csv (optional)
#   * data_lamda_eval/methods/
#   * data_lamda_eval/overrides/
#   * data_lamda_eval/fcg/
#
# APK / processed txt stay local by design.
#
# Usage:
#   bash scripts/cloud_lamda_train_push.sh <user@host> <remote_repo_path> [eval_dir]
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash scripts/cloud_lamda_train_push.sh <user@host> <remote_repo_path> [eval_dir]"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
EVAL_DIR="${3:-data_lamda_eval}"
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
need "$REPO_ROOT/$EVAL_DIR/methods"
need "$REPO_ROOT/$EVAL_DIR/overrides"
need "$REPO_ROOT/$EVAL_DIR/fcg"

echo "[lamda-train-push] ensuring remote directories exist"
ssh "$REMOTE" "mkdir -p \
  '$REMOTE_REPO/assets' \
  '$REMOTE_REPO/configs' \
  '$REMOTE_REPO/$EVAL_DIR/methods' \
  '$REMOTE_REPO/$EVAL_DIR/overrides' \
  '$REMOTE_REPO/$EVAL_DIR/fcg' \
  '$REMOTE_REPO/$EVAL_DIR/embeddings/baseline' \
  '$REMOTE_REPO/$EVAL_DIR/embeddings/finetuned/train' \
  '$REMOTE_REPO/$EVAL_DIR/embeddings/finetuned/val' \
  '$REMOTE_REPO/$EVAL_DIR/embeddings/finetuned/test' \
  '$REMOTE_REPO/$EVAL_DIR/embeddings/finetuned/full240' \
  '$REMOTE_REPO/$EVAL_DIR/gnn_embeddings/relay_package/train' \
  '$REMOTE_REPO/$EVAL_DIR/gnn_embeddings/relay_package/val' \
  '$REMOTE_REPO/$EVAL_DIR/gnn_embeddings/relay_package/test' \
  '$REMOTE_REPO/$EVAL_DIR/gnn_embeddings/relay_package/full240' \
  '$REMOTE_REPO/$EVAL_DIR/checkpoints/p4_lamda_abe' \
  '$REMOTE_REPO/$EVAL_DIR/checkpoints/p6_lamda_relay_package' \
  '$REMOTE_REPO/$EVAL_DIR/reports'"

echo "[lamda-train-push] syncing assets/"
rsync -avzP "$REPO_ROOT/assets/" "$REMOTE:$REMOTE_REPO/assets/"

echo "[lamda-train-push] syncing lamda configs"
rsync -avzP \
  "$REPO_ROOT/configs/lamda_binary_eval_240_candidates.csv" \
  "$REPO_ROOT/configs/lamda_binary_eval_240_candidates.summary.json" \
  "$REPO_ROOT/configs/lamda_train_160.csv" \
  "$REPO_ROOT/configs/lamda_val_40.csv" \
  "$REPO_ROOT/configs/lamda_test_40.csv" \
  "$REPO_ROOT/configs/lamda_train_val_test.summary.json" \
  "$REMOTE:$REMOTE_REPO/configs/"

if [[ -f "$REPO_ROOT/metadata.csv" ]]; then
    echo "[lamda-train-push] syncing metadata.csv"
    rsync -avzP "$REPO_ROOT/metadata.csv" "$REMOTE:$REMOTE_REPO/"
fi

echo "[lamda-train-push] syncing methods/"
rsync -avzP "$REPO_ROOT/$EVAL_DIR/methods/" "$REMOTE:$REMOTE_REPO/$EVAL_DIR/methods/"

echo "[lamda-train-push] syncing overrides/"
rsync -avzP "$REPO_ROOT/$EVAL_DIR/overrides/" "$REMOTE:$REMOTE_REPO/$EVAL_DIR/overrides/"

echo "[lamda-train-push] syncing fcg/"
rsync -avzP "$REPO_ROOT/$EVAL_DIR/fcg/" "$REMOTE:$REMOTE_REPO/$EVAL_DIR/fcg/"

if [[ -f "$REPO_ROOT/$EVAL_DIR/lamda_full_240.sha.txt" ]]; then
    echo "[lamda-train-push] syncing lamda_full_240.sha.txt"
    rsync -avzP "$REPO_ROOT/$EVAL_DIR/lamda_full_240.sha.txt" "$REMOTE:$REMOTE_REPO/$EVAL_DIR/"
fi

echo "[lamda-train-push] done"
