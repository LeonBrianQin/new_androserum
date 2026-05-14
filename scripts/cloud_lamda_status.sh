#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Quick remote status check for the lamda cloud pipeline.
#
# Usage:
#   bash scripts/cloud_lamda_status.sh <user@host> <remote_repo_path> [eval_dir]
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash scripts/cloud_lamda_status.sh <user@host> <remote_repo_path> [eval_dir]"
    exit 1
fi

REMOTE="$1"
REMOTE_REPO="$2"
EVAL_DIR="${3:-data_lamda_eval}"

ssh "$REMOTE" "bash -lc '
cd \"$REMOTE_REPO\" || exit 1
echo \"[remote] pwd: \$(pwd)\"
echo \"[remote] eval dir: $EVAL_DIR\"
if [ -f \"$EVAL_DIR/lamda_run.pid\" ]; then
  pid=\$(cat \"$EVAL_DIR/lamda_run.pid\" 2>/dev/null || true)
  if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then
    echo \"[remote] lamda pid: \$pid (running)\"
  else
    echo \"[remote] lamda pid file exists but process is not running\"
  fi
fi
if [ -f \"$EVAL_DIR/lamda_train_only.pid\" ]; then
  pid=\$(cat \"$EVAL_DIR/lamda_train_only.pid\" 2>/dev/null || true)
  if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then
    echo \"[remote] lamda train-only pid: \$pid (running)\"
  else
    echo \"[remote] lamda train-only pid file exists but process is not running\"
  fi
fi
for p in \
  \"$EVAL_DIR/apks\" \
  \"$EVAL_DIR/processed\" \
  \"$EVAL_DIR/methods\" \
  \"$EVAL_DIR/overrides\" \
  \"$EVAL_DIR/embeddings/baseline\" \
  \"$EVAL_DIR/embeddings/finetuned/train\" \
  \"$EVAL_DIR/embeddings/finetuned/val\" \
  \"$EVAL_DIR/embeddings/finetuned/test\" \
  \"$EVAL_DIR/embeddings/finetuned/full240\" \
  \"$EVAL_DIR/fcg\" \
  \"$EVAL_DIR/gnn_embeddings/relay_package/train\" \
  \"$EVAL_DIR/gnn_embeddings/relay_package/val\" \
  \"$EVAL_DIR/gnn_embeddings/relay_package/test\" \
  \"$EVAL_DIR/gnn_embeddings/relay_package/full240\"; do
    if [ -d \"\$p\" ]; then
      printf \"%-50s %s\n\" \"\$p\" \"\$(find \"\$p\" -type f | wc -l)\"
    fi
  done
  echo
  if [ -f \"$EVAL_DIR/logs/lamda_run.log\" ]; then
    echo \"--- tail $EVAL_DIR/logs/lamda_run.log\"
    tail -n 40 \"$EVAL_DIR/logs/lamda_run.log\"
    echo
  fi
  if [ -f \"$EVAL_DIR/logs/lamda_train_only.log\" ]; then
    echo \"--- tail $EVAL_DIR/logs/lamda_train_only.log\"
    tail -n 40 \"$EVAL_DIR/logs/lamda_train_only.log\"
    echo
  fi
  for f in \
    \"$EVAL_DIR/checkpoints/p4_lamda_abe/contrastive_ab_summary.json\" \
    \"$EVAL_DIR/checkpoints/p6_lamda_relay_package/bgrl_graphsage_summary.json\"; do
    if [ -f \"\$f\" ]; then
      echo \"--- \$f\"
      sed -n \"1,80p\" \"\$f\"
      echo
    fi
  done
'"
