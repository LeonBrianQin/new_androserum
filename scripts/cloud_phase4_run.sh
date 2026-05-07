#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run the current recommended Phase 4 config on a remote GPU box.
#
# Intended to be launched *on the GPU server* after:
#   1) git clone
#   2) bash scripts/cloud_setup.sh
#   3) sync assets/ and data/methods/ from local
#
# Usage:
#   bash scripts/cloud_phase4_run.sh [RUN_NAME] [EXTRA RUN_ALL FLAGS...]
#
# Example:
#   bash scripts/cloud_phase4_run.sh p4_dev200_run1
#   bash scripts/cloud_phase4_run.sh p4_dev200_run2 --phase4_epochs=5
#
# Default behavior:
#   * Phase 4 only
#   * recommended local sweep defaults
#   * exports finetuned embeddings before you shut down the machine
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-p4_dev200_run1}"
if [[ $# -gt 0 ]]; then
    shift
fi

ENV_NAME="${ANDROSERUM_ENV:-androserum}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found; run bash scripts/cloud_setup.sh first"
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "data/checkpoints/$RUN_NAME" "data/embeddings/finetuned/$RUN_NAME"

python scripts/run_all.py \
    --do_download=False \
    --do_disassemble=False \
    --do_extract=False \
    --do_susi=False \
    --do_encode=False \
    --do_train=True \
    --device=cuda \
    --phase4_checkpoint_dir="data/checkpoints/$RUN_NAME" \
    --phase4_embeddings_dir="data/embeddings/finetuned/$RUN_NAME" \
    --phase4_export_after_train=True \
    "$@"
