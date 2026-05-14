#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot bootstrap for a new machine.
#
# This is the simplest "catch up to current progress" entry:
#   1) download lamda240 locally from AndroZoo
#   2) disassemble
#   3) extract methods
#   4) tag SuSi
#   5) extract overrides
#   6) extract FCG
#
# Usage:
#   bash scripts/17_new_device_lamda240_bootstrap.sh
#   bash scripts/17_new_device_lamda240_bootstrap.sh <eval_dir> <workers>
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
WORKERS="${2:-8}"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

if [[ ! -f ".env.local" ]]; then
    echo "[ERR] missing .env.local; create it from .env.example and add ANDROZOO_APIKEY"
    exit 1
fi

step "1/2 local download lamda240"
bash scripts/13_download_lamda240_local.sh "$EVAL_DIR/apks" "$WORKERS"

step "2/2 local preprocessing lamda240 (no download)"
bash scripts/14b_prepare_lamda240_local_no_download.sh "$EVAL_DIR"

echo
echo "[new-device-bootstrap] done"
echo "[new-device-bootstrap] next: push methods/overrides/fcg to cloud with scripts/cloud_lamda_train_push.sh"
