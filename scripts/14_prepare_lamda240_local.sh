#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Local lamda240 preprocessing pipeline.
#
# Runs on the local machine:
#   0) download APKs
#   1) disassemble
#   2) extract methods
#   2b) tag SuSi
#   2c) extract overrides
#   5) extract FCG
#
# Stops before any GPU-heavy embedding / training work.
#
# Usage:
#   bash scripts/14_prepare_lamda240_local.sh
#   bash scripts/14_prepare_lamda240_local.sh <eval_dir> <workers>
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
WORKERS="${2:-8}"
APK_DIR="$EVAL_DIR/apks"
PROC_DIR="$EVAL_DIR/processed"
METHODS_DIR="$EVAL_DIR/methods"
OVERRIDES_DIR="$EVAL_DIR/overrides"
FCG_DIR="$EVAL_DIR/fcg"
SHA_FILE="$EVAL_DIR/lamda_full_240.sha.txt"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

step "0/6 download lamda240 APKs locally"
bash scripts/13_download_lamda240_local.sh "$APK_DIR" "$WORKERS"

step "1/6 disassemble APKs locally"
python scripts/01_disassemble_apks.py \
  --sha_file="$SHA_FILE" \
  --apks_dir="$APK_DIR" \
  --processed_dir="$PROC_DIR"

step "2/6 extract methods locally"
python scripts/02_extract_methods.py \
  --sha_file="$SHA_FILE" \
  --processed_dir="$PROC_DIR" \
  --methods_dir="$METHODS_DIR"

step "3/6 tag SuSi locally"
python scripts/02b_tag_susi.py \
  --sha_file="$SHA_FILE" \
  --methods_dir="$METHODS_DIR"

step "4/6 extract overrides locally"
python scripts/02c_extract_overrides.py \
  --sha_file="$SHA_FILE" \
  --apks_dir="$APK_DIR" \
  --out_dir="$OVERRIDES_DIR"

step "5/6 extract FCG locally"
python scripts/05_extract_fcg.py \
  --sha_file="$SHA_FILE" \
  --apks_dir="$APK_DIR" \
  --methods_dir="$METHODS_DIR" \
  --out_dir="$FCG_DIR"

step "6/6 local preprocessing summary"
echo "[lamda-local-prepare] APKs      : $(find "$APK_DIR" -maxdepth 1 -name '*.apk' | wc -l)"
echo "[lamda-local-prepare] processed : $(find "$PROC_DIR" -maxdepth 1 -name '*.txt' | wc -l)"
echo "[lamda-local-prepare] methods   : $(find "$METHODS_DIR" -maxdepth 1 -name '*.parquet' | wc -l)"
echo "[lamda-local-prepare] overrides : $(find "$OVERRIDES_DIR" -maxdepth 1 -name '*.parquet' | wc -l)"
echo "[lamda-local-prepare] fcg json  : $(find "$FCG_DIR" -maxdepth 1 -name '*.summary.json' | wc -l)"

echo
echo "[lamda-local-prepare] done"
echo "[lamda-local-prepare] next step: push local artifacts to cloud for training"
