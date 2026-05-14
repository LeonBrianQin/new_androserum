#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Local lamda240 preprocessing pipeline assuming APKs are already present.
#
# Runs on the local machine:
#   1) disassemble
#   2) extract methods
#   2b) tag SuSi
#   2c) extract overrides
#   5) extract FCG
#
# Usage:
#   bash scripts/14b_prepare_lamda240_local_no_download.sh
#   bash scripts/14b_prepare_lamda240_local_no_download.sh <eval_dir>
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
APK_DIR="$EVAL_DIR/apks"
PROC_DIR="$EVAL_DIR/processed"
METHODS_DIR="$EVAL_DIR/methods"
OVERRIDES_DIR="$EVAL_DIR/overrides"
FCG_DIR="$EVAL_DIR/fcg"
SHA_FILE="$EVAL_DIR/lamda_full_240.sha.txt"
CSV_PATH="configs/lamda_binary_eval_240_candidates.csv"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

if [[ ! -f "$CSV_PATH" ]]; then
    echo "[ERR] missing $CSV_PATH"
    exit 1
fi

if [[ ! -f "$SHA_FILE" ]]; then
    step "0/6 build lamda240 SHA list"
    mkdir -p "$EVAL_DIR"
    python3 - <<PY
import csv
from pathlib import Path

csv_path = Path("$CSV_PATH")
sha_path = Path("$SHA_FILE")
seen = set()
rows = []
with csv_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sha = row["sha256"].strip().upper()
        if sha and sha not in seen:
            seen.add(sha)
            rows.append(sha)
sha_path.write_text("".join(f"{sha}\n" for sha in rows), encoding="utf-8")
print({"sha_file": str(sha_path), "count": len(rows)})
PY
fi

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
