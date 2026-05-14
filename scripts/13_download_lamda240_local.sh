#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download the full lamda240 APK set locally from AndroZoo.
#
# What it does:
#   1) read configs/lamda_binary_eval_240_candidates.csv
#   2) write data_lamda_eval/lamda_full_240.sha.txt
#   3) load ANDROZOO_APIKEY from .env.local if present
#   4) call scripts/00_download_apks.py
#
# Usage:
#   bash scripts/13_download_lamda240_local.sh
#   bash scripts/13_download_lamda240_local.sh <out_dir> <workers>
#
# Example:
#   bash scripts/13_download_lamda240_local.sh data_lamda_eval/apks 8
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-data_lamda_eval/apks}"
WORKERS="${2:-8}"
EVAL_ROOT="$(dirname "$OUT_DIR")"
SHA_FILE="$EVAL_ROOT/lamda_full_240.sha.txt"
CSV_PATH="configs/lamda_binary_eval_240_candidates.csv"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

if [[ ! -f "$CSV_PATH" ]]; then
    echo "[ERR] missing $CSV_PATH"
    exit 1
fi

step "1/3 build lamda240 SHA list"
mkdir -p "$EVAL_ROOT"
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

step "2/3 load .env.local if present"
if [[ -f ".env.local" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env.local
    set +a
    echo "[lamda-local] loaded ANDROZOO_APIKEY from .env.local"
else
    echo "[lamda-local] .env.local not found; relying on current shell env"
fi

if [[ "${LAMDA_KEEP_PROXY:-0}" != "1" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
fi

step "3/3 download lamda240 APKs locally"
python scripts/00_download_apks.py \
  --sha_file="$SHA_FILE" \
  --out_dir="$OUT_DIR" \
  --workers="$WORKERS"

echo
echo "[lamda-local] done"
echo "[lamda-local] APK dir : $OUT_DIR"
echo "[lamda-local] SHA file : $SHA_FILE"
