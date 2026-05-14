#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Import lamda240 APKs downloaded on Windows into the WSL repo tree.
#
# Usage:
#   bash scripts/13c_import_lamda240_from_windows.sh
#   bash scripts/13c_import_lamda240_from_windows.sh <windows_apk_dir> <target_apk_dir>
# -----------------------------------------------------------------------------

set -euo pipefail

WIN_DIR="${1:-/mnt/c/Users/Lenovo/Downloads/lamda240_apks}"
TARGET_DIR="${2:-data_lamda_eval/apks}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

if [[ ! -d "$WIN_DIR" ]]; then
    echo "[ERR] windows apk dir not found: $WIN_DIR"
    exit 1
fi

step "1/2 import APKs from Windows into WSL repo"
mkdir -p "$TARGET_DIR"
cp -n "$WIN_DIR"/*.apk "$TARGET_DIR"/

step "2/2 imported count"
echo "[lamda-import] target apk count: $(find "$TARGET_DIR" -maxdepth 1 -name '*.apk' | wc -l)"
