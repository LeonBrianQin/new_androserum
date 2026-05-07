#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot bootstrap for a fresh GPU box (AutoDL, Lambda, Vast.ai, …).
#
# Assumes:
#   * Ubuntu / Debian-ish, with sudo (or already root, e.g. AutoDL containers)
#   * Conda preinstalled (almost every AutoDL image has it)
#   * NVIDIA driver + CUDA already provisioned by the cloud provider
#
# What it does:
#   1) apt-installs Java 11 (baksmali requires it)
#   2) creates conda env `androserum` (Python 3.10) if missing
#   3) installs the package in editable mode with `[dev]` extras
#   4) prints a checklist of the assets you still have to upload manually
#      (model weights / vocab / baksmali jar — gitignored, ~1.8 GB total)
#   5) prints how to populate `.env.local` with your AndroZoo key
#
# Run:    bash scripts/cloud_setup.sh
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ENV_NAME="${ANDROSERUM_ENV:-androserum}"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[ok]\033[0m %s\n"   "$*"; }

step "1/5  apt: install Java 11 (for baksmali)"
if command -v java >/dev/null 2>&1; then
    ok "java already present: $(java -version 2>&1 | head -n1)"
else
    if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq openjdk-11-jre-headless
    ok "installed openjdk-11-jre-headless"
fi

step "2/5  conda: create env '$ENV_NAME' (Python 3.10)"
if ! command -v conda >/dev/null 2>&1; then
    warn "conda not on PATH — install Miniconda first, then re-run this script."
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ok "env '$ENV_NAME' already exists"
else
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
ok "active env: $(python -V) @ $(which python)"

step "3/5  pip: install androserum in editable mode (+ dev extras)"
pip install --upgrade pip
pip install -e ".[dev]"
ok "package installed"

step "4/5  GPU sanity check"
python - <<'PY'
import torch
print(f"  torch    : {torch.__version__}")
print(f"  cuda ok  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device   : {torch.cuda.get_device_name(0)}")
    print(f"  vram     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

step "5/5  manual steps still required"
cat <<'EOF'

The following large/secret files are NOT in git. You must put them in
place before running the pipeline:

  assets/model_steps_604364.pt    (~1.8 GB) — DexBERT pretrained weights
  assets/vocab.txt                (~70 KB)  — DexBERT vocabulary
  assets/baksmali-2.5.2.jar       (~1.3 MB) — Smali disassembler

  -> easiest: rsync from your laptop:
        rsync -avzP /local/path/to/new_androserum/assets/  user@<host>:$(pwd)/assets/

  .env.local                                — AndroZoo API key
  -> cp .env.example .env.local && vim .env.local
  -> then: set -a; source .env.local; set +a

Then verify everything is wired up correctly:

  pytest tests/ -v -m "not slow"   # cheap unit tests
  pytest tests/test_encoder_load.py -v   # 1.8 GB ckpt smoke test (~10s on GPU)

Finally, fire the full pipeline:

  python scripts/run_all.py --batch_size=128 --workers=8

EOF
ok "bootstrap finished"
