#!/usr/bin/env bash
set -euo pipefail

# Install the Phase 6 PyG stack matching the *current* torch build.
#
# Usage:
#   conda activate DexBert
#   bash scripts/setup_gnn_env.sh
#
# Optional:
#   PYTHON_BIN=python bash scripts/setup_gnn_env.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

read -r TORCH_VERSION CUDA_TAG <<EOF
$("$PYTHON_BIN" - <<'PY'
import torch

torch_ver = torch.__version__.split("+", 1)[0]
cuda_ver = torch.version.cuda
if cuda_ver:
    cuda_tag = "cu" + cuda_ver.replace(".", "")
else:
    cuda_tag = "cpu"
print(torch_ver, cuda_tag)
PY
)
EOF

WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "[gnn-setup] python: $PYTHON_BIN"
echo "[gnn-setup] torch:  $TORCH_VERSION"
echo "[gnn-setup] build:  $CUDA_TAG"
echo "[gnn-setup] wheel:  $WHEEL_URL"

"$PYTHON_BIN" -m pip install \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_geometric \
  -f "$WHEEL_URL"

echo "[gnn-setup] done"
