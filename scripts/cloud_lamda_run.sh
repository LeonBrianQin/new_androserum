#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot lamda240 cloud pipeline:
#   1) materialize sha txt files from CSV splits
#   2) download / disassemble / extract / tag / override for full240
#   3) frozen baseline encode for full240
#   4) Phase 4 A+B+E train on train160
#   5) export finetuned embeddings for train/val/test/full240
#   6) Phase 5 FCG extract for full240
#   7) Phase 6 relay+package train on train160
#   8) export GNN embeddings for train/val/test/full240
#
# Usage:
#   bash scripts/cloud_lamda_run.sh [eval_dir]
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_DIR="${1:-data_lamda_eval}"
ENV_NAME="${ANDROSERUM_ENV:-androserum}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found; run bash scripts/cloud_setup.sh first"
    exit 1
fi

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
note() { printf "\033[1;32m[lamda]\033[0m %s\n" "$*"; }

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$EVAL_DIR" "$EVAL_DIR/reports"

TRAIN_SHA="$EVAL_DIR/lamda_train_160.sha.txt"
VAL_SHA="$EVAL_DIR/lamda_val_40.sha.txt"
TEST_SHA="$EVAL_DIR/lamda_test_40.sha.txt"
FULL_SHA="$EVAL_DIR/lamda_full_240.sha.txt"
P4_DIR="$EVAL_DIR/checkpoints/p4_lamda_abe"
P4_EMB_TRAIN="$EVAL_DIR/embeddings/finetuned/train"
P4_EMB_VAL="$EVAL_DIR/embeddings/finetuned/val"
P4_EMB_TEST="$EVAL_DIR/embeddings/finetuned/test"
P4_EMB_FULL="$EVAL_DIR/embeddings/finetuned/full240"
P6_CFG="$EVAL_DIR/train_gnn_bgrl_relay_package.yaml"
P6_DIR="$EVAL_DIR/checkpoints/p6_lamda_relay_package"
P6_EMB_TRAIN="$EVAL_DIR/gnn_embeddings/relay_package/train"
P6_EMB_VAL="$EVAL_DIR/gnn_embeddings/relay_package/val"
P6_EMB_TEST="$EVAL_DIR/gnn_embeddings/relay_package/test"
P6_EMB_FULL="$EVAL_DIR/gnn_embeddings/relay_package/full240"

step "0/9 build SHA lists from lamda CSV splits"
python - <<PY
from pathlib import Path
import pandas as pd

eval_dir = Path("$EVAL_DIR")
eval_dir.mkdir(parents=True, exist_ok=True)

def write_sha(csv_path: str, out_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    shas = [str(x).strip().upper() for x in df["sha256"].tolist()]
    Path(out_path).write_text("".join(f"{sha}\n" for sha in shas), encoding="utf-8")
    return shas

train = write_sha("configs/lamda_train_160.csv", "$TRAIN_SHA")
val = write_sha("configs/lamda_val_40.csv", "$VAL_SHA")
test = write_sha("configs/lamda_test_40.csv", "$TEST_SHA")
full = []
seen = set()
for sha in train + val + test:
    if sha not in seen:
        seen.add(sha)
        full.append(sha)
Path("$FULL_SHA").write_text("".join(f"{sha}\n" for sha in full), encoding="utf-8")
print({"train": len(train), "val": len(val), "test": len(test), "full": len(full)})
PY

step "1/9 Phase 0 download full lamda240"
python scripts/00_download_apks.py \
  --sha_file="$FULL_SHA" \
  --out_dir="$EVAL_DIR/apks" \
  --workers=8

step "2/9 Phase 1-2-2b-2c preprocess full lamda240"
python scripts/01_disassemble_apks.py \
  --sha_file="$FULL_SHA" \
  --apks_dir="$EVAL_DIR/apks" \
  --processed_dir="$EVAL_DIR/processed"
python scripts/02_extract_methods.py \
  --sha_file="$FULL_SHA" \
  --processed_dir="$EVAL_DIR/processed" \
  --methods_dir="$EVAL_DIR/methods"
python scripts/02b_tag_susi.py \
  --sha_file="$FULL_SHA" \
  --methods_dir="$EVAL_DIR/methods"
python scripts/02c_extract_overrides.py \
  --sha_file="$FULL_SHA" \
  --apks_dir="$EVAL_DIR/apks" \
  --out_dir="$EVAL_DIR/overrides"

step "3/9 Phase 3 frozen baseline encode full lamda240"
python scripts/03_encode_methods.py \
  --sha_file="$FULL_SHA" \
  --methods_dir="$EVAL_DIR/methods" \
  --out_dir="$EVAL_DIR/embeddings/baseline" \
  --device=cuda \
  --batch_size=32

step "4/9 Phase 4 A+B+E train on train160"
python scripts/04_train_contrastive.py \
  --sha_file="$TRAIN_SHA" \
  --methods_dir="$EVAL_DIR/methods" \
  --checkpoint_dir="$P4_DIR" \
  --finetuned_dir="$P4_EMB_TRAIN" \
  --device=cuda \
  --batch_size=8 \
  --epochs=3 \
  --lr=3e-5 \
  --weight_decay=1e-2 \
  --temperature=0.07 \
  --projection_dim=256 \
  --freeze_n_layers=4 \
  --freeze_embeddings=true \
  --label_group_size=2 \
  --label_fraction=0.5 \
  --steps_per_epoch=1000 \
  --max_unlabeled_per_apk=256 \
  --unlabeled_keep_ratio=0.05 \
  --overrides_dir="$EVAL_DIR/overrides" \
  --use_signal_e=true \
  --num_workers=0 \
  --seed=13 \
  --log_every=20 \
  --grad_clip_norm=1.0 \
  --export_after_train=true

step "5/9 export Phase 4 finetuned embeddings for val/test/full240"
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$VAL_SHA" \
  --methods_dir="$EVAL_DIR/methods" \
  --out_dir="$P4_EMB_VAL" \
  --device=cuda \
  --batch_size=16
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$TEST_SHA" \
  --methods_dir="$EVAL_DIR/methods" \
  --out_dir="$P4_EMB_TEST" \
  --device=cuda \
  --batch_size=16
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$FULL_SHA" \
  --methods_dir="$EVAL_DIR/methods" \
  --out_dir="$P4_EMB_FULL" \
  --device=cuda \
  --batch_size=16

step "6/9 Phase 5 FCG extract full lamda240"
python scripts/05_extract_fcg.py \
  --sha_file="$FULL_SHA" \
  --apks_dir="$EVAL_DIR/apks" \
  --methods_dir="$EVAL_DIR/methods" \
  --out_dir="$EVAL_DIR/fcg"

step "7/9 build lamda Phase 6 relay+package config"
cat > "$P6_CFG" <<EOF
sha_file: $TRAIN_SHA
fcg_dir: $EVAL_DIR/fcg
embeddings_dir: $P4_EMB_TRAIN
checkpoint_dir: $P6_DIR
gnn_embeddings_dir: $P6_EMB_TRAIN
device: cuda
graph_mode: relay
external_prior_mode: package
add_reverse_edges: true
hidden_dim: 256
output_dim: 256
predictor_hidden_dim: 512
encoder_dropout: 0.1
edge_drop_prob: 0.3
feature_mask_prob: 0.2
lr: 0.001
weight_decay: 0.00001
epochs: 20
ema_decay: 0.99
limit: 0
seed: 13
log_every: 10
exclude_filtered_from_loss: true
loss_scope: internal
export_after_train: true
export_encoder: target
EOF
note "wrote $P6_CFG"

step "8/9 Phase 6 relay+package train on train160"
python scripts/06_train_gnn.py --cfg_path="$P6_CFG"

step "9/9 export Phase 6 GNN embeddings for val/test/full240"
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$VAL_SHA" \
  --fcg_dir="$EVAL_DIR/fcg" \
  --embeddings_dir="$P4_EMB_VAL" \
  --out_dir="$P6_EMB_VAL" \
  --device=cuda
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$TEST_SHA" \
  --fcg_dir="$EVAL_DIR/fcg" \
  --embeddings_dir="$P4_EMB_TEST" \
  --out_dir="$P6_EMB_TEST" \
  --device=cuda
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$FULL_SHA" \
  --fcg_dir="$EVAL_DIR/fcg" \
  --embeddings_dir="$P4_EMB_FULL" \
  --out_dir="$P6_EMB_FULL" \
  --device=cuda

note "lamda cloud pipeline finished"
