#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Cloud GPU pipeline for lamda240 after local preprocessing is already done.
#
# Assumes these are already present on the cloud box:
#   * assets/
#   * data_lamda_eval/methods/
#   * data_lamda_eval/overrides/
#   * data_lamda_eval/fcg/
#
# Runs on the cloud box:
#   0) build SHA lists from CSV splits
#   1) Phase 4 A+B+E train on train160
#   2) export Phase 4 embeddings for train/val/test/full240
#   3) Phase 6 relay+package train on train160
#   4) export Phase 6 embeddings for train/val/test/full240
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
note() { printf "\033[1;32m[lamda-train]\033[0m %s\n" "$*"; }

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$EVAL_DIR" "$EVAL_DIR/reports"

TRAIN_SHA="$EVAL_DIR/lamda_train_160.sha.txt"
VAL_SHA="$EVAL_DIR/lamda_val_40.sha.txt"
TEST_SHA="$EVAL_DIR/lamda_test_40.sha.txt"
FULL_SHA="$EVAL_DIR/lamda_full_240.sha.txt"
METHODS_DIR="$EVAL_DIR/methods"
OVERRIDES_DIR="$EVAL_DIR/overrides"
FCG_DIR="$EVAL_DIR/fcg"
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
P4_BATCH_SIZE="${LAMDA_P4_BATCH_SIZE:-16}"
P4_EXPORT_BATCH_SIZE="${LAMDA_P4_EXPORT_BATCH_SIZE:-32}"
P4_EPOCHS="${LAMDA_P4_EPOCHS:-3}"
P6_EPOCHS="${LAMDA_P6_EPOCHS:-20}"

step "0/4 build SHA lists from lamda CSV splits"
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

step "1/4 Phase 4 A+B+E train on train160"
python scripts/04_train_contrastive.py \
  --sha_file="$TRAIN_SHA" \
  --methods_dir="$METHODS_DIR" \
  --checkpoint_dir="$P4_DIR" \
  --finetuned_dir="$P4_EMB_TRAIN" \
  --device=cuda \
  --batch_size="$P4_BATCH_SIZE" \
  --epochs="$P4_EPOCHS" \
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
  --overrides_dir="$OVERRIDES_DIR" \
  --use_signal_e=true \
  --num_workers=0 \
  --seed=13 \
  --log_every=20 \
  --grad_clip_norm=1.0 \
  --export_after_train=true

step "2/4 export Phase 4 embeddings for val/test/full240"
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$VAL_SHA" \
  --methods_dir="$METHODS_DIR" \
  --out_dir="$P4_EMB_VAL" \
  --device=cuda \
  --batch_size="$P4_EXPORT_BATCH_SIZE"
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$TEST_SHA" \
  --methods_dir="$METHODS_DIR" \
  --out_dir="$P4_EMB_TEST" \
  --device=cuda \
  --batch_size="$P4_EXPORT_BATCH_SIZE"
python scripts/04b_export_finetuned_embeddings.py \
  --checkpoint_path="$P4_DIR/contrastive_ab_best.pt" \
  --sha_file="$FULL_SHA" \
  --methods_dir="$METHODS_DIR" \
  --out_dir="$P4_EMB_FULL" \
  --device=cuda \
  --batch_size="$P4_EXPORT_BATCH_SIZE"

step "3/4 Phase 6 relay+package train on train160"
cat > "$P6_CFG" <<EOF
sha_file: $TRAIN_SHA
fcg_dir: $FCG_DIR
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
epochs: $P6_EPOCHS
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
python scripts/06_train_gnn.py --cfg_path="$P6_CFG"

step "4/4 export Phase 6 embeddings for val/test/full240"
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$VAL_SHA" \
  --fcg_dir="$FCG_DIR" \
  --embeddings_dir="$P4_EMB_VAL" \
  --out_dir="$P6_EMB_VAL" \
  --device=cuda
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$TEST_SHA" \
  --fcg_dir="$FCG_DIR" \
  --embeddings_dir="$P4_EMB_TEST" \
  --out_dir="$P6_EMB_TEST" \
  --device=cuda
python scripts/06b_export_gnn_embeddings.py \
  --checkpoint_path="$P6_DIR/bgrl_graphsage_best.pt" \
  --sha_file="$FULL_SHA" \
  --fcg_dir="$FCG_DIR" \
  --embeddings_dir="$P4_EMB_FULL" \
  --out_dir="$P6_EMB_FULL" \
  --device=cuda

note "cloud train-only pipeline finished"
