# Androserum

> Android method-level semantic clustering based on **DexBERT** + **Function Call Graph** + **self-supervised GNN**.

Given a corpus of Android APKs, learn a semantic vector for **each method** and cluster methods by **functional similarity** (network / crypto / UI / IO / ...) without supervision.

The full project background, design decisions, and execution plan live in [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md).

## Pipeline (Phases 0–7)

```
Phase 0: APK download (AndroZoo)
Phase 1: Disassemble (baksmali → smali → instructions .txt)
Phase 2: Method extraction + filtering + SuSi tagging
Phase 3: Frozen DexBERT embeddings (baseline, 768-d)
Phase 4: Contrastive fine-tuning (SimCSE + SuSi same-category)   ← core
Phase 5: FCG extraction (Androguard) + method-id alignment
Phase 6: Self-supervised GNN over FCG (BGRL, 2-layer GAT)        ← optional
Phase 7: Clustering + evaluation (UMAP + HDBSCAN; silhouette / NMI / ARI)
```

## Project Layout

```
new_androserum/
├── assets/                    # gitignored: model weights, vocab, baksmali jar
├── configs/                   # encoder + train hyperparameters (versioned)
├── data/                      # gitignored: apks, processed, embeddings, fcg
├── docs/                      # decision log + design notes
├── notebooks/                 # exploration / visualization
├── scripts/                   # entrypoints: 00_download → 07_cluster_eval
├── src/androserum/            # main package (importable as `androserum`)
│   ├── encoder/               # DexBERT transformer (migrated, TF-free)
│   ├── data/                  # download / disassemble / method extract / SuSi tag
│   ├── inference/             # frozen-encoder embeddings (Phase 3)
│   ├── train/                 # contrastive fine-tuning (Phase 4)
│   ├── fcg/                   # FCG extract + method-id alignment (Phase 5)
│   ├── gnn/                   # self-supervised GNN (Phase 6)
│   ├── cluster/               # UMAP + HDBSCAN + evaluation (Phase 7)
│   └── utils/                 # IO / logging / sanity checks
├── tests/                     # pytest
└── third_party/               # SuSi categories table etc.
```

## Setup

### Prerequisites

- Python 3.10+
- Java 11 (for baksmali)
- (Optional) NVIDIA GPU for Phase 3/4/6 — RTX 3090/4090 (24 GB VRAM) recommended

### Install (local dev)

```bash
conda activate DexBert         # reuse existing env (see PROJECT_HANDOFF.md §8.1)
cd /usr/local/python_projects/new_androserum
pip install -e ".[dev]"
```

### Download assets (gitignored, fetch manually)

The `assets/` directory holds 3 files that are **NOT** in version control:

| File | Source | Approx. size |
|------|--------|-------------|
| `assets/model_steps_604364.pt` | DexBERT pretrained weights | ~1.8 GB |
| `assets/vocab.txt`             | DexBERT vocab (9,537 tokens) | ~70 KB |
| `assets/baksmali-2.5.2.jar`    | Smali disassembler | ~1.3 MB |

DexBERT weights (Google Drive): <https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view>
baksmali: <https://github.com/JesusFreke/smali>

### Configure secrets (`ANDROZOO_APIKEY`)

API keys live in `.env.local` (gitignored), not source code:

```bash
cp .env.example .env.local
vim .env.local                    # paste your real ANDROZOO_APIKEY
set -a; source .env.local; set +a # export to current shell
```

Apply for an AndroZoo academic key: <https://androzoo.uni.lu/access>.

### Verify install

```bash
pytest tests/ -v -m "not slow"        # quick unit tests (~5s)
pytest tests/test_encoder_load.py -v  # 1.8 GB ckpt smoke test (~10s on GPU)
# expected: vocab=9537, [UNK]=0%, missing=0, MLM forward OK
```

## Cloud GPU Quick Start (AutoDL / Lambda / Vast)

Phase 3 (frozen DexBERT encoding) and especially Phase 4 (contrastive
fine-tuning) benefit a lot from a 24 GB GPU. To get this repo running on a
fresh box:

```bash
# 1. (on your laptop) push the latest code
git push origin main

# 2. (on the GPU box) clone + bootstrap
git clone https://github.com/LeonBrianQin/new_androserum.git
cd new_androserum
bash scripts/cloud_setup.sh           # apt java + conda env + pip install + GPU check

# 3. (on your laptop) upload the gitignored assets/ directory
rsync -avzP assets/  <user>@<host>:/path/to/new_androserum/assets/
# this is ~1.8 GB, takes 2-5 min on a decent uplink

# 4. (on the GPU box) configure secrets
cp .env.example .env.local
vim .env.local                        # paste ANDROZOO_APIKEY
set -a; source .env.local; set +a

# 5. fire the pipeline
conda activate androserum
python scripts/run_all.py --batch_size=128 --workers=8
```

Estimated wall-clock for the 200-APK dev set on a single RTX 4090
(24 GB): ~1.5 h end-to-end (Phase 0–3). Reference: ~5 h on a laptop
RTX 4060 (8 GB).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Citation

This project builds on:

- Sun, T. et al. *DexBERT: Effective, Task-Agnostic and Fine-grained Representation Learning of Android Bytecode.* IEEE Transactions on Software Engineering, 2023.
- Original repo: <https://github.com/TiezhuSun/DexBERT>
