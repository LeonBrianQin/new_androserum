# 项目交接文档：Android Method Clustering (基于 DexBERT)

> **本文档是一份完整的项目背景 + 决策快照 + 启动指南。**
> 如果你（agent 或人）是第一次接触这个项目，**只需要读这一份文档，就能在 5 分钟内接上工作。**
>
> 文档生成于：在 `/usr/local/python_projects/DexBERT/` 下与上一位 agent 的多轮讨论。
> 用途：作为搬迁到新项目仓库时的"完整上下文 snapshot"。

---

## 0. 给新 Agent 的 TL;DR

如果你是新窗口里被丢进来读这份文档的 agent，请按以下顺序行事：

1. **读完本文档（约 10–15 分钟）**：依次理解"项目目标 → 关键决策 → 流水线 → 当前状态 → 下一步任务"。
2. **不要重新论证已经做过的决策**：本文档"决策日志"一节列出的每一个选择都有充分的讨论依据，除非用户明确要求重审某项决策，否则按当前结论继续。
3. **下一步要做的事在第 7 节"立即可启动的下一步"**：直接对照执行。
4. **当前状态在第 5 节**：§5.0 为 DexBERT 旧仓库参考；**§「新项目 new_androserum」为真实进度快照**，续接时优先读后者 + §11 最新一条。
5. **如果用户想跳过某 Phase 或调整流程**：本文档只是当前共识的快照，不是契约，按用户最新意图为准。

---

## 1. 项目目标

**一句话**：基于 Android APK 反编译产出的 Smali 字节码，结合 **DexBERT 预训练模型** 与 **函数调用图（FCG）**，对每个**函数（method）节点**学习语义向量，并按"功能性语义"进行**无监督聚类**，得到诸如"网络请求类 / 加密类 / UI 类 / IO 类"等功能簇。

### 输入

- 一组 Android APK 文件（300–500 个，跨年份、跨类型）

### 输出

- 每个 APK 的 FCG（function call graph），节点 = method
- 每个 method 节点的语义向量（功能感知 + FCG 上下文感知）
- 节点聚类结果：每簇代表一类"功能相似的方法"
- 聚类质量指标：silhouette、NMI（基于 SuSi 弱标签）

### 不做什么

- 不做监督分类（如恶意/非恶意、缺陷/非缺陷）—— DexBERT 原作者的下游任务
- 不做 class 级聚类
- 不基于 DexBERT 仓库做 fork 修改 —— 而是新建项目，仅复用其 encoder

---

## 2. 关键决策日志

以下决策在多轮讨论后已确认，**有充分依据，新 agent 默认遵循**。

### 决策 1：粒度选 method 级，不选 class 级

- **DexBERT 原作者下游任务**：class 级（一个 class → 一个 768 维向量）
- **本项目目标**：FCG 节点 = method（一个方法 → 一个向量）
- **为什么 class 不行**：一个 class 内的多个 method 共享同一个 class 向量，无法对单个节点聚类；class 向量是多个 method 语义的平均，丢失了"该方法独有的功能信号"
- **方案**：复用 DexBERT 的 transformer encoder，但**改输入切片粒度**——按 method 切，每方法独立 tokenize → forward → 取 [CLS] 768 维向量

### 决策 2：训练目标 = 对比学习（contrastive learning），不是从头训练

- **现状**：DexBERT 已用 MLM + NSP 在 1000 个 APK 上预训练好（`model_steps_604364.pt`）
- **问题**：MLM 学的是 smali "语法 / 统计共现"，不是"功能"
- **方案**：在 DexBERT encoder 之上做**对比学习微调**，把"功能相似"信号通过正负例 pair 注入向量空间
- **不重训预训练，只微调**

### 决策 3：监督信号采用渐进升级策略 A → A+B+E → A+B+C+E

五种候选监督信号：

| 标签 | 信号 | 成本 | 强度 | 是否采用 |
|------|------|------|------|---------|
| **A** | SimCSE 自配对（同方法两次 dropout 互为正例） | 0 | 弱（regularizer） | **必用，地基** |
| **B** | SuSi 同 API 类别（都调网络/加密/IO 等同类 API 的方法互为正例） | 低（半天） | **强** | **MVP 起手用** |
| **C** | 库指纹（LibScout/LibRadar 识别"不同 APK 的同库同方法"） | 高（2–3 天搭建） | **最强** | 论文质量需要时再上 |
| **D** | FCG 1-hop 邻居互为正例 | 低 | 中（有噪） | **不用**（与 Phase 6 GNN 重复） |
| **E** | 同 override（都 override 同一父类方法） | 极低 | 中 | 第二轮加，几乎免费 |

**升级路径**：

```
Week 1 (MVP):       A + B           → 跑通 + 出 baseline 指标
Week 2 (提升):      A + B + E       → 加近免费的 override 信号
(可选) Week 3+:     A + B + C + E   → 上 LibScout，论文级
```

每加一种信号都做 ablation，不一次性堆叠所有信号。

### 决策 4：信号 D 不放 Phase 4，留给 Phase 6 的 GNN

- D（FCG 邻居正例）和 GNN 的消息传递在做同一件事
- 在两个 Phase 同时用会**双重平滑**，把内容向量"糊"掉
- **分工**：Phase 4 让 encoder 学"内容功能"，Phase 6 让 GNN 学"图上下文"，互不重复

### 决策 5：Phase 6 GNN 选 BGRL（推荐）或 DGI（可选）

| 选项 | 优点 | 缺点 |
|------|------|------|
| **BGRL** | 无负样本、训练稳定、batch 压力小 | 实现略新 |
| DGI | 经典、文档多 | 需要 corrupted graph 采样 |
| GraphCL | graph-level 对比 | 不太适合 node clustering |
| VGAE | 重构边 | cluster 效果一般 |

**首选 BGRL**：在稀疏标签 + 节点聚类场景下表现稳定。架构 = 2 层 GAT，hidden 256，4 heads。

### 决策 6：FCG 抽取用 Androguard，不上 Soot

- Androguard：纯 Python，`pip install androguard` 一行装好，`dx.get_call_graph()` 直接返回 NetworkX 图
- Soot：精度更高（SPARK / k-CFA），但 JVM + 依赖重，输出还要解析
- **聚类任务对边的精度不敏感**（GNN 会做 2–3 跳消息传递，少一两条 virtual dispatch 边不影响整体）
- **不做多工具对比**，单一工具一条流水线

### 决策 7：数据规模目标 300–500 APK

| Phase | 最少 | 推荐 |
|-------|------|------|
| Phase 3 推理 baseline | 1 | 几个 |
| Phase 4 对比微调 | 50 APK | **200–500 APK** |
| Phase 6 GNN 训练 | 100 图 | **300–500 图** |
| 整体 | — | **300 APK 起步**，多样性比绝对数量重要 |

每个 APK 过滤后约 3000–8000 个 method，300 APK ≈ 100–200 万 method 节点，足够对比学习。

可以直接复用 `Data/data/pretraining_apks.txt` 里的 1000 个 sha 列表（DexBERT 作者验证过反编译能成功）。

### 决策 8：GPU 选 RTX 3090/4090（24GB），不用 A100

- DexBERT 是 8 层 × 768 dim 的中型模型，FP16 训练 batch 64 在 24GB 显存里绰绰有余
- A100 的 40/80GB 显存对此任务**纯属浪费**，4090 算力已经接近 A100 一半价格三分之一
- 推荐 AutoDL / 矩池云 / RunPod 租 4090 按需使用
- 总 GPU 时间预算：**20–30 小时**（4090 约 ¥80–200 总成本）

### 决策 9：本地做 CPU 任务，云上只跑 GPU 任务

- **本地**：Phase 0（下载）/ 1（反编译）/ 2（method 切分）/ 5（FCG 抽取）/ 7（聚类）
- **云 GPU**：Phase 3（推理）/ 4（对比学习）/ 6（GNN）
- **不要把 APK 原文件传上云**，只传 `processed/*.txt` 和 `methods/*.parquet`

### 决策 10：新建独立项目，不基于 DexBERT 仓库 fork

- DexBERT 仓库 ~70% 代码（4 个下游任务 + MIL + TF checkpoint 转换）用不上
- DexBERT 的 `train.py → checkpoint.py → tensorflow` 是隐性依赖污染源
- **新建项目可以避开 tensorflow 依赖**，只搬必需的 encoder 文件
- DexBERT 仓库降级为 "reference / asset 来源"

---

## 3. 完整流水线（Phase 0–7）

```
┌─ Phase 0: 数据获取 ────────────────────────────────────────────────┐
│  AndroZoo 下载 APK (本地, 几小时)                                   │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 1: 反编译 + 指令文本化 ─────────────────────────────────────┐
│  baksmali: APK → smali → processed/<sha>.txt (本地, ~1 小时 / 500 个)│
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 2: Method 切分 + 过滤 + SuSi 打标 ─────────────────────────┐
│  <sha>.txt → methods/<sha>.parquet (本地, ~30 分钟)                │
│  字段: apk_sha, class, method_sig, full_id, instructions,          │
│        api_calls, susi_cats, susi_dominant_cat                      │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 3: 初始向量 (Frozen DexBERT, baseline) ────────────────────┐
│  每个 method 取 [CLS] → 768 维向量                                 │
│  embeddings/baseline/<sha>.npz (云 GPU, ~30 分钟)                  │
│  双重作用: ① ablation baseline ② Phase 4 训练初始化                 │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 4: 对比学习微调 (核心) ─────────────────────────────────────┐
│  正例: A (SimCSE) + B (SuSi 同类别) [+ E (同 override)]            │
│  负例: in-batch + hard mining                                       │
│  Loss: InfoNCE / SupCon, τ=0.07                                     │
│  Architecture: encoder + Linear(768→256) projection head           │
│  Hyperparams: bs=64, lr=2e-5, epochs=10, freeze 前 4 层            │
│  → encoder_finetuned.pt + embeddings/finetuned/<sha>.npy          │
│  (云 GPU, 6–12 小时)                                                │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 5: FCG 抽取 + method 对齐 ─────────────────────────────────┐
│  Androguard: AnalyzeAPK → dx.get_call_graph() → NetworkX           │
│  关键工程: method 签名标准化 (与 Phase 2 的 full_id 对齐)            │
│  → fcg/<sha>.gpickle (本地, ~1 小时)                                │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 6: 自监督 GNN over FCG ────────────────────────────────────┐
│  节点特征 = Phase 4 输出向量 (768d)                                 │
│  模型: 2 层 GAT, hidden 256, 4 heads                                │
│  目标: BGRL (推荐) - online + target encoder, EMA, 双视图增强        │
│  增强: 边 dropout 30% + 特征 mask 20%                               │
│  → gnn_embeddings/<sha>.npy (256d, 云 GPU, 2–4 小时)               │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─ Phase 7: 聚类 + 评估 ─────────────────────────────────────────────┐
│  UMAP (n_neighbors=30, min_dist=0.0, cosine)                       │
│  HDBSCAN (min_cluster_size=50, min_samples=5)                      │
│  指标: silhouette / DBI / NMI(vs SuSi cat) / ARI                    │
│  定性: 每簇 top-10 API + 代表 method (本地, ~30 分钟)                │
└─────────────────────────────────────────────────────────────────────┘
```

总周期估计：**1.5–2 周**（租 GPU 情况下）；纯 CPU 不可行。

---

## 4. 文件级"用 / 不用 / 借鉴"清单

来自 DexBERT 仓库（`/usr/local/python_projects/DexBERT/`）：

### ✅ 必须搬到新项目（核心 encoder）

| 来源文件 | 新位置 | 修改 |
|---------|--------|------|
| `Model/models.py` | `src/amc/encoder/models.py` | 改 `from utils import` 为 `from .utils import` |
| `Model/tokenization.py` | `src/amc/encoder/tokenization.py` | 直接拷 |
| `Model/utils.py` | `src/amc/encoder/utils.py` | 瘦身：只留 `set_seeds / split_last / merge_last / truncate_tokens / get_device`，删 `find_sublist / get_random_word* / get_logger` |
| `Model/config/DexBERT/bert_base.json` | `configs/encoder_base.json` | 直接拷 |

### ✅ 必须搬（数据预处理）

| 来源文件 | 新位置 |
|---------|--------|
| `Data/disassemble.py` | `src/amc/data/disassemble.py`（jar 路径改成可配置） |
| `Data/instruction_generator.py` | `src/amc/data/instruction_generator.py` |
| `Data/baksmali-2.5.2.jar` | `assets/baksmali-2.5.2.jar`（gitignore） |
| `save_dir/DexBERT/model_steps_604364.pt` | `assets/model_steps_604364.pt`（gitignore） |
| `save_dir/DexBERT/vocab.txt` | `assets/vocab.txt`（gitignore） |

### ✅ 必须搬（已经写好的工具脚本）

| 来源文件 | 新位置 | 备注 |
|---------|--------|------|
| `androzoo_download_by_sha.py` | `src/amc/data/androzoo.py` | 已具备：API key 走环境变量 / 三种输入方式 / sha256 校验 / 重试 |
| `process_apk.py` | `src/amc/data/apk_processor.py` | 已具备：APK → smali → 指令 .txt |
| `inspect_checkpoint.py` | `src/amc/utils/inspect.py` | 调试工具 |
| `sanity_check_vocab.py` | `src/amc/utils/sanity.py` | vocab + ckpt 端到端验证 |

### ❌ 不要搬（下游任务 + TF 污染）

下面这些不要 import / 不要复制：

- `Model/MaliciousCodeLocalization.py` / `MaliciousClassDetection_FirstState768.py` — class 级 MIL 任务
- `Model/AppDefectDetection.py` / `_FirstState768.py` — 缺陷预测
- `Model/ComponentTypeClassification_FirstState768.py` — 组件类型分类
- `Model/InferBERT.py` — 你的 sanity_check_vocab.py 已替代
- `Model/task_modules.py` — 仅给原下游任务用
- `Model/checkpoint.py` — **import tensorflow**，污染源
- `Model/train.py` — 依赖 checkpoint.py，间接拽 TF
- `Model/pretrainDexBERT.py` / `.sh` — 预训练流程，你不重训
- `Model/optim.py` — 自家优化器，用 `torch.optim.AdamW` 更标准
- `Model/count_flops.py` — FLOP 计数
- `Data/data4*.py` — 各下游任务的数据脚本
- `Data/generate_vocab.py` — **依赖 tensorflow_text**，且你已有 vocab

### 🟡 借鉴但重写（不直接 import）

| 文件 | 借鉴点 |
|------|--------|
| `Model/MaliciousCodeLocalization.py::compute_class_embeddings` | batch 推理套路；改 class 级为 method 级 |
| `Model/pretrainDexBERT.py::BertAEModel4Pretrain` | encoder 接 [CLS] 池化的写法；对比学习只保留 transformer + projection head |
| `Model/pretrainDexBERT.py::SentPairDataLoader` | 解析 `processed/*.txt` 的格式（ClassName/MethodName/空行） |

---

## 5. 当前状态（已完成的工作）

### 已下载 / 已验证的资源

```
/usr/local/python_projects/DexBERT/
├── save_dir/DexBERT/
│   ├── model_steps_604364.pt    # 预训练权重, OrderedDict, 完整 (151 张量, vocab=9537, dim=768)
│   └── vocab.txt                 # 词表, 9537 行, 已验证与 .pt 完全配对
├── Data/baksmali-2.5.2.jar      # 反编译工具
├── downloaded_samples/
│   └── 0D64...DC4A.apk          # 1 个测试 APK (3.1 MB)
└── processed/
    ├── 0D64...DC4A.txt          # 1 个已反编译指令文本 (3.96 MB, 680 classes, 106942 行)
    └── 0D64...DC4A/             # 反编译出的 smali 目录 (--keep_smali 保留的)
```

### 已写好可直接用的脚本

- `androzoo_download_by_sha.py`：按 sha256 列表从 AndroZoo 下载 APK
- `process_apk.py`：APK → smali → 指令 .txt（可单文件 / 批量）
- `inspect_checkpoint.py`：检查 .pt 文件内容
- `sanity_check_vocab.py`：端到端验证 vocab + .pt 配对

### 已验证的端到端事实

- ✅ 反编译流程通：1.5 秒处理 1 个 APK
- ✅ 词表与权重配对正确：[UNK] 比例 0.0%，state_dict 加载 missing=0/unexpected=0
- ✅ MLM forward 正常：top-1 预测正确（"invoke" → "invoke"，logit +38.57）
- ✅ Conda 环境 `DexBert` 已配置：torch + fire + tensorboardX 都装好
- ✅ Java 11.0.27 可用

### Conda 环境

```bash
conda activate DexBert  # 名字就叫 DexBert (大小写敏感)
# 已装: torch, fire, tensorboardX
# 缺: androguard, torch-geometric, umap-learn, hdbscan, pyarrow, pandas
```

### 新项目 `new_androserum`（`/usr/local/python_projects/new_androserum/`，截至 2026-05-08）

> **注意**：下文才是当前仓库的真实进度；上面 §5 的 DexBERT 目录仍作 asset / 格式参考。

**远程与 Git**：

- GitHub：<https://github.com/LeonBrianQin/new_androserum>，`main` 已与本地对齐。
- 推荐 `origin` 使用 **SSH**：`git@github.com:LeonBrianQin/new_androserum.git`（HTTPS push 在无凭证环境下会失败）。
- 2026-05-08 时在 `43635fb` 之后共有 **6 个 commits**（自旧到新）：
  - `9c2c20f` — 安全：移除源码中的 AndroZoo 默认 key；`.env.example`；`logs/` + `.env.*` 收紧 `.gitignore`；`scripts/cloud_setup.sh`；README 上云 Quick Start + 修正 ckpt 体积说明
  - `edd3788` — Phase 2：`schema` / `method_extractor` / `method_parquet`；`apk_processor` 按 `pyproject.toml` 找项目根与 `assets/baksmali`；配套 pytest
  - `86e1861` — Phase 2b：SuSi 解析与打标；`third_party/susi/` 内嵌上游 Android 4.2 列表（~2.7 MB，免弱网重复下载）
  - `4675363` — Phase 3：`inference/frozen_encode.py` + package export
  - `a6f325d` — `scripts/00_*`…`03_*`、`run_all.py`、`configs/sha_dev_200.txt`
  - `00ad80e` — Phase 4：A+B 对比学习训练模块、`scripts/04_train_contrastive.py`、`run_all.py` 接入 `do_train/phase4_*` 参数、`tests/test_phase4_ab.py`、`scripts/cloud_phase4_{push,run,pull}.sh`

**已实现流水线（Phase 0 → 4）**：

| Phase | 入口 / 核心模块 | 产出（均在 `data/`，gitignored） |
|------|------------------|----------------------------------|
| 0 | `scripts/00_download_apks.py` → `androserum.data.androzoo` | `data/apks/<SHA>.apk` |
| 1 | `scripts/01_disassemble_apks.py` → `apk_processor` | `data/processed/<SHA>.txt` |
| 2 | `scripts/02_extract_methods.py` | `data/methods/<SHA>.parquet` |
| 2b | `scripts/02b_tag_susi.py` | Parquet 内 `susi_cats` / `susi_dominant_cat` |
| 3 | `scripts/03_encode_methods.py` → `frozen_encode` | `data/embeddings/baseline/<SHA>.npz`（`full_id` + 768-d `embedding`） |
| 4 | `scripts/04_train_contrastive.py` → `src/androserum/train/` | `data/checkpoints/<run>/contrastive_ab_{best,last,summary}.pt/json` + `data/embeddings/finetuned/<run>/<SHA>.npz` |
| 串跑 | `scripts/run_all.py`（`fire`，每步可 `do_*=false` / `do_train=true` 控制） | 上述全集 |

**开发集**：`configs/sha_dev_200.txt`（200 个 SHA，来自 DexBERT 预训练列表子集）。

**当前已有结果**：

- `data/embeddings/baseline/` 现有 `201` 个 `.npz`：其中 **200 个**对应 `sha_dev_200.txt`，另有一个历史测试样本 `0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.npz`
- `data/embeddings/finetuned/p4_dev200_run1/` 已完整导出 **200** 个 `.npz`，与 `sha_dev_200.txt` 一一对应
- `data/checkpoints/p4_dev200_run1/contrastive_ab_summary.json` 显示首轮全量 `dev200` 训练结果：
  - `best_mean_loss = 0.8541`
  - 3 个 epoch：`1.1348 → 0.8948 → 0.8541`
  - `exported_npz_files = 200`

**安全（必用）**：

- AndroZoo：仅 **`--apikey` 或环境变量 `ANDROZOO_APIKEY`**。模板：`cp .env.example .env.local`，`set -a; source .env.local; set +a`。
- **轮换 key**：commit `b306b26` 的 `androzoo.py` **曾硬编码过 key**，公开历史中仍存在 — 到 <https://androzoo.uni.lu/access> **作废并重发**，勿依赖“删历史”。

**上云（AutoDL 等）**：

- 通用 bootstrap：`bash scripts/cloud_setup.sh`（Java 11 + conda env `androserum` + `pip install -e ".[dev]"` + GPU 自检清单）
- 当前最省钱路径：**只上云跑 Phase 4**
  - 本机推最小输入：`bash scripts/cloud_phase4_push.sh <user@host> /usr/local/python_projects/new_androserum`
  - 云端跑推荐默认参数：`bash scripts/cloud_phase4_run.sh p4_dev200_run1`
  - 结果拉回本机：`bash scripts/cloud_phase4_pull.sh <user@host> /usr/local/python_projects/new_androserum p4_dev200_run1`
- 注意：若用 `rsync` 手工同步代码到云端，排除规则应写成 **`/data/`**，不要写裸 `data/`，否则会误伤 `src/androserum/data/`

**测试**：`tests/test_schema.py`、`test_method_extractor.py`、`test_method_parquet.py`、`test_susi_index.py`、`tests/test_phase4_ab.py`；全量含大 ckpt：`pytest tests/test_encoder_load.py`。

**进行中 / 下一窗口**：

- ✅ Phase 4（A+B）已实现，并已在 AutoDL 5090 上完成首轮全量 `dev200` 训练与导出
- 当前应先做 **结果验收 / 对比分析**：以 `sha_dev_200.txt` 对齐 `baseline` 和 `finetuned/p4_dev200_run1`，再比较 frozen vs finetuned 的聚类代理指标 / 邻近语义表现
- 之后的下一工程块：**Phase 5** — FCG 抽取与 method-id 对齐
- Phase 5–7：未启动。

**杂项**：

- 长批处理不要用 IDE 附带 `debugpy` 起主进程；后台可用 `nohup python -u scripts/run_all.py ...`。
- 若 shell 包装导致 `git commit` 报 `unknown option trailer`，用 **`/usr/bin/git`** 调用。

---

## 6. 新项目目标结构

新项目应该建在 `/usr/local/python_projects/android-method-clustering/`（或同级目录，名字可改）。

```
android-method-clustering/
├── README.md
├── pyproject.toml                      # 见下方依赖清单
├── .gitignore                          # 忽略 assets/ data/ *.pt *.apk
├── LICENSE                              # MIT 或 Apache-2.0
│
├── PROJECT_HANDOFF.md                   # ← 本文档（拷过来）
├── docs/
│   └── DECISIONS.md                     # 决策日志（可摘自本文档第 2 节）
│
├── configs/
│   ├── encoder_base.json                # 来自 DexBERT bert_base.json
│   ├── train_contrastive.yaml           # Phase 4 超参（见决策 3 / 流程图）
│   └── train_gnn.yaml                   # Phase 6 超参
│
├── assets/                              # gitignore, README 里写下载方法
│   ├── model_steps_604364.pt            # 拷自 DexBERT
│   ├── vocab.txt                        # 拷自 DexBERT
│   └── baksmali-2.5.2.jar               # 拷自 DexBERT
│
├── src/amc/                             # 主包
│   ├── __init__.py
│   │
│   ├── encoder/                         # ← DexBERT 核心
│   │   ├── __init__.py
│   │   ├── models.py                    # 拷自 Model/models.py
│   │   ├── tokenization.py              # 拷自 Model/tokenization.py
│   │   ├── utils.py                     # 瘦身后的 utils
│   │   └── loader.py                    # 新写: load_pretrained_encoder()
│   │
│   ├── data/                            # Phase 0-2
│   │   ├── __init__.py
│   │   ├── androzoo.py                  # 拷自 androzoo_download_by_sha.py
│   │   ├── disassemble.py               # 拷自 Data/disassemble.py
│   │   ├── instruction_generator.py     # 拷自 Data/
│   │   ├── apk_processor.py             # 拷自 process_apk.py
│   │   ├── method_extractor.py          # 新写: Phase 2 method 切分
│   │   ├── susi_tagger.py               # 新写: SuSi 类别打标
│   │   └── schema.py                    # method 数据结构 (pydantic/dataclass)
│   │
│   ├── inference/                       # Phase 3
│   │   ├── __init__.py
│   │   └── frozen_encode.py             # 新写: encode_methods()
│   │
│   ├── train/                           # Phase 4
│   │   ├── __init__.py
│   │   ├── contrastive_model.py         # encoder + projection head
│   │   ├── losses.py                    # InfoNCE / SupCon
│   │   ├── samplers.py                  # 正负例采样 (A/B/E)
│   │   ├── dataset.py                   # method-level Dataset
│   │   └── trainer.py
│   │
│   ├── fcg/                             # Phase 5
│   │   ├── __init__.py
│   │   ├── extract.py                   # Androguard 抽 FCG
│   │   └── align.py                     # method id 标准化 + 对齐
│   │
│   ├── gnn/                             # Phase 6
│   │   ├── __init__.py
│   │   ├── models.py                    # GAT / GraphSAGE
│   │   ├── losses.py                    # BGRL / DGI
│   │   ├── dataset.py                   # PyG Dataset
│   │   └── trainer.py
│   │
│   ├── cluster/                         # Phase 7
│   │   ├── __init__.py
│   │   ├── reduce.py                    # UMAP
│   │   ├── cluster.py                   # HDBSCAN / KMeans
│   │   └── eval.py                      # silhouette / NMI / ARI
│   │
│   └── utils/
│       ├── io.py                        # parquet/npy/pickle helpers
│       ├── logging.py
│       ├── sanity.py                    # 拷自 sanity_check_vocab.py
│       └── inspect.py                   # 拷自 inspect_checkpoint.py
│
├── third_party/
│   └── susi_categories.json             # SuSi API → 类别表
│
├── scripts/                             # 一键 entrypoint
│   ├── 00_download.py
│   ├── 01_disassemble.py
│   ├── 02_extract_methods.py
│   ├── 03_baseline_embeddings.py
│   ├── 04_train_contrastive.py
│   ├── 05_extract_fcg.py
│   ├── 06_train_gnn.py
│   ├── 07_cluster_eval.py
│   ├── setup_assets.sh                  # 一键从 DexBERT 拷 assets
│   └── run_all.sh
│
├── notebooks/
│   ├── 01_inspect_method_distribution.ipynb
│   ├── 02_visualize_clusters.ipynb
│   └── 03_ablation_summary.ipynb
│
├── tests/
│   ├── test_encoder_load.py             # sanity: vocab+pt 加载
│   ├── test_method_extractor.py
│   ├── test_susi_tagger.py
│   └── test_fcg_align.py
│
└── data/                                # gitignore
    ├── apks/                            # APK 文件
    ├── processed/                       # <sha>.txt
    ├── methods/                         # <sha>.parquet
    ├── embeddings/
    │   ├── baseline/
    │   └── finetuned/
    ├── fcg/                             # <sha>.gpickle
    └── checkpoints/
```

### `pyproject.toml` 起手依赖

```toml
[project]
name = "amc"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy",
    "pandas",
    "pyarrow",                # parquet
    "tqdm",
    "requests",
    "fire",                   # CLI
    "scikit-learn",
    "umap-learn",
    "hdbscan",
    "androguard>=4.0",        # FCG
    "networkx",
    "matplotlib",
    "pyyaml",
    "pydantic",
    # GNN 依赖等到 Phase 6 再加，会和 torch 版本绑定:
    # "torch-geometric",
    # "torch-scatter",
    # "torch-sparse",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "ipykernel"]
```

**注意**：tensorflow 一行都不加，是切干净 DexBERT 隐性依赖的关键。

### `.gitignore` 必含

```
assets/
data/
*.pt
*.apk
*.txt          # processed/ 里的, 注意 README 等正常 .md 不在 gitignore
*.parquet
*.npy
*.gpickle
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
```

---

## 7. 立即可启动的下一步

**2026-05-08**：仓库 `main` 已含 Phase 0–4 可运行代码与脚本，且 `Phase 4` 首轮全量 `dev200` 实验已完成（见 §5「新项目 new_androserum」与 §11 最新一条）。本节 **§7.1–7.4** 仍是早期历史步骤，仅作溯源；**接手开发请以 §11 最新一条为起点，当前实际下一步是 Phase 4 结果验收 / 对比分析，然后推进 Phase 5。**

### 任务 7.1（历史）：建立新项目骨架

按第 6 节的目录结构建仓库，并把 DexBERT 那 6 个核心文件 + 4 个工具脚本 + 3 个 asset 拷过来。

具体可执行步骤：

```bash
# 1. 创建项目根目录
cd /usr/local/python_projects/
mkdir android-method-clustering
cd android-method-clustering
git init

# 2. 创建目录骨架
mkdir -p src/amc/{encoder,data,inference,train,fcg,gnn,cluster,utils}
mkdir -p configs assets data/{apks,processed,methods,embeddings/{baseline,finetuned},fcg,checkpoints} \
         third_party scripts notebooks tests docs

# 3. 拷贝 encoder 核心 (3 文件 + 1 配置)
SRC=/usr/local/python_projects/DexBERT
cp $SRC/Model/models.py        src/amc/encoder/models.py
cp $SRC/Model/tokenization.py  src/amc/encoder/tokenization.py
cp $SRC/Model/utils.py         src/amc/encoder/utils.py
cp $SRC/Model/config/DexBERT/bert_base.json configs/encoder_base.json

# 4. 修复 import (encoder/models.py 里的 utils 改相对导入)
# 用 sed 或者手改: from utils import split_last, merge_last
#                  → from .utils import split_last, merge_last

# 5. 拷贝数据预处理 (2 文件 + 1 jar + 2 模型 asset)
cp $SRC/Data/disassemble.py            src/amc/data/disassemble.py
cp $SRC/Data/instruction_generator.py  src/amc/data/instruction_generator.py
cp $SRC/Data/baksmali-2.5.2.jar         assets/
cp $SRC/save_dir/DexBERT/vocab.txt      assets/
cp $SRC/save_dir/DexBERT/model_steps_604364.pt  assets/

# 6. 拷贝你已有的工具脚本
cp $SRC/androzoo_download_by_sha.py    src/amc/data/androzoo.py
cp $SRC/process_apk.py                  src/amc/data/apk_processor.py
cp $SRC/inspect_checkpoint.py           src/amc/utils/inspect.py
cp $SRC/sanity_check_vocab.py           src/amc/utils/sanity.py

# 7. 把本文档拷过来
cp $SRC/PROJECT_HANDOFF.md  ./PROJECT_HANDOFF.md
```

### 任务 7.2（紧接着）：写 `encoder/loader.py`

封装"加载 .pt + vocab + 实例化 transformer"为统一 API：

```python
# src/amc/encoder/loader.py 期望 API
def load_pretrained_encoder(
    cfg_path: str = "configs/encoder_base.json",
    weights_path: str = "assets/model_steps_604364.pt",
    vocab_path: str = "assets/vocab.txt",
    device: str = "cpu",
) -> tuple[Transformer, FullTokenizer]:
    """加载 DexBERT 预训练 encoder, 返回 (transformer, tokenizer)."""
    ...
```

之后 inference / train 都通过这一个入口加载，避免重复 load_state_dict 样板。

### 任务 7.3（验证）：跑通 sanity test

把 `src/amc/utils/sanity.py` 改造成一个 pytest 用例，能跑通就证明搬迁成功：

```bash
pytest tests/test_encoder_load.py -v
# 期望输出: vocab=9537, [UNK]=0%, missing=0, MLM forward OK
```

### 任务 7.4（进入 Phase 2）：写 `method_extractor.py` 和 `susi_tagger.py`

这是流水线第一个新代码。设计规格：

**输入**：`data/processed/<sha>.txt`（DexBERT 标准格式）

**输出**：`data/methods/<sha>.parquet`，每行字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| apk_sha | str | 来自文件名 |
| class_name | str | `Lcom/foo/Bar;` |
| method_sig | str | `m(II)V` |
| full_id | str | `Lcom/foo/Bar;->m(II)V`（FCG 节点身份） |
| instructions | list[str] | smali 指令行 |
| n_instr | int | 指令数 |
| api_calls | list[str] | 形如 `Lpkg/Cls;->m(...)` 的调用列表 |
| susi_cats | list[str] | 命中的 SuSi 类别集合 |
| susi_dominant_cat | str/None | 最频繁的类别（弱监督主标签） |
| filtered | bool | 是否被 trivial 过滤标记 |

**过滤规则**：

- `n_instr < 5` → `filtered=True`
- class 名含 `R$` 或 `BuildConfig` → `filtered=True`
- method 名以 `access$` / `lambda$` / `$values` / `<clinit>` 开头 → `filtered=True`

**API 抽取正则**：

```python
import re
API_RE = re.compile(r"invoke-(?:virtual|direct|static|interface|super)(?:/range)?\s+\{[^}]*\},\s+(L\S+;->\S+)")
```

**SuSi 类别表来源**：

- 论文：Rasthofer et al. *A Machine-learning Approach for Classifying and Categorizing Android Sources and Sinks*
- 代码：<https://github.com/secure-software-engineering/SuSi>
- 取 `sources_and_sinks.txt`，把 API 路径 → 类别构建一个 hashmap

### 任务 7.5（已完成）：Phase 4 — 对比学习微调

Phase 4 A+B 路线现已落地：

- 训练模块：`src/androserum/train/{dataset,samplers,losses,contrastive_model,trainer}.py`
- 入口：`scripts/04_train_contrastive.py` 与 `scripts/run_all.py --do_train=True`
- 本地小规模 sweep 结论：`batch_size=8`、`lr=3e-5`、`label_fraction=0.5` 为当前推荐默认值
- AutoDL 5090 首轮全量运行：`p4_dev200_run1`
  - checkpoint：`data/checkpoints/p4_dev200_run1/`
  - finetuned embeddings：`data/embeddings/finetuned/p4_dev200_run1/`（200 个文件）

### 任务 7.6（当前）：Phase 4 结果验收 / baseline 对齐

建议按以下顺序继续：

```bash
# 1. 对齐 baseline 与本轮 finetuned 的文件名集合（以 sha_dev_200.txt 为准）
# 注意 baseline/ 目录里多一个历史测试样本:
#   0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.npz

# 2. 读取本轮 summary
sed -n '1,220p' data/checkpoints/p4_dev200_run1/contrastive_ab_summary.json

# 3. 比较 frozen vs finetuned 的代理指标 / 近邻语义表现
# （聚类前可先做简单 sanity: 文件对齐、向量 shape、代表 method 近邻）

# 4. 在确认 Phase 4 输出可用后，启动 Phase 5:
#    FCG 抽取 + method-id 对齐
```

### 任务 7.7（历史备忘）

早期文档曾写「注释掉已跑通的 test 再进 Phase 3」— 现已由 pytest markers（`slow`）与模块化测试替代，**无需**注释测试文件。

---

## 8. 重要"避坑"提醒

### 8.1 conda 环境

直接复用现有的 `DexBert` 环境即可，**不需要新建**。conda env 是按名字全局共享的，跨项目通用。

```bash
conda activate DexBert
# 在新项目里继续装:
pip install androguard pandas pyarrow umap-learn hdbscan
```

### 8.2 method 签名对齐

Phase 5 的 FCG 节点身份必须与 Phase 2 的 `full_id` **字符级一致**。Androguard 用 `Lcom/foo/Bar;->m(II)V`，instruction_generator 写出来的是 `com/foo/Bar` + `<init>(...)V` —— **必须写一个 normalize 函数让两边格式 100% 一致**。这是 Phase 5 最容易踩的坑。

### 8.3 不要把 `.pt / .apk / processed/*.txt` 提交进 git

这些文件加起来几个 GB，提交进 git 会让仓库无法 clone。`.gitignore` 必须配齐。`assets/` 也要 gitignore，README 里写下载方法。

### 8.4 不要 import `pretrainDexBERT.py`

它会传染性 import `train.py → checkpoint.py → tensorflow`。如果需要它的 `BertAEModel4Pretrain`，就把那 25 行类定义直接复制到新文件（已在 `sanity_check_vocab.py` 里这么做了）。

### 8.5 batch sampler 要避免"伪 negative"

Phase 4 用 SuSi 类别做正例时：如果 batch 里同时有 3 个 SuSi 类别相同的 method，但只把其中一个当 anchor 的正例，其他两个会被当 negative，**这是错的**。

解决方法：
- 用 **SupCon (supervised contrastive)** 形式直接处理多正例
- 或采样时按 SuSi 类别 group sampling，保证 batch 内同类别数量平衡
- 或多正例 InfoNCE：分子 sum over all positives

### 8.6 Phase 4 训练时一定要冻底层

DexBERT encoder 已经预训练好了，对比微调时如果全开放训练，会**遗忘大量预训练知识**。建议冻结前 4 层 transformer，只微调后 4 层 + projection head。

---

## 9. 关键参考资料

### DexBERT 论文

- Sun, Tiezhu et al. *DexBERT: Effective, Task-Agnostic and Fine-grained Representation Learning of Android Bytecode*. IEEE Transactions on Software Engineering, 2023.
- 仓库：<https://github.com/TiezhuSun/DexBERT>
- 预训练权重 Drive 链接：<https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view>

### 对比学习 / 表示学习

- SimCSE: <https://arxiv.org/abs/2104.08821>
- SupCon: <https://arxiv.org/abs/2004.11362>

### 图自监督 / GNN

- BGRL: *Bootstrapped Representation Learning on Graphs* — <https://arxiv.org/abs/2102.06514>
- DGI: *Deep Graph Infomax* — <https://arxiv.org/abs/1809.10341>
- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io/>

### Android 静态分析

- Androguard: <https://github.com/androguard/androguard>
- baksmali: <https://github.com/JesusFreke/smali>（已 archived，但 jar 仍可用）
- SuSi (sensitive API categories): <https://github.com/secure-software-engineering/SuSi>
- LibScout (库识别, 后期可选): <https://github.com/reddr/LibScout>
- AndroZoo: <https://androzoo.uni.lu/>

### 聚类与降维

- UMAP: <https://umap-learn.readthedocs.io/>
- HDBSCAN: <https://hdbscan.readthedocs.io/>

---

## 10. 联系 / 元信息

- 当前工作目录（旧项目，仅作 reference / asset 来源）：`/usr/local/python_projects/DexBERT/`
- 新项目实际位置：`/usr/local/python_projects/new_androserum/`（**注意**：handoff 早期版本里写的 `android-method-clustering` 这个名字最终没用，仓库名跟 GitHub 一致叫 `new_androserum`，Python 包名叫 `androserum`）
- GitHub 远程仓库：<https://github.com/LeonBrianQin/new_androserum>
- Conda 环境名：`DexBert`
- Python 版本：3.10+（DexBert 环境）
- Java：11.0.27（已装）
- AndroZoo API key：环境变量 `ANDROZOO_APIKEY`（旧项目源码里曾硬编码过一份，建议视为已泄露并去 AndroZoo 重置）

---

## 11. 工作日志（status updates）

> 每次会话结束追加一条，让本文档永远反映最新进度。
> 任何 agent 接手时先读最新一条，立刻知道项目当前位置和下一步要做什么。

---

### 2026-05-06（会话 1）— 项目骨架 + DexBERT encoder 迁移 + 工程保险

**完成的工作**：

- ✅ 新建独立项目 `/usr/local/python_projects/new_androserum/`，关联到 GitHub <https://github.com/LeonBrianQin/new_androserum>，已推送 3 个 commits
- ✅ 项目骨架按 §6 设计搭好：`src/androserum/` 下 8 个子模块（encoder / data / inference / train / fcg / gnn / cluster / utils）+ `configs/` + `assets/` + `data/` + `tests/` 等
- ✅ Python 包名最终用 **`androserum`**（不是 §6 里写的 `amc`），对应目录 `src/androserum/`，`pyproject.toml` 里 `name = "androserum"`
- ✅ DexBERT encoder 完整迁移到 `src/androserum/encoder/`，切干净 tensorflow 依赖
  - 拷贝并按需修改的文件：`models.py`、`tokenization.py`、`utils.py`、`configs/encoder_base.json`
  - `models.py` 第 12 行 `from utils import` 已修为 `from .utils import`（相对导入）
- ✅ 数据预处理代码迁移到 `src/androserum/data/`：`disassemble.py`、`instruction_generator.py`、`androzoo.py`、`apk_processor.py`
- ✅ 工具脚本迁移到 `src/androserum/utils/`：`inspect.py`（.pt 检查）、`sanity.py`（vocab + ckpt 端到端校验）
- ✅ 3 个 asset 拷到 gitignored `assets/`：
  - `model_steps_604364.pt` 实际 **1.8 GB**（不是 §5 估的 330 MB，里面除 model state dict 还含 optimizer state 等训练副产物）
  - `vocab.txt`（9,537 tokens）
  - `baksmali-2.5.2.jar`
- ✅ 新写 `src/androserum/encoder/loader.py` 统一加载入口
  - 公共 API：`load_pretrained_encoder(device="cuda") -> (transformer, tokenizer, cfg)`
  - 自动从 `BertAEModel4Pretrain` 完整 state_dict 中抽取 `transformer.*` 部分载到 `models.Transformer`
  - `_project_root()` 解析路径，任意 cwd 调用都能找到 `assets/` 和 `configs/`
- ✅ `pyproject.toml`（`name = "androserum"`，license `Apache-2.0`，**不含 tensorflow**）+ `.gitignore`（用 `/assets/` `/data/` 开头斜杠避免误屏蔽 `src/androserum/data/`）+ `README.md`
- ✅ `pip install -e . --no-deps` 注册项目，conda DexBert 环境里任意目录可 import
- ✅ **工程保险**（§7.3）：`tests/test_encoder_load.py` 7 个 pytest 用例，11 秒全绿，注册 `slow` marker（`pytest -m 'not slow'` 可跳过）

**关键事实（已验证可工作）**：

- vocab = 9,537 tokens，5 个保留 token（[PAD]/[UNK]/[CLS]/[SEP]/[MASK]）都在
- cfg：`dim=768`, `n_layers=8`, `n_heads=8`, `max_len=512`
- transformer 参数 = **54,980,352**（~55M）
- forward `[CLS][SEP]+pad` 输入 → 输出 shape `(1, 512, 768)`，CLS L2 norm = 5.768
- pytest 11.16 秒全绿

**踩坑记录（避免重复）**：

1. `.gitignore` 写 `data/`（不带斜杠）会**递归匹配任何深度的 data 子目录**，误屏蔽了 `src/androserum/data/` 包，导致首次 commit 漏掉 5 个 .py 文件 → 改为 `/data/` `/assets/`（开头斜杠 = 只匹配项目根）+ `git commit --amend` 修复
2. `git push` 偶发 `gnutls_handshake() failed: TLS connection was non-properly terminated` → 网络抖动，重试 1–2 次即可
3. Cursor 编辑器右上角的 ▷ "Run Python File" 按钮默认开新 (base) 终端，**既不激活 conda 也不跑 pytest**，要跑 pytest 必须在已激活 `(DexBert)` 的终端里手敲 `pytest tests/ -v`

**Git 历史**（远程 `main` 分支）：

| Commit | 内容 |
|--------|------|
| `84166f2` | Initial commit（GitHub 创建仓库时自动给的 LICENSE）|
| `b306b26` | `feat: scaffold androserum + migrate DexBERT encoder + add loader` |
| `bfe774b` | `test: add encoder load sanity tests` |

**遗留小事**（非阻塞，下次顺手做）：

- ~~`README.md` 里 `.pt` 大小写的 "~330 MB"~~ — **已在 2026-05-07 修正为 ~1.8 GB**
- 5 个空目录（`docs/`、`notebooks/`、`scripts/`、`third_party/`、`tests/` 之外的）GitHub 上不显示（git 不追踪空目录）；写 Phase 2 第一个真实文件时自然激活

**下一步起点（任何 agent 续上时从这里开始）**：

➡️ **Phase 4：对比学习微调**（design 见 §2 决策 3、§8.5–8.6；数据依赖：`data/methods/*.parquet` + 可选 baseline `embeddings/baseline/*.npz` 作初始化）。

会话 1 下方原「Phase 2 子任务表」**已完成**，保留仅作记录。

---

### 2026-05-07（会话 2）— Phase 2 / 2b / 3 落地 + 脚本化 + 上云准备 + Git 同步

**完成的工作**：

- ✅ **Phase 2**：`src/androserum/data/schema.py`（`MethodRecord`，pydantic）；`method_extractor.py`（processed `.txt` → rows）；`method_parquet.py`（pyarrow 读写）；`class_name` 正则允许 Dalvik 路径中的 `-`（避免 Play Services 等校验失败）
- ✅ **Phase 2b**：`susi_index.py`（SuSi develop 分支 Android 4.2 列表；行末 ` -> CATEGORY`，签名为 Soot→Dalvik 规范化）；`susi_tagger.py`（`susi_cats` / `susi_dominant_cat`）；`third_party/susi/Ouput_Cat{Sources,Sinks}_v0_9.txt` 入库（弱网友好）
- ✅ **Phase 3**：`src/androserum/inference/frozen_encode.py`（冻结 DexBERT → `[CLS]`，写出 `data/embeddings/baseline/<SHA>.npz`：`full_id` + `embedding`）；`inference/__init__.py` 导出
- ✅ **入口脚本**（`main()` + `fire`，无参可 IDE 直跑）：`scripts/00_download_apks.py` … `03_encode_methods.py`；`scripts/run_all.py` 串联 Phase 0–3；`configs/sha_dev_200.txt` 开发集
- ✅ **工程**：`apk_processor.py` 去掉旧 `sys.path`，`baksmali` 路径相对项目根；`data/__init__.py` 导出公共 API；`.gitignore` 增加 `*.npz`、`/logs/`、`.env.*`（保留 `.env.example`）
- ✅ **安全**：`androzoo.py` 与 `00_download_apks.py` **不再**含默认 API key；`.env.example` 文档化 `ANDROZOO_APIKEY`（**务必在 AndroZoo 轮换 key**：历史 `b306b26` 曾泄露）
- ✅ **上云**：`scripts/cloud_setup.sh`（Java 11 + conda + editable install + GPU 自检）；README「Cloud GPU Quick Start」+ DexBERT ckpt 体积更正为 ~1.8 GB
- ✅ **测试**：`tests/test_schema.py`、`test_method_extractor.py`、`test_method_parquet.py`、`test_susi_index.py`
- ✅ **远程**：`main` 推送至 GitHub；本机 `origin` 建议 SSH（`git@github.com:LeonBrianQin/new_androserum.git`）

**运行期观察（dev 200，供接续时对照）**：

- 本机会话末：`run_all.py` 可处于 Phase 3（`batch_size=64` 时约数小时级）；各步 `skip_existing`，日志与 pid 通常在 `logs/`（gitignored）
- Phase 3 瓶颈常含 **CPU 侧单线程 tokenize**；上 4090 需配合多进程 DataLoader 等才易吃满算力

**踩坑（新增）**：

1. 部分环境里 `git commit` 被注入不兼容参数 → 使用 **`/usr/bin/git commit`**
2. VS Code/Cursor 调试器附加长任务 → 进程易睡死在 futex；长任务用终端 `nohup` 或普通运行

**Git（`main` 新近线）**：

| Commit | 内容 |
|--------|------|
| `9c2c20f` | chore(security): AndroZoo key 清理 + cloud bootstrap |
| `edd3788` | feat(data): Phase 2 method extraction + parquet schema |
| `86e1861` | feat(data): Phase 2b SuSi tagger + cached lists |
| `4675363` | feat(inference): Phase 3 frozen encoder |
| `a6f325d` | feat(scripts): Phase 0–3 runners + sha_dev_200 |

**下一步起点**：

➡️ **Phase 4**：数据集（多 parquet 合并或 Iterable）、SupCon/InfoNCE、SimCSE dropout 双视图、SuSi 类别多正例、冻结前 N 层（§8.6）、训练脚本与 checkpoint 写出路径；评估可与 Phase 7 预留接口对齐。

---

### 2026-05-08（会话 3）— Phase 4 A+B 落地 + 小规模调参 + AutoDL 5090 全量 dev200 跑通

**完成的工作**：

- ✅ **Phase 4 训练模块落地**：`src/androserum/train/`
  - `dataset.py`：从 `methods/*.parquet` 构建 sampled in-memory 训练池；保留全部可用 SuSi 标注样本，并对无标签样本做有界抽样，避免内存爆炸
  - `samplers.py`：按 SuSi 类别分组采样，保证 batch 内有稳定的 `B` 正例
  - `losses.py`：A+B 多正例对比损失（A = SimCSE dropout 双视图；B = 同 `susi_dominant_cat`）
  - `contrastive_model.py`：DexBERT encoder + `Linear(768→256)` projection head；支持冻结前 4 层
  - `trainer.py`：训练循环、checkpoint、summary、导出 finetuned embeddings
- ✅ **Phase 4 入口**：`scripts/04_train_contrastive.py`；`scripts/run_all.py` 已接入 `do_train` 与一整套 `phase4_*` 参数
- ✅ **测试**：新增 `tests/test_phase4_ab.py`（6 passed）
- ✅ **默认配置**：新增 `configs/train_contrastive_ab.yaml`
- ✅ **上云辅助脚本**：
  - `scripts/cloud_phase4_push.sh`：本机推 `assets/` + `data/methods/`
  - `scripts/cloud_phase4_run.sh`：云端只跑 Phase 4 并导出 embeddings
  - `scripts/cloud_phase4_pull.sh`：把 checkpoint + finetuned embeddings 拉回本地
- ✅ **README**：补充 “Cheapest Cloud Path: Phase 4 Only”

**小规模调参结论（10 APK）**：

- batch 尺度：
  - `batch_size=8` 稳定
  - `batch_size=16` 在本机 RTX 4060 Laptop 8GB 上显存几乎打满，速度明显变慢，不作为默认值
- `label_fraction`：
  - `0.5` 优于 `0.75`（后者 `B` 信号更密，但本轮 mean loss 更高）
- learning rate：
  - `1e-5` → `mean_loss = 2.4170`
  - `2e-5` → `mean_loss = 2.2640`
  - `3e-5` → `mean_loss = 2.1783`
- 因此当前推荐默认值：
  - `batch_size=8`
  - `lr=3e-5`
  - `label_fraction=0.5`
  - `epochs=3`
  - `steps_per_epoch=1000`

**AutoDL 5090 全量运行（`p4_dev200_run1`）**：

- 运行方式：只上传 `assets/`（~1.8GB）和 `data/methods/`（~170MB），不重跑 Phase 0–3
- 结果已完整拉回本地：
  - `data/checkpoints/p4_dev200_run1/contrastive_ab_best.pt`
  - `data/checkpoints/p4_dev200_run1/contrastive_ab_last.pt`
  - `data/checkpoints/p4_dev200_run1/contrastive_ab_summary.json`
  - `data/embeddings/finetuned/p4_dev200_run1/` 共 **200** 个 `.npz`
- `summary` 关键值：
  - `best_mean_loss = 0.8541`
  - epoch history：`1.1348 → 0.8948 → 0.8541`
  - `samples_total = 25035`
  - `samples_labeled = 10717`
  - `susi_labels_usable = 17`

**踩坑（新增）**：

1. 若云端 `git clone` 访问 GitHub 超时，可直接用 `rsync` 同步代码，绕过 GitHub
2. `rsync --exclude 'data/'` 会误伤 `src/androserum/data/`，必须写成 **`--exclude '/data/'`**
3. `cloud_phase4_pull.sh` 中途断网后，直接重跑同一条命令即可，`rsync` 会跳过已完整拉回的文件
4. `Pseudo-terminal will not be allocated because stdin is not a terminal.` 是 SSH here-doc 常见提示，**不是错误**

**Git（`main` 新近线）**：

| Commit | 内容 |
|--------|------|
| `00ad80e` | `feat(train): add Phase 4 contrastive training and cloud helpers` |

**下一步起点**：

➡️ **先做 Phase 4 结果验收 / 对齐分析**：以 `sha_dev_200.txt` 对齐 `baseline` 与 `finetuned/p4_dev200_run1` 文件集合，确认 frozen vs finetuned 的比较口径，再开始评估代理指标与近邻语义。  
➡️ **随后进入 Phase 5**：FCG 抽取 + method-id 对齐。

---

**文档版本**：v1.3
**最后更新**：2026-05-08（会话 3：Phase 4 代码、调参与 AutoDL dev200 结果同步）
**下次更新触发条件**：Phase 4 结果分析完成 / Phase 5 首跑通 / 重大决策变更
