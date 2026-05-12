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

### 新项目 `new_androserum`（`/usr/local/python_projects/new_androserum/`，截至 2026-05-09）

> **注意**：下文才是当前仓库的真实进度；上面 §5 的 DexBERT 目录仍作 asset / 格式参考。

**远程与 Git**：

- GitHub：<https://github.com/LeonBrianQin/new_androserum>
- 推荐 `origin` 使用 **SSH**：`git@github.com:LeonBrianQin/new_androserum.git`（HTTPS push 在无凭证环境下会失败）。
- 当前关键 commits（自旧到新）：
  - `00ad80e` — Phase 4：A+B 对比学习训练模块与云端脚本
  - `43b6d8f` — docs：Phase 4 dev200 完成
  - `295507c` — docs：记录 A+B 的 `5k/10k/50k` 对比
  - `963a7a3` — Phase 4：信号 `E`（override）接线
  - `5fb5c93` — docs：同步 A+B+E 三路对比命名与结果
  - `d2583c9` — Phase 4：信号 `C`（library key）支持与 `A+B+C+E` 训练链路
  - `ded4d78` — Phase 5：FCG 抽取 / method-id 对齐 scaffold

**已实现流水线（Phase 0 → 5，Phase 5 已完成首跑）**：

| Phase | 入口 / 核心模块 | 产出（均在 `data/`，gitignored） |
|------|------------------|----------------------------------|
| 0 | `scripts/00_download_apks.py` → `androserum.data.androzoo` | `data/apks/<SHA>.apk` |
| 1 | `scripts/01_disassemble_apks.py` → `apk_processor` | `data/processed/<SHA>.txt` |
| 2 | `scripts/02_extract_methods.py` | `data/methods/<SHA>.parquet` |
| 2b | `scripts/02b_tag_susi.py` | Parquet 内 `susi_cats` / `susi_dominant_cat` |
| 2c | `scripts/02c_extract_overrides.py` → `override_index` | `data/overrides/<SHA>.parquet` |
| 2d | `scripts/02d_extract_library_keys.py` | `data/library_keys/<SHA>.parquet` |
| 3 | `scripts/03_encode_methods.py` → `frozen_encode` | `data/embeddings/baseline/<SHA>.npz`（`full_id` + 768-d `embedding`） |
| 4 | `scripts/04_train_contrastive.py` → `src/androserum/train/` | `data/checkpoints/<run>/contrastive_ab_{best,last,summary}.pt/json` + `data/embeddings/finetuned/<run>/<SHA>.npz` |
| 5 | `scripts/05_extract_fcg.py` → `src/androserum/fcg/` | `data/fcg/<SHA>.{aligned_nodes,internal_edges,boundary_edges,summary}` |
| 串跑 | `scripts/run_all.py`（`fire`，每步可 `do_*=false` / `do_train=true` / `do_fcg=true` 控制） | 上述全集 |

**开发集**：`configs/sha_dev_200.txt`（200 个 SHA，来自 DexBERT 预训练列表子集）。

**当前已有结果**：

- `data/embeddings/baseline/` 现有 `201` 个 `.npz`：其中 **200 个**对应 `sha_dev_200.txt`，另有一个历史测试样本 `0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.npz`
- `data/embeddings/finetuned/p4_dev200_run1/` 已完整导出 **200** 个 `.npz`，与 `sha_dev_200.txt` 一一对应
- `data/embeddings/finetuned/p4_dev200_abe_run1/` 已完整导出 **200** 个 `.npz`
- `data/embeddings/finetuned/p4_dev200_abce_run1/` 已完整导出 **200** 个 `.npz`
- `data/checkpoints/p4_dev200_run1/contrastive_ab_summary.json` 显示首轮全量 `dev200` 训练结果：
  - `best_mean_loss = 0.8541`
  - 3 个 epoch：`1.1348 → 0.8948 → 0.8541`
  - `exported_npz_files = 200`
- `data/checkpoints/p4_dev200_abe_run1/contrastive_ab_summary.json`：
  - `best_mean_loss = 1.0583`
  - `override_keys_total = 3148`
  - `override_keys_usable = 578`
- `data/checkpoints/p4_dev200_abce_run1/contrastive_ab_summary.json`：
  - `best_mean_loss = 1.1836`
  - `library_keys_total = 9274`
  - `library_keys_usable = 570`
  - `override_keys_total = 3148`
  - `override_keys_usable = 578`
- `data/reports/` 现有五路同口径对比（`50k` 为主）：
  - `frozen_vs_ab_dev200_50k.{png,json}`：frozen DexBERT baseline vs A+B
  - `frozen_vs_abe_dev200_50k.{png,json}`：frozen DexBERT baseline vs A+B+E
  - `ab_vs_abe_dev200_50k.{png,json}`：A+B vs A+B+E
  - `frozen_vs_abce_dev200_50k.{png,json}`：frozen DexBERT baseline vs A+B+C+E
  - `ab_vs_abce_dev200_50k.{png,json}`：A+B vs A+B+C+E
  - `abe_vs_abce_dev200_50k.{png,json}`：A+B+E vs A+B+C+E
- 当前 `50k` 对比结果（以 SuSi 弱标签为代理）：
  - frozen baseline：`NMI = 0.1301`，`ARI = -0.0151`，`silhouette_non_noise = 0.3233`
  - A+B：`NMI = 0.6062`，`ARI = 0.2626`，`silhouette_non_noise = 0.6264`
  - A+B+E：`NMI = 0.7137`，`ARI = 0.6760`，`silhouette_non_noise = 0.6847`
  - A+B+C+E：`NMI = 0.7054`，`ARI = 0.7185`，`silhouette_non_noise = 0.6104`
  - 结论：
    - A+B 相比 frozen baseline 已显著提升功能弱标签一致性
    - A+B+E 又进一步显著优于 A+B，说明 `E`（同 override）对功能性区分有增益
    - A+B+C+E 相比 A+B 仍明显更强，但相对 A+B+E 呈现**混合效应**：
      - `ARI` 继续上升（更多 pairwise 一致性）
      - `NMI` 小幅下降、`silhouette` 明显下降（全局簇几何更松）
    - 对“单 APK 内功能性相似方法聚类”这个原始目标而言，**A+B+E 目前是更均衡的 Phase 4 主线**；`A+B+C+E` 保留为实验支线
- `data/overrides/` 已为 `sha_dev_200.txt` 生成 **200** 个 sidecar parquet（`data/overrides/<SHA>.parquet`）
  - 字段：`apk_sha`, `full_id`, `override_keys`
  - 当前规则已排除 `<init>`、`<clinit>`、`private`、`static` 方法，避免把明显噪声带入 `E`
- `data/library_keys/` 当前有 **196** 个 sidecar parquet（4 个空 schema APK 无 sidecar）
  - 当前本地数据的 key 形态为 `LIBSCOUT::...`
  - 训练池中 `library_keys_usable = 570`，多为极小组（中位数 `2`）
- `data/fcg/` 已完成 `dev200` 首轮本地 CPU 抽取：
  - 共 `200` 个 `<SHA>.summary.json`
  - 共 `800` 个 Phase 5 sidecar 文件（每 APK 4 个）
  - 所有 summary 均满足：`missing_graph_nodes_count = 0`
  - 所有非空 APK 均满足：`aligned_graph_present = methods_rows`
  - `methods_rows = 0` 的 APK 共 `4` 个，与此前 Phase 4 中的 empty-schema APK 一致
  - `extra_internal_graph_nodes_count` 普遍较大，说明 Androguard 看到的 APK 内部方法口径**宽于** Phase 2/3 的 methods/embeddings 口径；当前 Phase 5 采用“只保留有 embedding 的对齐节点 + 记录 boundary edges”的策略

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

**测试**：`tests/test_schema.py`、`test_method_extractor.py`、`test_method_parquet.py`、`test_susi_index.py`、`tests/test_phase4_ab.py`、`tests/test_fcg_extract.py`；全量含大 ckpt：`pytest tests/test_encoder_load.py`。

**进行中 / 下一窗口**：

- ✅ Phase 4（A+B / A+B+E / A+B+C+E）训练与导出链路已全部打通
- ✅ baseline vs AB / ABE / ABCE 的 `50k` 同口径代理评估已完成
- ✅ Phase 5 FCG 抽取 / method-id 对齐已在 `dev200` 上完成首轮本地 CPU 运行
- 当前推荐主线：
  - **Phase 4 内容 encoder：A+B+E**
  - **下一工程块：Phase 6（FCG + GNN / BGRL）**
- Phase 6–7：尚未启动。

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

**2026-05-09**：仓库 `main` 已含 Phase 0–5 的可运行代码；`Phase 4` 的 `AB / ABE / ABCE` 实验与同口径代理评估已完成，`Phase 5` 的 `dev200` 首轮本地 CPU 抽取 / 对齐也已完成并通过验收（见 §5「新项目 new_androserum」与 §11 最新一条）。本节 **§7.1–7.4** 仍是早期历史步骤，仅作溯源；**接手开发请以 §11 最新一条为起点，当前实际下一步是开始 Phase 6（FCG + GNN / BGRL）。**

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

### 任务 7.6（已完成）：Phase 4 结果验收 / ABCE 复核

已完成的验收内容：

```bash
# 1. 对齐 baseline / A+B / A+B+E / A+B+C+E 的文件名集合

# 2. 读取各轮 summary
sed -n '1,220p' data/checkpoints/p4_dev200_run1/contrastive_ab_summary.json
sed -n '1,220p' data/checkpoints/p4_dev200_abe_run1/contrastive_ab_summary.json
sed -n '1,220p' data/checkpoints/p4_dev200_abce_run1/contrastive_ab_summary.json

# 3. 同口径 compare（50k）
python scripts/05_compare_baseline_finetuned.py \
  --baseline_dir=data/embeddings/finetuned/p4_dev200_abe_run1 \
  --finetuned_dir=data/embeddings/finetuned/p4_dev200_abce_run1 \
  --output_stem=abe_vs_abce_dev200_50k --sample_size=50000

# 4. 结论
#    当前内容 encoder 主线仍以 A+B+E 为准，ABCE 保留实验支线
```

### 任务 7.7（当前）：Phase 5 — FCG 抽取 + method-id 对齐

当前已完成的工程准备：

- 新增 `src/androserum/fcg/extract.py`
  - `AnalyzeAPK(...) -> dx.get_call_graph(...)`
  - Androguard method object → 项目 `full_id` 标准化
  - 将内部方法节点**严格对齐回** `methods/<SHA>.parquet` 行序
- 新增 `scripts/05_extract_fcg.py`
  - 输出：
    - `data/fcg/<SHA>.aligned_nodes.parquet`
    - `data/fcg/<SHA>.internal_edges.parquet`
    - `data/fcg/<SHA>.boundary_edges.parquet`
    - `data/fcg/<SHA>.summary.json`
- `scripts/run_all.py` 已接入 `do_fcg`
- 新增 `tests/test_fcg_extract.py`（toy graph 对齐单测已通过）

接手时建议按以下顺序继续：

```bash
# 1. 先小规模 dry-run（例如 limit=3）
python scripts/05_extract_fcg.py --limit=3

# 2. 检查 summary
sed -n '1,220p' data/fcg/<SHA>.summary.json

# 3. 核对对齐质量
#    - missing_graph_nodes_count 是否接近 0
#    - extra_internal_graph_nodes_count 是否接近 0
#    - aligned_nodes 行数是否等于 methods parquet / embedding 行数

# 4. 确认无结构性问题后，再跑 dev200 全量
python scripts/05_extract_fcg.py --sha_file=configs/sha_dev_200.txt
```

### 任务 7.8（历史备忘）

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
- ✅ **A+B 有效性对比**：新增 `scripts/05_compare_baseline_finetuned.py`
  - 处理了 `baseline/` 多一个历史测试样本的问题（自动忽略）
  - 处理了 4 个空 schema parquet（自动跳过）
  - 产出了 `5k / 10k / 50k` 三版 baseline vs finetuned 对比图和 JSON 指标

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

**A+B 有效性验证（baseline vs finetuned）**：

- `5k` 快速对比：
  - baseline：`NMI = 0.2244`，`ARI = 0.0494`
  - finetuned：`NMI = 0.5115`，`ARI = 0.3076`
- `10k` 更稳对比：
  - baseline：`NMI = 0.1693`，`ARI = 0.0177`
  - finetuned：`NMI = 0.6176`，`ARI = 0.3485`
- `50k` 同口径主对比：
  - baseline：`NMI = 0.1301`，`ARI = -0.0151`，`silhouette_non_noise = 0.3233`
  - finetuned：`NMI = 0.6062`，`ARI = 0.2626`，`silhouette_non_noise = 0.6264`
- 解释：
  - 虽然 baseline 在部分小样本图上可能显得“更几何规整”，但 A+B finetuned 在 `NMI/ARI` 上对 SuSi 功能弱标签的一致性提升非常明显
  - 因此当前可以把 **A+B** 视为已建立的有效 baseline，后续应在此基础上尝试 **A+B+E**

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

➡️ **当前应把 `A+B+E` 结果拉回并固定命名后，基于 `frozen vs A+B vs A+B+E` 三路结果继续做分析 / 写作。**  
➡️ **如果后续扩大到更大规模（例如 1 万 APK），再考虑更激进的 5090 吞吐优化（更大 batch、更高 num_workers、token 缓存）。**

---

### 2026-05-08（会话 4）— 信号 E 落地（override sidecar + A+B+E 接线）与本地验证

**完成的工作**：

- ✅ 新增 `src/androserum/data/override_index.py`
  - 用 `androguard` 从 APK class hierarchy 中抽取 `override_keys`
  - 输出 sidecar 记录：`apk_sha`, `full_id`, `override_keys`
- ✅ 新增 `scripts/02c_extract_overrides.py`
  - 输入：`data/apks/<SHA>.apk`
  - 输出：`data/overrides/<SHA>.parquet`
- ✅ `run_all.py` 接入 `do_overrides`
  - 可用 `--do_overrides=True` 批量生成 overrides sidecar
- ✅ `dataset.py` 接入 `override_keys`
  - `MethodTextSample` / `ContrastiveBatch` 新增 `override_keys`
  - `ContrastiveMethodDataset.from_methods_dir(...)` 可读取 `overrides_dir`
- ✅ `losses.py` 扩展到 A+B+E
  - `build_abe_positive_mask(...)`
  - `count_e_positive_pairs(...)`
  - `abe_contrastive_loss(...)`
- ✅ `samplers.py` 新增 `PositiveGroupBatchSampler`
  - 可同时从 SuSi label group 与 override group 采样正例
- ✅ `trainer.py` 接入 `use_signal_e`
  - 新增配置：`overrides_dir`, `use_signal_e`
  - 训练日志新增 `e_pairs=...`
- ✅ `scripts/cloud_phase4_push.sh` 更新
  - 若存在 `data/overrides/`，会一起同步上云

**override 定义细化**：

- 当前方法必须满足以下条件才参与 `E`：
  - 不是 `<init>`
  - 不是 `<clinit>`
  - 不是 `private`
  - 不是 `static`
- `override_key` 形如：
  - `Ljava/util/concurrent/Executor;->execute(Ljava/lang/Runnable;)V`
  - `Landroid/app/Application;->onCreate()V`

**本地验证（未上云）**：

- ✅ `tests/test_phase4_ab.py` 已扩展到 `A+B+E`（10 passed）
- ✅ smoke test：
  - `limit=3`, `epochs=1`, `steps_per_epoch=3`, `batch_size=4`, `use_signal_e=True`
  - 成功看到 `e_pairs > 0`
- ✅ 随机小规模干跑：
  - `configs/sha_dev_10_random.txt`
  - `limit=10`, `epochs=1`, `steps_per_epoch=20`, `batch_size=8`, `use_signal_e=True`
  - 成功跑完整个 epoch，无结构性错误
- ✅ `data/overrides/` 全量抽取已完成：
  - 共 `200` 个 parquet

**当前判断**：

- `E` 的数据层 / 训练层都已打通
- 本地 smoke test 与随机小规模干跑都通过
- 可以进入 AutoDL 200 样本全量 `A+B+E` 训练阶段

---

### 2026-05-08（会话 5）— AutoDL 上 A+B+E 全量训练完成 + 三路结果对比

**完成的工作**：

- ✅ AutoDL 5090 上完成 `p4_dev200_abe_run1`
  - 配置：`batch_size=8`, `lr=3e-5`, `label_fraction=0.5`, `use_signal_e=true`
  - `data/checkpoints/p4_dev200_abe_run1/contrastive_ab_summary.json`
  - `data/embeddings/finetuned/p4_dev200_abe_run1/` 共 `200` 个 `.npz`
- ✅ 本地拉回 A+B+E 全量结果
- ✅ 完成三路 `50k` 同口径对比：
  - `frozen_vs_ab_dev200_50k`
  - `frozen_vs_abe_dev200_50k`
  - `ab_vs_abe_dev200_50k`

**A+B+E 训练摘要**：

- `best_mean_loss = 1.0583`
- epoch history：`1.2924 → 1.0922 → 1.0583`
- `override_keys_total = 3148`
- `override_keys_usable = 578`
- `exported_npz_files = 200`

**三路对比结论（50k）**：

- frozen baseline vs A+B：
  - frozen：`NMI = 0.1301`, `ARI = -0.0151`, `silhouette = 0.3233`
  - A+B：`NMI = 0.6062`, `ARI = 0.2626`, `silhouette = 0.6264`
- frozen baseline vs A+B+E：
  - frozen：`NMI = 0.1301`, `ARI = -0.0151`, `silhouette = 0.3233`
  - A+B+E：`NMI = 0.7137`, `ARI = 0.6760`, `silhouette = 0.6847`
- A+B vs A+B+E：
  - A+B：`NMI = 0.6062`, `ARI = 0.2626`, `silhouette = 0.6264`
  - A+B+E：`NMI = 0.7137`, `ARI = 0.6760`, `silhouette = 0.6847`

**解释**：

- A+B 相比 frozen baseline 已经明显增强功能性区分
- 再加上 E（同 override）后，`NMI/ARI/silhouette` 都进一步提升
- 当前可以初步认定：`E` 提供的“框架角色/职责入口”弱监督，对功能语义表示学习是有增益的

**下一步起点**：

➡️ **围绕这三路结果做更系统的分析 / 记录 / 写作**（例如固定图名、整理实验表格、补充近邻案例）。  
➡️ **如果后续目标转向更大规模（1 万 APK），再设计 5090 的吞吐优化版本。**

---

### 2026-05-09（会话 6）— ABCE 结果回填验收 + Phase 5 scaffold 落地（未执行）

**完成的工作**：

- ✅ 解包并回填 `p4_dev200_abce_run1`
  - `data/checkpoints/p4_dev200_abce_run1/contrastive_ab_{best,last,summary}.pt/json`
  - `data/embeddings/finetuned/p4_dev200_abce_run1/` 共 `200` 个 `.npz`
- ✅ 将 compare 脚本泛化为可比较任意两组 embedding：
  - `scripts/05_compare_baseline_finetuned.py`
  - 支持自定义 `baseline_label` / `finetuned_label`
  - 自动记录 extra files ignored
- ✅ 完成 `ABCE` 的三组 `50k` 同口径对比：
  - `frozen_vs_abce_dev200_50k`
  - `ab_vs_abce_dev200_50k`
  - `abe_vs_abce_dev200_50k`
- ✅ 新增 `Phase 5` 工程 scaffold（**代码已写，尚未对真实 APK 执行**）：
  - `src/androserum/fcg/extract.py`
  - `scripts/05_extract_fcg.py`
  - `scripts/run_all.py --do_fcg=true`
  - `tests/test_fcg_extract.py`
- ✅ 轻量验证通过：
  - `pytest tests/test_fcg_extract.py`
  - `pytest tests/test_phase4_ab.py tests/test_schema.py tests/test_method_parquet.py tests/test_method_extractor.py tests/test_susi_index.py`

**ABCE 结果结论**：

- frozen baseline vs ABCE：
  - frozen：`NMI = 0.1301`, `ARI = -0.0151`, `silhouette = 0.3233`
  - ABCE：`NMI = 0.7054`, `ARI = 0.7185`, `silhouette = 0.6104`
- A+B vs ABCE：
  - A+B：`NMI = 0.6062`, `ARI = 0.2626`, `silhouette = 0.6264`
  - ABCE：`NMI = 0.7054`, `ARI = 0.7185`, `silhouette = 0.6104`
- A+B+E vs ABCE：
  - A+B+E：`NMI = 0.7137`, `ARI = 0.6760`, `silhouette = 0.6847`
  - ABCE：`NMI = 0.7054`, `ARI = 0.7185`, `silhouette = 0.6104`

**解释 / 当前策略**：

- `C`（library key）继续提高了 `ARI`，说明 pairwise 一致性更强
- 但相对 `A+B+E`，`NMI` 小降且 `silhouette` 明显下降，说明全局簇几何更松
- 结合原始目标“**单 APK 内功能性相似方法聚类**”，当前更推荐：
  - **Phase 4 主线：A+B+E**
  - **Phase 5 / 6 下一步：在 A+B+E embedding 之上接 FCG / GNN**
  - `ABCE` 保留为实验支线，而不是默认替代 `ABE`

**下一步起点**：

➡️ **首次执行 Phase 5（先 `limit=3`，再 `dev200`）并验收 `full_id` 对齐质量。**  
➡️ **若 Phase 5 对齐稳定，再进入 Phase 6：BGRL / GAT 的 FCG 上下文增强。**

---

### 2026-05-09（会话 7）— Phase 5 dev200 本地 CPU 首跑完成 + 对齐验收通过

**完成的工作**：

- ✅ 先跑 `Phase 5 limit=3` smoke test
  - 初次发现 `missing_graph_nodes_count > 0`
  - 原因定位为：Androguard `descriptor` 字符串里带空格，导致 `full_id` 假 mismatch
- ✅ 修正 `src/androserum/fcg/extract.py::method_to_full_id(...)`
  - 统一去掉 descriptor 中的 ASCII 空格
- ✅ 新增回归测试：
  - `tests/test_fcg_extract.py::test_method_to_full_id_strips_androguard_descriptor_spaces`
- ✅ 重跑 `Phase 5 limit=3`
  - `missing_graph_nodes_count = 0`
  - 对齐 smoke test 通过
- ✅ 本地 CPU 完成 `dev200` 全量 `Phase 5`
  - 命令：`python scripts/05_extract_fcg.py --sha_file=configs/sha_dev_200.txt`
  - 耗时：约 `34m45s`
  - 结果：`requested_shas = 200`, `written_bundles = 197`, `skipped_existing = 3`
  - 连同先前 `limit=3` 的结果，`data/fcg/` 现已覆盖完整 `dev200`

**验收结论**：

- `200` 个 APK 的 `summary.json` 已全部生成
- 所有 summary 都满足：
  - `missing_graph_nodes_count = 0`
  - `aligned_graph_present = methods_rows`
- `methods_rows = 0` 的 APK 共 `4` 个：
  - `DF18596BBEDD9C699DCF728F2ECFCE46C30049D72B832A7C00C2594096A71390`
  - `F3A942078517D3646E0AD3E036B9FCDEC3EB4DB88998861E53FB55A05B16483A`
  - `3A605CB4FA25EFFFE0C6B00B439B13074330BFDD46035C0AC81872EFCBC96E65`
  - `3A2E881B1C430CD5D7E642393BC3237B8801A4C2AFF1EA277348CD7B9F61C5B1`
- 这 `4` 个正是此前 Phase 4 中已知的 empty-schema APK，不是新的失败样本
- `extra_internal_graph_nodes_count` 普遍较大（中位数约 `23705`）
  - 这说明 Androguard 图包含的 APK 内部方法口径，显著宽于 Phase 2/3 已有 methods/embeddings 的口径
  - 当前 Phase 5 设计已把这类“图里有、embedding 里没有”的节点转为 boundary context，不阻塞后续 Phase 6

**当前判断**：

- `Phase 5` 已经可以视为**跑通且验收通过**
- 对“把 `A+B+E` 向量接上 FCG，再送入 GNN”这件事，当前主要 blocker 已从 **数据/对齐** 转移到 **Phase 6 代码本身尚未实现**

**下一步起点**：

➡️ **开始实现 Phase 6：A+B+E node features + Phase 5 aligned FCG + BGRL / GraphSAGE。**

---

### 2026-05-11（会话 8）— Phase 6 首轮 GraphSAGE+BGRL 跑通 + 三路对比完成

**完成的工作**：

- ✅ 新增 `Phase 6` 训练入口与模块：
  - `scripts/06_train_gnn.py`
  - `src/androserum/gnn/augment.py`
  - `src/androserum/gnn/bgrl.py`
  - `src/androserum/gnn/dataset.py`
  - `src/androserum/gnn/deps.py`
  - `src/androserum/gnn/models.py`
  - `src/androserum/gnn/trainer.py`
  - `tests/test_gnn_dataset.py`
- ✅ `scripts/run_all.py` 已接入 `do_gnn` 与 `phase6_cfg_path`
- ✅ `pyproject.toml` 新增可选依赖组 `gnn`，并引导使用 `scripts/setup_gnn_env.sh`
- ✅ `Phase 6` 当前实现为 **GraphSAGE + BGRL**
  - 不是 handoff 早期设计草案中的 `GAT`
  - 支持 BGRL online / target encoder、EMA、双视图增强、训练后导出 `gnn_embeddings`

**当前支持的三种图设定**：

1. `internal_only`
   - 只使用对齐后的内部 method 节点与内部边
2. `relay`
   - 保留 boundary relay nodes
   - `external_prior_mode = none`
3. `relay + package prior`
   - 保留 boundary relay nodes
   - 外部节点加入 package-family prior（当前 `external_family_vocab = 359`）

对应配置文件：

- `configs/train_gnn_bgrl.yaml`
- `configs/train_gnn_bgrl_relay.yaml`
- `configs/train_gnn_bgrl_relay_package.yaml`

**训练数据与覆盖范围**：

- 输入 embedding：`data/embeddings/finetuned/p4_dev200_abe_run1`
- 图输入：`data/fcg/`
- `dev200` 中有效图共 `196`
- 跳过 `zero_method_graphs = 4`
- `missing_graph_or_embedding_inputs = 0`
- `methods_rows_total = 739405`
- `methods_rows_median = 1088`
- `extra_internal_nodes_median = 23877`

**训练结果摘要**：

- `gnn_bgrl_internal_only`
  - `best_mean_loss = 0.1228`
  - `exported_npz_files = 196`
- `gnn_bgrl_relay`
  - `best_mean_loss = 0.0895`
  - `exported_npz_files = 196`
- `gnn_bgrl_relay_package`
  - `best_mean_loss = 0.0991`
  - `exported_npz_files = 196`

**已产出的 Phase 6 embedding 与 checkpoint**：

- `data/checkpoints/gnn_bgrl_internal_only/`
- `data/checkpoints/gnn_bgrl_relay/`
- `data/checkpoints/gnn_bgrl_relay_package/`
- `data/gnn_embeddings/gnn_bgrl_internal_only/`
- `data/gnn_embeddings/gnn_bgrl_relay/`
- `data/gnn_embeddings/gnn_bgrl_relay_package/`

**已完成的 50k 同口径报告**：

- `abe_vs_gnn_internal_only_dev200_50k`
- `abe_vs_gnn_relay_dev200_50k`
- `abe_vs_gnn_relay_package_dev200_50k`
- `gnn_internal_only_vs_relay_dev200_50k`
- `gnn_relay_vs_relay_package_dev200_50k`

**关键指标（Phase 4 A+B+E vs Phase 6）**：

- Phase 4 `A+B+E` baseline：
  - `NMI = 0.7137`
  - `ARI = 0.6760`
  - `silhouette = 0.6847`
- Phase 6 `internal_only`：
  - `NMI = 0.5074`
  - `ARI = 0.2124`
  - `silhouette = 0.4092`
- Phase 6 `relay`：
  - `NMI = 0.8397`
  - `ARI = 0.8758`
  - `silhouette = 0.4194`
- Phase 6 `relay + package prior`：
  - `NMI = 0.8497`
  - `ARI = 0.9102`
  - `silhouette = 0.4805`

**当前结论**：

- `internal_only` 明显退化，不建议作为主线
- `relay` 相比 `A+B+E` 明显提升了 `NMI / ARI`
- `relay + package prior` 是当前最强的 `Phase 6` 版本
- 但 `Phase 6` 虽然显著提升了 `NMI / ARI`，其 `silhouette` 明显低于纯 `Phase 4 A+B+E`
- 因此当前最合理的判断是：
  - **Phase 6 已经跑通并拿到了有效结果**
  - **但还存在“标签一致性更强、几何簇更松”的 tradeoff，需要继续解释与筛选主线版本**

**对 handoff 中旧判断的修正**：

- 旧版 handoff 把 `Phase 6` 记为“尚未实现 / 下一步”
- 这已经过时
- 截至当前仓库状态，`Phase 6` 已完成：
  - 数据集构建
  - 模型训练
  - embedding 导出
  - 多设定对比报告

**当前推荐主线**：

➡️ `Phase 4 A+B+E` 仍是强内容语义 baseline  
➡️ `Phase 6 relay + package prior` 是当前最值得继续深挖的图增强版本

**下一步起点**：

➡️ **解释并验证 `relay/package` 为何大幅提升 `NMI/ARI` 却压低 `silhouette`。**  
➡️ **决定论文/主线报告中，Phase 6 最终采用“以标签一致性为主”的版本，还是继续调图增强以兼顾几何聚类质量。**

---

### 2026-05-11（会话 9）— 单 APK 内部聚类 smoke test 落地 + 多样本定性检查

**完成的工作**：

- ✅ 新增 `scripts/07_cluster_single_apk.py`
  - 读取单个 APK 的 `methods/<SHA>.parquet`
  - 对齐 `Phase 6 relay/package` 的 `gnn_embeddings/<SHA>.npz`
  - 在 **单 APK 内部** 做 `UMAP + HDBSCAN`
  - 输出 `data/reports/single_apk_clusters/<SHA>.cluster_summary.json`
  - 每个 cluster 汇总：
    - `dominant_susi_labels`
    - `top_api_calls`
    - `top_classes`
    - `top_method_names`
    - `example_methods`

**已做的 smoke test / qualitative check**：

已对以下若干 APK 做了单样本内部聚类摘要：

- `4773AE2BE40E7C0D2EDF32CE99DA271F0062B6F5996C054B162A5EB5EDF91348`
- `E82480287CBD2D4341AE5B010F4F6D2E3504FD4B6C6D0C999F8D70ABAB11B764`
- `CD798050FBA9B1337A1720ADB020EFB038296A8840872F95AAFE25176D23404E`
- `747CADE4A974B1E9D79CBC912FCAA293F0CDF95BCCAE7EBD9C56A9976392D237`
- `EAD6B5FA0C3EE35F00B4E5888CAD4AE0B4A8AB50DC591331C3C507BE0F74D160`
- `E8EB3C40A4BFCBCA37BF81C7B7093BACC245505D7E86494C2C97520D4D005D5C`

**定性结论（当前最重要）**：

- ✅ 这条主线 **已经能在单 APK 内部稳定聚出一部分“功能一致”的方法簇**
- 反复观察到的较清晰功能簇包括：
  - `I/O / 文件操作`
  - `网络连接 / URL / HttpURLConnection`
  - `反射 / 动态代理 / JNI`
  - `广告 SDK 事件回调 / logging / 状态管理`
  - `UI / draw / layout / scroll / view behavior`
- ⚠️ 同时也反复观察到一些**解释价值较低的样板簇**：
  - 大量 `<init>` 构造器簇
  - `Enum.valueOf` / `createFromParcel` / lambda 编译产物簇
  - 混淆严重的 `a/b/c/zza/zzb` 工具簇

**当前判断**：

- 这一步 smoke test 已经足够说明：
  - `A+B+E + relay/package` 这条表示学习路线**不只是全局指标好看**
  - 它已经能够在真实 APK 内部产生**可人工解释的功能簇**
- 但目前还不能简单把“最大簇 = 最有价值簇”
- 后续需要补一个“**值得人工解释的 cluster 筛选准则**”，把样板代码簇和真正功能簇区分开来

**新的下一步起点**：

➡️ **研究并设计“值得人工解释的 cluster 筛选准则”**（例如弱标签集中度、API 集中度、非样板代码比例、簇的信息量）。  
➡️ **再评估是否要从“全 APK 全量聚类”转向“围绕敏感 API / 关键功能点抽局部子图再聚类”的分析路线。**

---

### 2026-05-11（会话 10）— 行为子图路线定稿：以 SAPI 为锚点的自适应子图生长

**最新路线确认**：

- ✅ 不再采用“全量 method 聚类”作为主分析单元
- ✅ 改为“以 SAPI / 关键 API 为锚点，在全 FCG 上自适应生长行为子图（BU）”
- ✅ 外部节点纳入 anchor 候选池与扩张候选池，但要单独标注置信度
- ✅ 扩张不使用固定 `1-hop/2-hop`，而采用 `gain` 驱动的 best-first growth
- ✅ 以 `conductance + 信息增益 + 样板惩罚` 共同决定边界与停止条件

**当前明确的实现分工**：

- `Phase 0`：全图 Anchor 发现
  - 内部方法与外部节点都可成为候选 anchor
  - 工具标定的 SAPI 映射仍是基础
- `Phase 1`：行为子图初始化
  - 从 anchor 出发建立初始 frontier
- `Phase 2`：自适应生长
  - 沿调用边扩张，但以 gain 排序而不是固定 hop
- `Phase 3`：边界裁剪
  - 结合 conductance / Δinfo / 噪声比例自动停边
- `Phase 4`：信息量筛选
  - 低信息子图直接丢弃
- `Phase 5`：重叠管理与输出
  - 允许多行为重叠，但保留主行为标签

**当前实现起点**：

- 先实现 `Phase 0`
- 目标是：在单 APK 的全 FCG 中识别出 SAPI / 关键 API anchors，并生成带置信度的候选锚点列表
- 之后再接入 Phase 1 的子图生长

**后续研究重点**：

- 如何定义“值得人工解释”的子图筛选准则
- 如何比较不同扩张策略与不同阈值的效果
- 如何在保留外部上下文节点的同时，避免样板代码和库代码污染行为子图

---

### 2026-05-11（会话 11）— Phase 0 anchor 优化 + dev200 批量评估完成

**已实现的优化**：

- ✅ Phase 0 从“单一 SAPI seed lookup”升级为“三层候选输出”
  - `hard anchors`：exact SuSi 命中且不属于明显 boilerplate 的节点
  - `context candidates`：有上下文意义、但暂不作为正式 seed 的节点
  - `rejected`：留作后续阶段/实验接口
- ✅ 外部节点不再误入 `hard anchors`
  - 外部节点仍保留在候选池中，供 Phase 1/2 扩张使用
- ✅ 给 anchor 记录保留了后续评估接口
  - `anchor_kind`
  - `source`
  - `exact_match`
  - `constraint_flags`
  - `score_components`
  - `future_eval`

**批量评估（dev200，200 APK）**：

- `apk_coverage = 0.89`
  - 178/200 个 APK 至少有 1 个 hard anchor
- `hard_anchor_total = 2017`
- `hard_anchor_mean_per_apk = 10.085`
- `hard_anchor_category_counts_top10`：
  - `NO_CATEGORY: 480`
  - `MIXED: 364`
  - `LOG: 343`
  - `NETWORK_INFORMATION: 229`
  - `DATABASE_INFORMATION: 140`
  - `LOCATION_INFORMATION: 98`
  - `CALENDAR_INFORMATION: 90`
  - `NETWORK: 86`
  - `FILE: 66`
  - `BLUETOOTH_INFORMATION: 32`

**当前解释**：

- SuSi 作为 hard anchor 的种子来源是**高精度、可复现、可批量跑通**的
- 但它不是最终行为定义
- 后续必须依赖 Phase 1/2 的自适应扩张和信息量筛选来减少 `NO_CATEGORY / MIXED / boilerplate` 污染

**Phase 0 当前状态**：

- 已可批处理 200 APK
- 已有量化评估
- 已保留后续阶段接口
- 下一步正式进入 `Phase 1`：基于 hard/context anchors 的自适应行为子图生长

---

### 2026-05-12（会话 12）— Phase 0.5 + Phase 1/2 一体化校准与 dev200 全量行为子图挖掘

**本轮目标**：

- 把 `Phase 1`（扩张）与 `Phase 2`（边界控制）合并为一个统一的“**增长即边界判断**”过程
- 在 `dev200` 上系统讨论阈值对行为子图规模、纯度与物理意义的影响
- 给出“**参数走廊**”和一个默认工作点，而不是只报一个拍脑袋的单点参数

#### 1. 新增的实现模块

- `src/androserum/behavior/clues.py`
  - 轻量版 `Phase 0.5` clue extraction
  - 从 `methods/*.parquet` 中基于 `instructions / api_calls / susi_cats` 提取：
    - `has_network_api`
    - `has_file_api`
    - `has_reflection_api`
    - `has_db_api`
    - `has_location_api`
    - `has_identifier_api`
    - `has_log_api`
    - `url_like_strings`
    - `file_like_strings`
    - `clue_score`
- `scripts/08b_extract_behavior_clues.py`
  - 单样本 clue 提取
- `scripts/08c_batch_extract_behavior_clues.py`
  - `dev200` 批量 clue 提取
- `src/androserum/behavior/growth.py`
  - `Phase 1/2` 一体化行为子图增长
  - 加入：
    - `quality_score`
    - `conductance_proxy`
    - `info_score`
    - `boilerplate_ratio`
    - `min_nodes_target` warm-up
    - `quality_delta` stop rule
    - 叶子边界裁剪
    - `behavior_label` / `behavior_label_reason`
- `scripts/10_grow_behavior_unit.py`
  - 单 anchor → 单 BU
- `scripts/10b_grow_representative_behavior_units.py`
  - 单 APK → 多个代表性 BU
- `scripts/10c_batch_grow_behavior_units.py`
  - `dev200` 全量行为子图挖掘
- `scripts/11_sweep_behavior_growth.py`
  - `dev200` 参数 sweep

#### 2. Phase 0.5（clue）批量结果

- `dev200` 批量 clue 提取完成
- 汇总：
  - `n_apks = 200`
  - `total_network_like_methods = 5223`
  - `total_file_like_methods = 13511`
  - `total_reflection_like_methods = 2520`
  - `total_db_like_methods = 4277`
  - `total_location_like_methods = 382`
  - `total_identifier_like_methods = 323`

**说明**：

- 这一步提供了后续 `Phase 1/2` 所需的 lightweight symbolic clues
- 不依赖 manifest 大工程，先用现有 `methods parquet` 直接补足 `clue(u,S)`

#### 3. 为什么不再把 Phase 1 / Phase 2 分开

之前的实现曾出现两种极端：

- 过松：BU 机械地长成固定 `26` 节点左右的图块
- 过紧：BU 普遍塌成 `2` 节点碎片

因此，本轮明确采用：

> **增长与边界一体化**

即：

- 每次不是单独问“这个节点像不像”
- 而是问“把它加入当前子图后，**整个子图质量是否提升**”

使用的整体质量函数包含：

- `semantic_cohesion`
- `clue_mean`
- `edge_density`
- `conductance_proxy`
- `boilerplate_ratio`
- `external_ratio`

同时引入：

- `quality_delta = quality(S ∪ {u}) - quality(S)`
- 若 `quality_delta` 低于阈值，则停止扩张

并补一个 `min_nodes_target = 6` 的 warm-up 机制，避免 BU 过早塌缩成仅 2 个点。

#### 4. 先做的小样本 pilot 发现

对两个代表性样本：

- `4773AE2BE40E7C0D2EDF32CE99DA271F0062B6F5996C054B162A5EB5EDF91348`
- `747CADE4A974B1E9D79CBC912FCAA293F0CDF95BCCAE7EBD9C56A9976392D237`

做了缩窄参数网格试验，重点观察：

- `mean_size`
- `median_size`
- `tiny_ratio (<=3 nodes)`
- `target_ratio (4~15 nodes)`
- `large_ratio (>=20 nodes)`
- `mean_conductance`
- `mean_info_score`
- `label_alignment`

发现：

- `tau_candidate_sim` 在 `[0.06, 0.10]` 范围内影响很小
- `tau_add` 在 `[0.0, 0.01]` 范围内也不是主导因素
- **真正主导 BU 是否“太碎 / 更完整”的，是 `tau_quality_delta`**

#### 5. dev200 上的系统参数 sweep（缩窄后）

在 `dev200` 上只扫：

- `tau_add = 0.01`
- `tau_quality_delta ∈ {0.005, 0.0, -0.005, -0.01}`
- `tau_candidate_sim ∈ {0.10, 0.08}`
- `min_nodes_target = 6`

主要结果如下：

##### 当 `tau_quality_delta = 0.005`

- `mean_size = 3.1567`
- `median_size = 2`
- `tiny_ratio = 0.6159`
- `target_ratio = 0.3841`
- `mean_conductance = 0.8991`
- `mean_info_score = 1.5137`
- `label_alignment = 1.0`

##### 当 `tau_quality_delta = 0.0`

- `mean_size = 3.2176`
- `median_size = 2`
- `tiny_ratio = 0.6083`
- `target_ratio = 0.3917`
- `mean_conductance = 0.8972`
- `mean_info_score = 1.5238`
- `label_alignment = 1.0`

##### 当 `tau_quality_delta = -0.005`

- `mean_size = 3.6387`
- `median_size = 2`
- `tiny_ratio = 0.5789`
- `target_ratio = 0.4113`
- `large_ratio = 0.0076`
- `mean_conductance = 0.8886`
- `mean_info_score = 1.5732`
- `label_alignment = 1.0`

##### 当 `tau_quality_delta = -0.01`

- `mean_size = 5.0501`
- `median_size = 3`
- `tiny_ratio = 0.5365`
- `target_ratio = 0.4070`
- `large_ratio = 0.0522`
- `mean_conductance = 0.8710`
- `mean_info_score = 1.6906`
- `label_alignment = 1.0`

#### 6. 对这些数字的解释（这一段非常重要）

**观察 1：**
`tau_candidate_sim` 在当前缩窄范围内几乎不敏感。

这表明：

- 真正进入 frontier 的有效节点，语义相似度本来就偏高
- 所以 `tau_candidate_sim` 在这一阶段更多是保守过滤，而不是主导 BU 形状

**观察 2：**
`tau_quality_delta` 是最关键的主调参数。

它控制的是：

- 要不要允许子图跨过局部的小质量波动，继续长成更完整的行为片段

当 `tau_quality_delta` 过高时：

- BU 很容易在 `2~3` 节点就停止
- 这种结果虽然边界干净，但更像“局部调用边”而不是“行为片段”

当 `tau_quality_delta` 适度放宽时：

- `FILE / NETWORK / DATABASE / LOCATION` 等 BU 更容易长成 `5~8` 节点的局部链条
- 更接近“可解释的静态行为单元”

但继续放宽也会出现代价：

- `large_ratio` 增加
- 子图更可能吸入弱相关上下文

所以我们最终不追求单一最优点，而是给出一个：

> **参数走廊（working corridor）**

#### 7. 最终推荐的参数走廊与默认工作点

##### 推荐走廊

- `tau_quality_delta ∈ [-0.01, -0.005]`
- `tau_candidate_sim ∈ [0.08, 0.10]`
- `tau_add ∈ [0.0, 0.01]`
- `min_nodes_target ∈ [5, 6]`

##### 更保守的工作点

- `tau_add = 0.01`
- `tau_quality_delta = -0.005`
- `tau_candidate_sim = 0.08`
- `min_nodes_target = 6`

特点：

- 更少的大 BU
- 更稳
- 但仍有较多 `2` 节点碎片

##### 更符合“物理行为意义”的默认工作点

- `tau_add = 0.01`
- `tau_quality_delta = -0.01`
- `tau_candidate_sim = 0.10`
- `min_nodes_target = 6`

选择理由：

- 它不是最保守的
- 但更容易让 BU 从碎片化走向局部行为链
- 在 `dev200` 上：
  - `mean_size = 5.0501`
  - `median_size = 3`
  - `meaningful_ratio = 0.4015`
- 比保守点（`tau_quality_delta = -0.005`）的
  - `mean_size = 3.6387`
  - `meaningful_ratio = 0.3624`
更符合“行为单元应具备最小物理行为意义”的目标

#### 8. 用默认工作点在 dev200 上全量挖掘行为子图的结果

对应目录：

- `data/reports/behavior_unit_sets_dev200_qd_neg01/`

汇总：

- `n_apks = 200`
- `total_units = 919`
- `mean_size = 5.0392`
- `median_size = 3`
- `tiny_ratio = 0.5386`
- `target_ratio = 0.4048`
- `large_ratio = 0.0522`
- `mean_conductance = 0.8715`
- `mean_info_score = 1.6886`
- `meaningful_units = 369`
- `meaningful_ratio = 0.4015`

`behavior_label_counts` 中较常见类别包括：

- `LOGGING: 212`
- `NETWORK_INFORMATION: 78`
- `NETWORK: 75`
- `FILE: 102`
- `DATABASE_INFORMATION: 41`
- `LOCATION_INFORMATION: 32`
- `DATABASE: 28`
- `REFLECTION: 24`

#### 9. 对“物理意义”的实际检查

从全量结果中抽查不同类别的 `meaningful_examples` 后，观察到：

##### `NETWORK`

典型链条会包含：

- `openConnection`
- `connection`
- `getRequestMethod`
- `getUrl`
- `getConnectTimeout`

这类子图很像一段真实的请求建立与配置过程。

##### `FILE`

典型链条会包含：

- `save`
- `writeFile`
- `decodeFrame`
- 某些 `run()` 回调

这更像一段局部的文件写入 / 缓冲 / 输出链条。

##### `DATABASE`

典型链条会包含：

- `getWritableDatabase`
- `query`
- 数据处理/持久化相关方法

这类 BU 从静态结构和 clue 上都比较像数据库读写行为片段。

##### `LOCATION_INFORMATION`

典型链条会包含：

- `fillCellLocationDetails`
- `fillWifiDetails`
- 位置服务检查 / 环境收集逻辑

##### `LOGGING`

这一类最需要谨慎。

- 有些 `LOGGING` BU 确实像 callback / state / persistence 的辅助链
- 但也有一部分更像“生命周期 + 状态更新”的泛化簇

因此：

- `FILE / NETWORK / DATABASE / LOCATION_INFORMATION`
  的物理意义最稳定
- `LOGGING / MIXED / NO_CATEGORY` 的解释还需要继续细化

#### 10. 当前最可信的结论

- ✅ 当前 pipeline 已经可以从 `dev200` 中系统抽取出一批具有**静态物理行为意义**的局部子图
- ✅ 参数不是拍脑袋选的，而是经过 `dev200` 校准后，在“子图完整性 / 边界清晰性 / 物理可解释性”之间形成的折中
- ⚠️ 这种“物理意义”并不意味着 100% 对应真实运行时行为
  - 它更准确地表示：
    - **结构上连通**
    - **语义上集中**
    - **线索上统一**
    - **人可解释**
  的静态行为单元

#### 11. 推荐的当前默认工作点（供后续实验统一使用）

- `tau_add = 0.01`
- `tau_quality_delta = -0.01`
- `tau_candidate_sim = 0.10`
- `min_nodes_target = 6`

后续如无特别说明，建议统一使用这组默认参数继续：

- 批量挖行为子图
- 做重叠 / prototype / cluster 级分析
- 以及和后续“局部子图知识化”路线对接

---

**文档版本**：v1.7
**最后更新**：2026-05-11（会话 11：Phase 0 anchor 优化 + dev200 批量评估完成）
**下次更新触发条件**：Phase 1 扩张规则成型 / 子图评价指标定稿 / 重大决策变更
