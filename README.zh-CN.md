# SGDiff 复现仓库（中文说明）

**课程项目交付入口**　[Katz 与 SP4LP 鲁棒性、组件消融、攻击图及复算说明](results/peer-protocol/README.md)。Cora、CiteSeer 和 PubMed 统一调用 Lin Junxiong 的原生评测器。下文保留早期 SGDiff 复现记录。

本组负责 **Katz 和 SP4LP 两个主模型**，横向比较采用 Lin Junxiong 的 14 条件评测。普通 GAE 已归档。[团队建议名单、评测对齐与新结果](results/peer-protocol/README.md)。

旧版观测缺失实验、既有接口及归档结果见[统一实验协议](docs/paired_reproduction.zh-CN.md)。

论文：**Sub-graph Based Diffusion Model for Link Prediction**（Li et al., LoG 2024）。

论文引用的官方仓库 <https://github.com/hzlihang99/SGDiff> 目前只有两行 README、没有任何代码，
因此本仓库依据论文正文（第 3 节 + 附录 A.1/A.2）**从零重写**了完整方法：

1. 提取 `(u, v)` 的 k 跳包围子图，并使用 DRNL 双半径节点标签（并保证目标边在 BFS 阶段就被隐藏，避免标签泄露）；
2. 子图邻接矩阵上的**离散扩散模型**（D3PM，二元状态、均匀转移），条件于类别 `y`；
3. 节点特征上的**高斯扩散模型**（DDPM），条件于邻接矩阵与类别；
4. 用可学习参数 `{η₁, η₂, δ}` 按贝叶斯公式融合两路条件似然（论文式 12-13）。

全部实现只依赖 PyTorch（无需 PyTorch Geometric / DGL），CPU 可跑。英文版见 [`README.md`](README.md)。

## 与论文的对齐

本仓库严格按 `li25a.pdf` 的表 3、表 4 与 A.3 节设置参数，逐项对照见
[`docs/alignment.md`](docs/alignment.md)。

| 论文（表 4 / A.3） | 本仓库 |
|--------------------|--------|
| 子图跳数 `k = 1`（所有数据集） | `subgraph.hops: 1` |
| 每跳采样上限 `ns` = −1 / 20 / 20 / 10 / 40 / 5 | 各数据集 `subgraph.max_nodes_per_hop` |
| 结构去噪器为 **Transformer**，8 头、2 层 | `structure.arch: transformer` |
| 注意力宽度 256 / 64 / 64，MLP 宽度 256 / 128 / 128 | `structure.node_attn`、`edge_attn`、`global_attn`、`*_mlp` |
| 结构扩散步数 20 / 20 / 10 / 5 / 10 / 20 | 各数据集 `structure.num_steps` |
| 特征 GCN 256 / 256 / 64，2 层，步数 100 / 100 / 50 | `feature.hidden_dim`、`num_layers`、`num_steps` |
| USAir / NS / Router 不使用特征支路 | `feature.enabled: false` |
| 结构去噪器以**节点/边交叉熵**训练，推理按 Algorithm 1 的后验 KL 打分 | `structure.loss_mode: ce`、`class_scores` |
| 先验 `p(y) = 0.5` | `LikelihoodFusion.log_prior`（均匀） |
| 80/5/15 划分、等量负样本、AUC/AP/Hit@100、10 个随机种子 | `data.*`、`scripts/run_experiments.py --seeds` |

> **方法状态（2026-11-15）：** `sgdiff-paper-v2`（修正实现：式 11 的特征 GCN、
> categorical 节点/边扩散、共享噪声打分、两阶段训练）已在 8×H20 GPU 上以
> `paper` 档（严格表 4：全量划分、子图不截断、全部 MC 步、10 种子）完整重跑
> 域内网格与 RQ1/RQ2/RQ3 全部实验。2026-11-13 审计确认早期实现不忠实，
> 其全部产物已于 2026-11-15 删除。

## 目录结构

```
sgdiff-repro/
├── sgdiff/                     # 可安装的库
│   ├── api.py                  # Python API：train / predict / evaluate
│   ├── config.py               # 类型化 YAML 配置
│   ├── cli/                    # `sgdiff` 命令行（prepare/train/evaluate/transfer）
│   ├── data/                   # datasets / download / graph / splits / subgraph / attack
│   ├── models/                 # SGDiff、结构扩散、特征扩散、Transformer 去噪器、融合
│   │   └── baselines/          # GCN / GAT / SAGE / VGAE / SEAL / PENCIL
│   ├── engine/                 # factory / batch / trainer / evaluator / benchmark / ablation
│   └── utils/                  # 指标、随机种子、日志、JSON 读写
├── configs/                    # 每个数据集 + smoke / transfer / lowdata / attack 的 YAML
├── data/                       # 原始与处理后数据（git 忽略）+ README
├── scripts/                    # run_experiments / run_baselines / run_benchmark /
│                               # run_ablation / collect_results / run_all_experiments.sh
├── examples/                   # 4 个可直接运行的示例脚本
├── tests/                      # 52 个 pytest 单元/端到端测试
├── docs/                       # method / alignment / api / reproduction / benchmark / ablation
├── checkpoints/                # 训练权重（git 忽略）
├── runs/                       # 每次运行：metrics.jsonl + best.pt（git 忽略）
└── results/                    # 运行产出的指标 JSON / 日志
```

## 安装与准备数据

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m sgdiff prepare --datasets cora citeseer pubmed usair ns
```

| 数据集 | 节点 | 无向边 | 节点特征 | 来源 |
|--------|-----:|-------:|:--------:|------|
| Cora / Citeseer / Pubmed | 2708 / 3327 / 19717 | 5278 / 4552 / 44324 | ✔ | Planetoid |
| USAir | 332 | 2126 | ✘ | networkrepository (USAir97) |
| NS | 1589 | 2742 | ✘ | KONECT (dimacs10-netscience) |
| Router | 5022 | 12516 | ✘ | SEAL 数据包，需手动放置（见 `data/README.md`） |

下载器带镜像回退（`ghproxy.net`），并会缓存为 `data/processed/*.npz`。

## 接口

### 命令行（`python -m sgdiff …`）

| 命令 | 作用 | 主要参数 |
|------|------|----------|
| `prepare` | 下载并预处理图，缓存为 `.npz` | `--datasets …`、`--root`、`--no-download` |
| `train` | 训练单个数据集（写 `runs/<name>/best.pt` + `metrics.jsonl`） | `--config`、`--set KEY=VALUE…`、`--run-dir`、`--max-steps` |
| `evaluate` | 评测 checkpoint，可叠加结构攻击 | `--checkpoint`、`--dataset`、`--split`、`--max-pairs`、`--attack {none,random,dice}`、`--attack-mode {flip,insert,delete}`、`--ratio`、`--device`、`--output` |
| `transfer` | 零样本跨数据集打分 | `--checkpoint`、`--targets …`、`--root`、`--max-pairs`、`--device`、`--output` |

任意配置项都可在命令行覆盖：

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

### 预算档（`scripts/run_experiments.py --profile`）

三档保持架构、划分与打分规则不变，只改算力预算。

| 档位 | 训练对 | epoch（预训练） | MC 步 | 子图上限 | 硬件 |
|------|-------:|----------------:|------:|---------:|------|
| `paper` | 全量 | 50 (10) | 全部 `ds_t` | 无 | 需 GPU（hub 的 1-hop 子图超 150 节点） |
| `full` | 全量 | 20 (4) | 8 | 60 节点 | 笔记本 CPU，10-30 分钟/数据集-种子 |
| `cpu` | 2 000 | 8 (3) | 4 | 40 节点 | 笔记本 CPU，2-10 分钟/数据集-种子 |

### 实验脚本

| 脚本 | 作用 | 示例 |
|------|------|------|
| `run_experiments.py` | 域内网格 + RQ1/RQ2/RQ3 | `--datasets cora citeseer usair ns pubmed --seeds 0 1 2 --profile full` |
| 同上 | RQ1 单源迁移 | `--transfer cora --targets citeseer pubmed usair ns` |
| 同上 | RQ1 全矩阵（每个源训练一次） | `--transfer-matrix --datasets cora citeseer pubmed usair ns --targets cora citeseer pubmed usair ns` |
| 同上 | RQ2 1% 子图 | `--lowdata cora --seeds 0 1 2` |
| 同上 | RQ3 攻击 | `--attack cora --ratios 0.1 0.25 0.5` |
| `run_baselines.py` | 同划分下的基线 | `--dataset cora --baselines gcn gat sage vgae --epochs 200` |
| `run_benchmark.py` | 同数据对比 + 划分/子图指纹 | `--dataset cora --profile cpu --seeds 0` |
| `run_ablation.py` | 似然是否真的条件于链接 | `--dataset cora --profile cpu --seed 0` |
| `collect_results.py` | 汇总为 markdown；`--merge <profile>` 由 per-seed 文件重建均值/方差 | `--merge full` |
| `run_all_experiments.sh` | RQ1+RQ2+RQ3 一键跑 | `bash scripts/run_all_experiments.sh cpu "0 1 2"` |

`--merge` 的存在原因：`run_experiments.py` 每次调用都会重写
`results/summary_indomain_<profile>.json`，分批跑数据集时前面几批会被覆盖。

## 本机实测（`sgdiff-paper-v2`，`paper` 档）

以下全部数字来自修正实现 `sgdiff-paper-v2`、`paper` 档（严格表 4：全量
训练/测试划分、子图不截断、全部 MC 步），8×NVIDIA H20，2026-11-14/15。
逐种子数据源：`results/sgdiff-paper-v2/{dataset}_seed{K}_paper.json`。
**是本机数字；论文表 2 没有 clean AUC 列可对比**（其 0.884/0.871/0.951
是 RF@25% 攻击下的数字，不能当 clean 参照）：

| 数据集 | 种子 | AUC | AP | Hit@100 |
|--------|-----:|----:|---:|--------:|
| Cora | 0-9 | **0.862 ± 0.008** | 0.875 | 0.953 |
| Citeseer | 0-9 | **0.833 ± 0.011** | 0.858 | 0.980 |
| Pubmed | 0-9 | **0.902 ± 0.002** | 0.900 | 0.988 |
| USAir | 0-9 | **0.868 ± 0.087** | 0.883 | 0.962 |
| NS | 0-9 | **0.965 ± 0.007** | 0.974 | 1.000 |

公式级修正对有特征的引文图影响最大（旧实现已删除的记录为 Cora 0.730 /
Citeseer 0.778 / Pubmed 0.827，修正后 +5.5 到 +13.2 个点）。

同划分 Cora 基线（`results/sgdiff-paper-v2/baselines_cora.json` + 3 种子
robustness 运行）：GCN 0.823 / GAT 0.849 / SAGE 0.743 / VGAE 0.824 /
SEAL 0.888 / PENCIL 0.883。**SEAL 与 PENCIL 在 clean AUC 上仍高于 SGDiff v2
（0.858）**；SGDiff 的优势在 Hit@100 排序指标，以及攻击下的稳定性。

### 研究问题（`paper` 档）

| 研究问题 | 实验 | 结果 |
|---|---|---|
| RQ1 | 5 源 × 5 目标（Router 不可得）、10 种子、全量测试集、不含节点特征、`source_ns` | 如 citeseer→cora **0.834**、citeseer→ns **0.957**、usair→pubmed **0.799** |
| RQ2 | Cora 仅用 1% 训练子图，10 种子 | AUC **0.711 ± 0.092** |
| RQ3 | Cora，SGDiff 在 RF/DICE/EA @10/25/50% 下，3 种子 | EA@25% **0.834**（clean 0.858）；最差 DICE@50% 0.751 |
| RQ3 | Cora，7 个模型 × RF/DICE/EA，3 种子 | `results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.{json,md}` |

RQ1 一段话总结：结构支路（无任何节点特征）确实可以零样本迁移——citeseer
训练的模型在 cora 上 0.834、在 ns 上 0.957；即使 332 节点的 USAir 源也能在
19717 节点的 Pubmed 上拿到 0.799。cora 是最难的目标（0.603–0.834），
与其 hub 密集的 1-hop 邻域一致。

### 已完成 / 未完成

| 实验 | 状态 | 产物 |
|------|------|------|
| 域内网格 `paper` | 完成：5 数据集 × 种子 0-9 | `results/sgdiff-paper-v2/{ds}_seed{K}_paper.json` |
| RQ1 迁移 | 完成：5 源 × 10 种子，`source_ns` | `results/sgdiff-paper-v2/transfer_{src}_seed{K}_paper.json` |
| RQ2 低数据 | 完成：Cora × 10 种子 | `results/sgdiff-paper-v2/lowdata_cora_seed{K}_paper.json` |
| RQ3 攻击（SGDiff） | 完成：3 种子 | `results/sgdiff-paper-v2/attack_cora_seed{K}_paper.json` |
| RQ3 全模型鲁棒性 | 完成：3 种子 | `results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.{json,md}` |
| 基线（Cora） | 完成 | `results/sgdiff-paper-v2/baselines_cora.json` |
| 似然消融 / 探针链 | 未在 v2 上重跑（旧产物已于 2026-11-15 删除） | `docs/ablation.md`、`docs/attribution.md` |

## 论文本身的漏洞与欠定义之处

复现过程中暴露出以下论文未定的协议细节。它们让"是否复现成功"部分不可判定——
以下不是"论文错了"的断言，而是复现缺少的必要信息。

1. **没有参考实现。** 论文链接的官方仓库只有一个两行 README，所有细节只能从正文推断。
2. **优化细节完全未报告。** batch size、学习率、epoch 数、两阶段日程、`{η₁, η₂, δ}` 的拟合方式
   都没有；我们只能自选（见 `docs/alignment.md` 第 4 节），而它们能让 Cora 变动好几个点。
3. **没有方差、没有评测集规模。** 4.1 节说用了 10 个种子，但表里只给单个数，也没说测试对数量
   与负采样规则。
4. **RQ1 没说明是否剔除对角线。** "在全部六个图上测"把真正的迁移行和域内行（source=target）
   混在一起；我们剔除对角线并单独报出。
5. **RQ1 的特征处理不是中立的。** SGDIFF/SEAL 完全不给特征，VGAE 用"随机投影 + 行归一化"，
   无特征图用高斯随机向量——检验的是结构支路，随机投影无法对齐跨数据集语义。
6. **"无需重训"仍隐含目标端知识。** 表 4 的 `ns` 是 per-dataset 的。论文没说测试时用哪一套；
   我们的矩阵采用最严格的 `source_ns` 读法。
7. **Source/Target 是无权算术平均。** 目标难度差异极大，平均值取决于集合里放了哪些数据集；
   完整矩阵才是可靠产物。

### 实现侧备注（诚实清单）

* **`k = 1` 是论文的设定，不是偏差。** 表 4 所有数据集都是 `k=1`；它解释了 1-hop SEAL
  为何是强基线，不构成与论文的偏离。
* **KL 项在 log 空间计算。** float32 下 `1 - (1 - 1e-8)` 舍入成 `0` 会使
  `log((1-pq)/(1-sigmoid(z)))` 变 `inf`；已加非有限步保护。
* **负样本各 split 独立采样、未共享去重集合**，测试负样本可能同时出现在训练负样本里
  （小图上 ≤ 3.8%）。重叠会小幅抬高 AUC；为保持可复现，记录但不改动。

## 文档

| 文档 | 内容 |
|------|------|
| `docs/method.md` | 论文公式 → 代码的逐项映射（DRNL、D3PM 后验闭式推导、融合式） |
| `docs/alignment.md` | 表 3/表 4 的超参逐项对照 + 已知偏差 |
| `docs/api.md` | Python API / CLI / 内部模块的接口说明 |
| `docs/reproduction.md` | 实验协议、实测数值、精度差距归因、bug 记录 |
| `docs/benchmark.md`、`docs/ablation.md` | 同数据对比、似然消融 |
| `data/README.md` | 数据集来源、格式、Router 的手动获取方式 |

## 引用

请引用原论文（见 `CITATION.cff`）。本仓库代码以 MIT 协议发布。
## GCN 消融整合入口

新增 [GCN 三组输入消融与跨数据集迁移](experiments/gcn_ablation/README.md)，复用本仓库 GCN 编码器及内积解码器。两套实验的划分、传播图和指标存在差异；比较历史结果前请阅读 [重合与协议审查](docs/gcn_ablation_integration.md)。

## 课程基线、SP4LP与观测缺失实验

新增CN、AA、PA、属性点积/余弦、逻辑回归融合及SP4LP三个版本，接入现有benchmark和攻击评价。SGDiff v2核心保持不变。运行方式和历史结果边界见[集成说明](docs/course_sp4lp.md)。

[八种方法的统一鲁棒性实验、四人分工与三组消融](docs/team_study.zh-CN.md)
