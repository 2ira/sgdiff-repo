# SGDiff 复现仓库（中文说明）

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
| 训练目标 = VLB + 节点/边交叉熵之和 | `structure.aux_ce_weight: 1.0` |
| 先验 `p(y) = 0.5` | `LikelihoodFusion.log_prior`（均匀） |
| 80/5/15 划分、等量负样本、AUC/AP/Hit@100、10 个随机种子 | `data.*`、`scripts/run_experiments.py --seeds` |

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

## 本机实测

域内链接预测，`full` 档，多个种子的均值 ± 标准差。数据源
`results/summary_indomain_full.json`（可用 `python scripts/collect_results.py --merge full` 重建）。
**是本机数字，不是论文数字**：

| 数据集 | 种子 | AUC | AP | Hit@100 | 论文（clean AUC） |
|--------|-----:|----:|---:|--------:|------------------:|
| Cora | 0/1/2 | 0.730 ± 0.030 | 0.794 | 0.973 | 0.884 |
| Citeseer | 0/1/2 | 0.778 ± 0.016 | 0.814 | 0.937 | 0.871 |
| Pubmed | 0 | 0.827 | 0.817 | 0.900 | 0.951 |
| USAir | 0/1/2 | 0.929 ± 0.002 | 0.924 | 0.967 | ~0.95 |
| NS | 0/1/2 | 0.985 ± 0.002 | 0.987 | 1.000 | ~0.97 |

无节点特征的 USAir / NS 与论文持平或略好；有特征的三个引文网络偏低
（Cora 低 15 点、Citeseer 低 9 点、Pubmed 低 12 点）。**落后的恰好是"有节点特征"的三个数据集**，
而无特征的两个数据集能对上——这说明问题出在特征支路，而不是结构支路。

从 `cpu` 档换到 `full` 档（全量训练对、20 epoch、8 个 MC 步、60 节点上限）：
Citeseer +4.0、NS +2.1、USAir 无变化、Cora −2.1（Cora 的种子间方差 0.690/0.736/0.763
已大于档位差）。`results/summary_indomain_cpu.json` 保留了更省的档位。

同划分的 Cora 基线（`results/baselines_cora_full.json`、`results/baselines_cora_seal.json`）：

| 模型 | AUC | AP |
|------|----:|----:|
| GCN | 0.811 | 0.826 |
| GAT | 0.846 | 0.858 |
| SAGE | 0.749 | 0.758 |
| VGAE（`kl_weight=1e-3`） | 0.798 | 0.810 |
| SEAL（1-hop 子图，20 epoch） | 0.855 | 0.885 |

SEAL 在这套 1-hop 子图上**强于** SGDiff（0.855 vs 0.730）：两者读的是同一批子图，
说明在表 4 的 `k=1` 设定下，有监督子图分类器比生成式似然更占优。SEAL 自己的协议
（3 000 子图 / 2 跳 / ranking loss）本仓库未复现。

### 研究问题（`cpu` 档）

RQ1–RQ3 每个 arm 都要独立训练一次，因此保留便宜的预算；同一组对比内部预算一致。

| 研究问题 | 实验 | 结果 |
|---|---|---|
| RQ1 | 5 源 × 5 目标（Router 不可得）、3 种子、全量测试集、不含节点特征、剔除对角线 | 源平均 **0.71-0.84**，目标平均 **0.67-0.92**（`results/transfer_matrix_cpu.json`） |
| RQ2 | Cora 仅用 1% 训练子图（40 epoch） | AUC 0.725 ± 0.058 |
| RQ3 | Cora 随机翻转 10% / 25% / 50% | AUC 0.698 / 0.651 / 0.577 |
| RQ3 | Cora DICE 10% / 25% / 50% | AUC 0.692 / 0.643 / 0.584 |

RQ1 一段话总结：迁移始终达不到域内水平，要损失 1-13 个点
（NS 作目标 0.924 vs 域内 0.964；Citeseer 0.673 vs 0.738；Pubmed 0.682-0.738 vs 0.817），
而且有一格跌到随机水平（Pubmed → NS `0.494 ± 0.038`）。**域内越强的源迁移越差**
（源平均：Citeseer 0.838 > Cora 0.824 > USAir 0.759 > Pubmed 0.742 > NS 0.713，
几乎与域内排名相反），说明小而准的源过拟合了自己的度分布。用目标的表 4 `ns` 替代源的 `ns`
会让低度目标变好（USAir +2.6）、Pubmed 变差（−5.6），可见"无需重训"仍隐含一个
per-dataset 旋钮。完整矩阵与两个变体见 `docs/reproduction.md` 第 4 节。

### 已完成 / 未完成

| 实验 | 状态 | 产物 |
|------|------|------|
| 域内网格 `full` | 完成：Cora/Citeseer/USAir/NS × 种子 0-2，Pubmed × 种子 0 | `results/summary_indomain_full.json` |
| 域内网格 `cpu` | 完成：5 数据集 × 种子 0-2 | `results/summary_indomain_cpu.json` |
| 域内网格 `paper` | 未跑：需要 GPU（1-hop 子图不设上限） | — |
| 基线（Cora） | 完成：GCN/GAT/SAGE/VGAE 200 epoch、SEAL 20 epoch、PENCIL 在 benchmark 中 | `results/baselines_cora_{full,seal}.json` |
| 同数据 benchmark | 完成 | `results/benchmark_cora.{json,md}`、`docs/benchmark.md` |
| 似然消融 | 完成 | `results/ablation_cora{,_full}.{json,md}`、`docs/ablation.md` |
| RQ1 单源迁移 | 完成（`cpu`，种子 0，每类 500 对），已被矩阵取代 | `results/transfer_cora_seed0_cpu.json` |
| RQ1 全矩阵 | 完成：5 源 × 5 目标、3 种子、全量测试集、`source_ns` + `target_ns` 两版 | `results/transfer_matrix_cpu.json` |
| RQ2 / RQ3 | 完成，3 种子（`cpu`） | `results/summary_{lowdata,attack}_cora.json` |

## 论文本身的漏洞与欠定义之处

本仓库在**有节点特征**的三个数据集上比论文低（Cora −15、Citeseer −9、Pubmed −12 个 AUC 点），
而在**无特征**的两个数据集上与论文持平。追这个差距时遇到以下问题，它们让"是否复现成功"
变得无法判定，而不只是困难。以下不是"论文错了"的断言，而是复现时缺少的必要信息。

1. **没有参考实现。** 论文链接的官方仓库只有一个两行 README，下面所有细节都只能从正文推断。
2. **优化细节完全未报告。** batch size、学习率、epoch 数、两阶段日程、`{η₁, η₂, δ}` 的拟合方式
   都没有；我们只能自选（见 `docs/alignment.md` 第 4 节），而它们能让 Cora 变动好几个点。
3. **没有方差、没有评测集规模。** 4.1 节说用了 10 个种子，但表里只给单个数，也没说测试对数量
   与负采样规则。没有这些，2-3 个点的差异既不能说"复现失败"，也不能说"复现成功"。
4. **RQ1 没说明是否剔除对角线。** "在全部六个图上测"把真正的迁移行和域内行（source=target）
   混在一起；Source/Target 平均会因此变化。我们剔除对角线并单独报出。
5. **RQ1 的特征处理不是中立的。** SGDIFF/SEAL 完全不给特征，VGAE 用"随机投影 + 行归一化"，
   无特征图用高斯随机向量。这套设计检验的是**结构支路**能否迁移，而不是特征支路；而且随机投影
   无法对齐跨数据集的语义，VGAE 的迁移分数基本是噪声，不能当作其编码器可迁移的证据。
6. **"无需重训"仍隐含目标端知识。** 表 4 的 `ns` 是 per-dataset 的。若测试时用目标的 `ns`，
   就不是纯零样本；论文没说用哪一套。我们两版都报（`results/transfer_matrix_cpu.json`：
   `source_ns` 与 `target_ns`）。
7. **Source/Target 是无权算术平均。** 目标难度差异极大（我们实测单目标 AUC 跨度 0.80-0.99），
   平均值取决于目标集合里放了哪些数据集；完整的矩阵才是可靠产物。

### 我们这边的同类问题（诚实清单）

* **特征支路从未学到东西。** 每个有特征的数据集上，DDPM 损失在 20 个 epoch 内始终是
  `1.0000 ± 0.0003`，即去噪器输出≈0、特征似然是常数。原因：Cora 的 `eps` 是 1433 维，
  但只能通过 `1433 → 256` 的投影到达输出，无法重建，于是"预测 0"成为最优解。
  加上 `sqrt(1-ā)·x_t` 的跳连后损失能下降（0.74 → 0.69），但 Cora AUC 反而从 0.736 掉到 0.704，
  说明该支路在当前预算下本身没有信息量。下文数字都是在"支路存在但无效"的状态下测得的。
* **`k = 1` 是论文的设定，不是偏差。** 它只能解释我们为什么低于 SEAL 的 2-hop 结果，
  不能解释我们为什么低于同样用 `k = 1` 的论文。
* **KL 曾经产生 `inf`。** float32 下 `1 - (1 - 1e-8)` 会舍入成 `0`，
  `log((1-pq)/(1-sigmoid(z)))` 直接变 `inf`；NS 种子 1 因此参数变 NaN、AUC 0.223。
  已改为 log 空间计算并跳过非有限步，重跑得 0.985。
* **负样本三次采样未共享去重集合**，测试负样本可能同时出现在训练负样本里
  （Cora 1/792、USAir 12/319）。重叠不超过测试负样本的 3.8%，且会**抬高** AUC，
  因此不能解释精度差；为保持结果可复现，记录但不改动。

完整清单见 `docs/reproduction.md` 的 bug journal。

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
