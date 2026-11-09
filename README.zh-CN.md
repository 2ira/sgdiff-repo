# SGDiff 复现仓库（中文说明）

论文：**Sub-graph Based Diffusion Model for Link Prediction**（Li et al., LoG 2024）。

论文引用的官方仓库 <https://github.com/hzlihang99/SGDiff> 目前只有两行 README、没有任何代码，
因此本仓库依据论文正文（第 3 节 + 附录 A.1/A.2）**从零重写**了完整方法：

1. 提取 `(u, v)` 的 k 跳包围子图，并使用 DRNL 双半径节点标签（并保证目标边在 BFS 阶段就被隐藏，避免标签泄露）；
2. 子图邻接矩阵上的**离散扩散模型**（D3PM，二元状态、均匀转移），条件于类别 `y`；
3. 节点特征上的**高斯扩散模型**（DDPM），条件于邻接矩阵与类别；
4. 用可学习参数 `{η₁, η₂, δ}` 按贝叶斯公式融合两路条件似然（论文式 12-13）。

全部实现只依赖 PyTorch（无需 PyTorch Geometric / DGL），CPU 可跑。

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
sgdiff/            # 可安装的库：api / config / data / models / engine / cli / utils
configs/           # 各数据集与实验（迁移、1% 数据、对抗攻击）的 YAML 配置
data/              # 原始与处理后的数据（git 忽略，可用命令重新拉取）
scripts/           # 数据准备、基线模型、一键跑完 RQ1/RQ2/RQ3
examples/          # 4 个可直接运行的示例脚本
tests/             # 31 个 pytest 单元/端到端测试
docs/              # 方法-代码映射、API 手册、复现记录（含已测数值与偏差说明）
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

## 常用命令

```bash
# 冒烟测试（合成图，CPU 约 30 秒）
python -m sgdiff train --config configs/smoke_synthetic.yaml

# 在 Cora 上训练 + 测试（下面是 CPU 档；去掉 --set 即表 4 全量设置，需要 GPU）
python -m sgdiff train --config configs/sgdiff_cora.yaml --run-dir runs \
  --set data.max_train_pairs=2000 data.max_eval_pairs=500 train.epochs=8 \
        train.pretrain_epochs=3 structure.num_mc_samples=4 \
        feature.num_mc_samples=4 subgraph.max_nodes_total=40
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --split test --device cpu

# RQ1 跨数据集零样本迁移
python -m sgdiff transfer --checkpoint runs/cora-*/best.pt --targets citeseer usair

# RQ3 结构攻击下的鲁棒性
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack random --ratio 0.1

# 基线模型（GCN / GAT / SAGE / VGAE / SEAL）
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal

# 论文实验网格：5 数据集 × 3 种子 + 迁移 / 1% 数据 / 攻击
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed --seeds 0 1 2
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5
```

> `--profile paper` 使用表 4 的完整设置（全量数据、全部 MC 步），需要 GPU；
> `--profile cpu`（默认）保持协议与结构不变，只缩小预算（2000 训练对、8 epoch、
> 4 个 MC 步、子图 40 节点上限），笔记本即可跑完。

任意配置项都可以在命令行覆盖：

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

## 本机实测（CPU 预算档 `--profile cpu`，3 个随机种子均值）

| 实验 | AUC | AP | Hit@100 |
|------|----:|---:|--------:|
| Cora | 0.751 | 0.809 | 0.960 |
| Citeseer | 0.738 | 0.787 | 0.903 |
| Pubmed | 0.816（2 种子） | 0.813 | 0.810 |
| USAir | 0.929 | 0.923 | 0.960 |
| NS | 0.964 | 0.973 | 1.000 |

无节点特征的 USAir / NS 与论文数值基本一致；两个引文网络低 10–13 个 AUC 点，
原因按影响从大到小为：`k=1` 子图（表 4；同预算下 `subgraph.hops=2` 时 Cora 可达 0.86）、
仅 2000 个训练对、8 个 epoch、4 个 Monte-Carlo 步。详细分析与标准差见
`docs/reproduction.md`。

三个研究问题（同预算，结果在 `results/`）：

| 研究问题 | 实验 | AUC |
|---|---|---|
| RQ1 | Cora → Pubmed / NS / USAir / Citeseer（零样本、不含节点特征） | 0.830 / 0.947 / 0.800 / 0.733 |
| RQ2 | Cora 仅用 1% 训练子图 | 0.725 ± 0.058 |
| RQ3 | Cora 随机翻转 10% / 25% / 50% | 0.698 / 0.651 / 0.577 |
| RQ3 | Cora DICE 10% / 25% / 50% | 0.692 / 0.643 / 0.584 |

## 文档

| 文档 | 内容 |
|------|------|
| `docs/method.md` | 论文公式 → 代码的逐项映射（含 DRNL、D3PM 后验闭式推导、融合式） |
| `docs/api.md` | Python API / CLI / 内部模块的接口说明 |
| `docs/reproduction.md` | 实验协议、实测数值、与论文的偏差与局限、bug 记录 |
| `data/README.md` | 数据集来源、格式、Router 的手动获取方式 |

## 引用

请引用原论文（见 `CITATION.cff`）。本仓库代码以 MIT 协议发布。
