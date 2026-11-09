# SGDiff 复现仓库（中文说明）

论文：**Sub-graph Based Diffusion Model for Link Prediction**（Li et al., LoG 2024）。

论文引用的官方仓库 <https://github.com/hzlihang99/SGDiff> 目前只有两行 README、没有任何代码，
因此本仓库依据论文正文（第 3 节 + 附录 A.1/A.2）**从零重写**了完整方法：

1. 提取 `(u, v)` 的 k 跳包围子图，并使用 DRNL 双半径节点标签（并保证目标边在 BFS 阶段就被隐藏，避免标签泄露）；
2. 子图邻接矩阵上的**离散扩散模型**（D3PM，二元状态、均匀转移），条件于类别 `y`；
3. 节点特征上的**高斯扩散模型**（DDPM），条件于邻接矩阵与类别；
4. 用可学习参数 `{η₁, η₂, δ}` 按贝叶斯公式融合两路条件似然（论文式 12-13）。

全部实现只依赖 PyTorch（无需 PyTorch Geometric / DGL），CPU 可跑。

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

# 在 Cora 上训练 + 测试
python -m sgdiff train    --config configs/sgdiff_cora.yaml --run-dir runs
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --split test --device cpu

# RQ1 跨数据集零样本迁移
python -m sgdiff transfer --checkpoint runs/cora-*/best.pt --targets citeseer usair

# RQ3 结构攻击下的鲁棒性
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack random --ratio 0.1

# 基线模型（GCN / GAT / SAGE / VGAE / SEAL）
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal
```

任意配置项都可以在命令行覆盖：

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

## 本机实测（CPU、小预算：2000 训练对、9 个 epoch）

| 实验 | AUC | AP | Hit@100 |
|------|----:|---:|--------:|
| Cora 同分布测试 | 0.860 | 0.885 | 0.96 |
| Citeseer 同分布测试 | 0.805 | 0.854 | 0.97 |
| Cora → Citeseer 零样本迁移 | 0.834 | 0.857 | 0.73 |
| Cora → USAir 零样本迁移 | 0.674 | 0.619 | 0.62 |
| Cora 随机攻击 10% | 0.889 | 0.914 | — |

这些数值**不是**论文报告值，只用于验证实现正确、可学习。与论文的差距主要来自训练预算
（子图数量、epoch、MC 采样步数）；完整的偏差说明、已修复的 bug 记录见
`docs/reproduction.md`。

## 文档

| 文档 | 内容 |
|------|------|
| `docs/method.md` | 论文公式 → 代码的逐项映射（含 DRNL、D3PM 后验闭式推导、融合式） |
| `docs/api.md` | Python API / CLI / 内部模块的接口说明 |
| `docs/reproduction.md` | 实验协议、实测数值、与论文的偏差与局限、bug 记录 |
| `data/README.md` | 数据集来源、格式、Router 的手动获取方式 |

## 引用

请引用原论文（见 `CITATION.cff`）。本仓库代码以 MIT 协议发布。
