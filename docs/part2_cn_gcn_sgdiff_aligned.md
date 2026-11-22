# Part 2: CN、GCN 与 SGDiff 的统一鲁棒性实验

本分支补齐剩余三个建议方法，并固定调用 Junxiong 提交
`95083da808e500c02da458e205f4b861ceb80871` 的原生划分、扰动、评测与四项指标。
所有方法使用相同的 Cora、CiteSeer、PubMed seed 0/1/2 测试正负对和相同攻击图。

## 当前定稿范围

三数据集现已全部完成：7 个配置 × 3 个数据集 × 3 个 seed × 14 个条件，共 882 个
完整单元，且所有攻击图 SHA-256 与 pinned evaluator 匹配。三数据集定稿表位于
`results/part2-peer-cn-gcn-sgdiff-all/`。

## 正式矩阵

- 数据集：Cora、CiteSeer、PubMed。
- 独立种子：0、1、2。
- 条件（每个 seed 14 个）：clean；RF、DICE、EA 各 10%/25%/50%；随机删边 50%；
  高度优先删边 50%；随机插边 50%；节点属性整行置换。
- 指标：AUROC、AP、Hits@100、Precision@100。
- 测试上下文：train positive + validation positive；所有 validation/test 候选受保护。
- EA：Cora/CiteSeer 复用 Junxiong 发布的 CUDA 图；PubMed 复用 Jiatong 发布的 H200 图。
  因而模型之间逐 seed、逐条件的图指纹完全一致。

## 模型与消融

| family | full | 组件对照 | 训练 |
|---|---|---|---|
| CN | 共同邻居数 | 无可训练组件；与 NCNC 做机制比较 | 不训练 |
| GCN | 两层 GCN + dot-product decoder | no-neighbors（相同 MLP 宽度）；no-attributes（常数节点输入） | 三个版本分别训练 |
| SGDiff | structure + feature + Bayesian fusion | structure-only；feature-only | 只训练一次；三种读出共享同一前向证据 |

GCN 三个版本保持隐藏宽度 64、输出宽度 64、两层、dropout 0.5；Adam lr=0.01，
最多 100 轮，每 5 轮验证，30 轮无改进耐心。训练图仅有 train positives；验证图仍仅有
train positives；测试及攻击图包含 train + validation positives。

SGDiff 使用仓库修正后的 `sgdiff-paper-v2`：完整训练 split、论文 Table-4 的子图和扩散步数、
batch 32、枚举全部扩散时间步而非抽样子集。扩散最多 50 轮并按验证 diffusion loss 早停。论文没有给出
可核实的 epoch 数；本地 3060 的正式预算将冻结扩散后的 fusion 校准固定为 5 轮，并按验证损失
选择其中最佳 checkpoint。该选择保留完整数据和全时间步证据，同时避免验证表现已经平台化后
继续空跑约数小时；结果不能伪称为仓库旧 H20 运行的 20 轮 fusion 原样复跑。

## 本地资源记录

环境：NVIDIA GeForce RTX 3060 Laptop GPU 6 GiB，PyTorch 2.5.1+cu121，CUDA 12.1。

- CN：参数为零，不训练；CPU 稀疏邻接计数。攻击图仍来自统一 CUDA/H200 缓存。
- GCN：正式 Cora full 单 seed 实测约 2.3 秒（最大 3.6 秒），三数据集 full 版本峰值约
  0.51 GiB；三个版本、三个数据集、三个 seed 的训练和 504 个 CN/GCN 评测单元合计约 75 秒。
- SGDiff：Cora 扩散阶段约 24--28 秒/epoch，通常约 31 轮早停；全时间步 fusion 约
  9--10 分钟/epoch；14 个条件的三种共享读出约 15 分钟/seed。训练时整卡占用约 2.9 GiB。
- PubMed EA 生成峰值约 8.77 GiB，超出本机 6 GiB；正式实验只读取团队发布的 H200 缓存，
  不在本机重新生成不同的攻击图。

## 运行

```powershell
E:\codex\sgdiff-part2-aligned\.venv-cuda\Scripts\python.exe `
  scripts\run_part2_peer_models.py run `
  --models cn gcn sgdiff --datasets cora citeseer pubmed --seeds 0 1 2 `
  --peer E:\codex\lin-eval `
  --peer-ea-assets E:\codex\lin-peer-assets\experiments\new_link_models\benchmarks\ea_graphs `
  --device cuda --sgdiff-finetune-epochs 5 --num-workers 0
```

入口支持断点恢复。训练权重和逐条件预测留在 `runs/part2-peer/`；紧凑的逐 seed 表、均值、
样本标准差、clean-to-attack drop、full-minus-component 配对差和 provenance 发布到
`results/part2-peer-cn-gcn-sgdiff/`。
