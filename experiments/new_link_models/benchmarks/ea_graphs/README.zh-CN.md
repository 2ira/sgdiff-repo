# 三模型实验实际使用的 18 张 EA 攻击图

这里上传的是原实验缓存的**逐字节副本**，不是重新生成的攻击图，也不是筛选出表现较好的图。
18 张 = Cora/Citeseer × 种子 0/1/2 × 编辑预算 10%/25%/50%。同一张图供三个模型及各自
两个消融版本共同使用，每图对应 9 次评估；共对应报告中的 162 次 EA 评估。

## 文件与加载

- `graphs/`：18 个受扰图，每个 NPZ 含 `edges`、`base_sha256`、`statistics`。
- `clean/`：6 个对应的干净可见基图，同样保留原缓存内容。
- `splits/`：6 个原始划分，含 `train_pos/val_pos/test_pos/train_neg/val_neg/test_neg`。
- `manifest.json`：文件 SHA-256、边数组指纹、实际预算、使用这些图的运行记录和生成代码指纹。
- `verify.py`：只需要 NumPy，核对文件、划分、边集合、预算和候选保护，不运行攻击或训练。

边均是从 0 开始的节点编号，`int64` 数组，形状为 `(无向边数, 2)`，每行 `u < v`，
不含自环、反向副本或重复边。节点数 Cora=2708，Citeseer=3327；孤立节点仍计入节点数。
这不是图片文件，而是可直接用于模型推理的图边表。特征按原 Planetoid 节点顺序加载，不在此重复上传。

```python
from pathlib import Path
import json
import numpy as np

root = Path('experiments/new_link_models/benchmarks/ea_graphs')
with np.load(root / 'graphs/cora_s0_ea_0.1.npz', allow_pickle=False) as f:
    edges = f['edges']
    statistics = json.loads(str(f['statistics']))
with np.load(root / 'splits/cora_seed0.npz', allow_pickle=False) as f:
    test_pos, test_neg = f['test_pos'], f['test_neg']
# 如果框架需要双向 edge_index，补上反向边：
edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T
# 直接使用已加载的受扰图，不要再次调用攻击器；LPFormer 需针对该图重算 PPR。
```

克隆并切换到 `feat/ncnc-lpformer-refined-gae` 分支后，在仓库根目录核验：

```bash
python experiments/new_link_models/benchmarks/ea_graphs/verify.py
```

## 图是怎么生成的

1. 固定同一数据集/种子的正负样本划分；干净推理图仅含训练和验证正边。
2. 保护全部测试正负候选点对及验证负边，排除其被翻转的可能。
3. 在可见图上训练 32 维线性代理 `Z = row_normalize(A) W`，Adam 学习率 0.01，60 步。
   每步抽取 1024 个现有正边和 1024 个真实非边作为负样本；不使用测试候选标签。
4. 对代理损失计算邻接梯度，用 `(grad + grad.T) * (1 - 2*A)` 为无向翻边打分，
   在允许的上三角候选中一次性选择 top-k；k=`round(预算比例 × 可见无向边数)`。
5. 不固定删除/插入配额。本轮 18 张图的 top-k **全部为插边**，删除数均为 0。
   同种子各预算分别按固定种子运行/缓存，不根据模型测试分数挑选；应以缓存为唯一复用依据，
   不依赖跨设备重跑梯度攻击后能逐边相同。

实现见 [`protocol.py::_ea_candidates/perturb`](../../protocol.py)，缓存使用见
[`run.py::evaluate`](../../run.py)。它是线性代理迁移攻击，不是针对 NCNC/LPFormer/
Refined-GAE 分别优化的白盒攻击，也不是 DeepRobust 的原版实现。

| 数据集 | 干净可见边数 | 10% 插入数 | 25% 插入数 | 50% 插入数 |
|---|---:|---:|---:|---:|
| Cora | 4486 | 449 | 1122 | 2243 |
| Citeseer | 3869 | 387 | 967 | 1934 |

## 与原仓库协议的实质差异

以下比较针对本分支基于的原仓库实现 `sgdiff/data/attack.py::embedding_attack` 和
`sgdiff/engine/robustness.py::run_robustness`；不声称代表合作者未提供的其他版本。

| 项目 | 原仓库实现 | 本轮三个新模型的实现 |
|---|---|---|
| 攻击输入 | 完整图；攻击后移除测试正边 | 先移除测试正边，在训练+验证图上攻击 |
| 预算分母 | 完整图边数 | 可见推理图边数 |
| 测试候选保护 | 攻击器没有测试候选保护参数；之后移除测试正边 | 攻击前保护测试正负候选及验证负边 |
| 删除/插入配额 | 固定约一半删除、一半插入 | 统一翻边排序，不固定配额，本轮全部为插入 |
| 代理每步样本 | 2048 正例 + 2048 随机点对负例；负例不逐一过滤非边/自环 | 1024 正例 + 1024 过滤后的真实非边 |
| 共同部分 | 干净权重下的推理扰动；32维线性代理、60步、三档预算 | 相同总体设置，但不等于相同攻击协议或相同图 |

**这些差异不只是随机数导致，不能把两批 EA 结果直接拼接成同协议排名。**
保护测试候选可作为独立设计，但更换图的时序、分母、采样和翻边配额会改变攻击难度；
不能据此直接断言本轮更严格或更合理。此前只概括成“统一协议”不够明确。

若目的是与原仓库已有模型联合比较，主结果应统一原始划分、候选点对、攻击缓存和预算语义。
这些图可以供合作者复核本轮独立协议，不能冒充原仓库攻击图。若决定采用本轮候选保护方案，
则应让原仓库模型也在同一批图上重新评估；若训练划分不同，还需重新训练。
现有报告的数字保持原样，原协议下的三模型对照尚未完成。
