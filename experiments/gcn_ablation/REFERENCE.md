# SGDiff 数据集上的 GCN 链路预测消融及跨图迁移

本项目基于《Sub-graph Based Diffusion Model for Link Prediction》(Li et al., LoG 2024 / PMLR 269)。复现任务是 **GCN 链路预测基线及额外消融**，不实现 SGDiff 扩散模型。作者仓库目前仅有 README，因此这是依据论文实验协议独立实现的基线，不能声称精确复现其表中数值。

已完成 10 个随机种子、240 个模型、660 条评估记录。查看 [完整实验报告](benchmarks/report.md)、[逐种子结果](benchmarks/per_seed.csv) 和 [迁移 AUC 图](benchmarks/figures/transfer_auc.png)。

仓库包含源代码和已完成的结果摘要；原始数据、训练权重、精确划分与逐模型预测由运行脚本生成。`benchmarks/` 是已有实验记录，复现命令会创建独立的 `results/` 目录。

## 数据与原文协议

- 数据集：Cora、Citeseer、Pubmed、Router、USAir、NS。
- 引文数据来自 [Planetoid 原始数据](https://github.com/kimiyoung/planetoid/tree/master/data)，后三个来自 [SEAL 原始数据](https://github.com/muhanzhang/SEAL/tree/master/Python/data)。下载脚本固定 commit；每个原始文件的 SHA256 和 URL 保存在 `data/manifest.json`。
- 无向化、去自环、去重后按正边随机划分 80%/5%/15%。验证/测试数向下取整，余下用于训练。不使用节点分类标签，不使用 Planetoid 节点分类 mask。
- 训练、验证、测试负边各与相应正边等量；均匀从完整图的非边中无放回抽取，三个集合之间互不重叠。训练负边固定。
- 所有模型仅以训练正边构建传播邻接矩阵。验证、测试边不会参与传播或度数编码。测试时不加回验证正边。
- 三组共享每个 seed 的划分、负样本、评估候选池；保存于 `results/splits`。seed=0..9，共 10 次，报告均值和样本标准差。
- 论文表 3 的边数主要按邻接矩阵非零条目（通常双向）计数。原始字典存在重复条目；去重后 Citeseer 仍有124个自环，Pubmed有3个自环。因此清洗后的无向边数与将表中数字直接除以二不总一致。原始及清洗计数均保存。

## 图内消融 (`within`)

| 组别 | 输入 | 编码器 | 数据集 |
|---|---|---|---|
| structure | 单位矩阵 X=I，第一层权重等价于可学习节点嵌入 | 两层 GCN | 全六个 |
| feature | 行归一化的原始节点特征 X，传播矩阵 I | 两层 MLP | 三个引文网络 |
| combined | 相同原始特征 X | 两层 GCN | 三个引文网络 |

GCN：`Z = A_norm ReLU(A_norm X W0 + b0) W1 + b1`（激活后 dropout），`A_norm=D^-1/2(A_train+I)D^-1/2`。feature-only 将传播矩阵替换为 I，等价于相同深度的 MLP。所有组使用相同的内积链路解码器和 BCEWithLogitsLoss。

X=I 的参数数目受节点数影响，因此 structure-only 与属性模型参数数目并不相同，结果中记录实际参数量；不能将差异全部归因于输入信息量。

## 跨图零样本迁移 (`transfer`)

节点编号在不同图之间不对应，所以图内的可学习 ID 嵌入不能直接迁移。迁移采用单独注明的输入设置：

- 结构输入 S：训练可见图上的节点度数 one-hot，32 维，度数 >=31 截到第31类。编码规则固定，对所有图相同。
- 属性输入 P：原始属性先行归一化，乘固定高斯随机矩阵投到64维，再做 L2 行归一化。投影按 seed 和数据集名固定，源图与目标图各有自己的投影，不训练、不利用链接标签。
- structure：`GCN([S, 0])`；feature：`MLP([0, P])`；combined：`GCN([S, P])`。三组输入均为96维，隐藏层/输出层/参数量相同。
- 每个模型仅在源图训练，用源图验证 AUC 选 checkpoint，然后冻结参数，在各目标图上直接测试。目标图提供其80%的可见结构及可用属性，属于新图上的归纳式编码；没有目标图训练或目标验证调参。
- structure 提供完整6×6矩阵；feature、combined 提供3×3引文网络矩阵。Router、USAir、NS 无原始属性，属性相关单元标记 N/A，绝不将节点编号、标签、随机向量当成原始属性。
- 保留迁移矩阵对角线作为同一迁移输入协议下的图内对照。它与 `within` 的输入不同，不能直接当作同一模型比较。
- **随机投影只对齐维度，不对齐词表语义。** 属性相关迁移是该设定下的可运行基线，较弱结果可能由属性语义不对齐造成，不能据此推出特征本身无用。论文4.2也使用随机投影思路，但未提供完整 GCN 实现；本项目度数编码等选择不应被表述为作者原设定。

## 训练与指标

统一超参数：隐藏64维、输出32维、dropout=0.3、Adam lr=0.01、weight_decay=0；最多200轮，每5轮验证一次，30轮无验证AUC提升停止。所有组同一组超参数，不做测试集调参。每次训练保存最优权重、学习曲线、运行配置和目标图预测分数。零权重衰减与 GAE 原作者默认值相同；在 Cora seed0 的源图验证检查中，0 比0.0005表现更好，故正式实验统一采用0（不做逐组超参数搜索）。初步验证运行未纳入仓库或正式统计。这是一次小规模验证选择，并非完整嵌套超参数优化。

AUC 为 ROC-AUC，AP 为 Average Precision。Hits@100 定义为每条正边得分严格大于同一 split 的**全局负边池第100高得分**的比例；同分按失败处理。如果负边池少于100，Hits@100恒为1；当前六个数据集默认验证/测试划分的负边池均不少于100条。AUC用于选模型。论文未给出 Hits 的完整候选池/同分细则，本项目显式固定该定义，避免与其他定义混用。

平均迁移结果同时给出含对角线与排除对角线的版本；后者为真正跨数据集的均值。每个 seed 先在相应目标集合取均值，再跨 seed 计算统计，避免将目标组合错误当独立重复。

## 运行

需要 Python 3.10 或更高版本。首次运行前先下载数据，然后运行测试和实验。

```bash
git clone https://github.com/linjunxiong/GCN.git
cd GCN
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python experiment.py > run.log 2>&1
python summarize.py
```

单个种子/小规模检查（务必使用独立输出目录）：

```bash
python experiment.py --seeds 0 --datasets Cora USAir --epochs 20 --patience 10 --output smoke_results
```

默认 CPU 单线程，便于复现并避免小图的线程开销；支持 `--threads`。当前共享环境的旧 ONNX 与新版 protobuf 存在兼容问题，上述环境变量用于仅在本进程内兼容；pytest 变量用于关闭与本项目无关的全局插件。已完成的训练单元自动跳过，配置或训练代码变更时要求新输出目录。输出包括 `results/summary.csv`、`results/per_seed.csv`、`results/transfer_averages.csv`、`results/report.md`、`results/figures/`。报告由实际完整运行结果生成，不包含虚构数值。

`requirements.lock.txt` 记录本机实际版本，其中 PyTorch 为环境预装的 NVIDIA 构建，不保证能从普通 PyPI 重装；跨环境安装用 `requirements.txt`。已有实验的精确运行环境记录于 `benchmarks/runtime.json`。重新加载权重验证某个模型的预测：

```bash
python replay.py --protocol transfer --source Cora --mode combined --seed 0
```

实际运行中，部分种子使用独立目录的单种子进程并行计算，再由 `merge_shards.py` 校验除种子集合、输出路径外配置完全相同后合并；已有实验的原始分片配置保存在 `benchmarks/shard_configs`。串行运行默认命令可复现同一实验集合，无需并行。

相关来源：[论文作者仓库](https://github.com/hzlihang99/SGDiff)、[GCN/GAE 原作者实现](https://github.com/tkipf/gae)。
