# 对齐 main 的输入与攻击协议：补实验说明

本轮固定 `main=d4b150c121625001f52ead23abf15aead1a4b665`，不改写上一轮候选保护协议的结果。
范围为 Cora、Citeseer、PubMed，三个模型、完整模型及两个组件变体、种子 0/1/2：
**81 组新训练、729 次评估**。结果以输出目录的 `complete.json` 与审计为准；本文件不是完成状态声明。

## 对齐内容

- 数据使用 main 原始加载器；原始 Planetoid 文件固定在
  `af6abf468a772cc055bec88c52e4d6f51b7c37e3`。使用 main `make_link_split` 的完整
  80/5/15 划分及互斥负样本，不重新抽样供不同模型使用。
- 属性按 main `node_feature_matrix` 逐列标准化。上一轮用原始属性，不能直接复用旧权重，
  因此全部从头训练。类别标签只用于 main DICE，模型不读取类别。
- 直接调用 main `apply_attack`，在**完整图**生成 RF/DICE/EA，再移除测试正边。
  预算分母为完整图边数，不增加测试负边保护，不修补原算法中可能发生的删除/重插抵消。
- 测试直接调用 main `_cap_pairs`：每类前 500 个点对。所有模型及组件共享相同缓存。
- AUC/AP 调用 main `summarize_scores`；保留 main `hit@100` 原值，但明确它实际是池化
  Precision@100。额外记录严格全局负样本 Hits@100，避免名称混用。
- 验证复现 main 节点对基线 `train_node_pair_baseline` 的**实际代码**：在 train+val
  上下文中验证，每 10 epoch 按验证 AUC 选点，耐心 30 epoch。该上下文含验证正边；
  不能称为严格隐藏验证边的协议。上一轮独立实验的验证上下文仅含训练边。
- 冻结选定权重后只扰动推理图。输入图变化时重算 LPFormer PPR，所有组件共享图缓存。

## 保留的模型专属设置

原仓库没有这三个作者架构的原生训练器。此处调用固定版本作者模型，保留各自的数据集专属
维度、学习率、dropout、优化器与学习率衰减，最多 100 epoch；不宣称所有模型的训练器逐行相同。
共同正例 batch size 为 512，并移除批次目标正边；LPFormer 的训练图 PPR 仍可能包含批次目标信息。
不把“调用原始攻击器”扩大解释成“复现原论文全部表格”。

PubMed 使用作者现成配置，而不是直接套 Citeseer 参数：

| 模型 | 作者来源 | 本轮采用的关键配置 |
|---|---|---|
| NCNC | `ncn/README.md` Pubmed 命令 | hidden256，GCN dropout0.1，xdropout0.3，pt0.5，scale5.3，offset0.5，alpha0.3，GCN lr0.0097 / predictor lr0.002 |
| LPFormer | `lpformer/scripts/replicate_existing.sh` Pubmed 命令 | dim128，lr0.001，dropout0.3，weight decay0.0001，PPR eps1e-5，候选阈值0.01 |
| Refined-GAE | `refined_gae/ReadMe.md` PubMed 命令 | hidden512，2层传播，dropout0.4，lr0.001，GELU，两层解码器，指数初始化深度权重 |

作者原始 batch size、负样本数、随机种子数等与这里的统一协议有差异，运行配置记录所有采用值。
模型实现位于 `repository_pubmed.py`，不修改上一轮固定的作者适配器。

## PubMed 在 main 中的已有内容

- `README.md`、`docs/reproduction.md`、`results/sgdiff-paper-v2/pubmed_seed{0..9}_paper.json`
  已有 SGDiff 10 种子域内结果；`transfer_pubmed_seed{0..9}_paper.json` 等文件记录迁移。
- `configs/sgdiff_pubmed.yaml` 的 1-hop、每跳20、结构扩散10步、特征64×2及扩散50步、
  训练50 epoch/预训练10 epoch属于 SGDiff 配置，不应套到三个新架构上。
- 规范化 PubMed 为19717节点、44324无向边、500维属性；main 已说明与论文的3条边差异。
- main `EA_MAX_NODES=4000`，PubMed 的 EA **按原规则跳过**，不是未实现或漏跑。
  PubMed 每次训练评估 clean + RF三档 + DICE三档，共7个条件；另两图各10个条件。

## 图缓存与可比性

历史 main 的结果 JSON 没有保存图本体，合作者也尚未提供攻击缓存路径。本轮用固定 main
实现新生成共享缓存。相同算法、种子不保证跨硬件/依赖环境逐边相同，不能声称重放了历史图。
若要严格合表比较，应让其他模型也加载同一批图、同一组500+500测试点对，或核对其已有图指纹。

保存攻击前后两层边表：`raw_edges` 为原攻击器的完整图结果，`edges` 为移除测试正边后的推理图。
原始要求预算、完整图实际编辑、可见图实际编辑分别记录。不要把生成缓存后再移除测试边造成的
预算变化当作新攻击算法，也不要悄悄保护原协议未保护的测试负边。

## 执行

先按 main 数据布局准备 `raw/{dataset}/ind.*` 和 `processed/{dataset}.npz`，再在仓库根目录运行：

```bash
python -m experiments.new_link_models.repository_protocol prepare \
  --data-root /path/to/main-data --output /path/to/main-supplement --device cuda
python -m experiments.new_link_models.repository_protocol suite \
  --source-root /path/to/pinned-authors --output /path/to/main-supplement --device cuda --jobs 2
python -m experiments.new_link_models.repository_replay \
  --source-root /path/to/pinned-authors --output /path/to/main-supplement --device cuda
python -m experiments.new_link_models.repository_results \
  --output /path/to/main-supplement --publish experiments/new_link_models/benchmarks/repository_protocol
```

依赖与上一轮作者模型环境一致。PPR 按图的精确指纹缓存；可以复用相同图的已验证 PPR，
不能复用旧训练权重。新输出目录独立保存检查点、完整预测、训练记录和配置。
