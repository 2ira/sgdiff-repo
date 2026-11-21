# Katz 与 SP4LP 两个主模型：按 Lin Junxiong 的协议重新评测

## 八模型建议与已确定分工

| 席位 | 方法 | 状态 |
|---|---|---|
| 1–3 | NCNC、LPFormer、Refined-GAE | Lin Junxiong 已完成三种方法的鲁棒性与组件消融 |
| 4–5 | Katz、SP4LP | 本组负责的两种主模型，各有完整模型和两个组件对照 |
| 6 | SGDiff | 建议由其他成员负责。作为主论文模型，可用结构分数、属性分数及融合分数研究两类证据如何变化；算力需求最高。 |
| 7 | 属性余弦（Feature Cosine） | 建议由其他成员负责。它只用两端的词向量，固定测试对后删边不会改变分数，特征打乱则改变分数。无需训练；如需组件对照，可去掉范数归一化，比较原始点积。 |
| 8 | SEAL-style | 建议由其他成员负责。在和 SGDiff 相近的包围子图上做判别式预测，可重训去 DRNL 角色标签、去节点属性两个版本，与路径模型及扩散模型形成对照。 |

后三席是推荐选项，尚待组内认领。本次实验范围仅 Katz 和 SP4LP。普通 GAE 的既有结果保存在[观测缺失实验归档](../paired-reproduction/README.md)。仓库已有[SEAL-style 实现](../../sgdiff/models/baselines/seal.py)和[属性基线实现](../../sgdiff/engine/course_baselines.py)。

## Cora 与 CiteSeer 的横向比较

我们直接调用 Lin 分支固定提交的 evaluate、split_data、perturb、metrics。Katz 含 H=2/3/5；SP4LP 含完整、仅端点、无原始属性。2 张图 × 3 个种子 × 6 个配置 × 14 个条件 = **504 组实测结果**。所有 AUROC、AP、Hits@100、Precision@100 均从新保存的逐对预测复算。Cora 每种子有 792 正、792 负，CiteSeer 有 683 正、683 负。每个种子的六份原始正负划分与本组训练使用的完整划分逐元素一致。

SP4LP 复用本组在各自干净训练划分上训练的 18 套权重（训练查询上限 2,000，最多 300 轮）。Lin 的三种学习模型用其原批次目标边遮蔽、最多 100 轮的训练配方。这里统一测试查询、指标和扰动规则，各模型保留自己的训练预算。Katz 直接数游走。

**合并表格时选取图指纹一致的条件。** 每种子 11 个非 EA 条件的边指纹都与对方归档相同；特征打乱还复用了同一版本的特征置换代码、节点顺序与种子。三档 EA 用同学原代码在 CPU 上生成，18 张 EA 图均与对方发布的指纹不同。EA 可以在这次生成的同一张图内比较 SP4LP 和 Katz；与 Lin 的三模型做逐图对照时应使用其原始 EA 边缓存。总计 396 组来自边指纹一致的条件，108 组属于单列的 CPU EA。[同学的逐图审计](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/benchmarks/protocol_audit.json)

表中是三个种子的平均 AUROC（0–1）。Lin 的报告表以百分数展示；引用前需统一量纲。

| 数据集 | 完整方法 | 干净 | 随机删边 50% | 高度优先删边 50% | 特征整行打乱 |
|---|---|---:|---:|---:|---:|
| Cora | Katz | 0.8434 | 0.6590 | 0.6278 | 0.8434 |
| Cora | SP4LP | 0.9141 | 0.8361 | 0.8228 | 0.8024 |
| CiteSeer | Katz | 0.7778 | 0.6444 | 0.5740 | 0.7778 |
| CiteSeer | SP4LP | 0.9133 | 0.8969 | 0.8970 | 0.6236 |

Katz 在两张图上都对高度优先删边更敏感。CiteSeer 上 SP4LP 对两类删边的平均分相近，而打乱论文和词向量的对应关系会明显降低表现。本组旧版 21 条件实验采用低度优先删边、属性置零与联合缺失，回答的是另一种观测机制下的问题。

[逐种子指标和边指纹](per_seed.csv) · [均值、样本标准差、降幅与配对组件差](summary.csv) · [划分、特征和权重指纹](provenance.json)

## 同学报告的消融是否充分

按课程项目的组件分析要求，报告有完整的模型内对照。每个完整模型对应两个独立重训变体，2 张图、3 个种子、14 个条件都有实测。我们以逐种子表核对了 54 条训练记录、756 个唯一评测单元，以及 168 行配对汇总中的 AUROC/AP/Hits@100 均值与样本标准差。独立核对基于对方发布的表格；其检查点重载验证由对方提供审计记录。

| 模型 | 两个组件对照 | 可支持的结论与边界 |
|---|---|---|
| NCNC | 去补全；改用端点预测头 | 干净 Cora 上，补全贡献平均约 +0.06 个 AUROC 百分点，三个种子有正有负。端点预测头仍使用图编码，不能把差异全部归因于共同邻居数量。 |
| LPFormer | 去 PPR 位置编码；将学习的注意力改为均匀权重 | 干净 Cora 去位置编码平均损失约 0.98 个百分点，三个种子同向。去位置编码仍保留 PPR 候选筛选。均匀权重保留候选和值投影。 |
| Refined-GAE | 去 LayerNorm；用点积替换 Hadamard-MLP 解码器 | 干净 Cora 上，完整解码器相对点积平均提高约 1.57 个百分点，三个种子同向。去归一化同时涉及编码器和解码器；解码器替换也改变容量，两者都只能解释整项改动。 |

两数据集三种子的单组件对照足以报告本设置的机制差异；组件交互、跨数据集泛化或统计显著性需要额外证据。[模型组件定义](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/MODEL_GUIDE.zh-CN.md) · [原始报告及配对表](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/benchmarks/report.md)

## 复算

需要本仓库的实验依赖、已解压的 [SP4LP 权重归档](../paired-reproduction/seven-methods/archives)、本组的数据目录，以及同学的固定提交。以下示例从本分支根目录执行。

```bash
git worktree add --detach ../lin-eval 95083da808e500c02da458e205f4b861ceb80871
(cd ../lin-eval && python -m experiments.new_link_models.prepare --root ../lin-peer-data)
python scripts/evaluate_peer_protocol.py --peer ../lin-eval --peer-data ../lin-peer-data --data-root data --output runs/lin-eval --tables results/peer-protocol
python scripts/summarize_peer_protocol.py results/peer-protocol
```

[504 组逐对预测、测试划分和实际使用的攻击图](predictions-and-graphs.zip)。解压到上述命令的 output 目录可复用已核验的 CPU 攻击图。
