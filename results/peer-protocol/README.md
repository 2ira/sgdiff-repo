# Katz 与 SP4LP 两个主模型：按 Lin Junxiong 的协议重新评测

**SP4LP 全量训练更新**：[新版报告、检查点及 378 组评测](../sp4lp-full-training/README.md)已补做完整训练正负对、六候选配置验证集选择、三种子与两项独立重训消融。新版继续直接调用本文固定的 Junxiong 评测器并复用相同图与查询。本文以下 SP4LP 表格保留为旧训练历史，Katz 结果继续适用。

**正式评测入口**　三个数据集均直接调用 Junxiong 的 [`experiments.new_link_models.run.evaluate`](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/run.py#L123-L169)。[本分支入口](../../scripts/evaluate_peer_protocol.py)在运行时核对评测器和协议文件与固定提交的内容一致。框架负责生成或读取攻击图、拼接固定正负查询、分批预测及计算四项指标。本组提供权重、图数据与模型打分接口。[三数据集的框架与历史分数核验](pubmed/native_evaluator_audit.json)。

## 八模型建议与已确定分工

| 分工 | 课程基线 | 研究模型 | 状态 |
|---|---|---|---|
| Lin Junxiong | — | NCNC、LPFormer、Refined-GAE | 三个模型已有鲁棒性和组件消融报告 |
| 本组 | Katz | SP4LP | 两个主模型各有完整模型和两个组件对照 |
| 建议其他成员认领 | 共同邻居（CN）、GCN | SGDiff | 三个建议席位，待组内确认 |

共八席，**三种课程基线、五种研究模型**。三种课程基线的依据如下。

- **共同邻居（CN）**：第 6 讲第 11–12 页用共同邻居说明缺失边。CN 直接数两端共同邻居，适合与 NCNC 成对分析：同样删边后，学习式共同邻居聚合和补全是否比直接计数保留更多预测能力？当测试候选边均已隐藏时，两步 Katz 恰好等于 CN 乘以同一个正常数，因此二者的 AUROC 和 AP 应一致。已在 PubMed 种子 0 的干净图与高度优先删边图上逐对核对，每图 13,298 对。主表的完整 Katz 则累计到五步。[计数实现](../../sgdiff/engine/course_baselines.py)
- **Katz**：第 6 讲第 13 页介绍按长度加权累加路径的相似度，举例为 Leicht 等人的方法。本仓库实现同一思想的截断 Katz，完整配置累计到五步游走。[课程方法实现说明](../../docs/course_sp4lp.md)
- **GCN**：第 10 讲第 13–14 页介绍多轮邻居聚合。仓库用 GCN 编码节点、用点积预测连接，可以观察删边或打乱节点属性对标准消息传递的影响。[GCN 与点积实现](../../sgdiff/models/baselines/node_pair.py)

SGDiff 保留主论文的直接比较对象。研究模型覆盖 2024 年的 [NCNC](https://openreview.net/forum?id=sNFLN3itAd)、[LPFormer](https://github.com/HarryShomer/LPFormer)，以及 [Refined-GAE（CIKM 2025）](https://github.com/GraphPKU/Refined-GAE)、[SP4LP（2025 年预印本）](https://arxiv.org/abs/2507.07138) 和 [SGDiff（2025 年论文集）](https://proceedings.mlr.press/v269/li25a.html)。普通 GAE 的早期结果保存在[观测缺失实验归档](../paired-reproduction/README.md)。

## Cora 与 CiteSeer 的横向比较

我们直接调用 Lin 分支固定提交的 evaluate、split_data、perturb、metrics。Katz 含 H=2/3/5；SP4LP 含完整、仅端点、无原始属性。2 张图 × 3 个种子 × 6 个配置 × 14 个条件 = **504 组实测结果**。所有 AUROC、AP、Hits@100、Precision@100 均从新保存的逐对预测复算。Cora 每种子有 792 正、792 负，CiteSeer 有 683 正、683 负。每个种子的六份原始正负划分与本组训练使用的完整划分逐元素一致。

SP4LP 复用本组在各自干净训练划分上训练的 18 套权重（训练查询上限 2,000，最多 300 轮）。Lin 的三种学习模型用其原批次目标边遮蔽、最多 100 轮的训练配方。这里统一测试查询、指标和扰动规则，各模型保留自己的训练预算。Katz 直接数游走。**SP4LP 的训练还采用宽度 64、每轮一次参数更新，训练充分性缺少逐轮日志证据，跨模型排名须结合这些差异解释。** [训练配置核查](five-models/README.md#训练配置对排名的影响)。

**五模型在第一版协议下的全部 14 个条件已对齐。** 本次复用 Junxiong 第一版的 18 张 EA 原图和此前的模型权重，重评两个方法及各自消融，共 108 组。原图文件逐字节一致，6 份划分逐元素一致；本组 504 组结果的边指纹全部匹配第一版审计。其余 396 组非 EA 记录保留原值。Cora、CiteSeer 在 EA 50% 下，完整 SP4LP 的三种子平均 AUROC 分别为 0.8953、0.8948，完整 Katz 为 0.8214、0.7745。[五模型对比、组件结果和核验明细](five-models/README.md)。

表中是三个种子的平均 AUROC（0–1）。Lin 的报告表以百分数展示；引用前需统一量纲。

| 数据集 | 完整方法 | 干净 | 随机删边 50% | 高度优先删边 50% | 特征整行打乱 |
|---|---|---:|---:|---:|---:|
| Cora | Katz | 0.8434 | 0.6590 | 0.6278 | 0.8434 |
| Cora | SP4LP | 0.9141 | 0.8361 | 0.8228 | 0.8024 |
| CiteSeer | Katz | 0.7778 | 0.6444 | 0.5740 | 0.7778 |
| CiteSeer | SP4LP | 0.9133 | 0.8969 | 0.8970 | 0.6236 |

Katz 在两张图上都对高度优先删边更敏感。CiteSeer 上 SP4LP 对两类删边的平均分相近，而打乱论文和词向量的对应关系会明显降低表现。本组旧版 21 条件实验采用低度优先删边、属性置零与联合缺失，回答的是另一种观测机制下的问题。

[逐种子指标和边指纹](per_seed.csv) · [均值、样本标准差、降幅与配对组件差](summary.csv) · [划分、特征和权重指纹](provenance.json)

## PubMed 扩展

本组另在 PubMed 完成 Katz 和 SP4LP 的完整配置与四种组件对照，共 252 组已核验结果。三个种子的平均干净 AUROC 分别为 0.8256、0.9666；高度优先删边 50% 后为 0.7046、0.9210。PubMed 的 14 个条件包含 H200 生成的 EA 10%、25%、50%。[九张可复用 EA 图及固定测试划分](pubmed/pubmed_ea_graphs.zip) · [PubMed 扩展报告、权重与预测](pubmed/README.md)。

## 同学报告的消融是否充分

按课程项目的组件分析要求，报告有完整的模型内对照。每个完整模型对应两个独立重训变体，2 张图、3 个种子、14 个条件都有实测。我们以逐种子表核对了 54 条训练记录、756 个唯一评测单元，以及 168 行配对汇总中的 AUROC/AP/Hits@100 均值与样本标准差。独立核对基于对方发布的表格；其检查点重载验证由对方提供审计记录。

| 模型 | 两个组件对照 | 可支持的结论与边界 |
|---|---|---|
| NCNC | 去补全；改用端点预测头 | 干净 Cora 上，补全贡献平均约 +0.06 个 AUROC 百分点，三个种子有正有负。端点预测头仍使用图编码，不能把差异全部归因于共同邻居数量。 |
| LPFormer | 去 PPR 位置编码；将学习的注意力改为均匀权重 | 干净 Cora 去位置编码平均损失约 0.98 个百分点，三个种子同向。去位置编码仍保留 PPR 候选筛选。均匀权重保留候选和值投影。 |
| Refined-GAE | 去 LayerNorm；用点积替换 Hadamard-MLP 解码器 | 干净 Cora 上，完整解码器相对点积平均提高约 1.57 个百分点，三个种子同向。去归一化同时涉及编码器和解码器；解码器替换也改变容量，两者都只能解释整项改动。 |

两数据集三种子的单组件对照足以报告本设置的机制差异；组件交互、跨数据集泛化或统计显著性需要额外证据。[模型组件定义](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/MODEL_GUIDE.zh-CN.md) · [原始报告及配对表](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/benchmarks/report.md)

## 复算

从本分支根目录执行。使用已安装本仓库依赖的 Python 环境，PubMed 原始数据位于项目的数据根目录。请用新的输出目录，以免复用历史 CPU EA 预测。先解压本组各数据集已有的权重，再运行同一个入口。三个数据集都由 Junxiong 的原生 `evaluate` 驱动，训练配方仍保留各方法原有设置。

```bash
git fetch origin feat/ncnc-lpformer-refined-gae
git worktree add --detach ../lin-eval 95083da808e500c02da458e205f4b861ceb80871
(cd ../lin-eval && python -m experiments.new_link_models.prepare --root ../lin-peer-data)
for archive in results/paired-reproduction/seven-methods/archives/*_seed*.zip results/peer-protocol/pubmed/pubmed_seed*.zip; do
    unzip -oq "$archive" 'runs/pair/*' -d .
done
unzip -oq results/peer-protocol/pubmed/pubmed_ea_graphs.zip -d runs/pubmed-cache
unzip -oq results/peer-protocol/predictions-and-graphs.zip 'attacks/*' -d runs/lin-first-report
python scripts/evaluate_peer_protocol.py --peer ../lin-eval --peer-data ../lin-peer-data --data-root data --output runs/lin-first-report --tables runs/peer-tables --pubmed-ea-cache runs/pubmed-cache/attacks
python scripts/verify_pubmed_extension.py --peer ../lin-eval --output runs/lin-first-report/pubmed --tables runs/peer-tables/pubmed --ea-cache runs/pubmed-cache/attacks
```

[Cora 与 CiteSeer 的逐对预测、测试划分和实际使用的攻击图](predictions-and-graphs.zip) · [PubMed 归档与复用说明](pubmed/README.md)。
