# 三个作者模型：统一鲁棒性、组件消融与重要节点 AUROC

这层扩展复用已有 `experiments/node_rank_auc/evaluate.py` 和固定原生评测器，保持 Katz / HL-GNN 的既有训练与计分模块不变。结果放在 `results/new-models-node-rank/`；本次更新发布到 `feat/ncnc-lpformer-refined-gae`，协议来源为 `unified-robustness-ablation@bef9674`。

- Cora/Citeseer：复用第一轮候选保护协议54份权重及756份真实逐边分数。
- PubMed：27次新训练、378组14条件评估，使用统一分支已发布的图，绝不重生成EA。
- 三模型共有1134份预测，Degree/PageRank × k0..100共229068行逐seed曲线。排名、查询和图均与已发布Katz/HL-GNN核验。
- 新计分不加载模型、不需要GPU。`frozen_rank.py`在核验训练图和中心性数值后复用已有排名，以避免不同依赖版本交换PageRank近并列节点；输出排名CSV必须与既有交付逐字节相同。

## CPU复算已发布分数

Python3.10+，需要 NumPy/SciPy/scikit-learn/NetworkX；不需要Torch、PyG、作者代码或GPU。使用空输出目录：

```bash
python -m experiments.new_models_rank.replay \
  --predictions results/new-models-node-rank/predictions.zip \
  --rank-archive results/node-rank-auc/per_seed_curves.zip \
  --expected results/new-models-node-rank/per_seed_curves.zip \
  --output /tmp/new-model-rank-replay
```

它逐文件校验分数指纹，核对固定查询和训练图，重新运行原计分模块，逐行比较229068个点的AUROC、类别数、低支持与NA状态。k100必须匹配原全局AUROC。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  experiments/node_rank_auc/test_evaluate.py \
  experiments/node_rank_auc/test_hlgnn_adapter.py \
  experiments/new_models_rank/test_frozen_rank.py
```

## 重做PubMed训练

需要固定作者源码与原生评测器95083da的独立checkout（不要修改其文件）。作者版本及安装依赖沿用此前三模型实现；PubMed适配器在 `pubmed_models.py`。本分支已有 `experiments/new_link_models`，因此 `peer.py` 将固定peer路径放到导入搜索顺序最前，并逐文件核验实际加载的原生模块；使用新的Python进程运行训练入口，避免提前导入另一版本。已有模型代码不作替换。

固定原生checkout中的 [requirements](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/requirements.txt) 和 [完整安装说明](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/README.md) 包含PyG/DGL/Numba依赖及NVIDIA ABI注意点。作者源码固定为：NCNC `11d597013750da17ce7468e344bec756a7af39a4`，LPFormer `3ca80aba3bd1ba4903ec5c042608e4a034110c9f`，Refined-GAE `5bcf00ee42134a31995bbc7bcc5359810aa99b6c`。从该peer checkout运行：

```bash
python -m experiments.new_link_models.sources --root /path/to/pinned-authors
```

该命令下载固定commit并验证所有Python文件的SHA256。三个模型构造时也会逐文件检查，禁止静默使用新的作者版本。仅CPU分数复算不需要这些源码或训练依赖。

先使用 `experiments.new_models_rank.stage_graphs`（从原归档导入脚本适配） 导入117张归档图和9份划分，按其现有说明把 `predictions-and-graphs.zip` 与三个 `pubmed_seed*.zip` 放入同一archives目录。数据必须匹配HL-GNN已发布的原始属性与划分指纹。 原图来自[已发布Cora/Citeseer归档](https://github.com/2ira/sgdiff-repo/blob/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/peer-protocol/predictions-and-graphs.zip)和[PubMed归档目录](https://github.com/2ira/sgdiff-repo/tree/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/peer-protocol/pubmed)中的三个`pubmed_seed*.zip`。下载后放入同一个archives目录；仅CPU复算不需要下载这些图。

```bash
python -m experiments.new_models_rank.stage_graphs \
  --peer /path/to/peer-at-95083da --data-root /path/to/data \
  --archives /path/to/archives \
  --historical results/new-models-node-rank/per_seed.csv --output /path/to/evaluation
python -m experiments.new_models_rank.suite \
  --peer /path/to/peer-at-95083da --data-root /path/to/data \
  --source-root /path/to/pinned-authors --output /path/to/evaluation --jobs 2
```

`prepare_ppr.py`可预计算PPR并复用图指纹完全一致的缓存，不能复用旧main模型权重。每个训练配置有独立worker锁；辅助队列只改变调度，不增加seed或重复拟合。训练仍调用固定原生 `fit`，测试调用固定原生 `evaluate`，并将任何新攻击生成尝试设为错误。

三个模型均保留完整模型及两项组件：NCNC noCNC/noCN，LPFormer Uniform/noPE，Refined-GAE noLN/Dot。PubMed使用作者专属配置，raw属性、train-only验证、完整6649正+6649负测试；不是旧main的列标准化、验证正边可见或500+500候选实验。LPFormer训练PPR仍可能带批次目标信息，属于保留作者实现的已披露限制。

## 原始结果适配、汇总与报告

`prepare_scores.py`核验旧54个模型的源码/划分、所有权重指纹与实际预测，再生成附带原始顺序证据的manifest。`finish.py`在27个新模型完成后调用它、原计分模块、通用汇总、制图和归档。它不调用Git提交或推送。`replay_models.py`独立重载新检查点，在clean、EA50、feature_shuffle验证保存分数。

`summary.csv`的扰动变化为 perturbed−clean；组件表为 Full−Variant，正负含义不同。所有差值先同seed相减，再求样本SD。主AUROC图纵轴固定0–1，低支持以空心点显示；训练曲线另行标注放大轴。完整源数据/训练记录/指标及全部k保留，报告不以汇总均值造曲线。

原EA是线性代理近似，实际27图全为插边；删边表现单独从delete/delete_hub读取。冻结重要节点子集不能消除原随机负例的度偏差，也不是天然更优的总体指标。

## TeX与PDF

`tex_report.py --output results/new-models-node-rank`从真实CSV生成完整TeX表；该目录下运行`tectonic report.tex`，或XeLaTeX→BibTeX→XeLaTeX两遍。使用CTeX/Fandol中文字体。`report_sources.py --results results/new-models-node-rank`打包17张PDF图、文献、全部已展开表格与编译说明；`report_sources.zip`可独立解压编译，不需GPU或原数据。训练记录和逐边预测分别在另外两个归档中，模型权重保留在本机，仓库只提供其路径、大小和SHA256清单。

报告编译完成后，可额外安装`PyMuPDF`，运行`python -m experiments.new_models_rank.verify_report`：核对全部表行、图表、归档完整性、源包字节、PDF嵌入字体、文本页面边界和TeX警告。输出`verification.json`及实现指纹；该核验不启动训练、不更改Git。`report.log`保留编译记录。
