# NCNC / LPFormer / Refined-GAE：统一鲁棒性与重要节点 AUROC

2026-11-26 实验更新，交付分支 `feat/ncnc-lpformer-refined-gae`。对齐 `unified-robustness-ablation@bef9674` 的 Katz / 重训 HL-GNN，使用原生评测器 `95083da808e500c02da458e205f4b861ceb80871`。

- [完整中文报告（先图后表）](report.md)
- [PDF报告](report.pdf) · [TeX主文件](report.tex) · [可独立编译的TeX源包](report_sources.zip)
- [代码、协议边界与无需GPU的复算说明](../../experiments/new_models_rank/README.md)
- [原重要节点协议](https://github.com/2ira/sgdiff-repo/blob/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/node-rank-auc/PROTOCOL.zh-CN.md) · [HL-GNN报告](https://github.com/2ira/sgdiff-repo/blob/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/hlgnn-retrained/report.md)

Cora/Citeseer复用54个历史候选保护协议模型与756组真实预测，PubMed新增27次拟合与378组评估。81个模型涵盖三个数据集、三个模型、每模型三种组件版本、三个seed；每个模型14条件，共1134组实际预测。117张结构图、9份split、查询顺序及两种节点排名与已有Katz/HL-GNN匹配。新增229068行逐seed条件曲线，包括k0的NA点。

|文件|内容|
|---|---|
|`per_seed.csv` / `robustness_summary.csv`|1134行原生逐seed指标 / 378行三seed全局汇总|
|`paired_component_deltas.csv`|Full−Variant，先同seed相减，252行全局组件效应|
|`summary.csv` / `delta_summary.csv`|全部k的AUROC与扰动−clean配对变化|
|`key_points.csv`|k=1/5/10/20/50/100|
|`conditional_component_deltas.csv`|全部条件、全部k的50904行配对组件效应|
|`five_model_full_summary.csv`|Katz、重训HL-GNN与三个完整模型的统一条件汇总|
|`per_seed_curves.zip`|229068行原始曲线、9份冻结排名、逐输入审计|
|`predictions.zip`|1134份实际逐边分数、9份完整split、可移植manifest及源指纹|
|`training_records.zip`|81份训练history、配置、receipt及新PubMed日志|
|`training_summary.csv` / `learned_parameters.csv`|训练选择与可描述的分支/传播深度参数|
|`checkpoints.json`|本机持久权重路径、大小和SHA256；权重不提交进Git|
|`delivery_audit.json` / `replay_audit.json`|完整矩阵、k0/k100与无GPU逐点复算核验|
|`new_checkpoint_replay_audit.json`|27份新权重独立重载、81组预测核验|
|`reused_checkpoint_replay_audit.json`|历史54份权重独立重载核验记录|
|`runtime.json` / `code_manifest.json` / `SHA256SUMS`|环境、实现与文件指纹|
|`verification.json` / `portable_compile_audit.json`|全部表格、17图、67页PDF与独立源包编译核验|

这是对同一批预测的附加计分：仅用干净训练图冻结Degree/PageRank，从原正负测试边共同筛出至少一个端点命中Top-k%节点的边。不按预测分数挑选、不重采负例、不在每个k攻击。k100等于全局AUROC，k0为NA；低支持点明确标记，三seed误差是样本SD。

本轮复用的EA图实际均为插边，不能据此代表删边鲁棒性；随机删边和高度相关删边为独立条件。新PubMed使用原始属性、train-only验证、6649正+6649负测试候选，与此前main补实验的标准化特征和500+500候选结果分开。这里只宣称统一组内协议的架构复现，不宣称复刻模型论文原表，也不作三seed的强显著性或因果泛化声明。

共享计分器和冻结排名/汇总已一并引入本分支，来源文件与原始提交逐字节核验，见`integration_audit.json`。原生评测器通过`peer.py`绑定固定95083da checkout，适配本分支已有同名模型包的情况。
