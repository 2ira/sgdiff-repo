# 重要节点相关边的 AUROC：统一附加计分层

主报告见 [HL-GNN 重训与组件消融](../../results/hlgnn-retrained/report.md) 和 [Katz + HL-GNN 新指标](../../results/node-rank-auc/README.md)。

本模块不训练、不生成攻击、不重采样测试边。Degree / PageRank 仅由 `train_pos` 计算，按 Top k% 节点筛选至少一端命中的原测试候选边；k=0..100，k=100 必须还原原生全局 AUROC。完整约定见 [协议](PROTOCOL.zh-CN.md)。

## 不用 GPU，复算发布的全部分数

在仓库根目录创建 Python 3.11 环境。`requirements.lock.txt` 记录本次本地计分环境，GPU 训练环境另见训练工件。

```bash
python3.11 -m venv .venv-rank
.venv-rank/bin/python -m pip install -r experiments/node_rank_auc/requirements.lock.txt
.venv-rank/bin/python -m pytest -q experiments/node_rank_auc
.venv-rank/bin/python experiments/node_rank_auc/replay.py --output /tmp/hlgnn-katz-rank-replay
.venv-rank/bin/python experiments/node_rank_auc/summarize.py --results /tmp/hlgnn-katz-rank-replay/curves
.venv-rank/bin/python experiments/node_rank_auc/verify_replay.py \
  --replay-output /tmp/hlgnn-katz-rank-replay --output /tmp/replay-audit.json
```

`replay.py` 从仓库内两个发布归档恢复原逐边分数，验证 HL-GNN queries 顺序和历史攻击图哈希，为 Katz 与 HL-GNN 共 756 组预测计算两种排名 × 101 个 k。每次使用新输出目录，禁止静默覆盖。

`summary.csv` 包含全部模型与组件。`summarize.py` 自带的图展示 Katz；HL-GNN 对比与组件图由 `report.py` 绘制，发布结果包含 PNG/PDF。标准差为三个 seed 的样本标准差，非置信区间。

## 接入另一种模型

导出原测试候选边顺序对应的 `positive` 和 `negative` 分数，按 `evaluate.py` 的 manifest 输入；必须提供明确的查询顺序证据、原生全局 AUC、划分文件与攻击图哈希。`prepare_hlgnn.py` 是已验证的适配示例。任何只有汇总 AUC 的结果都无法用于子集曲线。

```bash
python experiments/node_rank_auc/evaluate.py \
  --manifest /path/to/dataset_seed0.json --output /path/to/new-curves
```

固定原排序/标签、k=100 全局还原、查询顺序核验、低样本点标记均属于验收条件。Top k% 节点不代表 Top k% 测试边；该指标补充总体 AUC，不取代它。
