# HL-GNN 工件恢复性重训与交付说明

本轮为恢复丢失的模型权重和逐边预测而重训，不重新做模型选型或参数搜索。保持历史 `readme_exact` 配方、同数据划分、同负例、同原生评测器、同攻击图。重训结果单独发布，旧汇总保留在 `results/hlgnn-historical/`。

## 服务器及准确环境

- SSH 连接信息保存在本地 `final_project/node_rank_auc/RETRAINING_PROGRESS.md`；公开仓库只记录运行环境与持久工件路径，不包含登录凭据。
- GPU：单张 NVIDIA H200，143771 MiB。
- 本轮实际解释器：**`/usr/bin/python`，无额外虚拟环境**。Python 3.12.3，PyTorch `2.7.0a0+ecf3bae40a.nv25.02`，PyG 2.5.3，NumPy 1.26.4，SciPy 1.12.0，scikit-learn 1.5.2。完整版本以 `training/environment.json` 为准。
- 持久化根目录：`/nas/jiatong/data8002/hlgnn_retrain_20261124`。
- 代码：`code/`；原生评测器：`peer/`，固定提交 `95083da808e500c02da458e205f4b861ceb80871`。
- 数据：`data/`，Planetoid 原始文件来自提交 `af6abf468a772cc055bec88c52e4d6f51b7c37e3`，逐文件 SHA256 见 `data_receipt.json`。
- 权重：`training/models/{dataset}_readme_exact_{variant}_s{seed}/best.pt`。
- 原生评测：`evaluation/runs/{dataset}_{variant}_s{seed}/`，含 `queries.npz`、`receipt.json`、`predictions_*.npz`、`evaluation_*.json`。
- 日志：`suite.log`、`training/logs/`；状态：`training/status.json`；完成标志：`training/TRAINING_COMPLETE` 只代表训练结束，评测还应检查 `evaluation/audit.json` 的 378 格。

## 重训合同

3 数据集 × 3 既定种子 × 3 版本 = 27 次独立拟合。正边固定 80% / 5% / 15% 划分，train / validation / test 各自配等量固定、互不重叠的负例。每轮训练使用全部固定正负对，正例 batch=512，批次正边从传播图移除。Adam lr=0.001、weight_decay=0、gradient clipping=1，禁用 TF32。每 5 epoch 用干净验证 AUROC 选最佳；最多 150 epoch；15 个验证检查无改善时早停。其余参数见模型 `configuration()` 和每个运行 `receipt.json`。

完整版 full：20-hop 可学习跳权；fixed_weights：20-hop 跳权冻结；shallow_k2：2-hop 可学习跳权。三个版本独立训练，没有推断时临时关掉组件。RWR 在 K=2 时末跳初值为 0.64，在 K=20 时第 2 跳为 0.128；浅层配置因此也包含初始化尾项变化，不将结果解释为完全隔离的纯深度效应。

历史 seed 0 配置选择已经完成，本轮固定使用胜出的 readme_exact。先分别跑每个数据集 full seed 0，与历史验证 AUROC 作差异诊断（绝对差超过 0.005 时暂停检查）；这不是使用测试集挑选重训次数。通过后继续原 seed 1/2 和两个组件变体。保存所有拟合结果，不挑好 seed。

## 复用旧攻击，杜绝重生成

先运行 `scripts/stage_hlgnn_archived_inputs.py` 导入公开原图：Cora/CiteSeer 来自 `results/peer-protocol/predictions-and-graphs.zip`；PubMed 来自相应 `pubmed_seed{seed}.zip`。图和固定特征置换都核验历史指纹。117 张结构缓存图 = 3 数据集 × 3 seeds × 13 个非特征条件；另有 9 个确定性 feature_shuffle 条件，每个模型均评 14 条件。

`bind_peer` 核对 Git 提交与 `run.py`、`protocol.py` 原文。新脚本只负责排程、工件读取和审计，模型推断仍进入 Junxiong 原生 `run.evaluate`。PubMed 归档的 `graph_*.npz` 只含边与置换，导入时补充缓存元数据并核验边哈希，不运行攻击算法。

## 一次重训入口

以下路径对应本次实际服务器目录（将代码根目录置于当前目录）。原生 peer checkout、data 和攻击缓存需先准备并核验。

```bash
cd /nas/jiatong/data8002/hlgnn_retrain_20261124/code
/usr/bin/python -u scripts/retrain_hlgnn_archived.py \
  --peer ../peer --data-root ../data \
  --historical results/hlgnn-robustness/training_metadata.json \
  --output ../training --evaluation ../evaluation
```

发布代码使用 `results/hlgnn-historical/training_metadata.json` 保存相同历史记录；重跑发布仓库时改用此路径。`training/selection.json` 从历史 `selection.json` 复制；它是历史选择记录，不代表本轮重跑过候选配置。

## 保存与发布

GitHub 保留同一个 `unified-robustness-ablation` 分支，发布代码、模型环境、训练日志、所有逐边预测、数据/图/权重哈希、完整结果表、新指标曲线与 Markdown 报告。大模型权重留在 NAS，发布路径和 SHA256；关掉容器前须确认 `/nas` 为持久挂载。

本轮不重复 SGDiff、SP4LP 或队友模型的训练，也不声称重新验证 HL-GNN 原论文指标。Katz 为已有分数重算，HL-GNN 为新训练新推断；报告明确区分二者。

调度补充：三个 seed-0 验证门槛均通过后，在同一 H200 上最多并行 3 个独立拟合进程。`worker.lock` 对每个数据集/版本/seed 互斥，主队列不会与辅助队列重复训练；训练函数 AST 与恢复代码完全一致，见 `training_source_audit.json`。

## 从发布仓库准备相同输入

已有公开攻击图 ZIP 在仓库 `results/peer-protocol/` 和 `results/peer-protocol/pubmed/`，可把它们链接到同一个 `archives/` 目录。下载数据时若需要代理，在 shell 加载用户的 VPN 脚本；不把代理凭据写入代码。

```bash
python scripts/prepare_hlgnn_retrain_data.py \
  --workspace /path/to/new-retrain \
  --peer /path/to/peer-at-95083da \
  --historical results/hlgnn-historical
python scripts/stage_hlgnn_archived_inputs.py \
  --peer /path/to/peer-at-95083da \
  --data-root /path/to/new-retrain/data \
  --archives /path/to/archives \
  --historical results/hlgnn-historical/per_seed.csv \
  --output /path/to/new-retrain/evaluation
```

辅助调度参数 `--prefetch-seed 1` / `--prefetch-seed 2` 只在全部三个 full seed-0 诊断通过后允许执行，且共享每个拟合的互斥锁。它不额外增加 seed 或训练版本。默认单队列也能完成全部实验。
