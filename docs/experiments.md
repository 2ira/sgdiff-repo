# 实验总览：跑过什么、结果在哪、怎么复现

**新增三模型实验（2026-11-18）**：NCNC / LPFormer / Refined-GAE 的完整结果见
[完整实验报告](../experiments/new_link_models/benchmarks/report.md)。共 54 组训练、756 次评估，
报告展开全部攻击预算、随机/按度删边、插边、特征打乱、组件消融和逐种子指标。
该实验采用独立统一协议；下文为原有 SGDiff v2 实验记录。

> **本页只记录 `sgdiff-paper-v2`（修正实现）的实验。** 2026-11-13 审计确认
> 早期实现不忠实（categorical diffusion/后验错误、feature denoiser 不符
> Eq. 11、两类噪声独立、训练调度非论文默认），旧实现的全部产物已于
> 2026-11-15 删除；旧结论一律作废，不再在本页保留。

全部实验于 2026-11-14/15 在 8×NVIDIA H20 上以 `--profile paper`
（严格 Table 4 配置：全量训练/测试划分、子图不截断、全部 `ds_t` 蒙特卡洛
步、两阶段训练）完成。入口调度：`scripts/repro_queue.sh`（in-domain）与
`scripts/rq_queue.sh`（RQ 网格），flock 保护的作业列表并行分发。

## 1. 复现层（6 组实验）

| 实验 | 协议 | 规模 | 产出 |
|------|------|------|------|
| in-domain 网格 | Sec. 4.1，10 seeds | 5 数据集 × 10 seeds = 50 runs | `results/sgdiff-paper-v2/{ds}_seed{K}_paper.json` |
| RQ1 零样本迁移 | Sec. 4.2，structure-only 训练，source_ns | 5 源 × 10 seeds = 50 runs | `transfer_{src}_seed{K}_paper.json` |
| RQ2 1% 低数据 | Sec. 4.3，`train_ratio=0.01` | cora × 10 seeds | `lowdata_cora_seed{K}_paper.json` |
| RQ3 SGDiff 攻击 | Sec. 4.4，RF/DICE/EA × 0.1/0.25/0.5，evasion | cora × 3 seeds | `attack_cora_seed{K}_paper.json` |
| RQ3 全模型鲁棒性 | 同上 + 7 模型 zoo | cora × 3 seeds | `robustness_cora__v2_seed{K}.{json,md}` |
| baselines | 同一 split 同一测试对 | gcn/gat/sage/vgae/seal | `baselines_cora.json` |

每个 JSON 内嵌完整 protocol 指纹（损失模式、DRNL 细节、特征维度、
子图分布 marginals 等）；checkpoint 与 metrics 全部在
`runs/sgdiff-paper-v2/`（in-domain 的 cora 另有 `_bak` 备份）。

## 2. 关键数字（详见 `reproduction.md` §0）

* in-domain AUC：cora 0.862 / citeseer 0.833 / pubmed 0.902 / usair 0.868 / ns 0.965
* RQ1：citeseer→cora 0.834、citeseer→ns 0.957、usair→pubmed 0.799 等
* RQ2：1% cora AUC 0.711 ± 0.092
* RQ3：EA@25% 仅损失 2.4 点（0.858→0.834）；SEAL/PENCIL 绝对值更高

## 3. 没做、做不了的

* **Router** 数据不可分发，矩阵只覆盖论文 6 图中的 5 个。
* **EA 攻击**是本地线性代理，非作者/DeepRobust 实现，Table 2 EA 列不构成
  对齐目标。
* **`target_ns` 迁移变体**（假设目标图子图统计已知）未在 paper profile 重跑。
* **似然消融**（`run_ablation.py`）与**机制归因探针链**（`run_probe.py` 等）
  尚未在 v2 上重跑；工具就绪，checkpoint 在 `runs/sgdiff-paper-v2/`。

## 4. 复现命令

见 [`reproduction.md`](reproduction.md) §3；单条命令即可重跑任一格子。

## 5. 建议阅读顺序

1. [`reproduction.md`](reproduction.md) §0 —— 全部数字与协议
2. [`robustness.md`](robustness.md) —— RQ3 全模型表与损伤分析
3. [`benchmark.md`](benchmark.md) —— 数据一致性对比合约与 v2 基线
4. [`findings.md`](findings.md) —— 按论文主张逐条给结论
5. [`alignment.md`](alignment.md) —— 论文公式 → 代码的逐条映射
