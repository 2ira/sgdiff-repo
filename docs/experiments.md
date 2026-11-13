# 实验总览：跑过什么、在哪分析、看到什么

> **历史实现记录，不能据此判断原论文。** 2026-11-13 已核对原文并修正实现为
> `sgdiff-paper-v2`；下表仍为旧实现的产物，相关 SGDiff 结论须重新跑。
> 此前“论文 clean AUC”的 0.884 / 0.871 / 0.951 实为 Table 2 RF@25% 数值，
> Figure 2 是 1% 训练；所谓差 9–15 点的直接比较已撤回。新结果单独写入
> `results/sgdiff-paper-v2/`，不混入本页旧产物。

本仓库的全部实验分两层。**单一入口是 `results/probe_report.html`**：第 00 节是复现层，
第 01-15 节是归因层，数字全部由 `scripts/make_probe_report.py` 从 `results/` 里的产物现算，
所以不会和 JSON 漂移。本文档是同一份清单的 markdown 版（便于 diff 与 grep）。

| 层 | 问的问题 | 结论入口 | 逐实验细节 |
|---|---|---|---|
| **复现层**（能不能复现论文） | 论文的 C1-C4 成立吗？模型在这个预算下什么水平？ | [`findings.md`](findings.md)（1 页，claim by claim） | [`reproduction.md`](reproduction.md)、[`benchmark.md`](benchmark.md)、[`ablation.md`](ablation.md)、[`robustness.md`](robustness.md) |
| **归因层**（为什么） | 每个模型在读图的哪个性质？伤害从哪进、能不能修？ | [`attribution.md`](attribution.md) §8 | [`attribution.md`](attribution.md) §1-§7；HTML 第 01-15 节 |

下面两张表就是这两层的完整清单。数值均可从 `results/` 里同名产物复算；"详述"列给出读得下去的那一节。

## 1. 复现层（9 个实验）

注：**逐模型 × 逐数据集**的单元格（Cora / Citeseer / USAir × 7 模型，格内附"该模型 − GCN"的差）
见 `results/probe_report.html` 00.2；Pubmed / NS 只跑了 SGDiff，所以那两列只有一行。
论文 Table 1 包含逐模型迁移值，Table 2 包含逐模型攻击值，Figure 2 是低数据实验；
不能将这些值与全量训练 clean 相减。HTML 00.2 仅展示旧实现各模型间的差，
不是与论文逐格差；必须先匹配实验协议后再比较。

| # | 实验 | 脚本 / 产物 | 观察 | 详述 |
|---|---|---|---|---|
| 1 | in-domain 网格，5 数据集 × 3 种子 | `run_experiments.py` → `summary_indomain_{full,cpu}.json` | 旧实现 clean：Cora 0.730±0.030、Citeseer 0.778、Pubmed 0.827（seed 0）、USAir 0.929、NS 0.985。此前论文参照来自不同协议，已撤回逐格差与“仅属性图存在缺口”的推断 | `reproduction.md` §3 |
| 2 | 同数据、同切分的 7 模型对比 | `run_benchmark.py` → `benchmark_cora.{json,md}` | SGDiff AUC 0.731 **垫底**，但 AP 第二（0.793）、Hit@100 第一（0.950）；代价是 2.28 M 参数（≈40× SEAL）和 104 s（≈100×）。`full` 预算下 SEAL 在同一批 1-hop 子图上能到 0.855，所以差距不是数据 | `benchmark.md` §5 |
| 3 | 似然审计（打分时 + 重训） | `run_ablation.py` → `ablation_cora[_full].{json,md}` | 条件被真用（`same_condition` 0.718 → 0.495）；**深度是装饰**（单步 0.730 / 最贵步 0.728）；DRNL 标签无信息（打乱后 0.729 vs 0.718）；特征流在随机水平（`feature_only` 0.467，方差占比 4.3e-6）；随机条件重训时冻结去噪器 0.475 vs 不冻结 0.720 → 起作用的是融合层的交叉熵 | `ablation.md` §3-§6 |
| 4 | 鲁棒性网格，7 模型 × RF/DICE/EA × 3 预算 × 3 种子 | `run_robustness.py` → `robustness_cora.{json,md}` | SGDiff 相对掉 **−19 % / −19 % / −6 %**，是唯一跌破 0.70 的模型；GCN −6/−8/−1。论文 C3（"似然打分对结构攻击鲁棒"）方向相反 | `robustness.md` §2-§4 |
| 5 | 防御对照：RGCN(2019) + GDDM + GPR-GAE，配对跑 | `run_robustness.py --defense ... --tag purify2` → `robustness_cora_purify2.{json,md}` | 18 个受防御格中 17 个低于对应未防御格，**包括 clean**（唯一例外 RGCN + GPR on clean，+0.001，在种子噪声内）；原因是可测的：GPR 把边精度 0.72 提到 0.94 的代价是删掉一半真边，而 RF/DICE 是删除主导的扰动 → 删除主导的净化补不回来。RGCN clean 比 GCN 低 7.7 点，方差惩罚只动 ≤0.006 | `robustness.md` §5 |
| 6 | RGCN 训练预算敏感性 | `run_robustness.py --tag rgcn1000` → `robustness_cora_rgcn1000.{json,md}` | 1000 epoch 不改变 §5 的结论（"没训够"不是解释） | `robustness.md` §5.3 |
| 7 | RQ1 零样本迁移，5×5 矩阵（Router 不可得 → 缺第 6 个数据集） | `run_experiments.py --transfer-matrix` → `transfer_matrix_cpu.json` | 目标均值比同数据 in-domain 低 1-13 点（源均值 0.713-0.838）；作为源 Citeseer 0.838 最好、NS 0.713 最差，**近似 in-domain 排序的镜像**（小图准确源过拟合自己的度分布）；最差单元 NS→Pubmed 0.494 = 随机。`source_ns` 还是 `target_ns` 最多移动 5.6 点 → "无需重训"里藏着一个逐数据集旋钮 | `reproduction.md` §4 |
| 8 | RQ2 1 % 标签 | `run_experiments.py --lowdata` → `summary_lowdata_cora.json` | 40 epoch 下 0.725±0.058 vs 全量 0.751±0.017（成立）；8 epoch 崩到 0.49 → **claim 只在长 schedule 下成立**，而论文没报 schedule | `reproduction.md` §4 |
| 9 | 全图 baseline 对照 | `run_baselines.py` → `baselines_cora_full.json`、`baselines_cora_seal.json` | 同预算下 GAT 0.846 / GCN 0.811 / VGAE 0.798 / SAGE 0.749，SEAL 在 `full` 1-hop 上 0.855 —— 没有任何配置让 SGDiff 的 AUC 领先 | `reproduction.md` §5 |

## 2. 归因层（1 个主网格 + 12 个子实验）

主网格先用同一套探针把"谁读什么"钉住，后面每个子实验都是对它的一个追问或反证。

| # | 实验 | 脚本 / 产物 | 观察 | 详述 |
|---|---|---|---|---|
| A | 主网格：7 模型 × 7 探针，统一 50 % 预算，5 种子 | `run_probe.py` → `probe_cora.{json,md}` | SGDiff 在 `del` 0.729→**0.493**、`del_hub`→0.565（最脆）；GCN/GAT/VGAE/SAGE 在 `feature_shuffle` 崩到 0.637/0.622/0.597/0.581；PENCIL/SEAL 对特征打乱相对稳（0.748/0.733）；SGDiff 在 `ins_hub` 反而**升**到 0.779 | `attribution.md` §3 |
| B | 同探针搬到无属性无标签的 USAir | `run_probe.py --dataset usair` → `probe_usair.{json,md}` | 同样 50 % 预算伤害小得多；`ins_hub` 的伤害跟"度归一连通性"走 —— 在无属性图上，度通道就是属性通道 | `attribution.md` §5 |
| C | 跨图稳定性（3 个不同性质的图） | `probe_citeseer/usair.{json,md}` | 排序与符号一致，量级按 §5/§7 的机制预测偏移 | `attribution.md` §6 |
| D | 层级定位（每层漂移 + 干净层上训的线性探针） | `run_deep_probe.py` → `deep_cora.{json,md}` | 属性伤害发生在**第一次聚合之前**（`feature_shuffle` L1 漂移 ≈1.0，可读信号 0.839→0.614）；结构伤害在**第二跳**进入（`del`：L1 几乎不动，L2 0.839→0.809）；GAT 的 L2 掉得是 GCN 的两倍 | §7.1 |
| E | 换传播算子（同权重、只改推理期算子） | `deep_*` | 我原先的假设**错了**：对称归一化不是鲁棒性载体（sym 0.790 / row-mean 0.777 / 未归一 0.773 相差 <0.02）；载体是**自环**（去掉 −0.072）。USAir 上自环优势消失（−0.004）→ 这条是 Cora 的，不是 GNN 的 | §7.2 |
| F | 删三角边 vs 删桥边（同预算） | `deep_*` | 桥边不亚于三角边（每个模型都是），"三角是证据"在这一粒度不成立；USAir 只有 83 条桥边 → 该列退化，作为零结果连同适用条件一起报 | §7.3 |
| G | 修复实验（因果检验：同一干预，看是否只救被预测的模型） | `deep_*` | CN 补边只救 SGDiff（+0.113，且高于 CN 启发式自身的 0.572）；结构平滑只救属性模型（GCN +0.054 / SAGE +0.065 / VGAE +0.058，SGDiff 恰好 0.000）；CN 先验加进分数几乎无效（+0.000~0.013）。同行不同号，这是全文最干净的证伪检验 | §7.4 |
| H | 模型侧修复：k=2 重训 | `deep_*` | 抬天花板不抬地板：SEAL clean 0.781→0.856，但删除缺口 0.141→**0.189**（Citeseer 更狠，0.086→0.182）→ "跳数不够"不是脆弱的原因 | §7.5 |
| I | 组件级重训网格（8 种子 × 3 图，一次只改一个部件） | `run_redesign.py` → `redesign_cora.{json,md}` | 唯一显著变差的是 `noloop`（`del` 0.056→0.114）；`depth-L3` 降结构伤害但**升**特征伤害（0.167→0.225）；`+cnfeat+smooth` 只把特征伤害压到 0.130（clean 不掉）。没有单一部件能同时改善两侧 | §7.6 |
| J | 运行点：伤害是分层还是整体平移 | `run_score_dump.py` → `run_operating_point.py` → `operating_cora.{json,md}` | 是**分层**：SGDiff 在 CN 存活的半数上 −0.15σ、CN→0 的半数上 **+2.18σ**；GCN/VGAE 两半都平（−0.12/−0.04）→ 它们的删除伤害不是证据饥饿，这解释了 §7.6 为什么补结构对它们无效。另一个反例：SGDiff `ins_hub` AUC **+0.023** 却 TPR@1 %FPR **−0.262** | §7.7 |
| K | 自适应攻击（贪婪 score-feedback，匹配预算） | `run_adaptive_attack.py` → `adaptive_cora.{json,md}` | 固定探针是**下界**：10 % 自适应删除 = 固定 50 % 探针伤害的 5.2×（Cora GCN）、16.7×（Citeseer SAGE）；20 % 预算把 GCN 从 0.826 打到 0.51（随机探针 50 % 都到不了 0.04） | §7.8 |
| L | 零模型校准（负结果） | `run_evidence_calibration.py` → `calibration_cora.{json,md}` | 结构先验能解释 SGDiff 分数的 R²=0.656、GCN 0.301；但减去先验只沿前沿移动、从不抬高。根因是结构性的：Cora **54 % 的正样本本来 CN=0**（负样本 99 %），"证据消失"不是可观测异常 | §7.9 |
| M | 配对 bootstrap（误差棒到底多大） | `run_bootstrap_ci.py` → `significance_cora.{json,md}` | 0.12-0.21 量级的 AUC 差异在 pair 重采样下 95 % 区间排除 0；0.020 量级的差异被显式标为不稳健 | §7.10 |
| N | 证据路由修复（负结果） | `run_gated_repair.py` → `gated_cora.{json,md}` | 路由买到了 k=2 的天花板（clean +0.06），但继承其删除缺口：比 k=1 差 **−0.067 [−0.097, −0.037]**。原因是门在需要它时退化：`CN=0` 覆盖 clean 的 77 %、删除后的 93-94 % | §7.11 |
| O | 反事实拼接门（负结果，§7.11 自己提出的最后一个候选） | `run_splice_gate.py` → `splice_cora.{json,md}` | 拼接一个合成共享邻居后 Δ>0 覆盖 **97-100 %** 的 pair，且 Δ 预测逐 pair 掉分的 AUC 只有 **0.445-0.531**（随机）→ 对"响应"设阈和对"状态"设阈一样饱和。至此"per-pair 门"路线两个方向都关闭 | §7.12 |

## 3. 没做、做不了的

| 项 | 状态 | 影响 |
|---|---|---|
| `paper` 预算（无上限 1-hop、全 MC 步） | **未跑**：M 系列 CPU 上 >2.5 h/模型 | §4 的"深度是装饰"只对 `cpu`/`full` 成立，对论文 schedule 未验证 |
| 第 6 个数据集（Router） | 不可分发 → 迁移矩阵是 5×5 | RQ1 的"零样本"少一列 |
| 特征 DDPM 有效的情形 | 未发生：MSE ≈1.0（预测 0） | 论文的"贝叶斯分解"在本机不可观测 |

## 4. 已清理的低价值产物

以下实验已判定为无对应结论或已被取代，产物与脚本输出已删，**不要重跑**：`ablation_hops2`（旧 hops 探针，被 deep probe §7.5 取代）、`*_synthetic_smoke` 与 `robustness_cora_seedcheck`（单种子自检）、`attack_cora_*_seed0_cpu`（被 `summary_attack_cora` + robustness 网格取代）、`transfer_*_seed0_cpu` 单次迁移（被 5×5 矩阵取代）、以及无引用的运行日志。

## 5. 建议阅读顺序

1. [`findings.md`](findings.md) §1 —— 四条 claim 的判决表（5 分钟）。
2. 本文件 §2 的 A/E/G/J/K 五行 —— 归因链的骨架：读什么 → 载体是什么 → 因果检验 → 伤害在哪 → 固定探针低估了多少。
3. [`attribution.md`](attribution.md) §8 —— 这些机制如何改写鲁棒性叙事。
4. 要成稿或分享，直接开 `results/probe_report.html`（自包含，无 CDN）。
