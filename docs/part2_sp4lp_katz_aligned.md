# Part 2：SP4LP 与 Katz 对齐鲁棒性实验

本研究在团队 Line C 的统一 benchmark 上重做两个互补模型：可训练的局部最短路径模型 SP4LP，以及免训练的全局衰减路径计数 Katz。四套主实验已经完成：Cora/Citeseer × 完整属性/固定 40% 整节点属性缺失；每套包含 3 个 seed、clean 与 RF/DICE/近似 EA 的 10%/25%/50% 攻击。

## 1. 对齐协议与证据边界

- 数据集为 Cora、Citeseer，seed 为 0/1/2。每个 seed 共享 edge split、最多 2,000 个训练 pair、500 正 + 500 负测试 pair、攻击图和候选 pair。
- 攻击是 evasion：模型只在 clean graph 上训练一次，测试时冻结参数；结构攻击后重新构造 GCN 输入和最短路径。RF/DICE/EA 均使用 10%/25%/50% edge-flip budget。
- SP4LP 最多训练 300 epoch，学习率 0.001，以 clean validation AUROC 每 5 epoch 检查、patience=6。Katz 不训练。
- 本机无 CUDA，本结果使用同一个 `cpu` budget profile 但实际在 CPU 执行；团队 Line C 数字在 CUDA 执行。因此可比较协议和趋势，但不能把两批数无标记地合并成同硬件显著性结论。
- SP4LP 是宽度 64 的仓库适配版，不是论文 Cora 宽度 256 + HeaRT negatives 的完整复现；EA 是仓库的 embedding-surrogate 近似攻击，不是外部库的原版攻击。

自动审计：

- cora/observed: 210/210 cells, finite metrics, 3 seeds
- cora/missing40: 210/210 cells, finite metrics, 3 seeds
- citeseer/observed: 210/210 cells, finite metrics, 3 seeds
- citeseer/missing40: 210/210 cells, finite metrics, 3 seeds
- cora: attack metadata shared; Katz and featureless are exactly feature-mask invariant
- citeseer: attack metadata shared; Katz and featureless are exactly feature-mask invariant
- total: 840/840 distinct model/seed/condition cells

## 2. 两个模型到底在学什么

### SP4LP

SP4LP 把 link prediction 拆成两条信息流。第一条用一层 GCN 把节点属性与一跳邻居聚合成 endpoint embedding；第二条在移除待预测边后，用 BFS 找到两个端点之间的精确最短路径，把路径上的节点 embedding 作为序列交给一层 Transformer。最终把 endpoint 对表示与 path 表示拼接，经 MLP 输出边分数。它的偏置是：一条边是否存在，既取决于端点自身/邻域，也取决于连接二者的最短结构证据。

`sp4lp_endpoints` 删除整个 path tokenizer/Transformer，只保留同一个 GCN 与 endpoint MLP，用来回答“显式最短路径序列是否真的有用”；`sp4lp_featureless` 把原始属性替换为常数 1，但保留 GCN、路径与 Transformer，用来回答“提升来自节点属性还是纯结构”。两者都重新独立训练，不是把完整 checkpoint 临时关掉组件。

需要留意：路径构造会隐藏 query edge；但与常规 full-graph link prediction 一样，训练正边仍存在于 GCN 的训练图中。这个设定与团队 harness 一致，却不等价于严格 inductive 或全边遮蔽训练。

### Katz

Katz 对候选节点对 `(u,v)` 计算 `Σ_l β^l (A^l)[u,v]`：`A^l` 是长度为 l 的 walk 数，β 让长 walk 递减。这里 query edge 已从测试图移除，所以 H2 基本就是缩放后的 common-neighbour 证据；H3 加入三跳 walk，H5 再加入四、五跳 walk。β 取 `0.85/max_degree`，保证对应无限级数收敛；`katz_decay50` 仅把分子改为 0.50。Katz 完全不读属性、没有学习参数，并在每个 attacked graph 上重新计算，因此测的是结构本身被扰动后的可恢复性，而不是冻结神经网络对 distribution shift 的适应。

## 3. 主结果

表中括号是相对本模型 clean AUC 的绝对下降；均值 ± sample standard deviation 来自 3 个 seed。

### cora: 完整属性

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| `sp4lp` | 0.919 ± 0.004 | 0.858 ± 0.014 (−0.061) | 0.837 ± 0.007 (−0.083) | 0.897 ± 0.009 (−0.022) |
| `sp4lp_endpoints` | 0.891 ± 0.010 | 0.851 ± 0.013 (−0.040) | 0.843 ± 0.009 (−0.048) | 0.868 ± 0.015 (−0.023) |
| `sp4lp_featureless` | 0.905 ± 0.007 | 0.809 ± 0.019 (−0.096) | 0.784 ± 0.013 (−0.122) | 0.850 ± 0.006 (−0.056) |
| `katz` | 0.851 ± 0.007 | 0.768 ± 0.008 (−0.083) | 0.755 ± 0.013 (−0.097) | 0.786 ± 0.016 (−0.065) |
| `katz_h3` | 0.807 ± 0.006 | 0.731 ± 0.010 (−0.076) | 0.715 ± 0.001 (−0.091) | 0.764 ± 0.008 (−0.043) |
| `katz_h2` | 0.729 ± 0.003 | 0.653 ± 0.007 (−0.076) | 0.642 ± 0.005 (−0.087) | 0.704 ± 0.003 (−0.025) |

### cora: 40% 整节点属性缺失

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| `sp4lp` | 0.873 ± 0.005 | 0.792 ± 0.005 (−0.081) | 0.770 ± 0.012 (−0.102) | 0.843 ± 0.015 (−0.029) |
| `sp4lp_endpoints` | 0.832 ± 0.005 | 0.765 ± 0.014 (−0.067) | 0.766 ± 0.001 (−0.067) | 0.798 ± 0.006 (−0.035) |
| `sp4lp_featureless` | 0.905 ± 0.007 | 0.809 ± 0.019 (−0.096) | 0.784 ± 0.013 (−0.122) | 0.850 ± 0.006 (−0.056) |
| `katz` | 0.851 ± 0.007 | 0.768 ± 0.008 (−0.083) | 0.755 ± 0.013 (−0.097) | 0.786 ± 0.016 (−0.065) |
| `katz_h3` | 0.807 ± 0.006 | 0.731 ± 0.010 (−0.076) | 0.715 ± 0.001 (−0.091) | 0.764 ± 0.008 (−0.043) |
| `katz_h2` | 0.729 ± 0.003 | 0.653 ± 0.007 (−0.076) | 0.642 ± 0.005 (−0.087) | 0.704 ± 0.003 (−0.025) |

### citeseer: 完整属性

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| `sp4lp` | 0.913 ± 0.011 | 0.884 ± 0.012 (−0.029) | 0.877 ± 0.012 (−0.036) | 0.890 ± 0.014 (−0.023) |
| `sp4lp_endpoints` | 0.922 ± 0.009 | 0.900 ± 0.012 (−0.022) | 0.892 ± 0.013 (−0.030) | 0.903 ± 0.005 (−0.019) |
| `sp4lp_featureless` | 0.856 ± 0.017 | 0.747 ± 0.014 (−0.109) | 0.747 ± 0.008 (−0.109) | 0.800 ± 0.012 (−0.056) |
| `katz` | 0.776 ± 0.009 | 0.710 ± 0.012 (−0.067) | 0.702 ± 0.006 (−0.075) | 0.725 ± 0.007 (−0.051) |
| `katz_h3` | 0.756 ± 0.010 | 0.685 ± 0.011 (−0.072) | 0.686 ± 0.006 (−0.070) | 0.710 ± 0.002 (−0.046) |
| `katz_h2` | 0.667 ± 0.008 | 0.603 ± 0.006 (−0.064) | 0.606 ± 0.012 (−0.062) | 0.661 ± 0.011 (−0.007) |

### citeseer: 40% 整节点属性缺失

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| `sp4lp` | 0.842 ± 0.003 | 0.794 ± 0.007 (−0.048) | 0.792 ± 0.007 (−0.050) | 0.804 ± 0.001 (−0.039) |
| `sp4lp_endpoints` | 0.832 ± 0.007 | 0.797 ± 0.011 (−0.035) | 0.786 ± 0.013 (−0.046) | 0.805 ± 0.011 (−0.027) |
| `sp4lp_featureless` | 0.856 ± 0.017 | 0.747 ± 0.014 (−0.109) | 0.747 ± 0.008 (−0.109) | 0.800 ± 0.012 (−0.056) |
| `katz` | 0.776 ± 0.009 | 0.710 ± 0.012 (−0.067) | 0.702 ± 0.006 (−0.075) | 0.725 ± 0.007 (−0.051) |
| `katz_h3` | 0.756 ± 0.010 | 0.685 ± 0.011 (−0.072) | 0.686 ± 0.006 (−0.070) | 0.710 ± 0.002 (−0.046) |
| `katz_h2` | 0.667 ± 0.008 | 0.603 ± 0.006 (−0.064) | 0.606 ± 0.012 (−0.062) | 0.661 ± 0.011 (−0.007) |

## 4. SP4LP 组件消融

下面是严格按 seed 配对后的 AUC 差：正数表示完整 SP4LP 更好。

### 显式最短路径分支：`sp4lp − sp4lp_endpoints`

| Dataset / attributes | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| cora / observed | +0.028 | +0.007 | -0.007 | +0.029 |
| cora / missing40 | +0.040 | +0.026 | +0.005 | +0.046 |
| citeseer / observed | -0.009 | -0.016 | -0.015 | -0.013 |
| citeseer / missing40 | +0.010 | -0.003 | +0.006 | -0.001 |

结论：路径分支在 Cora clean 和 EA 下有稳定可见增益，属性缺失时增益更大；但在 Citeseer 完整属性下反而平均更差。它不是普适增益组件，当前较小预算下可能因为路径序列较噪或 Transformer 优化方差而输给更简单的 endpoint head。不能只用 Cora 宣称路径模块全面有效。

### 原始节点属性：`sp4lp − sp4lp_featureless`

| Dataset / attributes | Clean | RF 50% | DICE 50% | EA 50% |
|---|---:|---:|---:|---:|
| cora / observed | +0.014 | +0.049 | +0.053 | +0.047 |
| cora / missing40 | -0.033 | -0.018 | -0.014 | -0.006 |
| citeseer / observed | +0.056 | +0.137 | +0.130 | +0.090 |
| citeseer / missing40 | -0.014 | +0.047 | +0.045 | +0.003 |

结论：完整属性在 Citeseer 尤其重要，并显著缓冲 50% RF/DICE；Cora 也在强攻击下受益。可是把 40% 节点特征直接置零后，full SP4LP 的 clean AUC 在两数据集都低于 featureless，说明残缺特征可成为噪声，简单地保留零行不一定胜过明确不用属性。Citeseer 强攻击下残余属性仍有帮助，说明“是否用属性”与“攻击强度”存在交互。

## 5. Katz 组件消融


| Dataset | H2 clean | H3 clean | H5 clean | H3−H2 | H5−H3 | H5(0.85)−H5(0.50) |
|---|---:|---:|---:|---:|---:|---:|
| cora | 0.729 | 0.807 | 0.851 | +0.078 | +0.044 | -0.0000 |
| citeseer | 0.667 | 0.756 | 0.776 | +0.089 | +0.020 | -0.0000 |

结论：主要收益来自加入更长路径，而不是在 0.50–0.85 范围内调衰减。H2→H3 在 Cora/Citeseer 分别增加约 0.078/0.089，H3→H5 再增加约 0.044/0.020；两种 β 的排序几乎完全相同，所以 AUROC 基本一致。这不表示 β 数值无影响，而是本候选集上它没有改变多少 pairwise ranking。

## 6. 与团队 benchmark 的关系

本分支从 Line C benchmark commit 派生，复用了相同数据拆分、缺失掩码规则、攻击生成、候选 pair、预算和 JSON schema，因此实验设计已经对齐。团队 CUDA 表中 Cora/Citeseer 完整属性的 SEAL clean AUC 为 0.866/0.861；本次 CPU SP4LP 为 0.919/0.913，Katz H5 为 0.851/0.776。强攻击下 SP4LP 的绝对 AUC 也普遍较高。不过 SP4LP 与团队学习模型不在同一设备上训练，因此这些数字只作为同协议上下文；正式总表应增加 hardware/provenance 列，或让团队在同一 CUDA 环境重跑本分支后再做严格排序。

## 7. 整体发现与下一步

1. **效果上**：SP4LP 是这两个模型中更强的主模型；Katz H5 是稳定、透明、免训练的结构基线。
2. **鲁棒性上**：EA 对本次 SP4LP/Katz 通常比 RF/DICE 温和；应同时报告 attacked AUC 和 clean drop，不能只看下降量。
3. **组件上**：路径深度对 Katz 明确重要；SP4LP 的显式路径分支具有数据集依赖；完整节点属性在强结构攻击下尤其有价值，但简单零填充的残缺属性会伤害 clean 表现。
4. **成本上**：Katz 每 seed 约 0.1–0.2 秒；完整 SP4LP 单 seed 的训练均值/最大值约为 Cora 34.5/51.8 秒、Citeseer 113.2/175.4 秒。本次是正常训练并受 early stopping 控制，不是论文级宽度和完整负采样预算，所以不应与“跑一天”的 full-paper run 等同。
5. **若要继续强化**：优先在团队 CUDA 环境原样重跑 3 个 SP4LP 配置（full/endpoints/featureless，必要时另加宽度 256），并增加多重 missingness 比例或更合理的 feature imputation；不要再无目的重复 Katz。

## 8. 可复现产物

- 原始结果：`results/part2-aligned/cpu/robustness_{dataset}_{observed|missing40}_cpu.{json,md}`
- 全条件 tidy 表：`results/part2-aligned/cpu/part2_aligned_auc.csv`
- 配对消融表：`results/part2-aligned/cpu/part2_aligned_ablations.csv`
- 重新生成本报告：`python scripts/analyze_part2_aligned.py`
