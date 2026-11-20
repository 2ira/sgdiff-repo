# SGDiff 与 SP4LP 的干净基准核查

11 月 20 日的表把八种方法放进了统一的极小预算。它能检查数据和打分是否接通，却不足以代表训练充分时的模型排序。本次核查先逐种子读取原始组内结果，再固定查询分别改变训练数据和推断预算。两路使用同一个答案集合时，AUROC 才能直接配对。

## 组内原始记录

| 干净 AUROC | Cora | CiteSeer | 对照范围 |
|---|---:|---:|---|
| SGDiff v2，仓库 paper 档 | 0.862 ± 0.009 | 0.833 ± 0.011 | 各 10 个逐种子文件，样本标准差；仓库独立复现 |
| SGDiff v2，组内 Line C | 0.772 | 0.783 | 2,000 训练对、3 种子、每类最多 500 个测试对 |
| 完整 SP4LP，组内 Part 2 | 0.918 | 0.912 | 与 Line C 同档的干净划分、候选与监督预算 |
| SP4LP 仅端点，Part 2 | 0.890 | 0.919 | 独立训练 |
| SP4LP 无属性，Part 2 | 0.905 | 0.855 | 独立训练 |

Line C 与 Part 2 来自不同 CUDA 设备。paper 档和 Part 2 采用不同训练、评测预算，不能将对应数字当成同协议的模型差。早期 train-only 上下文的 4,000 训练对实验中，Cora 完整／端点／无属性为 0.914／0.892／0.883，PubMed 为 0.971／0.931／0.955。数据支持用户记得的那次排序。到了 Part 2 的 CiteSeer 干净条件，仅端点平均领先完整约 0.7 个百分点。

[逐种子摘录](archived_clean.csv) · [SGDiff paper 配置原始结果](../../sgdiff-paper-v2) · [Line C 原始结果](../../line-c-fsd/cuda) · [Part 2 原始结果](../../part2-aligned/cuda-local) · [早期观测实验原表](../../../experiments/course_sp4lp/historical_2026-11-11/summary.csv)

## SP4LP 控制实验

在 Cora 上重新训练 3 种子 × 2 档训练查询数 × 2 档训练轮数上限 × 3 个版本。每个种子固定干净训练图、模型宽度、学习率、验证的 128 个查询和早停规则；另对两个固定测试队列各打一次分。训练样本按同一个种子从同一训练边划分取出。每个 epoch 梯度累积后只更新一次参数。

下表沿用 11 月 20 日的 128 个测试对（64 正、64 负），报告三种子平均 AUROC。

| 训练对 | epoch 上限 | 完整 | 仅端点 | 无属性 |
|---:|---:|---:|---:|---:|
| 256 | 30 | 0.816 | 0.866 | 0.890 |
| 256 | 300 | 0.818 | 0.871 | 0.881 |
| 2000 | 30 | 0.914 | 0.888 | 0.878 |
| 2000 | 300 | 0.917 | 0.887 | 0.877 |

固定 30 轮上限，训练规模增大后，完整模型对端点版的平均差从约 −5.0 点变成 +2.7 点。只放宽轮数上限，256 对训练的完整模型收益很小。使用旧分支的 1,000 个测试对，2,000 对训练 / 30 轮时三种版本分别为 0.921、0.884、0.898，排序仍然相同。两个测试队列的大小和内容同时变化。上述实验固定了验证队列，不能把它当成重新抽取验证集的敏感性分析。两种消融都会改变网络的输入或容量。

[全部 36 次训练在两个测试队列上的得分](sp_cora_controls.csv)

## SGDiff 控制实验

先加载 11 月 20 日训练好的六个检查点。使用库中原生 `score_pairs`，以相同查询顺序、分批、随机种子和 MC=2 评分，在测试的六个检查点上逐查询复得原表分数。这样可以排除这六组干净结果因新入口把正负类颠倒或抄错 AUROC。再只改变每个分支用于评分的时间步数量，权重、节点属性和边集合保持相同。

| 原 256 对训练的检查点 | 2 步 | 4 步 | 全部时间步（结构 20、属性 100） |
|---|---:|---:|---:|
| Cora，3 种子平均 AUROC | 0.515 | 0.546 | 0.670 |
| CiteSeer，3 种子平均 AUROC | 0.552 | 0.570 | 0.713 |

实现用等间隔时间点，每个时间点抽噪声，两个类别共用噪声。取两个时间点时只覆盖首尾。增加时间步会同时改变时间点覆盖和噪声平均，当前实验不能把两者贡献拆开。六次小预算预训练中，五次在第 4 轮才达到已观察到的最低验证损失。

再在 **Cora seed 0** 单独把训练查询从 256 增到 2,000，维持子图上限 16、4+4 轮、训练 MC=2、相同 128 个验证／测试查询。测试 MC=2 时，从 **0.523 升到 0.719**；新检查点在 MC=4、全部时间步时分别为 0.754、0.729。SGDiff 按批次更新，增加训练查询也增加了每轮优化步数，因此只能把这里的变化归到整体训练预算。较大的评分预算也不会保证每个检查点都单调改善。

[逐种子控制实验](sg_controls.json)

## 复跑

从仓库根目录操作。下例使用 Cora seed 0。SP4LP 命令对其他种子重复即可。SGDiff 的 checkpoint 路径指向对应的本地诊断运行；没有检查点时可省略该参数重新训练。原始控制实验使用 CPU，环境沿用上一级报告。

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/audit_team_budget.py sp4lp --dataset cora --seed 0 --out runs/budget-check/sp0.json
python scripts/audit_team_budget.py sgdiff --dataset cora --seed 0 --checkpoint /path/to/local/sgdiff-s0/best.pt --out runs/budget-check/sg0.json
python scripts/audit_team_budget.py sgdiff --dataset cora --seed 0 --train-pairs 2000 --out runs/budget-check/sg2000.json
python scripts/run_team_study.py --profile reference --datasets cora --seeds 0 --output runs/reference-check
python scripts/summarize_team_study.py runs/reference-check/reference --datasets cora --seeds 0
```
