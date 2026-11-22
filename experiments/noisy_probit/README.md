# Noisy reciprocal probit：复现实验

最新研究：[六参数方向相关通道](directional_channel/README.md)。以下为早期实验的历史说明。

这是独立于 SGDiff / Line E 的探索目录。起点是用户提供的
一份小组提供的 reciprocal-probit 报告（原文不随仓库分发），没有使用同学未提供的源代码。
研究定位、文献和最小协议见 [RESEARCH.md](RESEARCH.md)。

## 环境

本次使用 Python 3.12.15、NumPy 2.3.1、SciPy 1.17.0；pytest 用于数值测试。
所有拟合为 CPU float64，设置 BLAS 线程数为 1；进程数只影响运行时间。
无需 GPU、下载数据或访问真实网络。

从仓库根目录运行：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest experiments/noisy_probit/test_model.py experiments/noisy_probit/test_components.py -q

.venv/bin/python -m experiments.noisy_probit.run --replicates 100 --workers 4 --output results/noisy_probit/minimal
.venv/bin/python -m experiments.noisy_probit.run --config experiments/noisy_probit/extension_cases.json --replicates 100 --workers 6 --output results/noisy_probit/extension
.venv/bin/python -m experiments.noisy_probit.run --config experiments/noisy_probit/scaling_cases.json --replicates 30 --workers 4 --output results/noisy_probit/scaling

.venv/bin/python -m experiments.noisy_probit.audit_design
.venv/bin/python -m experiments.noisy_probit.profile --replicates 100 --points 161 --workers 6 --output results/noisy_probit/rho_integrated_final
.venv/bin/python -m experiments.noisy_probit.full_channel

.venv/bin/python -m experiments.noisy_probit.mcmc --input results/noisy_probit/minimal/moderate/seed202101.json --output results/noisy_probit/mcmc_moderate --warmup 2000 --draws 8000
.venv/bin/python -m experiments.noisy_probit.mcmc --input results/noisy_probit/minimal/strong/seed202101.json --output results/noisy_probit/mcmc_strong --warmup 2000 --draws 8000

.venv/bin/python -m experiments.noisy_probit.deliver
```

第一次复现可用 `--replicates 5 --output results/noisy_probit/new_smoke` 检查基础拟合；
后续脚本默认读取上述正式路径，不把 smoke 数据混入正式结果。
基础运行器验证配置/代码哈希，支持中断续跑；改变核心代码须换输出目录。
`audit_design` 和 `full_channel` 会重算其输出；不要在运行过程中修改源文件。

## 文件与方法

- `model.py`：Owen T 双变量正态 CDF、四状态概率、独立测量通道、核验似然。
- `experiment.py`：数据生成、五种拟合与同中心条件协方差消融。
- `run.py`：配对重复、配置、随机种子和核心源文件哈希。
- `report.py`：逐参数汇总、Wilson 区间、数值失败和 manifest 校验。
- `mcmc.py`：四链随机游走 Metropolis、rank-split R-hat 与 bulk ESS。
- `profile.py`：rho 一维积分，nuisance 条件 Laplace；记录数值尾部裁剪。
- `audit_design.py`：零核验和同预算观测状态分层，复用随机核验结果。
- `full_channel.py`：16 维联合拟合，允许依赖错误的完整 4x4 通道。
- `deliver.py`：验证完整正式结果、计算派生统计、生成 slides 中的表格/数据。

## 正式输出

`results/noisy_probit/` 下：

- `minimal`：3×100 场景重复；`extension`：9×100；`scaling`：1×30。
- 每个基础结果保留 mode、估计、区间、Hessian、梯度、耗时与失败信息。
- `audit_design`、`rho_integrated_final`、`full_channel`：复用正式网络的追加方法。
- `mcmc_moderate`、`mcmc_strong`：固定 seed=202101 的采样及诊断。
- `profile_grid_check_final.json`：161/321 网格和更宽左边界的固定数据诊断。
- `synthesis.json`：slides 使用的全部汇总与额外核验设计对比。

`pilot`、`profile_check*`、`rho_integrated` 是早期/中断探索，不用于正式结论。
中断的旧积分运行曾与代码编辑重叠，故整体标记为弃用；最终目录由固定版本重新全跑。

## 解释约定

1. 基础方法点估计是变换坐标 mode 映回原尺度；一维积分/MCMC 报后验均值。
2. 95% 后验区间的频率覆盖按“网络重复”计算，不能把不同参数混为独立样本。
3. 报告的 coverage 条件于成功拟合；certified_coverage 将失败计入分母。
   基础 RMSE 包含全部返回的有限点估计。积分无效拟合若存在，RMSE 只能用有效值。
4. Oracle 仅在同质独立错误场景中是真正的已知通道 oracle。在异质或相关错误压力
   测试中，它只是把基准 FP/FN 固定后的错误模型，不是正确模型上界。
5. 所有 dyads 条件独立，未模拟节点效应或网络传递性；独立通道的错误率限制在 (0,.5)。
6. 1,230 是场景--重复单元数，场景间共享随机种子，不是全部相互独立。
7. 完整通道与一维积分均为结果驱动的探索性追加，未在新保留场景验证。

## Slides

`deliverables/noisy_probit_slides/slides.tex` 是中文 Beamer 原生源文件，图形由 pgfplots
和导出的 `.dat` 构建，表格由结果自动生成。PDF 不是截图拼接；可修改文字、公式与数据。
构建和交付说明见该目录 `README.md`。
