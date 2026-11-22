# 查新与实验登记：带误报/漏报的互惠 probit 网络

检索日期：2026-11-21。本目录是新实验，独立于 SGDiff / Line D / Line E。
起点是用户提供的 `chapter3_report.pdf`，而非该报告未提供的源代码。
报告的数值只作为背景，不当作本实现的验证结果。

## 查新结论

不能主张“首个把误分类加到 probit / 网络模型的方法”。目前检索发现：

| 文献 | 已有贡献 | 与本实验的关系 |
|---|---|---|
| Hausman, Abrevaya, Scott-Morton (1998), *Misclassification of the dependent variable in a discrete-response setting*, JoE 87:239–269, DOI 10.1016/S0304-4076(98)00015-3 | 二元响应误报/漏报的参数化似然与识别条件 | 噪声通道不是新方法；单次观测不等于必然不可识别，非线性、协变量支持和单调性限制重要 |
| Meyer & Mittag, *Misclassification in Binary Choice Models*, NBER w20509 | 系统性误分类的偏差，核验数据与误报模型的价值 | 需要异质噪声压力测试，不能只验证独立同分布翻转 |
| Newman (2018), *Estimating network structure from unreliable measurements*, PRE 98:062321 | 网络测量误差下的结构与统计量推断 | 网络噪声建模已有完整框架 |
| Peixoto (2018), *Reconstructing networks with unknown and heterogeneous errors*, PRX 8:041011 | SBM 结构先验支持未知/异质误差、单次网络观测推断 | 不能主张未知误差或单网络推断是首次；与本模型的目标参数和假设不同 |
| Young, Cantwell & Newman (2021 publication, 2020 issue), *Bayesian inference of network structure from unreliable data*, J Complex Networks 8:cnaa046 | 一般观测模型下的贝叶斯网络重建 | 本研究专门关注协变量效应和 dyad 互惠参数，不作通用重建优越性声称 |
| Candelaria & Ura (2022 online), *Identification and inference of network formation games with misclassified links*, DOI 10.1016/j.jeconom.2022.07.007 | 误分类网络形成博弈的识别集与推断，涉及 reciprocity，允许协变量相关错误 | 最接近网络文献；其 reciprocity 是网络效用统计量，不等于本模型的潜在高斯相关系数 rho |
| Giordano, Broderick & Jordan (2018), *Covariances, Robustness, and Variational Bayes*, JMLR 19(51) | LRVB 协方差和局部敏感性 | 不能声称 LRVB 或线性响应是新增通用理论 |
| Lotspeich et al. (2022 preprint revision / CJS article DOI 10.1002/cjs.11772), *Optimal multi-wave validation of secondary use data with outcome and exposure misclassification* | 两阶段核验、似然推断和多轮优化核验设计 | 联合利用核验数据与目标方差选样也已有方法；本实验是向互惠 dyad 的具体适配 |

已阅读主要全文：Hausman 参数化识别段、Candelaria/Ura 引言与假设、Peixoto
测量模型与未知误差讨论、Lotspeich 两阶段似然和核验设计；其他条目以原始论文
摘要/出版记录定位。检索同时覆盖 2024–2026 与 bivariate probit / reciprocity /
misclassified links / validation / Bayesian network measurement error 组合。
这不是穷尽式系统综述；未找到完全相同实现不等于证明不存在。

## 可检验增量

将四状态 dyad 的精确边缘似然、少量核验和噪声率不确定性传播结合，测量
rho 的偏差—覆盖率—核验成本关系；进一步检验以 rho 信息为目标的核验能否
优于同预算随机核验。定位为可复现的方法适配与实证边界研究，暂不声称新理论。

1. 已知误报/漏报率可否纠正 naive rho 偏差？
2. 只有少量 gold-standard dyads 时，联合推断是否比固定估计噪声率更可靠？
3. 在相同联合后验 mode 上，条件协方差和边缘协方差差多少？这一步隔离
   噪声率不确定性传播，不与均值变化混淆。
4. 核验预算、样本规模、稀疏性、强互惠、异质错误会如何改变收益或暴露失败？

## 最小实验（在正式结果前登记）

- n=60，beta=(1,-0.5,2)，rho=0.8。不同无序 dyad 独立，与原报告模型一致。
- 三种噪声：clean (0,0)、moderate (FP=.05,FN=.20)、strong (FP=.10,FN=.30)。
- 随机核验 5% dyads，每个核验同时揭示两个方向的真实边。审计标签不供给
  naive/oracle 方法；clean 参考可见全部真边，明确不同信息预算。
- 100 个独立网络，配对基础随机数。方法包括 clean reference、naive、known-rate
  oracle、audit-rate plug-in、joint audit、joint mode with conditional covariance。
- 主要指标逐参数 bias、RMSE、95% 覆盖率和区间宽度；另报优化/Hessian 失败率，
  不能静默丢弃失败。覆盖率给二项 Wilson 区间；样本为网络而非参数总和。
- 固定先验、起点策略和优化阈值。所有方法使用相同数值概率实现。
- beta 使用原报告形式的 g-prior，rho 原生尺度均匀；噪声率分别服从 0.5×Beta(2,8)。
  在 atanh(rho)、logit(2×错误率) 坐标优化并计入 Jacobian。数值搜索限制
  atanh(rho)∈[-4,4]、logit(2×错误率)∈[-12,12]，撞界单列为失败。
  核验 plug-in 使用 Jeffreys 平滑的误分类频率，并明确不传播该估计的不确定性。
- 对数值 CDF、解析梯度、噪声矩阵和核验似然做独立测试；选定典型数据对
  collapsed posterior 做多链采样检验 Laplace，避免把近似误差误称模型鲁棒性。

## 扩展规则

先检查最小实验能否区分 naive 与 known-rate oracle，以及 joint 的数值稳定性。
确认后扩到 n=30/120/300、稀疏截距、rho=.2/.95、1%/10% 核验预算与
协变量相关或方向相关错误。错误模型不匹配的结果保留为负结果。
追加联合零核验和同预算选样对照。只做明确标记的仿真实验，不伪装真实应用。

## 结果驱动的追加分析（探索性，不属于最初登记）

1. 最小实验中 Joint 的 rho 覆盖率仅 85%，故追加 rho 一维积分 + nuisance
   条件 Laplace。该方法是已有近似思想的具体应用，不声称新算法。
   最终输出为 `rho_integrated_final`，161 点；点估计采用后验均值。
   默认 eta 区间 [-2,4]；极负 rho 的 CDF 尾部消减会使优化不稳定，因此对
   峰值附近的连续有效网格积分，只在切点相对密度 <1e-7 时裁去失败尾部，
   左边界相对密度须 <1e-6。所有裁剪均记录；这不是严格尾部概率上界。
   两个固定诊断数据集比较 161/321 点及左界 -2/-2.5，并与四链 MCMC 比较。
2. 核验对照实现为“观测四状态均衡分层”，不是信息最优核验。没有证据证明
   优于随机核验，因此目标信息驱动设计保留为未来问题，不作为已完成贡献。
3. 方向相关错误导致独立通道 Joint 的覆盖率为零后，追加完整 4x4 测量通道。
   每个真实状态对应一行 Dirichlet(.5,.5,.5,.5)；12 个自由 logits 加 beta/rho
   共 16 维，包含变换 Jacobian，联合估计。相同 5% 核验与相同网络上配对比较。
   同时保留独立错误和协变量异质错误对照；完整通道仍假设对 x 同质。
   本实验不能把整体收益唯一归因于通道结构，因维数和先验也随之变化。

## 当前证据的边界

- 1,230 个基础/扩展场景--重复单元（300+900+30），不是 1,230 个全部相互独立
  的数据集：场景间共享种子，保留配对。每个场景内网络独立。
- 额外 400 次核验设计拟合、200 次一维积分、300 次完整通道拟合，使用已有网络。
- 数值失败保留；摘要覆盖率条件于成功，另报失败计入分母的 certified coverage。
  n=300 的独立通道 Joint 全部成功，但部分基线未通过绝对梯度阈值。
- 组件表现来自同一开发轮次的模拟；未在保留场景或真实数据外部验证。
  未推导识别定理、最优核验定理、一般网络依赖下的频率覆盖保证。
- 不使用已中断的 `rho_integrated`、早期 `profile_check*`、旧 pilot 作为正式结果。
  最终代码及哈希只匹配 `rho_integrated_final`；中断探索记录保留以便审计。

## 来源链接

- https://economics.mit.edu/sites/default/files/2022-09/Misclassification%20of%20the%20Dependent%20Variable%20in%20a%20Discrete-Response%20Setting.pdf
- https://www.nber.org/papers/w20509
- https://arxiv.org/abs/1803.02427
- https://arxiv.org/html/1806.07956
- https://academic.oup.com/comnet/article/8/6/cnaa046/6161493
- https://wrap.warwick.ac.uk/id/eprint/167348/1/WRAP-Identification-and-inference-of-network-formation-games-with-misclassified-links-Candelaria-2022.pdf
- https://www.jmlr.org/papers/v19/17-670.html
- https://arxiv.org/html/2108.13263
