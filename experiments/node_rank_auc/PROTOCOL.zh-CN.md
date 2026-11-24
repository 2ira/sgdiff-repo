# 重要节点覆盖率—AUROC 评测协议 v1

制定日期：2026-11-24。用户已明确：关注高度数或高 PageRank 的节点，横轴 Top k% 节点，k=0–100，纵轴 AUROC。这里的 rank 指图结构重要性，不是模型预测分数；排名第一是最重要的节点。

## 研究问题

当只关注图中较重要节点相关的测试边时，模型的判别能力和扰动退化是否与全测试集不同？这是一项子集评测，补充原全局 AUROC，不替换原实验，也不预设复杂模型或某一模型应当获胜。

当前研究对象为 Katz、HL-GNN；先做 seed 0 验收，再扩展原 seed 0/1/2。Katz 复用可核验的原逐边分数。HL-GNN 的原分数与 checkpoint 均已丢失，用户于 2026-11-24 明确授权按历史配置恢复性重训，并重做完整鲁棒性与组件消融。重训流程独立于本计分模块，仍按原干净验证 AUROC 选权重；本模块不训练、不重新选权重。仅有论文表格或平均 AUROC 无法反推出子集曲线。该补充不改变本协议的排名、筛边或计分规则。

## 1. 节点重要性只由干净训练图决定

每个 dataset、split seed 独立计算并保存一次，所有方法、变体和扰动共用：

- 图为原协议的 `train_pos` 构成的简单无向图；保留原始全部 N 个节点，包括训练图孤立点。
- 不用测试正边、测试标签、模型分数、扰动后图计算节点排名。也不把验证正边加到“排名图”中。模型推断仍遵循 Junxiong 协议（clean test 可用 train+val）；排名图和推断图用途不同。
- **Degree**：无向图度数，降序。
- **PageRank**：无向边等价双向边，alpha=0.85，均匀 teleport 和 dangling 分配；tol=1e-12、max_iter=1000；不收敛必须报错，不回退成另一种排名。
- 分数并列时按 `SHA256("node-rank-v1|dataset|split_seed|node_id")` 升序打破并列；这不使用标签/预测。保存 tie key 与边界并列组大小，避免输入顺序决定结果。必要时再做并列处理敏感性分析，不能挑对某方法有利的排序。
- 度数、PageRank 分别报告；无向图中两者可能高度相关，不宣称它们必然捕捉完全不同机制。

参考：[NetworkX PageRank 定义与参数](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)。

## 2. 横轴、节点集合和测试边筛选

k 取整数 0,1,…,100。选取数量 m(k)=ceil(N*k/100)，k=0 时 m=0。按冻结排序取前 m 个节点，记为 S(k)。同时记录实际 m/N，因取整它与标称 k% 可能略有区别。

主分析 **incident / OR**：从原测试正边和原测试负边分别保留至少一个端点在 S(k) 中的边：

`selected(u,v,k) = (u in S(k)) OR (v in S(k))`

- 正负边使用完全相同筛选条件；不重采负例，不再人为调回 1:1。
- 两端都在 S 的一条边只计一次；无向边不能因方向重复计数。
- 子集随 k 增大而嵌套；每个 seed、排名类型、k 对所有方法/条件固定不变。
- 这是“与重要节点有关的所有预测”，不是“两个重要节点之间的预测”。后者为 AND 子集，本版不混到主图里。
- **k=100 必须与同一组逐边分数的原全局 AUROC 一致。** 与历史汇总表的比较须先统一 seed、模型版本和数值精度。
- **k=0 没有预测样本，AUROC=NA。** 不填 0/0.5，也不从 0 绘制虚构线段。

例如 1,000 个节点取 Top 10%=100 个。原测试集中筛出 180 条正边、60 条负边，则在这 240 条原预测分数上计算 AUC。不能只筛正边却保留全部负边，否则评价的是另一种难度。

## 3. 纵轴与边界情况

主指标为 **incident-edge pooled AUROC**：在该子集内随机取一条正边和一条负边，模型把正边打分更高的概率，并列计 0.5。

`AUC(k) = [sum_{p in P(k), n in Q(k)} (1[s(p)>s(n)] + 0.5*1[s(p)=s(n)]) / (|P(k)|*|Q(k)|)`

使用排序算法计算，不实际构造所有正负组合。AUROC 不是阈值 accuracy，也不是“对每个点先算 acc 再平均”。原实验主指标同样是边预测 AUROC/AP，并非全节点分类 accuracy。

- 任意分数非有限数、数组顺序或长度不一致直接失败。
- 子集任一类别为空：AUC=NA，明确原因，不跳过后假装所有 seed 齐全。
- 每点输出 n_selected_nodes、n_pos、n_neg、正负边覆盖率、类别比例。
- 正/负任一类少于 20 个时标记 `low_support`，数字可以留作探索，绘图不与充分样本混为强证据。20 是本项目预先规定的提示阈值，不是统计保证。
- 图的纵轴固定 0–1，可用第二张放大图，不能截轴掩盖下降。

不在本轮自动加入“节点 macro-AUC”：它需要每个节点都有正负例，有些节点不能计算，并且 k=100 也不等于原全局 AUROC。如以后做，单独命名并报告可评节点覆盖率。

参考：[scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。

## 4. 鲁棒性沿用原协议，固定测试对象

沿用 Junxiong 固定提交 `95083da808e500c02da458e205f4b861ceb80871` 的训练/验证/测试划分及逐条件图。当前 14 条件：

- clean；
- RF、DICE、EA，各 10%、25%、50%；
- delete_0.5、delete_hub_0.5、insert_0.5；
- feature_shuffle。

不生成新的 EA 图，不改变扰动比例，也不在每个 k 下重新扰动或删除“非重要节点”。完整扰动图用于模型推断，只在计分时筛测试边。因此每个模型/条件只需推断一次，101 个 k 不产生 101 次模型推断。

保存每个 k 的 **ΔAUC(k)=AUC_perturbed(k)-AUC_clean(k)**（负值代表退化）。不同 k 的绝对 AUC 变化也包含测试人群变化；同一 k 下的扰动差才是固定子集的鲁棒性变化。

参考：[Junxiong 原协议源码](https://github.com/2ira/sgdiff-repo/blob/95083da808e500c02da458e205f4b861ceb80871/experiments/new_link_models/protocol.py)。

## 5. 统计与图表

先独立算每个 seed，随后在同一 dataset/model/variant/condition/rank/k 汇总 mean 和 sample SD(ddof=1)。三个 seed 同时包含训练/划分变动，不当作严格独立节点样本来做显著性声明。只完成一个 seed 时不显示虚构 SD。

- 主图：每个数据集两列（Degree / PageRank），各方法的 clean 曲线。
- 扰动图：每个模型分别画 clean、EA50、hub-delete50、feature-shuffle；全 14 条件进入完整数值表；主图展示代表条件，避免一图塞过多线。
- 降幅表：同一 k 的配对 ΔAUC，先逐 seed 相减，再汇总；不从汇总 SD 直接相减推算配对 SD。
- 覆盖率图：每个 k 的正边和负边数/比例，用来解释 Top 1% 已覆盖较多 hub 相关边等现象。
- 关键表：k=1/5/10/20/50/100；完整 CSV 保留 0–100。
- 不预设曲线单调，也不把较高 AUC(k) 解释为选节点本身提升了模型。测试子集变了，模型没有变。

注意度相关负采样偏差：原随机负例在 hub 子集内可能与正例的度分布不同。第一轮保留原负例以保证与旧结果可核对，并记录覆盖率；如果要做度匹配负例，另设独立评价协议，不混进本曲线。

## 6. 可复用输入接口

本目录的 `evaluate.py` 只读取固定 split 和逐边原始分数，不依赖模型框架，不加载 GPU。未来任何方法只需导出相同字段；这是一层附加计分模块，不另写一套训练/攻击流程。

输入 manifest JSON：

```json
{
  "dataset": "cora",
  "seed": 0,
  "num_nodes": 2708,
  "split_path": "inputs/cora_seed0.npz",
  "protocol_commit": "95083da808e500c02da458e205f4b861ceb80871",
  "records": [
    {
      "model": "katz", "variant": "full", "condition": "clean",
      "scores_path": "inputs/katz_clean.npz",
      "pos_key": "pos", "neg_key": "neg",
      "positive_sha256": "原始导出 receipt 中的有序 test_pos 指纹",
      "negative_sha256": "原始导出 receipt 中的有序 test_neg 指纹",
      "query_order_evidence": "原始 archive/adapter 中按 split.test_pos, split.test_neg 逐对输出，附源文件位置",
      "expected_auc": 0.84
    }
  ]
}
```

示例 `expected_auc` 仅为接口占位，运行时必须填写该 seed/condition 的实际历史值或删除字段，不能把 0.84 当验收目标。路径相对 manifest 所在目录。split NPZ 必须含 `train_pos, test_pos, test_neg`；建议保留原始六个 split 数组。若 scores NPZ 含 `test_pos,test_neg` 也须逐项比对；否则必须有可审核的原始导出顺序证据，以及与当前 split 精确匹配的有序 test_pos/test_neg 指纹；仅长度相同或布尔声明不足以通过验收。

输出：

- `node_ranks.csv`：两种排名原分数、排序位置、tie key；
- `per_seed_curves.csv`：逐条件、逐 k AUC/覆盖率/低样本标记；
- `paired_deltas.csv`：相对于 clean 的逐 seed 差值；
- `audit.json`：输入文件、数组、实现 hash，依赖版本，100% 还原误差；
- 图与跨 seed 汇总由 `summarize.py` 从曲线表产生。

## 7. 验收后再扩展

1. 人工小图：验证 OR 规则、双端不重复、0/100 边界、并列 AUC、单类别 NA。
2. 调换分数输入顺序应被拒绝或由明确 ID 映射纠正，不能仅因长度相等就认为对齐。
3. 验证排名不因模型/扰动改变，子集严格嵌套；100% 精确复算全局 AUC。
4. 先用一个真实 seed 验证计分与原指标。恢复性重训另行按明确授权执行，不能把旧均值当作新分数。
5. 验收通过后扩展原 seeds/conditions。缺失原分数/权重时，需真实恢复或经授权重训，并单独标记新结果；不从报告均值造曲线。

## 8. 本轮实现与工件

训练与原生 14 条件评测：`scripts/retrain_hlgnn_archived.py`、`scripts/evaluate_hlgnn_full.py`，继续调用 Junxiong 固定提交的 `run.evaluate`。新计分层：`experiments/node_rank_auc/evaluate.py`。

重训环境为单张 H200、容器 `/usr/bin/python`，无额外训练 venv；确切版本、模型工件路径及复算指令见 [重训说明](../../docs/hlgnn_retraining_20261124.md)。结果入口为仓库 `results/hlgnn-retrained/` 与 `results/node-rank-auc/`。只重算已发布分数时不需要 GPU。

公开 GitHub 交付仍使用同一分支 `unified-robustness-ablation`。Katz 来源于原发布归档；新 HL-GNN 明确标记为重训结果，并保留旧成绩对照。旧 SP4LP 分数不会混作 HL-GNN。SSH 连接信息另存本地交接文档，不属于公开计分协议。
