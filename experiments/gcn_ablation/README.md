# GCN 输入消融与跨数据集迁移

这里整合 [linjunxiong/GCN](https://github.com/linjunxiong/GCN/tree/9aabfe3e41c1d80d6756fc332d8c5a77412ac670) 的实验。模型属于确定性的 GAE 框架：GCN 编码器 + 内积解码器 + 采样 BCE；feature-only 用两层 MLP 替代传播。GCN 是编码器，GAE 是整个链路预测框架，两者不是互斥的模型。本实验没有新增 VGAE；仓库已有独立的 VGAE 类。

**共享模型，保留实验协议。** `run_shared.py` 使用本仓库的 `GCNEncoder`、`NodePairLinkPredictor.score_pairs`，包括稀疏输入和单位矩阵输入的缓存传播路径。它继续使用原实验的划分、输入消融、迁移、早停与评估代码。完整对照见 [整合审查](../../docs/gcn_ablation_integration.md)。

## 输入与实验范围

| 实验 | structure | feature | combined |
|---|---|---|---|
| 图内 within | GCN(X=I)，第一层为可学习节点嵌入 | MLP(行归一化原始属性) | GCN(相同原始属性) |
| 跨图 transfer | GCN([32维度数编码, 0]) | MLP([0, 64维属性投影]) | GCN([度数编码, 属性投影]) |

数据集为 Cora、Citeseer、Pubmed、Router、USAir、NS。结构迁移为 6×6；后面三个没有原始属性，因此属性相关实验仅覆盖前三个，迁移为 3×3。目标图可见其 80% 训练结构，但不在目标图更新参数或选择模型；仅以源图验证 AUC 选择 checkpoint。属性随机投影只对齐维度，不对齐词表语义。图内 X=I 与属性模型参数量不同，记录在结果中；迁移三组参数量相同。

正边约 80/5/15 划分（验证和测试向下取整），负边与正边等量且三个 split 互不重叠。所有阶段传播和度数编码只使用训练正边。AUC/AP 使用 sklearn；Hits@100 为正边严格超过全局负边池第100高分的比例，负边少于100时恒为1。**这些细则与仓库现有 benchmark 不完全相同，不能直接拼接数值排名。**

## 运行

从仓库根目录执行，Python >=3.10：

```bash
python -m pip install -e .
python -m pip install -r experiments/gcn_ablation/requirements.txt
python experiments/gcn_ablation/prepare_data.py
python -m pytest tests/test_gcn_ablation_bridge.py
python experiments/gcn_ablation/run_shared.py
python experiments/gcn_ablation/summarize.py --output experiments/gcn_ablation/shared_results
```

数据下载使用固定 commit 的 Planetoid / SEAL 原始数据，保存 URL、SHA256 及清洗计数至 `data/manifest.json`。现有仓库数据加载器的下载源/特征处理不同，所以这里保留原下载脚本。数据、权重及预测由运行生成，不纳入 Git。

默认 10 个种子；CPU 单线程，隐藏64维、输出32维、dropout .3、Adam .01、weight_decay 0、最多200轮、每5轮验证、耐心30轮。小规模检查：

```bash
python experiments/gcn_ablation/run_shared.py --seeds 0 --datasets Cora USAir --epochs 20 --patience 10 --output experiments/gcn_ablation/smoke_results
python experiments/gcn_ablation/replay_shared.py --output experiments/gcn_ablation/smoke_results --protocol transfer --source Cora --mode structure --seed 0
python experiments/gcn_ablation/summarize.py --output experiments/gcn_ablation/smoke_results --allow-partial
```

共享实现默认输出 `shared_results/`，每个模型保存权重、训练历史、划分和各目标图预测；`backend.json` 额外记录模型后端及相关源文件哈希。已完成单元可续跑；配置或后端代码改变需要新输出目录。共享后端的权重使用 `replay_shared.py` 回放，不能用旧的 `replay.py`。可以使用 `python -m experiments.gcn_ablation.run_shared` / `replay_shared`。

某些共享环境预装的 pytest 插件和旧 ONNX/protobuf 冲突时，分别在命令前加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`；干净环境一般不需要。

## 原始结果与可追溯性

`benchmarks/` 保存原实现完成的 **10 个种子、240 个训练模型、660 条评估记录**及图表，来自上述 GCN 仓库 commit；**不是共享后端的全量重跑结果**。`experiment.py` 和其余原始实验脚本保留原样；该训练文件 SHA256 为 `39d54b45444a8dc5ada1aad8ada4e615d1846816a7241990e6ffd9a3096c590e`。运行旧实现用 `python experiments/gcn_ablation/experiment.py`，默认输出独立的 `results/`。

[REFERENCE.md](REFERENCE.md) 是原仓库说明的存档，里面的根目录命令需在本目录执行；其中关于外部仓库状态的描述也是原记录。详细报告见 [benchmarks/report.md](benchmarks/report.md)。存档中的绝对输出路径仅记录原运行环境。

原实现测试需要先准备 Cora 数据：

```bash
cd experiments/gcn_ablation
python -m pytest -q test_experiment.py
```

共享模型的数学等价性通过输出、梯度、初始化、dropout 随机状态和稀疏 X=I 检查；短训练验证见整合审查。未承诺不同实现/依赖版本的训练结果逐位相同。
