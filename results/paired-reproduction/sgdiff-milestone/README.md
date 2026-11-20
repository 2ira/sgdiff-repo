# 首个本地 SGDiff 校准检查点

Cora 种子 0，训练对 8,444 个，验证对 528 个。MPS 完成 50 轮完整扩散预训练，按验证损失恢复最佳扩散参数，开始最多 20 轮的冻结扩散融合校准。本目录固定记录第 1 轮融合得到的权重。CPU 对同一批 500 正、500 负测试对给出下列结果。历史 RQ3 的 Cora 种子 0 为 0.85149，三个种子均值为 0.85820。这个中间检查点不计入最终三种子汇总，训练继续按验证损失选权重。

| 读出 | 干净 AUROC / AP | 均匀删边 25% AUROC / AP | 低度删边 25% AUROC / AP |
|---|---:|---:|---:|
| sgdiff | 0.8678 / 0.8820 | 0.7902 / 0.8261 | 0.7698 / 0.8031 |
| sgdiff_structure | 0.8297 / 0.8728 | 0.7521 / 0.8090 | 0.7155 / 0.7775 |
| sgdiff_feature | 0.7567 / 0.7438 | 0.7259 / 0.7303 | 0.7209 / 0.7199 |

归档包含实际校准权重、测试对、三张可见图及每个候选在有边/无边两类下的原始分支分数。两路扩散模型在融合校准期间冻结，计算分支输出时的每个扩散参数与该校准权重逐字节校验一致。三种读出在同一份类条件分数上调用统一融合函数。解压到仓库根目录后可以从原始图重新计算全部分数。

```bash
unzip results/paired-reproduction/sgdiff-milestone/cora_seed0_fusion_epoch1.zip
python scripts/run_reproduction.py evaluate --models sgdiff --datasets cora --seeds 0 --device cpu --threads 2 --weights runs/sgdiff-milestone --output runs/sgdiff-milestone/replay --conditions clean random_delete_25 degree_delete_25
```

[精确指标](metrics.csv) · [模型与分支证据](cora_seed0_fusion_epoch1.zip) · [下载校验值](SHA256SUMS)
