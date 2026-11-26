# Shared important-node AUROC evaluator

The evaluator, input-order tests, and HL-GNN manifest adapter in this directory are copied unchanged from `unified-robustness-ablation@bef967467abd3fe825c08e243b3f7c7b6a2f250b`. They support the [three-model extension](../new_models_rank/README.md) on this feature branch. No training or attack generation occurs in the scoring layer.

See the [original protocol](https://github.com/2ira/sgdiff-repo/blob/bef967467abd3fe825c08e243b3f7c7b6a2f250b/results/node-rank-auc/PROTOCOL.zh-CN.md) and [upstream complete module](https://github.com/2ira/sgdiff-repo/tree/bef967467abd3fe825c08e243b3f7c7b6a2f250b/experiments/node_rank_auc). Published ranking and comparison inputs are in `results/node-rank-auc/`. Exact imported-file fingerprints are recorded in `results/new-models-node-rank/integration_audit.json`.
