# API reference

Three layers, from the most convenient to the most explicit.

## 1. Python API (`sgdiff.api`)

```python
from sgdiff import api

# Train: returns (model, summary, config)
model, summary, cfg = api.train(
    config_path="configs/sgdiff_cora.yaml",
    overrides=["train.epochs=20"],
)
print(summary["test"])            # {'auc': ..., 'ap': ..., 'hit@100': ...}
print(summary["run_dir"])         # runs/<name>/  (best.pt, metrics.jsonl, config.yaml)

# Data
graph = api.load_graph("cora", root="data")
split = api.make_split(graph, cfg)        # 80/5/15 + equal negatives

# Inference on arbitrary pairs
scores = api.predict(model, graph, [(0, 12), (7, 99)], cfg)

# Evaluation on explicit positive / negative sets
metrics = api.evaluate(model, graph, pos_pairs, neg_pairs, cfg)
```

| function | purpose |
|----------|---------|
| `api.prepare(cfg)` | graph + split + train/val/test datasets |
| `api.train(cfg=None, config_path=None, overrides=None, run_dir=None)` | full training run |
| `api.predict(model, graph, pairs, cfg, eval_graph=None, device=None)` | ranking scores |
| `api.evaluate(model, graph, pos, neg, cfg, eval_graph=None)` | AUC / AP / Hit@100 |
| `api.load_graph(name, root="data")` | dataset loading + caching |
| `api.load_split(cfg)` | `(graph, LinkSplit)` |
| `api.build_extractor(graph, cfg, seed=None, target_feature_dim=None)` | sub-graph extractor |
| `api.build_model(cfg, feature_dim, num_labels=None)` | instantiate `SGDiff` |
| `api.load_checkpoint(path, device="cpu")` | `(model, cfg)` from `best.pt` |

## 2. Command line (`python -m sgdiff` / `sgdiff`)

```bash
sgdiff prepare [--datasets cora usair ...] [--root data] [--no-download]

sgdiff train --config configs/sgdiff_cora.yaml \
             [--set section.key=value ...] [--run-dir runs] [--max-steps N]

sgdiff evaluate --checkpoint runs/cora/best.pt \
                [--dataset citeseer] [--split test] [--max-pairs 2000] \
                [--attack none|random|dice] [--attack-mode flip|insert|delete] \
                [--ratio 0.1] [--device cpu] [--output results/x.json]

sgdiff transfer --checkpoint runs/cora/best.pt \
                [--targets citeseer pubmed usair ns] [--max-pairs 500] \
                [--device cpu] [--output results/transfer.json]
```

Every sub-command prints a JSON summary on stdout.

## 3. Library internals

### Data

| object | description |
|--------|-------------|
| `sgdiff.data.GraphData` | undirected graph, optional features/labels, `.neighbors()`, `.drop_edges()`, `.add_edges()` |
| `sgdiff.data.Subgraph` | extracted sub-graph: `nodes`, `adj`, `labels`, `x`, `u_idx`, `v_idx` |
| `sgdiff.data.SubgraphExtractor(graph, hops=2, max_nodes_per_hop=50, max_label=16, feature_mode="auto", target_feature_dim=None)` | builds/caches sub-graphs |
| `sgdiff.data.make_link_split(graph, val_ratio=.05, test_ratio=.15, neg_ratio=1.0, seed=0)` | `LinkSplit` |
| `sgdiff.data.random_perturbation`, `dice_attack` | structural attacks |
| `sgdiff.data.load_graph(name, root="data", download=True, use_cache=True)` | registry loader |

### Models

| class | interface |
|-------|-----------|
| `SGDiff(feature_dim, num_labels=17, num_classes=2, structure_kwargs, feature_kwargs, fusion_kwargs)` | `.diffusion_losses(batch)`, `.class_scores(batch)`, `.forward(batch) -> {"logits", "feature_scores", "structure_scores"}`, `.predict_proba`, `.predict_logit`, `.train_logits` |
| `StructureDiffusionModel` | `.q_sample`, `.posterior_state1`, `.compute_kl`, `.loss`, `.log_likelihood` |
| `FeatureDiffusionModel` | `.q_sample`, `.compute_mse`, `.loss`, `.log_likelihood` |
| `LikelihoodFusion` | `.log_joint(feat, struct)`, `.forward(feat, struct)` |
| `baselines.NodePairLinkPredictor` | full-graph GCN / GAT / SAGE + dot-product decoder |
| `baselines.VGAE` | variational graph auto-encoder |
| `baselines.SEALBaseline` | sub-graph GCN classifier |

### Engine

| object | description |
|--------|-------------|
| `SubgraphBatch` | padded `(B, n, n)` adjacency, `(B, n, F)` features, DRNL labels, mask, labels, endpoints |
| `PairSubgraphDataset(pairs, labels, extractor)` | lazy, cached dataset of sub-graphs |
| `Trainer(model, cfg, device=None, run_dir=None)` | `.fit(train, val, test)` → history dict |
| `load_checkpoint(path, device)` | rebuild `(model, cfg)` |
| `score_pairs(model, extractor, pairs, batch_size=64, device=None, num_samples=None, output="logit", seed=0)` | score vector |
| `evaluate_link_prediction(model, extractor, pos, neg, ...)` | metrics dict |

### Configuration

`ExperimentConfig` is a tree of dataclasses (`data`, `subgraph`, `structure`,
`feature`, `model`, `train`) with `load_config(path, overrides)`, `to_dict()` and
`apply_overrides(["data.name=usair", "train.epochs=5"])`.
