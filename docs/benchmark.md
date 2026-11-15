# Data-identical benchmark

`scripts/run_benchmark.py` compares SGDiff with the discriminative baselines on
**the same split and the same sub-graphs**, and records fingerprints that make
the claim auditable. The harness lives in `sgdiff/engine/benchmark.py`.

## 1. The comparability contract

For every `(dataset, seed)` the harness

1. loads the graph once,
2. builds one 80/5/15 split (`make_link_split`, `neg_ratio` from the config),
3. builds **one** `prepare_datasets(...)` - enclosing sub-graphs, DRNL labels,
   node features and the train/eval graph split are shared by all models,
4. trains each model through its own loop but on those exact objects, and
5. evaluates all models on the same `(pairs, labels)` with the same metric code.

Two hashes are written next to the metrics:

| field | what it hashes | proves |
|-------|----------------|--------|
| `split_fingerprint` | the six positive/negative pair arrays (canonicalised, order-free) | every model saw the same links and the same negatives |
| `subgraph_fingerprint` | nodes / DRNL labels / adjacency / features of the first 32 test sub-graphs | every model saw the same inputs, not a re-sampled neighbourhood |

`_check_identity` raises if two models in the same seed disagree. Compare the
JSON of two runs to prove two *experiments* used the same data.

## 2. Model zoo

| name | kind | what it is | source |
|------|------|------------|--------|
| `sgdiff` | `sgdiff` | the paper's structure + feature diffusion with learned fusion | `models/sgdiff.py` |
| `seal` | `subgraph` | 4-layer GCN sub-graph classifier + DRNL | Zhang & Chen 2018 |
| `pencil` | `subgraph` | plain Transformer over the same sub-graphs, pair read-out | arXiv:2602.01553 |
| `gcn` / `gat` / `sage` | `node_pair` | full-graph encoder + dot-product decoder | Kipf 2017 / Velickovic 2018 / Hamilton 2017 |
| `vgae` | `node_pair` | variational graph auto-encoder | Kipf & Welling 2016 |

`kind` selects the training loop: `node_pair` and `subgraph` use
`sgdiff/engine/baseline_trainer.py`, `sgdiff` uses `Trainer` (two-stage
pre-train / fine-tune), exactly as in the reproduction experiments.

## 3. Running it

```bash
# CPU budget, one seed, full table
python scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0

# table-4 budget (GPU recommended)
python scripts/run_benchmark.py --dataset cora --profile paper --seeds 0 1 2

# 3 seeds, sub-graph models only, 2-hop sub-graphs
python scripts/run_benchmark.py --dataset citeseer --models sgdiff seal pencil \
    --profile cpu --seeds 0 1 2 --overrides subgraph.hops=2

# what is in the zoo
python scripts/run_benchmark.py --list-models

# smoke test on the synthetic graph (no downloads)
python scripts/run_benchmark.py --dataset synthetic --profile smoke \
    --models sgdiff seal pencil gcn
```

Outputs:

* `results/benchmark_<dataset>.json` - protocol, fingerprints, per-run metrics,
  parameter counts, wall-clock and the aggregate mean/std;
* `results/benchmark_<dataset>.md` - the markdown table.

Budget profiles (`paper` / `full` / `cpu` / `smoke`, defined in
`sgdiff/engine/benchmark.py`) mirror the ones in `scripts/run_experiments.py` so
the diffusion numbers stay comparable with the main reproduction report.

## 4. Adding a model

Add one `ModelSpec` to `build_registry` in `sgdiff/engine/benchmark.py`:

```python
"sp4lp": ModelSpec(
    "sp4lp",
    "subgraph",                       # reuse the shared sub-graph datasets
    lambda fd, c: SP4LPBaseline(fd, num_labels=c.subgraph.max_label + 1),
    note="shortest-path sequence model",
),
```

The `builder(feature_dim, config)` must return a module that either

* implements `forward(batch) -> (B, 2)` plus `predict_logit(batch, num_samples=None)`
  (sub-graph kind), or
* implements `encode` / `score_pairs` (node-pair kind), or
* is an `SGDiff` instance (sgdiff kind).

Nothing else needs to change: the loop, metric code, fingerprints and table
format are shared.

## 5. Results (Cora, `paper` profile, `sgdiff-paper-v2`)

The 2026-11-14/15 GPU grid at the paper profile (Table 4 config, full splits,
uncapped sub-graphs, all Monte-Carlo steps). All models share the same
80/5/15 split and the same test pairs (SGDiff + baselines via
`scripts/run_robustness.py --profile paper --seeds 0 1 2`, 3-seed means;
`run_baselines.py` for the single-seed baseline column):

| model | AUC (3 seeds) | AUC (baselines run) | Hit@100 |
|---|---|---|---|
| pencil | 0.883 | — | — |
| seal | 0.872 | 0.888 | 0.99 |
| **sgdiff (v2)** | **0.858** | — | 0.95 (in-domain grid) |
| gat | 0.850 | 0.849 | 0.97 |
| gcn | 0.815 | 0.823 | 0.97 |
| vgae | 0.816 | 0.824 | 0.94 |
| sage | 0.760 | 0.743 | 0.86 |

Artifacts: `results/sgdiff-paper-v2/robustness_cora__v2_seed{0,1,2}.json`
(includes per-model parameter counts, fit times and the clean-variant metrics),
`results/sgdiff-paper-v2/baselines_cora.json`. The old CPU-profile table
(seed 0, old SGDiff implementation) was removed with its artifacts on
2026-11-15; the fingerprints of that run no longer exist.

Reading of the paper-profile run:

* **SGDiff v2 is competitive on AUC and first-tier on Hit@100** (0.953 in the
  10-seed in-domain grid), but SEAL and PENCIL are the strongest clean AUC
  performers on this split.
* The fused generative score ranks the top of the pool sharply; under attack
  the same table degrades gracefully (see [`robustness.md`](robustness.md)).
* Parameter cost still differs by ~40x (2.28 M for SGDiff vs 54 k for SEAL);
  wall-clock per paper-profile fit is in the robustness JSON `fit_seconds`.

## 6. Caveats when reading the table

1. **Random negatives.** AUC/AP/Hit@100 are computed on an equal-size random
   negative pool, following the paper. This flatters every model relative to a
   full-ranking protocol (`Hits@K` over all non-edges, or the union-set protocol
   of *Mind the Links*, 2025). If the claim under test is "generative models
   generalise better", the ranking protocol matters - report both.
2. **Hit@100 is a top-of-pool recall here**, not a per-node retrieval metric;
   with 500 evaluation pairs the pool is smaller than 100 items in some runs.
3. **Optimisation budgets are ours.** The paper reports no batch size, learning
   rate or epoch count; the diffusion model may be under-trained relative to
   the authors' (undocumented) schedule.
4. **Monte-Carlo noise.** SGDiff's scores are estimates; the harness re-seeds
   before scoring, but single-seed differences below ~0.02 AUC are not
   meaningful.
5. **PENCIL is a re-implementation.** It keeps the DRNL label and endpoint role
   embeddings of the shared pipeline (`use_labels=False` disables them) and uses
   a symmetric pair read-out, so it is a fair *encoder* comparison rather than a
   byte-exact PENCIL reproduction.

## 7. Queued experiments

| experiment | why | hook |
|------------|-----|------|
| likelihood ablation under v2 | old-implementation audit withdrawn; rerun `scripts/run_ablation.py` on `sgdiff-paper-v2` | `engine/ablation.py` |
| flow-matching scorer | CPD-PIFM-style replacement of the DDPM backbone | new `ModelSpec` |
| DMNS negatives | diffusion-generated hard negatives | `data/splits.py` |
| EO-GNN baseline | re-test the automorphic-node argument | `models/baselines/` |
| full-ranking evaluation | avoid the random-negative protocol | new metric in `utils/metrics.py` |

See [`related_work.md`](related_work.md) for the papers behind each row.
