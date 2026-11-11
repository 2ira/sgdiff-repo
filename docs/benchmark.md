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

## 5. Results (Cora, `cpu` profile, seed 0)

Filled in by `python scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0`.
The CPU budget is 2 000 training pairs, 8 SGDiff epochs (3 pre-train), 4 Monte
Carlo steps and a 40-node sub-graph cap; baselines are not tuned per dataset
(fixed 300 epochs for full-graph encoders, 40 for sub-graph classifiers).

| model | AUC | AP | Hit@100 | #params | fit (s) |
|---|---|---|---|---|---|
| gcn | 0.8296 | 0.8352 | 0.9300 | 95,936 | 0.3 |
| gat | 0.7915 | 0.7739 | 0.8400 | 100,352 | 0.2 |
| seal | 0.7883 | 0.7950 | 0.8600 | 54,082 | 12.6 |
| pencil | 0.7755 | 0.7901 | 0.8400 | 176,834 | 16.2 |
| vgae | 0.7667 | 0.7743 | 0.9300 | 46,944 | 0.5 |
| sage | 0.7525 | 0.7537 | 0.8900 | 191,744 | 0.3 |
| sgdiff | 0.7314 | 0.7930 | **0.9500** | 2,283,006 | 104.4 |

All seven rows share `split_fingerprint = a4855f05c6d94de2`,
`subgraph_fingerprint = eeb0dee5e6caa792` and
`train_pairs_fingerprint = b55c4afa8cf099a0` (2 000 training pairs, 500
validation, 500 test) in `results/benchmark_cora.json` → `data_integrity`.

Reading of this single-seed CPU run:

* **SGDiff is behind on AUC but first on Hit@100 and second on AP.** Its fused
  likelihood ranks the very top of the pool better than any baseline but
  separates the bulk of the negatives worse - consistent with the paper's
  "generative scoring is sharper, not uniformly better" framing, and with the
  0.751 AUC recorded for the same CPU budget in `docs/reproduction.md`.
* The gap to `gcn` (9.8 AUC points) is *budget*, not architecture: SGDiff gets 8
  epochs because one epoch costs ~13 s on CPU, the GNN baselines get 300 epochs
  at <1 s each, and SGDiff's validation AUC was still improving at epoch 5.
* Parameter cost differs by ~40x (2.28 M vs 54 k for SEAL) and wall-clock by
  ~8-350x. Any "generative models are worth it" argument has to be made on the
  `paper` profile, not on this one.
* Do not over-read single-seed numbers; rerun with `--seeds 0 1 2` before
  quoting them.

## 6. Caveats when reading the table

1. **Random negatives.** AUC/AP/Hit@100 are computed on an equal-size random
   negative pool, following the paper. This flatters every model relative to a
   full-ranking protocol (`Hits@K` over all non-edges, or the union-set protocol
   of *Mind the Links*, 2025). If the claim under test is "generative models
   generalise better", the ranking protocol matters - report both.
2. **Hit@100 is a top-of-pool recall here**, not a per-node retrieval metric;
   with 500 evaluation pairs the pool is smaller than 100 items in some runs.
3. **Budget asymmetry.** SGDiff runs 8 epochs because each epoch is ~15 s on
   CPU; the full-graph baselines run 300 cheap epochs. Increasing SGDiff's
   budget raises its AUC (validation AUC was still climbing at epoch 5), so the
   table is a lower bound for the diffusion model.
4. **Monte-Carlo noise.** SGDiff's scores are estimates; the harness re-seeds
   before scoring, but single-seed differences below ~0.02 AUC are not
   meaningful. Use `--seeds 0 1 2` before drawing conclusions.
5. **Sub-graph cap.** The `cpu` profile caps sub-graphs at 40 nodes, which can
   cut off hub neighbourhoods; `paper` leaves them unbounded.
6. **PENCIL is a re-implementation.** It keeps the DRNL label and endpoint role
   embeddings of the shared pipeline (`use_labels=False` disables them) and uses
   a symmetric pair read-out, so it is a fair *encoder* comparison rather than a
   byte-exact PENCIL reproduction.

## 7. Queued experiments

| experiment | why | hook |
|------------|-----|------|
| ~~condition-shuffle ablation~~ | done: `scripts/run_ablation.py`, see [`ablation.md`](ablation.md) | `engine/ablation.py` |
| flow-matching scorer | CPD-PIFM-style replacement of the DDPM backbone | new `ModelSpec` |
| DMNS negatives | diffusion-generated hard negatives | `data/splits.py` |
| EO-GNN baseline | re-test the automorphic-node argument | `models/baselines/` |
| full-ranking evaluation | avoid the random-negative protocol | new metric in `utils/metrics.py` |

See [`related_work.md`](related_work.md) for the papers behind each row.
