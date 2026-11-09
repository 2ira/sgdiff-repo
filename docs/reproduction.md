# Reproduction report

Protocol and hyper-parameters follow `li25a.pdf` (Table 3/4, Sec. 4.1, App. A.2-A.4).
The parameter-by-parameter mapping is in [`alignment.md`](alignment.md).

## 1. What is implemented

| component | paper | code | status |
|-----------|-------|------|--------|
| enclosing sub-graph, DRNL | Sec. 3.1 | `data/subgraph.py` | aligned (`k=1`, per-dataset `ns`) |
| structure diffusion (D3PM) | Sec. 3.3 | `models/structure_diffusion.py` | aligned |
| transformer denoiser | App. A.3, Table 4 | `models/transformer_denoiser.py` | aligned |
| VLB + node/edge CE | App. A.3 | `StructureDiffusionModel.loss` | aligned |
| feature diffusion (DDPM) | Sec. 3.4 | `models/feature_diffusion.py` | aligned (off for USAir/NS/Router) |
| fusion `{η1,η2,δ}`, `p(y)=0.5` | Eq. 12-13 | `models/fusion.py` | aligned |
| AUC / AP / Hit@100 | Sec. 4.1 | `utils/metrics.py` | aligned |
| 10 seeds | Sec. 4.1 | `scripts/run_experiments.py` | supported |
| RQ1 transfer without features | Sec. 4.2 | `--transfer` | aligned |
| RQ2 1% training sub-graphs | Sec. 4.3 | `data.train_ratio` | aligned |
| RQ3 RF / DICE | Sec. 4.4 | `--attack` | RF + DICE (no EA) |
| baselines GCN/GAT/SAGE/VGAE/SEAL | Sec. 4.1 | `models/baselines/` | NeoGNN missing |

## 2. Two budgets

* `--profile paper` — Table 4 exactly: full training split, all `ds_t` Monte-Carlo
  steps, no sub-graph size cap. Needs a GPU (a 1-hop Cora sub-graph of a hub node
  easily exceeds 150 nodes, so the transformer attention is the bottleneck).
* `--profile cpu` — same architecture, schedule and scoring rule, with
  `2 000` training pairs, `8` epochs (3 pre-train), `4` Monte-Carlo steps and a
  `40`-node sub-graph cap. This is what produced the numbers below.

## 3. Measured results (CPU profile, 3 seeds unless stated)

In-domain link prediction (mean ± std over seeds):

| dataset | AUC | AP | Hit@100 | paper (clean AUC) |
|---------|-----|----|---------|-------------------|
| Cora | 0.751 ± 0.017 | 0.809 ± 0.014 | 0.960 | 0.884 |
| Citeseer | 0.738 ± 0.019 | 0.787 ± 0.022 | 0.903 | 0.871 |
| Pubmed | 0.822 (seed 0) | 0.821 | 0.830 | 0.951 |
| USAir | 0.929 ± 0.004 | 0.923 ± 0.006 | 0.960 | ~0.95 |
| NS | 0.964 ± 0.006 | 0.973 ± 0.004 | 1.000 | ~0.97 |

Paper values are the clean-Cora/Citeseer/Pubmed AUC column of Table 2 (the RF
25% column is within ~1 point of clean) and the USAir/NS in-domain AUC of
Figure 2. The attribute-free graphs (USAir, NS) match the paper; the citation
graphs are 10-13 points lower under the CPU budget.

### Where the gap comes from

1. **k = 1 sub-graphs.** With the target edge removed, a 1-hop sub-graph of a
   Cora positive contains the two endpoints and their (mostly disjoint)
   neighbourhoods; common-neighbour AUC on this split is 0.733, so the task is
   much harder than the 2-hop setting used by SEAL. Ablation (same CPU budget,
   Cora, seed 0): `hops=2` reaches **0.859-0.878** AUC versus 0.726-0.75 with
   `hops=1`.
   Table 4 specifies `k = 1`; the config exposes both.
2. **Training pairs.** 2 000 of 4 222 positives (plus 2 000 negatives) instead of
   the full split.
3. **Finetune epochs.** 5 (of 8) versus an unspecified, larger budget in the
   paper; validation AUC is still rising when the run stops.
4. **Monte-Carlo steps.** 4 of 20 at inference; on a fixed checkpoint this
   changes AUC by <0.02 (0.789 / 0.772 / 0.780 for 4 / 8 / 20 samples).

## 4. Research questions

### RQ1 - zero-shot cross-dataset transfer (Sec. 4.2)

Trained on Cora **without node features** (`feature.enabled=false`), evaluated
on every other dataset without any retraining:

| target | AUC | AP | paper trend |
|--------|----:|---:|-------------|
| Citeseer | 0.733 | 0.782 | SGDIFF transfers best from large sources |
| Pubmed | 0.830 | 0.849 | |
| USAir | 0.800 | 0.715 | the weakest target for SGDIFF in the paper |
| NS | 0.947 | 0.964 | |

The structure branch transfers well even though the feature branch is disabled,
which is the qualitative claim of Sec. 3.2.

### RQ2 - 1% training sub-graphs (Sec. 4.3)

Only 42 positive + 42 negative sub-graphs are used; the remaining training
edges stay in the graph (paper protocol).

| budget | Cora AUC | seeds |
|--------|---------:|-------|
| 100% pairs, 8 epochs | 0.751 ± 0.017 | 0,1,2 |
| 1% pairs, 40 epochs | 0.725 ± 0.058 | 0,1,2 (0.744 / 0.647 / 0.785) |

The 1% model is within ~2.5 points of the full model once the schedule is long
enough to see the tiny sample repeatedly. With the same 8-epoch schedule the 1%
run collapses to 0.49 (every sub-graph is seen twice); with 40 epochs the
variance across seeds grows, which is expected from 84 training sub-graphs.

### RQ3 - robustness under structure attacks (Sec. 4.4)

Cora checkpoint trained on the clean graph, inference graph perturbed:

| budget | clean | RF | DICE |
|-------:|------:|---:|-----:|
| 10% | 0.705 | 0.698 | 0.692 |
| 25% | 0.705 | 0.651 | 0.643 |
| 50% | 0.705 | 0.577 | 0.584 |

The model degrades gracefully at 10-25% budget and loses ~13 points at 50%,
qualitatively matching the paper's "generative models degrade less" finding
(their RF-50% drop on Cora is ~1.5 points, so the absolute gap is ours).

## 5. Baseline numbers (Cora, full graph)

| model | AUC | AP |
|-------|----:|----:|
| GCN | 0.811 | 0.826 |
| GAT | 0.846 | 0.858 |
| SAGE | 0.747 | 0.756 |
| VGAE (`kl_weight=1e-3`) | 0.797 | 0.805 |
| SEAL (3k sub-graphs, 10 epochs, 2 hops) | 0.852 | — |

## 6. Known deviations

1. Optimisation details (batch 32, Adam `1e-3`, fusion `1e-2`, 50-epoch budget
   in the `paper` profile) are not reported by the paper.
2. Embedding attack (EA) needs DeepRobust and is not implemented; RF and DICE
   share its interface.
3. `Router` is not redistributable (see `data/README.md`), so it is absent from
   the tables above.
4. Citeseer counts 4 552 undirected links versus the paper's 9 228/2 = 4 614
   because Planetoid's pickled graph has 248 self-loops and 56 asymmetric pairs.
5. `subgraph.max_nodes_total` in the CPU profile is a compute safeguard that the
   paper does not have.

## 7. Reproducing

```bash
python -m sgdiff prepare --datasets cora citeseer pubmed usair ns
python scripts/run_experiments.py --datasets cora citeseer pubmed usair ns --seeds 0 1 2
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5
bash scripts/run_all_experiments.sh cpu "0 1 2"
```

Each run writes `results/<dataset>_seed<k>_<profile>.json`; aggregates with
mean/std are stored in `results/summary_*.json`.

## 8. Bug journal (things that silently break this reproduction)

* Removing the target edge only from the induced adjacency still leaks the label
  through DRNL distances; the edge must be blocked inside the BFS.
* Capping `max_train_pairs` by slicing `[pos; neg]` keeps only positives.
* A uniform D3PM schedule with `β_end = 0.5` spends most steps in pure noise and
  the averaged KL loses the class signal; `β_end = 0.2` keeps `ᾱ_t` informative.
* Pure MLE pre-training produces nearly identical `p(A|y=0)` and `p(A|y=1)`
  (AUC ≈ 0.5); the cross-entropy on the fused likelihood is what makes the
  scores discriminative, so stage 2 must not be skipped.
* Unbounded 1-hop sub-graphs of hub nodes make transformer training intractable
  on CPU (one epoch took hours); `subgraph.max_nodes_total` fixes it.
* The likelihood scores are Monte-Carlo estimates: evaluation fixes the RNG seed
  (`score_pairs(..., seed=0)`) so checkpoints are reproducible.
