# Reproduction notes

## Status

| item | state |
|------|-------|
| sub-graph construction + DRNL | implemented and unit tested |
| discrete (D3PM) structure diffusion | implemented, closed-form posterior verified |
| Gaussian feature diffusion | implemented |
| learnable fusion (Eq. 12-13) | implemented |
| 80/5/15 split + equal negatives | implemented |
| AUC / AP / Hit@100 | implemented, dependency-free |
| cross-dataset transfer | implemented (incl. feature projection) |
| 1% training-data study | implemented (`data.train_ratio`) |
| adversarial robustness | implemented (random + DICE; **not** DeepRobust) |
| baselines | GCN, GAT, SAGE, VGAE, SEAL (NeoGNN omitted) |

The official repository referenced by the paper
(<https://github.com/hzlihang99/SGDiff>) contains only a two-line README at the
time of writing, so this project is an independent re-implementation from the
paper text (Sec. 3, Appendix A.1-A.2).

## Protocols

* Links are split 80/5/15; negatives are uniformly sampled non-edges, one per
  positive (paper Sec. 4.1).
* Training sub-graphs come from `G \ (E_val ∪ E_test)`, validation/testing
  sub-graphs from `G \ E_test`, so test links never appear in the encoders'
  neighbourhoods.
* The target edge `(u, v)` is hidden **during the BFS**, not only in the induced
  adjacency (see "Bug journal" below).
* `Hit@100` needs a candidate pool larger than 100; with fewer than 100
  positives in the pool the metric saturates (it is still printed).

## Reference numbers measured on this machine

CPU (Apple M-series, PyTorch 2.15, Python 3.14), deliberately small budgets —
these are **not** the paper's numbers, they only show the implementation learns:

| experiment | setting | AUC | AP | Hit@100 |
|------------|---------|----:|---:|--------:|
| Cora (in-domain) | 2k train pairs, 9 epochs | 0.860 | 0.885 | 0.96 |
| Cora (re-eval of that checkpoint, 100+100 pairs) | — | 0.895 | 0.925 | 0.80 |
| Citeseer (in-domain) | 2k train pairs, 9 epochs | 0.805 | 0.854 | 0.97 |
| USAir (in-domain) | 5 epochs, 1.7k train pairs, structural features | 0.925 | 0.926 | 0.99 |
| Cora → Citeseer (zero-shot) | no retraining | 0.834 | 0.857 | 0.73 |
| Cora → USAir (zero-shot) | no retraining | 0.674 | 0.619 | 0.62 |
| Cora, random attack 10% | same checkpoint | 0.889 | 0.914 | — |

Baselines on Cora:

| model | AUC | AP |
|-------|----:|---:|
| GCN (120 epochs) | 0.811 | 0.826 |
| GAT (120 epochs) | 0.846 | 0.858 |
| SAGE (120 epochs) | 0.747 | 0.756 |
| VGAE (`kl_weight=0.001`) | 0.797 | 0.805 |
| SEAL (3k sub-graphs, 10 epochs) | 0.852 | — |

USAir is the setting where SGDIFF is strongest in the paper as well (no node
attributes, so the DRNL/structural branch carries the signal).

Paper (Table 1, transfer setting): SGDIFF AUC between 72 and 97 depending on the
source/target pair, SEAL 56-98. Larger budgets (full 4.2k+4.2k training pairs,
50 epochs, more MC samples) should close part of the gap; the remaining
difference is dominated by the number of training sub-graphs, which we cap for
CPU-only replication.

## Deviations and limitations

1. **Training schedule.** Algorithm 1 describes inference. We use MLE
   pre-training followed by cross-entropy on the fused likelihood; the second
   stage is essential (see `docs/method.md`).
2. **Score normalisation.** The KL is averaged per candidate edge
   (`structure.normalize_score`); the paper sums over the sub-graph. Both are
   supported.
3. **Timestep averaging.** `num_mc_samples: null` averages over all `T` steps
   (paper). For large `T` this is slow; the shipped configs use 8 MC steps.
4. **Feature-space transfer.** Datasets have different attribute widths, so
   cross-dataset evaluation projects attributes with a seeded random projection
   (`SubgraphExtractor._project_to`). The paper does not describe this step.
5. **`Router`** is not auto-downloadable (SEAL data bundle, not redistributable);
   place `Router.mat` under `data/raw/router/`.
6. **Attacks.** DeepRobust is not a dependency; random and DICE perturbations
   share the same interface so a DeepRobust wrapper can be added in
   `sgdiff/data/attack.py`.
7. **Baselines** are re-implementations sharing the SGDiff data pipeline, not the
   authors' code, and NeoGNN is not implemented.

## Bug journal (useful when reading the code)

* Removing the target edge only from the induced matrix still leaks the label
  through DRNL distances (`u`/`v` both get label 1). The edge is therefore
  blocked inside the BFS (`sgdiff/data/subgraph.py`).
* Capping `max_train_pairs` by slicing `[pos; neg]` silently keeps only
  positives. `_cap_eval` now samples `cap/2` positives and `cap/2` negatives.
* A uniform D3PM schedule with `β_end = 0.5` spends most steps in pure noise and
  the averaged KL loses all class signal; the default is `β_end = 0.2`.
* `device="auto"` must be resolved before moving tensors
  (`sgdiff.utils.seed.resolve_device`).
* The likelihood scores are Monte-Carlo estimates; evaluation fixes the RNG seed
  (`score_pairs(..., seed=0)`) so checkpoints are reproducible.
