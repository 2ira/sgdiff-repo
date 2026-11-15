# Key findings

> **All numbers below are `sgdiff-paper-v2` at the `paper` profile**
> (Table 4 exactly: full splits, uncapped sub-graphs, all Monte-Carlo steps).
> The 2026-11-14/15 GPU grid: 5 datasets × 10 seeds in-domain, RQ1 transfer
> 5 sources × 10 seeds, RQ2 1% low-data × 10 seeds, RQ3 attack + all-model
> robustness × 3 seeds. The 2026-11-13 audit had found the earlier
> implementation unfaithful (categorical diffusion/posteriors, non-Eq. (11)
> feature denoiser, independent noise between classes); its artifacts were
> deleted on 2026-11-15 and nothing on this page relies on them.

One-page synthesis of what this reproduction measured, claim by claim. Every
number comes from an artifact in `results/sgdiff-paper-v2/` and is reproducible
with the commands in [`reproduction.md`](reproduction.md) §3; the detailed
tables live in [`reproduction.md`](reproduction.md) §0 and
[`robustness.md`](robustness.md).

## 1. Verdict on the paper's four claims

| claim | paper | fresh measurement | verdict |
|-------|-------|-------------------|---------|
| C1 zero-shot transfer | Sec. 4.2 | matrix §0.2: e.g. citeseer→cora 0.834, →ns 0.957 | **reproduced** |
| C2 1% training data | Sec. 4.3 / Fig. 2 | cora AUC 0.711 ± 0.092 | **reproduced** (incl. the large seed variance) |
| C3 robustness under attack | Sec. 4.4 / Table 2 | EA@25% costs 2.4 points (0.858→0.834) | **reproduced** (relative); SEAL/PENCIL stay stronger in absolute terms |
| in-domain accuracy | Table 2 / Sec. 4.1 | cora 0.862 / citeseer 0.833 / pubmed 0.902 / usair 0.868 / ns 0.965 | measured; no clean-AUC reference exists in the paper to compare against |

## 2. Is the model actually better than simpler baselines?

On the same Cora split (paper profile): SGDiff v2 0.858 vs SEAL 0.872 / PENCIL
0.883 / GAT 0.850 / GCN 0.815 / VGAE 0.816 / SAGE 0.760 (3-seed means, and
SEAL reaches 0.888 in `run_baselines.py`). SGDiff v2 sits in the top half of
the zoo but is not the strongest clean performer; its edge is the Hit@100
ranking metric (0.953-0.988 across datasets) and the robustness profile above.

## 3. Robustness: the paper's C3 holds relatively

See [`robustness.md`](robustness.md) for the full zoo table. SGDiff v2 loses
2.4 / 4.0 / 4.1 points under EA / DICE / RF at the 25% budget — the smallest
EA damage of any *generative* model in the zoo, and no catastrophic collapse
at the 50% budget. The absolute ranking under attack is PENCIL ≥ SEAL >
SGDiff ≈ GAT > GCN ≈ VGAE > SAGE.

## 4. Where the paper is under-specified

1. Optimisation (batch, learning rates, epochs) is not reported; our choices
   are recorded in every run's `resolved_config.json` and `protocol` block.
2. The EA attack implementation is not released; ours is a linear surrogate.
3. Which `ns` applies at transfer test time is not stated; the fresh matrix
   uses the strict `source_ns` reading.
4. Router is not redistributable, so 5 of 6 graphs are covered.

## 5. Reproducing every number

```bash
# in-domain grid      -> results/sgdiff-paper-v2/{ds}_seed{K}_paper.json
python scripts/run_experiments.py --datasets cora citeseer pubmed usair ns \
    --seeds 0 1 2 3 4 5 6 7 8 9 --profile paper
# RQ1 transfer        -> results/sgdiff-paper-v2/transfer_{src}_seed{K}_paper.json
python scripts/run_experiments.py --transfer citeseer \
    --targets cora citeseer pubmed usair ns --seeds 0 --profile paper
# RQ2 low data        -> results/sgdiff-paper-v2/lowdata_cora_seed{K}_paper.json
python scripts/run_experiments.py --lowdata cora --seeds 0 --profile paper
# RQ3 SGDiff attacks  -> results/sgdiff-paper-v2/attack_cora_seed{K}_paper.json
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 --profile paper
# RQ3 whole zoo       -> results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.{json,md}
python scripts/run_robustness.py --dataset cora --profile paper --seeds 0 1 2 \
    --output results/sgdiff-paper-v2 --tag _v2
# baselines           -> results/sgdiff-paper-v2/baselines_cora.json
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal \
    --output results/sgdiff-paper-v2/baselines_cora.json
```

## 6. What would change these conclusions

* The transfer matrix is the `source_ns` variant; if the authors used the
  target's `ns`, cells on very heterogeneous pairs (e.g. ns→pubmed) move.
* The EA surrogate is weaker than DeepRobust's attack; a parity EA could
  reorder the robustness table.
* Optimisation budgets are ours; the paper reports none, so its authors may
  have trained longer (Cora validation AUC was still improving at our budget).
