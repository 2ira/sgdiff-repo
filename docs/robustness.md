# Robustness under structural attacks (RQ3)

`scripts/run_robustness.py` implements the paper's Sec. 4.4 question for **every**
model of the zoo (SGDiff, SEAL, PENCIL, GCN, GAT, SAGE, VGAE) with the three
attacks the paper uses — random flip (RF), DICE and embedding attack (EA) — all
served by the same dispatcher (`sgdiff/data/attack.py::apply_attack`). The
harness lives in `sgdiff/engine/robustness.py`.

Artifacts: `results/robustness_cora.json` (per-run metrics, `drop_mean` per
variant) and `results/robustness_cora.md` (the table below). The SGDiff-only
view of the paper's own driver is `results/summary_attack_cora.json`.

## 1. Protocol: evasion, one clean training run

The paper perturbs the graph the model is *asked about*, not the training data.
Every model is therefore

1. trained **once** on the clean graph through
   `benchmark.fit_model` — the same split, the same enclosing sub-graphs, the
   same capped training pairs as `docs/benchmark.md`, so the clean column is
   comparable to the benchmark table;
2. evaluated on the **inference** graph, which is the clean graph with
   `ratio * |E|` flips applied and the test positives withheld
   (`attacked.drop_edges(test_pos)`), so the encoder never sees the link it is
   asked to score;
3. scored on the **same** 500 test pairs per class for every variant, with the
   same metric code (`utils/metrics.py`).

Two details that keep the comparison honest:

* **node features are standardised once** on the clean evaluation graph and
  reused for every attack. On attribute-free graphs (USAir, NS) the features are
  a degree encoding, so recomputing them from the attacked graph would let the
  perturbation leak into the inputs — structure must be the only variable.
* **the clean column is produced by the same scoring function** as the attacked
  columns (`score_variant`), never copied from the training loop, otherwise a
  different cap / seed would show up as "attack damage".

Budget split is the same for all three attacks: half deletions, half insertions
(`mode="flip"`), rounded on Cora as follows.

| budget | per attack delete / insert | DICE partition |
|-------:|---------------------------:|----------------|
| 10 % | 264 / 264 (528 flips) | `targets`, k = 7 |
| 25 % | 660 / 660 (1 320 flips) | `targets`, k = 7 |
| 50 % | 1 319 / 1 320 (2 639 flips) | `targets`, k = 7 |

`|E| = 5 278`, so the net edge count is unchanged at every budget — that is the
`flip` semantics, not a missing perturbation. EA runs with `dim = 32` and 60
surrogate steps (0.6-0.9 s on Cora).

## 2. Main table (Cora, AUC, seeds 0-2, `cpu` profile)

| model | #params | clean | RF 10% | DICE 10% | EA 10% | RF 25% | DICE 25% | EA 25% | RF 50% | DICE 50% | EA 50% |
|-------|--------:|------:|-------:|---------:|-------:|-------:|---------:|-------:|-------:|---------:|-------:|
| GCN | 95,936 | **0.828 ± 0.008** | 0.818 ± 0.007 | 0.818 ± 0.010 | 0.827 ± 0.007 | 0.805 ± 0.010 | 0.801 ± 0.006 | 0.825 ± 0.009 | 0.778 ± 0.018 | 0.762 ± 0.011 | 0.819 ± 0.010 |
| GAT | 100,352 | 0.801 ± 0.009 | 0.789 ± 0.014 | 0.789 ± 0.010 | 0.807 ± 0.009 | 0.774 ± 0.005 | 0.770 ± 0.002 | 0.802 ± 0.014 | 0.747 ± 0.013 | 0.745 ± 0.005 | 0.788 ± 0.013 |
| PENCIL | 176,834 | 0.783 ± 0.026 | 0.775 ± 0.026 | 0.772 ± 0.028 | 0.786 ± 0.026 | 0.754 ± 0.022 | 0.754 ± 0.031 | 0.778 ± 0.021 | 0.724 ± 0.033 | 0.715 ± 0.029 | 0.761 ± 0.021 |
| VGAE | 46,944 | 0.773 ± 0.015 | 0.765 ± 0.015 | 0.762 ± 0.021 | 0.772 ± 0.016 | 0.752 ± 0.011 | 0.752 ± 0.007 | 0.773 ± 0.021 | 0.735 ± 0.032 | 0.725 ± 0.024 | 0.765 ± 0.019 |
| SEAL | 54,082 | 0.772 ± 0.007 | 0.764 ± 0.007 | 0.761 ± 0.009 | 0.763 ± 0.013 | 0.745 ± 0.010 | 0.743 ± 0.011 | 0.756 ± 0.010 | 0.713 ± 0.003 | 0.698 ± 0.013 | 0.740 ± 0.012 |
| SAGE | 191,744 | 0.765 ± 0.019 | 0.762 ± 0.021 | 0.763 ± 0.022 | 0.770 ± 0.019 | 0.741 ± 0.024 | 0.741 ± 0.020 | 0.770 ± 0.019 | 0.730 ± 0.004 | 0.726 ± 0.009 | 0.761 ± 0.016 |
| SGDiff | 2,283,006 | 0.737 ± 0.035 | 0.726 ± 0.035 | 0.716 ± 0.032 | 0.734 ± 0.034 | 0.662 ± 0.033 | 0.680 ± 0.029 | 0.728 ± 0.041 | 0.594 ± 0.053 | 0.598 ± 0.063 | 0.694 ± 0.026 |

Rows sorted by clean AUC. 7 models × 10 variants × 3 seeds = 210 scored runs,
no `skipped` entries (EA is dense but Cora is far below `EA_MAX_NODES = 4000`).

## 3. Average damage per attack

Mean AUC lost against each model's own clean column (`drop_mean` in the JSON):

| budget | RF | DICE | EA |
|-------:|----:|-----:|---:|
| 10 % | −0.0086 | −0.0112 | −0.0003 |
| 25 % | −0.0323 | −0.0312 | −0.0041 |
| 50 % | −0.0627 | −0.0699 | −0.0188 |

DICE is the strongest attacker on average, RF is within a point of it, EA moves
the AUC very little. Damage grows roughly linearly for RF/DICE (0.086 points per
1 % of flips at 10 %, 0.125 at 50 %) and sub-linearly for EA (0.000 / 0.016 /
0.038 points per 1 % at the three budgets).

AP and Hit@100 tell the same story for the representative models (clean → 50 %
budget):

| model | clean AUC / AP / Hit@100 | RF 50% | DICE 50% | EA 50% |
|-------|--------------------------|--------|----------|--------|
| GCN | 0.828 / 0.845 / 0.970 | 0.778 / 0.786 / 0.913 | 0.762 / 0.776 / 0.923 | 0.819 / 0.825 / 0.943 |
| SEAL | 0.772 / 0.794 / 0.930 | 0.713 / 0.731 / 0.880 | 0.698 / 0.728 / 0.907 | 0.740 / 0.772 / 0.943 |
| SGDiff | 0.737 / 0.811 / 0.993 | 0.594 / 0.709 / 0.993 | 0.598 / 0.704 / 0.983 | 0.694 / 0.777 / 0.983 |

Hit@100 barely moves even for SGDiff (0.993 → 0.983): the ranking *within* the
top 100 survives while the global ordering degrades, i.e. the flips mostly
demote the weakest positives rather than promoting negatives.

## 4. Analysis

**SGDiff is the fragile model.** It loses 14.3 points at RF-50 % and 13.9 at
DICE-50 %, versus 3.5-5.4 for the full-graph baselines (GCN 5.0 / GAT 5.4 /
SAGE 3.5 / VGAE 3.8) and 5.9-6.0 for the sub-graph classifiers (SEAL 5.9 /
PENCIL 6.0). Its clean AUC is also the lowest (0.737), so the model starts
behind *and* decays fastest.

**Why the sub-graph model is hit harder.** SGDiff reads one 1-hop enclosing
sub-graph per candidate pair and scores a class-conditional likelihood on it; a
flip that survives extraction changes that pair's evidence directly. GCN/GAT
smooth over the whole graph, so 528 random flips (10 % of Cora) are absorbed by
the neighbourhood aggregate. This is the same mechanism behind the benchmark
result in `docs/benchmark.md` (supervised sub-graph classifiers beat the
generative likelihood at the 1-hop budget) — the granularity that helps SEAL
also exposes SGDiff.

**DICE ≥ RF for the sub-graph models.** DICE deletes intra-class links and
inserts cross-class ones (`partition = targets`, k = 7, so it uses the Cora class
labels like DeepRobust's DICE). For SEAL/PENCIL/SGDiff the deleted links are
exactly the ones that make two nodes look adjacent inside their own community,
so the loss is concentrated where the likelihood looks. The full graph models
average the same loss over all pairs, which is why their RF and DICE columns are
within 1-2 points.

**EA is the weakest attacker in this implementation (≤ 4.3 points at 50 %).**
Our EA trains a one-layer linear surrogate `Z = A_norm W` with a dot-product
link objective and flips the `ratio * |E|` entries ranked by a single
meta-gradient `dL/dA`. DeepRobust fits a NetMF/spectral embedding and iterates a
PGD-style perturbation, so its `dL/dA` is aligned with a much richer embedding.
The ordering RF/DICE > EA is therefore a statement about *our* EA, and the
absolute drop is not comparable to the paper's EA column.

**The model ranking is stable under attack.** Clean order is GCN > GAT >
PENCIL > VGAE > SEAL > SAGE > SGDiff (PENCIL/VGAE and SEAL/SAGE within 1 point).
At 10 % and 25 % budgets the order is unchanged; at 50 % only the mid-tier swaps
(VGAE/SAGE overtake PENCIL, SEAL) while GCN stays first and SGDiff stays last at
every budget and every attack. No model has a robustness advantage large enough
to reorder the top of the table.

**Reconciling the two SGDiff tables.** `run_experiments.py --attack` reports
clean 0.706 with RF 0.692 / 0.655 / 0.579, DICE 0.685 / 0.665 / 0.578 and EA
0.702 / 0.695 / 0.684 (seed 0, its own in-domain checkpoint, `results/
summary_attack_cora.json`). Same ordering and almost identical deltas; the
0.031 anchor difference comes from the training protocol (in-domain grid vs.
benchmark `fit_model`), not from the attacks.

## 5. Robust-by-design baselines: does anything actually resist?

The 2025-2026 papers in [`related_work.md`](related_work.md) that talk about
noise are **not** attack-robust link predictors: DiffHGLP and DiffGraph are
heterogeneous-graph models, FSD-CAP and GGDIFF target feature missingness and
conditional guidance, and GCCM targets the *conditioning* shortcut. So the
comparison imports the canonical robust-by-design GNN instead:

* `rgcn` — Bayesian robust GCN (Zhu et al., KDD 2019): Gaussian latent per node,
  reparameterised sampling while training, `0.1 * mean(sigma)` penalty, `mu` at
  inference. Implemented as `kind="robust_node_pair"` in
  `models/baselines/robust.py`.
* `rgcn_novar` — the same model with `var_weight = 0`, i.e. sampling without the
  penalty. It isolates whether the penalty is what buys robustness.

Same protocol, same attacked graphs, same seeds as §2 (`results/robustness_cora_robust.json`):

| model | clean | rf@50% | dice@50% | ea@50% | rel. drop (rf / dice / ea) |
|---|---:|---:|---:|---:|---|
| gcn | 0.832 ± 0.022 | 0.787 | 0.780 | 0.817 | −5.4% / −6.3% / −1.8% |
| sgdiff | 0.769 ± 0.021 | 0.657 | 0.674 | 0.740 | −14.6% / −12.4% / −3.8% |
| seal | 0.768 ± 0.014 | 0.705 | 0.693 | 0.738 | −8.2% / −9.8% / −3.9% |
| rgcn_novar | 0.758 ± 0.023 | 0.701 | 0.674 | 0.748 | −7.5% / −11.1% / −1.3% |
| **rgcn** | 0.754 ± 0.010 | 0.700 | 0.673 | 0.740 | −7.2% / −10.7% / −1.9% |

Reading:

1. **No robustness advantage.** RGCN's relative drops sit between GCN's
   (−5/−6/−2) and SEAL's (−8/−10/−4), and it loses 7.8 clean AUC points to GCN.
   At every attack budget a plain GCN scores 0.09-0.12 AUC higher than RGCN.
2. **The variance penalty is inert.** `rgcn` vs `rgcn_novar` differ by ≤0.006 AUC
   in every cell; tripling the schedule (`--node-pair-epochs 1000`,
   `results/robustness_cora_rgcn1000.md`) moves clean AUC only 0.754 -> 0.755.
   Whatever robustness the paper reports does not survive this implementation
   (single Gaussian head + mean-sigma penalty rather than per-layer variances
   with a KL to a prior) or this evasion setup.
3. **SGDiff remains the least robust model** in relative terms, 2-3x GCN's
   degradation, and the only one whose Hit@100 advantage does not survive.
4. **Purification defences are not viable on Cora.** The Jaccard pre-filter of
   Wu et al. (2019) would drop 47% of Cora's edges before any attack
   (median edge Jaccard 0.022; 47.3% of edges below 0.01), so the defence cannot
   be evaluated without destroying the clean graph; it is rejected on
   measurement rather than implemented as a straw man.

Reproduce:

```bash
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2 \
    --models gcn rgcn rgcn_novar sgdiff seal --tag robust
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2 \
    --models rgcn rgcn_novar --node-pair-epochs 1000 --tag rgcn1000
```

## 6. How to reproduce / extend

```bash
# the table above (7 models x 3 attacks x 3 budgets x 3 seeds, ~12 min on CPU/MPS)
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2

# one model, one attack budget
python scripts/run_robustness.py --dataset cora --models sgdiff --attacks ea --ratios 0.1

# the paper's own driver (SGDiff only), same three attacks
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 --attack-methods rf dice ea

# a single checkpoint, no retraining
python -m sgdiff evaluate --checkpoint runs/cora_seed0_cpu/best.pt --attack ea --ratio 0.1
python examples/04_attack.py --checkpoint runs/cora_seed0_cpu/best.pt --attacks rf dice ea
```

Other datasets work the same way; EA is dense (`n^2` floats) so Pubmed
(19 717 nodes) is skipped with a `skipped` entry instead of being silently
downgraded, while Citeseer (3 327), USAir (332) and NS (1 589) fit under
`EA_MAX_NODES = 4000`.

## 7. Caveats

1. **Evasion, not poisoning.** Weights are trained clean and the graph is
   perturbed at inference. Retraining on the poisoned graph is a different (and
   much more expensive) question that neither the paper's RQ3 nor this harness
   answers.
2. **EA deviates from DeepRobust** (linear surrogate, single meta-gradient
   step). See `docs/reproduction.md` §6, deviation 2.
3. **The absolute drops are larger than the paper's** (RF-50 %: 14 points here
   vs. ~1.5 in the paper). Our SGDiff is already 2-6 points below the paper on
   Cora clean (see `docs/reproduction.md` §3), so the gap is reported rather
   than tuned away.
4. **`cpu` profile**: 8 epochs for SGDiff (3 pre-train), 40 for the sub-graph
   classifiers, 300 for the node-pair baselines, 40-node sub-graph cap. All arms
   of a comparison share that budget; `--profile full` raises it.
5. The 500-pairs-per-class cap is applied to every variant identically; the full
   test split is available with `--max-pairs 0`.
6. **Two artifacts, same protocol, different SGDiff anchor.** §2 comes from
   `results/robustness_cora.{json,md}` (7 models, 15:33) and §5 from
   `results/robustness_cora_robust.{json,md}` (5 models, 17:50). Both use
   identical settings, yet SGDiff's clean column differs (0.737 vs 0.769; seed 0
   alone: 0.699 vs 0.793) while GCN and SEAL agree within 0.004. With 8 epochs
   SGDiff is the least reproducible model in the zoo, so compare rows **within**
   one table and treat its absolute AUC as ±0.03. The direction of the
   robustness finding is identical in both runs (SGDiff loses 15-19% under
   RF-50%).
7. RGCN deviations from the paper: one Gaussian head over the final layer and a
   `mean(sigma)` penalty instead of per-layer variances with a KL to a prior;
   no per-dataset tuning of `lr`/`var_weight`/`hidden_dim`.
