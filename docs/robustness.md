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
behind *and* decays fastest. [`attribution.md`](attribution.md) decomposes this
number: the RF / DICE damage is almost entirely its **deletion** half (Cora
`del@50%` −0.24 vs `ins@50%` +0.02), and it is an input-supply effect that scales
with graph sparsity.

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

## 5. Defences: a 2019 robust classifier vs 2025-2026 purifiers

The 2025-2026 papers in [`related_work.md`](related_work.md) are not
attack-robust link predictors, but the *defence* literature moved after 2024 from
"train a robust classifier" to **"purify the graph, leave the predictor
untouched"**. Two recent purifiers are implemented here, alongside the classic
robust classifier kept as a historical anchor:

| name | source | mechanism | kind |
|------|--------|-----------|------|
| `gddm` | GDDM, KDD 2026 (arXiv:2501.11568) | level-conditioned edge denoiser trained on corrupted graphs (random drop + insert); reverse chain keeps edges by probability, structure-driven refiner pins the edge budget to the clean graph, node-feature-constrained regularizer removes feature-inconsistent edges | purifier |
| `gpr` | GPR-GAE, ICML 2025 (arXiv:2605.23239) | graph auto-encoder with a generalized-PageRank filter bank (learnable mixture over 1..K hops), self-supervised reconstruction, plug-and-play | purifier |
| `rgcn` | RGCN, KDD 2019 | Gaussian latent + variance penalty | classifier |

Both purifiers are trained **self-supervised on the clean evaluation graph** (test
edges removed, so they cannot re-insert held-out positives), see no attacked graph
during training, and are applied to the *inference* graph only. Everything else -
split, predictor weights, scored pairs - is held fixed, so `rf@50%` and
`rf@50%|gddm` are a **paired** comparison on the same trained model.

### 5.1 A purifier can only be as good as its edge discrimination

Measured on Cora (clean 4 486 edges, RF-50 % graph 4 694 edges, 28 % of them
injected):

| purifier | AUC at telling clean edges from random pairs | precision after purification | recall of true edges |
|---|---:|---:|---:|
| `gddm` | 0.724 | 0.762 (from 0.719) | 0.60 |
| `gpr` | 0.993 (transductive memorisation) | 0.939 | 0.47 |

The trade-off is the whole story: to raise precision from 0.72 to 0.94 the GPR
purifier has to delete **half of the true edges**. For link prediction that is
fatal - the enclosing sub-graphs lose the structure the predictors read.

### 5.2 Result: purification makes every model worse

Cora, `cpu` profile, seeds 0-2, AUC (`results/robustness_cora_purify2.md`):

| model | clean | rf@25% | dice@25% | rf@50% | dice@50% | clean +gddm | rf@50% +gddm | dice@50% +gddm | clean +gpr | rf@50% +gpr | dice@50% +gpr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gcn | 0.832 | 0.812 | 0.812 | 0.787 | 0.780 | 0.819 | 0.774 | 0.773 | 0.828 | 0.758 | 0.753 |
| rgcn | 0.755 | 0.721 | 0.727 | 0.702 | 0.675 | 0.752 | 0.669 | 0.651 | 0.756 | 0.638 | 0.595 |
| sgdiff | 0.750 | 0.681 | 0.694 | 0.618 | 0.624 | 0.732 | 0.590 | 0.612 | 0.743 | 0.576 | 0.546 |

17 of the 18 defended cells are below their undefended counterpart (-0.005 to
-0.072 AUC); the exception is `rgcn` under GPR on the clean graph (0.755 → 0.756,
+0.001, inside the seed spread). The purifiers cost accuracy even on the **clean**
graph (-0.003 to -0.018), i.e. they are not free insurance.

Two mechanisms, both measured rather than assumed:

1. **The attack flips edges.** RF/DICE at 50 % *remove* about a quarter of the
   true edges while inserting a similar number, so the graph's edge count barely
   moves (4 486 -> 4 694). A purifier that mainly deletes therefore cannot restore
   the missing half of the damage, and the deletion it does perform is dominated by
   true edges (recall 0.47-0.60).
2. **Purifiers are transductive.** The GPR auto-encoder separates clean from random
   pairs at AUC 0.993 on the graph it was trained on, but inserting noise edges
   shifts its GPR-propagated embeddings and the calibrated threshold then removes
   true edges. More steps make it worse, so the reported runs use a single step.

### 5.3 The 2019 robust classifier does not help either

`rgcn` (clean 0.755) is 7.7 AUC points below plain GCN and degrades slightly faster
under RF-50 % (-7.0 % vs -5.4 %). Its variance penalty is inert: `rgcn` vs
`rgcn_novar` differ by <=0.006 AUC in every cell, and a 1000-epoch budget changes
nothing (`results/robustness_cora_rgcn1000.md`).

### 5.4 Reproducibility: one fix, one residual limit

The harness used to seed **once per seed** and then train models in list order, so
the RNG stream a model saw depended on how many models (and purifiers) ran before
it. `run_robustness` now calls `set_seed(seed)` **before every model fit and every
scoring call**, so a model's numbers no longer depend on the model list.

That fix does not make SGDiff bit-reproducible. Two runs of the same
single-seed protocol still differ (clean AUC 0.7031 vs 0.7167 for seed 0), and the
~0.013 gap is **device-level**: the sparse message passing accumulates with
`Tensor.index_add_`, whose atomic ordering is non-deterministic on MPS, so the same
weights produce slightly different logits. Consequences:

* seeds fix the *data* and the *initialisation*, not the floating-point order;
* use `--device cpu` when an exact repeat matters (slower, deterministic for these
  kernels);
* treat SGDiff's absolute AUC as +/-0.02 and prefer paired cells from one run,
  which is what the tables above do.

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

# 2025-2026 purifiers: every attack is scored with and without the defence
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2 \
    --models gcn rgcn sgdiff --attacks rf dice --ratios 0.25 0.5 \
    --defense gddm gpr --purifier-epochs 200 --tag purify2
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
6. **Seeding changed mid-project.** Before the fix in §5.4 the SGDiff anchor
   moved between runs of the same protocol (0.737 / 0.769 / 0.750) because the
   RNG stream depended on the model list; the residual +/-0.013 is MPS atomics
   (see §5.4). Artifacts predating the fix
   (`robustness_cora.md`, `robustness_cora_robust.md`, `_rgcn1000`) are internally
   consistent but not comparable with `_purify2`; use `_purify2` for cross-model
   claims.
7. RGCN deviations from the paper: one Gaussian head over the final layer and a
   `mean(sigma)` penalty instead of per-layer variances with a KL to a prior;
   no per-dataset tuning of `lr`/`var_weight`/`hidden_dim`.
