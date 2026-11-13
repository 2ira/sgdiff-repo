# Key findings

> **Historical implementation results — not valid evidence about the original SGDiff method.**
> The 2026-11-13 paper audit found incorrect categorical diffusion/posteriors,
> a non-Eq. (11) feature denoiser, independent noise between classes, and a
> non-paper-default training schedule. The verdicts below describe the old implementation
> only; they must be re-evaluated with `sgdiff-paper-v2` before judging the paper.

One-page synthesis of what this reproduction measured, claim by claim. Every
number below comes from an artifact in `results/` and is reproducible with the
command given in §6; the detailed arguments live in
[`reproduction.md`](reproduction.md), [`benchmark.md`](benchmark.md),
[`ablation.md`](ablation.md) and [`robustness.md`](robustness.md).

Budgets, because several of them differ:

| budget | train pairs | epochs | sub-graph cap | MC steps | used for |
|--------|------------:|-------:|--------------:|---------:|----------|
| `cpu` | 2 000 | 8 (3 pre) | 40 nodes | 4 | benchmark table, RQ1-RQ3, retrain ablations |
| `full` | 8 444 (all) | 20 (4 pre) | 60 nodes | 8 | in-domain summary, published ablations |
| `paper` | all | 50 | none | all | **not runnable here** (>2.5 h per model) |

## 1. Verdict on the paper's four claims

| claim | paper | this reproduction | evidence |
|-------|-------|-------------------|----------|
| **C1** structure alone transfers without re-training | Table 1 | **partially holds**: target averages land 1-13 points below in-domain (source averages 0.713-0.838), and one cell (NS -> Pubmed) is at chance (0.494) | `reproduction.md` §RQ1, `results/transfer_matrix_cpu.json` |
| **C2** generative scoring wins with 1% of the labels | Fig. 2 | **holds, but is schedule-sensitive**: 0.725 ± 0.058 vs 0.751 ± 0.017 at 40 epochs; at 8 epochs the 1% run collapses to 0.49 | `reproduction.md` §RQ2 |
| **C3** likelihood scoring is robust to structural attack | Sec. 4.4 | **reversed**: the least robust model of the nine tested, including a robust-by-design RGCN | §3 below, `results/robustness_cora*.md` |
| **C4** sub-graph modelling fixes automorphic nodes / memory | Sec. 2-3 | **not supported as stated** at `k = 1`: the DRNL labels carry no measurable information and the model costs 40x SEAL's parameters | §4 below, `ablation.md` |

Accuracy context (in-domain, `full`, seeds 0-2): Cora 0.730 ± 0.030, Citeseer
0.778 ± 0.016, USAir 0.929 ± 0.002, NS 0.985 ± 0.002, against the paper's
0.884 / 0.871 / ~0.95 / ~0.97. Attribute-free graphs match or beat the paper;
every attributed graph stays below it, which points at the feature stream (§5).

## 2. Is the model actually better than simpler baselines?

On one identical Cora split (`cpu`, seed 0, 500 test pairs, fingerprints in
`results/benchmark_cora.json`):

| model | AUC | AP | Hit@100 | #params | fit (s) |
|---|---|---|---|---|---|
| gcn | 0.8296 | 0.8352 | 0.9300 | 95 936 | 0.3 |
| gat | 0.7915 | 0.7739 | 0.8400 | 100 352 | 0.2 |
| seal | 0.7883 | 0.7950 | 0.8600 | 54 082 | 12.6 |
| pencil | 0.7755 | 0.7901 | 0.8400 | 176 834 | 16.2 |
| vgae | 0.7667 | 0.7743 | 0.9300 | 46 944 | 0.5 |
| sage | 0.7525 | 0.7537 | 0.8900 | 191 744 | 0.3 |
| **sgdiff** | **0.7314** | 0.7930 | **0.9500** | 2 283 006 | 104.4 |

SGDiff ranks last on AUC, second on AP and first on Hit@100: its fused score is
sharp at the top of the ranking and blunter over the bulk. It costs 40x the
parameters and ~100x the wall-clock of a sub-graph GCN. On the `full` profile
SEAL reaches 0.855 AUC on the *same* 1-hop sub-graphs (`README.md` §Measured
results), so the deficit is not the data.

## 3. Robustness: the paper's C3 is reversed

Evasion protocol, 3 seeds, `cpu` budget, AUC (`results/robustness_cora.md`):

| model | clean | RF 50% | DICE 50% | EA 50% | relative drop (RF / DICE / EA) |
|---|---:|---:|---:|---:|---|
| gcn | 0.828 ± 0.008 | 0.778 | 0.762 | 0.819 | −6% / −8% / −1% |
| gat | 0.801 ± 0.009 | 0.747 | 0.745 | 0.788 | −7% / −7% / −2% |
| pencil | 0.783 ± 0.026 | 0.724 | 0.715 | 0.761 | −8% / −9% / −3% |
| vgae | 0.773 ± 0.015 | 0.735 | 0.725 | 0.765 | −5% / −6% / −1% |
| seal | 0.772 ± 0.007 | 0.713 | 0.698 | 0.740 | −8% / −10% / −4% |
| sage | 0.765 ± 0.019 | 0.730 | 0.726 | 0.761 | −5% / −5% / −0.5% |
| **sgdiff** | 0.737 ± 0.035 | 0.594 | 0.598 | 0.694 | **−19% / −19% / −6%** |

SGDiff degrades two to three times faster than every other model under
random-flip and DICE, and it is the only model whose AUC falls below 0.70. Its
`Hit@100` advantage from §2 does not protect it. Any "generative scoring is more
robust" claim has to be made against a baseline set, and against this one it
does not hold.

Defences were then tested properly, with the 2025-2026 line rather than a 2019
model: **GDDM** (KDD 2026) and **GPR-GAE** (ICML 2025) purify the inference graph
before any predictor scores it, and **RGCN** (KDD 2019) is kept as the historical
anchor (`docs/robustness.md` §5, paired runs, 3 seeds):

| model | clean | rf@50% | dice@50% | clean +gddm | rf@50% +gddm | dice@50% +gddm | clean +gpr | rf@50% +gpr | dice@50% +gpr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gcn | 0.832 | 0.787 | 0.780 | 0.819 | 0.774 | 0.773 | 0.828 | 0.758 | 0.753 |
| rgcn (2019) | 0.755 | 0.702 | 0.675 | 0.752 | 0.669 | 0.651 | 0.756 | 0.638 | 0.595 |
| sgdiff | 0.750 | 0.618 | 0.624 | 0.732 | 0.590 | 0.612 | 0.743 | 0.576 | 0.546 |

Three findings, all measured:

1. **Purification lowers accuracy almost everywhere** - 17 of the 18 defended
   cells are below their undefended counterpart, including on the clean graph
   (the exception is `rgcn` + GPR clean, +0.001, inside the seed spread). The reason is
   measurable rather than mysterious: to raise edge precision from 0.72 to 0.94 the
   GPR purifier has to delete half of the true edges, and RF/DICE remove as many
   edges as they insert, so deletion-dominated purification cannot restore the
   damage.
2. **A 2019 robust classifier is not a fix either**: RGCN is 7.7 points below plain
   GCN clean and degrades slightly faster under RF-50 %; its variance penalty
   changes <=0.006 AUC.
3. **Plain GCN stays the most robust option** at every budget tested, and SGDiff is
   the least robust in relative terms (-17.6 % under RF-50 % here).

## 4. Why: the likelihood audit

Scoring-time and training-time ablations of one checkpoint (`full` budget, 500
test pairs):

| probe | result | reading |
|---|---|---|
| `same_condition` (both classes scored with `y = 0`) | 0.7179 -> 0.4951 | the condition is genuinely used; no leakage |
| `single_t_*` (one step at 5% / 50% / 100% of `T`) | 0.7298 / 0.7166 / 0.7282 | **depth is decorative** - the cheapest step is the best, spread 0.013, and it survives 4.2x data and 2.5x epochs |
| `label_shuffle` (DRNL permuted) | 0.7288 vs 0.7179 | the radius labels carry **no** usable information at `k = 1` |
| `feature_only` | 0.4674, score std 0.004 | feature stream at chance |
| fusion weights | `eta_1 = 0.61`, `eta_2 = 2.30`, feature variance share **4.3e-6** | the feature weight is large only because its score is nearly constant |
| retrain with a random stage-1 condition | frozen denoisers 0.4750 vs unfrozen 0.7201 | the fusion cross-entropy, not the likelihood, separates the classes |

So the model behaves as a **one-shot structural denoiser plus a
cross-entropy-tuned fusion**, not as a two-stage generative likelihood. That is
consistent with §2 (competitive AP/Hit@100, weak AUC) and §3 (fragile under
structure noise).

## 5. Where the paper is under-specified

* `ns` at test time for cross-dataset transfer: the `source_ns` vs `target_ns`
  choice moves target averages by up to 5.6 AUC points (`reproduction.md` §RQ1).
* Training schedule for the 1% study: the claim only survives a long schedule
  (40 epochs); the paper reports neither the schedule nor its sensitivity.
* The feature stream's role: with a dead feature DDPM (MSE ~1.0, it predicts
  zero) the "Bayesian decomposition" is unobservable, yet the paper's advantage
  over baselines is concentrated exactly there (§1).

## 6. Reproducing every number

```bash
pip install -e ".[dev]"

# in-domain grid -> results/summary_indomain_{cpu,full}.json
python scripts/run_experiments.py --datasets cora citeseer pubmed usair ns --seeds 0 1 2 --profile full

# data-identical model comparison -> results/benchmark_<dataset>.{json,md}
python scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0

# likelihood audit -> results/ablation_<dataset>[_<tag>].{json,md}
python scripts/run_ablation.py --dataset cora --profile cpu --seed 0
python scripts/run_ablation.py --dataset cora --profile full --seed 0 --no-retrain --tag full

# robustness of the whole zoo -> results/robustness_<dataset>.{json,md}
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2

# RQ1 / RQ2
python scripts/run_experiments.py --transfer-matrix --profile cpu --seeds 0 1 2
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2 --profile cpu
```

## 7. What would change these conclusions

* **The `paper` budget.** It never ran here: uncapped 1-hop Cora sub-graphs
  exceed 3.5 min per epoch on an M-series CPU (>2.5 h per model). The
  depth-insensitivity in §4 is established for `cpu` and `full`, not for the
  published schedule.
* **A working feature DDPM.** If it learned (MSE ~1.0 now), the fusion could put
  real weight behind it and the attributed-graph gap in §1 might close.
* **Larger evaluation pools.** AUC on 500 random-negative pairs is noisy at
  ±0.01, the size of the effects in §4; full-ranking protocols would be sharper.
* **Baseline tuning.** Baselines use fixed budgets (300 epochs full-graph, 40
  sub-graph) and were not tuned per dataset.
* **Five of six datasets.** Router is not redistributable, so the transfer
  matrix is 5x5.
