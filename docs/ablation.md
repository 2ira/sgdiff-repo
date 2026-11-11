# Likelihood ablations: is the score really conditioned?

`scripts/run_ablation.py` implements the check that GCCM (arXiv:2605.05689,
2026) argues every conditional-diffusion predictor needs: **does the reported
accuracy come from the conditional likelihood, or from machinery around it?**

Everything except the ablated quantity is held fixed: same checkpoint, same
evaluation pairs as `scripts/run_benchmark.py`, same fusion weights, same
seeds. The module is `sgdiff/engine/ablation.py`. Two budgets are reported: a
cheap CPU profile (§3) and the full training split (§4); the fusion weights and
the variance decomposition in §5 turn "the feature stream is dead" from an
inference into a reading.

## 1. Variants

| variant | what changes | what a *healthy* result looks like |
|---------|--------------|-----------------------------------|
| `reference` | nothing (paper scoring) | baseline |
| `same_condition` | both classes scored with `y = 0` | AUC collapses to ~0.5 - otherwise the class signal never came from `y` |
| `condition_flip` | class `c` scored with `y = 1-c` | AUC ≈ 1 - reference (monotone flip) |
| `label_shuffle` | DRNL labels permuted inside each sub-graph | noticeable drop if the structural prior matters |
| `single_t_low` | one diffusion step at 5% of `T` | drop if multi-step denoising does real work |
| `single_t_mid` | one step at 50% of `T` | same |
| `single_t_full` | one step at 100% of `T` | same |
| `structure_only` | feature likelihood zeroed in the fusion | tells which stream carries AUC |
| `feature_only` | structure likelihood zeroed | same |
| `trained_without_condition` | **retrain**: stage-1 condition randomized, denoisers frozen in stage 2 | ~0.5 |
| `trained_without_condition_unfrozen` | same retrain, stage 2 may back-propagate into the denoisers | shows whether the fusion loss re-teaches `y` |

`single_t_*` is the GCCM test proper: if the score is the same at `t = 0.05T`
and `t = T`, the denoiser is a deterministic regressor of the input sub-graph
and the "diffusion" contributes no extra information.

Freezing the denoisers in the clean retrain matters: with an unfrozen stage 2
the cross-entropy gradient flows into the denoisers and re-introduces the class
condition that stage 1 never saw, so the ablation silently measures nothing.
Both versions are reported for exactly that reason.

## 2. Running it

```bash
# cheap budget: 2 000 pairs, 8 epochs, 40-node cap (~12 min, includes retrains)
python scripts/run_ablation.py --dataset cora --profile cpu --seed 0

# full training split: 8 444 pairs, 20 epochs, 60-node cap (~75 min for the
# reference model + ~1 min of scoring); retrains omitted
python scripts/run_ablation.py --dataset cora --profile full --seed 0 \
    --no-retrain --tag full

python scripts/run_ablation.py --dataset synthetic --profile smoke   # ~1 min
```

Output: `results/ablation_<dataset>[_<tag>].{json,md}` with per-variant metrics,
score standard deviation, per-stream contributions and wall-clock.

The `paper` profile (50 epochs, **no** node cap) was measured and is not
runnable here: with uncapped 1-hop sub-graphs one epoch exceeded 3.5 min on an
M-series CPU, i.e. >2.5 h for the reference run alone, before the retrains. The
`full` profile is the repo's designated stand-in - same data and
hyper-parameters, shorter schedule, 60-node cap.

## 3. Results: cheap budget (Cora, `cpu`, seed 0, 500 test pairs)

| variant | training | scoring | AUC | AP | Hit@100 | score std |
|---|---|---|---|---|---|---|
| reference | condition=true | condition=true, labels=true, stream=both, t=MC-avg | 0.7248 | 0.7856 | 0.9500 | 2.0095 |
| same_condition | condition=true | condition=same | 0.5000 | 0.6109 | 0.6900 | 0.2037 |
| condition_flip | condition=true | condition=flip | 0.2747 | 0.3957 | 0.3800 | 2.0249 |
| label_shuffle | condition=true | labels=shuffle | 0.7171 | 0.7828 | 0.9500 | 2.0065 |
| single_t_low | condition=true | t=0.05·T | 0.7290 | 0.7827 | 0.9300 | 2.0204 |
| single_t_mid | condition=true | t=0.5·T | 0.7215 | 0.7834 | 0.9500 | 2.2351 |
| single_t_full | condition=true | t=1·T | 0.7188 | 0.7812 | 0.9200 | 2.2259 |
| structure_only | condition=true | stream=structure | 0.7237 | 0.7764 | 0.9500 | 2.0113 |
| feature_only | condition=true | stream=feature | 0.5204 | 0.5067 | 0.5000 | 0.0083 |
| trained_without_condition | condition=random, denoisers frozen in stage 2 | paper scoring | 0.4750 | 0.5026 | 0.5000 | 0.0268 |
| trained_without_condition_unfrozen | condition=random | paper scoring | 0.7201 | 0.7923 | 0.9500 | 1.2189 |

This table was produced before the fusion/variance reporting existed, so
`results/ablation_cora.json` has no `fusion` block; the weights are read from the
full-budget run in §5.

## 4. Results: full training split (Cora, `full`, seed 0, 500 test pairs)

Same checkpoint for every row; 20 epochs (4 pre-train), 8 444 training pairs,
8 Monte-Carlo steps, 60-node sub-graphs. The reference model needed 74.6 min to
train and 0.8 min for the nine evaluations (`results/ablation_cora_full.json`).

| variant | AUC | AP | Hit@100 | score std |
|---|---|---|---|---|
| reference (MC-avg) | 0.7179 | 0.7841 | 0.9500 | 2.0000 |
| same_condition | 0.4951 | 0.6017 | 0.6800 | 0.1505 |
| condition_flip | 0.2800 | 0.4432 | 0.4600 | 2.0009 |
| label_shuffle | 0.7288 | 0.8008 | 0.9400 | 1.9717 |
| single_t_low (0.05·T) | 0.7298 | 0.7678 | 0.9000 | 2.0568 |
| single_t_mid (0.5·T) | 0.7166 | 0.7967 | 0.9400 | 2.0341 |
| single_t_full (1·T) | 0.7282 | 0.8041 | 0.9300 | 2.0053 |
| structure_only | 0.7181 | 0.7782 | 0.9500 | 1.9945 |
| feature_only | 0.4674 | 0.4565 | 0.3600 | 0.0041 |

**The answer to the budget question is no: the `t`-flatness is not an artefact
of a short schedule.** With 4.2x the training pairs and 2.5x the epochs the four
scoring depths still span 0.013 AUC (0.7166-0.7298), and the cheapest setting
(one step at 5% of `T`) is the highest of the four. Training did move the
validation AUC (0.724 at epoch 7 -> 0.764 at epoch 15) and the structure loss
(0.59 -> 0.54), so the model was still learning; the *scoring* remained
depth-insensitive.

The extra training did not raise the test AUC of the reference model
(0.7179 here vs 0.7248 at the cheap budget). The two runs differ in data,
schedule *and* node cap, and 500-pair test pools are noisy, so this is not
evidence that more training hurts - only that this dataset saturates around
0.72 AUC under both budgets.

## 5. Fusion weights and what each stream contributes (full budget)

| model | eta_1 (feature) | eta_2 (structure) | delta | std(eta_1·d_feat) | std(eta_2·d_struct) | var share feat | corr |
|---|---|---|---|---|---|---|---|
| reference | 0.6082 | 2.3003 | -10.1464 | 0.0042 | 2.0011 | 0.000 | -0.013 |

`d_stream = score(class 1) - score(class 0)` over the test pairs.

Read it as follows:

* `eta_1` is **not** driven to zero (0.61 here, 0.87 at the cheap budget). The
  optimiser happily keeps a large weight on the feature likelihood.
* But the feature term's spread is 0.004 against 2.00 for the structure term:
  a **0.000 variance share**. The weight is large precisely because the stream
  is nearly constant - a constant times a weight is still a constant.
* `delta = -10.15` shows the fusion is compensating an offset, not calibrating
  two likelihoods to a common scale (AUC is rank-based, so `delta` cannot help
  accuracy).
* The two terms are essentially uncorrelated (r = -0.013), so the structure
  stream is not "explaining away" the feature stream either.

Consequence: reporting `eta_1` alone would have suggested the feature stream is
used; the variance decomposition shows it is not. That is why both are stored.

## 6. What it says

1. **The condition is genuinely used.** `same_condition` drops AUC to 0.495/0.500
   and shrinks the score spread from 2.00 to 0.15-0.20: the likelihood ratio
   really is driven by `y`. SGDIFF is *not* the failure mode GCCM describes at
   the conditioning level.
2. **The diffusion depth is decorative.** One step at 5% of `T` matches (and
   slightly beats) the paper's Monte-Carlo average, at both budgets and after
   2.5x the training. The score behaves like a one-shot denoising regressor:
   the extra steps add cost, not information. This is the GCCM shortcut in its
   `t` dimension and the single most important caveat for the "generative
   likelihood" framing.
3. **The DRNL labels carry no usable information** at `k = 1`. Permuting them
   inside each sub-graph costs nothing (full budget: 0.7288 shuffled vs 0.7179
   reference - the shuffled variant is nominally *higher*, i.e. within noise).
   The automorphic-node argument (C4) is therefore not supported at this depth.
4. **The feature stream is dead, but not because `eta_1` is zero.** See §5:
   `feature_only` is at chance (0.4674, score std 0.004) and the feature term
   contributes 0.000 of the logit variance while `eta_1 = 0.61`. The Bayesian
   decomposition into structure + feature is carried entirely by the structure
   term on Cora; the feature DDPM's MSE stays pinned at ~1.0 (it predicts zero)
   at both budgets.
5. **Stage 2 can manufacture conditioning.** Training stage 1 with a random
   condition and leaving stage 2 unfrozen still reaches 0.7201 AUC - the fusion
   cross-entropy re-teaches the denoisers the label they never saw during MLE.
   Freezing them collapses the score to chance (0.4750, score std 0.027). On
   this dataset the discriminative signal is produced by *the cross-entropy
   stage*, not by the class-conditional likelihood estimation.

## 7. Limitations

* Single seed per budget; the 500-pair test pools are noisy at the ±0.01 AUC
  level, which is the size of the effects claimed in items 2 and 3.
* The `paper` profile was not run (see §2 for the measured cost), so "decorative
  depth" is established for the `cpu` and `full` budgets, not for the published
  one.
* The retrains exist only at the cheap budget; the full-budget table therefore
  has no randomized-condition rows.
* `label_shuffle` keeps the label multiset and destroys only the node-to-radius
  correspondence.
* Model selection uses validation AUC on the same pool as the test metric family;
  the reference checkpoints were chosen by the trainer, not re-tuned per
  ablation.

## 8. Follow-ups this suggests

| follow-up | why |
|-----------|-----|
| swap the DDPM score for flow matching (CPD-PIFM) | makes the depth question moot by construction and is cheaper |
| fix the feature DDPM | MSE ~1.0 means it never learned; the "Bayesian decomposition" claim cannot be assessed until it does |
| per-timestep AUC curve (t = 1..T) | a single-step sweep would show where the (absent) information sits |
| full-ranking evaluation | AUC on a 500-pair random-negative pool is the weakest link in every number here |
