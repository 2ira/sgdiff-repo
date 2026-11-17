# Line C: label-free FSD + GAE robustness study

Completed: seven models, two datasets, two attribute settings, three seeds and ten structural conditions; all native metrics recovered.

## Method and scope

This study adapts **FSD-CAP stage 1** to the repository's link-prediction interface. The reference is [the author's code](https://github.com/ssjcode/FSD-CAP/blob/a6016131d1a1c362453b9568a87e5749ce07cbd8/fsdcap.py) and [paper](https://openreview.net/pdf?id=LAr64ymrMJ). CAP's node-class-supervised classifier is excluded, so the main comparison uses link labels only. This is a label-free FSD + GAE adaptation, not a reproduction of the complete FSD-CAP system or its published numbers.

Four controls share the same two-layer GAE (hidden 32, latent 16, inner-product decoder): `fsd_none` (zero fill), `fsd_fp` (ordinary propagation), `fsd_gamma1` (author's expanding-subgraph stage with gamma=1), and `fsd` (gamma=1.2). FSD uses 100 propagation iterations per stage and lambda=0.1. The power is **elementwise on normalized adjacency**, as in the author code; it is not a spectral fractional matrix power. Propagation clamps observed features and is recomputed on each attacked inference graph.

The other models are repository SEAL, corrected VGAE, and SGDiff. VGAE now samples with `exp(log_std)`, consistent with its KL term, and evaluates using the posterior mean. The four FSD controls isolate imputation components; GAE versus VGAE also changes the variance head and KL objective, so it does not isolate sampling alone.

## Shared protocol

- Cora and Citeseer; seeds 0, 1, 2.
- Complete attributes and fixed 40% whole-node attribute missingness. The same mask is applied before training to every model within a seed. Node-pair normalization uses observed rows only and leaves missing rows zero.
- Same repository edge splits, capped training-pair sampling, scored test pairs and attack graphs. Validation context contains training edges only; test context contains train + validation edges, with test positive edges removed.
- RF, DICE and the repository's approximate EA at 10%, 25%, 50% edge-flip budgets, plus clean evaluation. Weights remain frozen during structural attacks. DICE uses node labels as attacker information; models receive no additional node-class supervision.
- Repository `cpu` budget profile, executed on CUDA: at most 2,000 training pairs, validation subgraph cap 500 pairs, scored test cap 500 positives + 500 negatives, 1 hop and at most 40 nodes per subgraph, four Monte Carlo samples.
- Native training schedules and early stopping: node-pair models at most 300 epochs, SEAL at most 40, SGDiff 8 diffusion epochs followed by up to 20 fusion epochs. Native trainer-specific preprocessing and validation-pair sampling are retained.
- One NVIDIA RTX PRO 5000 72GB Blackwell; Python 3.11.15, PyTorch 2.7.0+cu128, NumPy 1.26.0, SciPy 1.15.3. Parameter-free FSD propagation uses CPU sparse matrices; learned encoders use CUDA.

## Results

AUC mean ± sample standard deviation across three seeds. All tables use native full-precision result JSON. [All budgets and raw metrics](https://github.com/2ira/sgdiff-repo/tree/feat/line-c-fsdcap/results/line-c-fsd/cuda).

### cora: complete attributes

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---|---|---|---|
| fsd_none | 0.799 ± 0.021 | 0.751 ± 0.024 | 0.747 ± 0.021 | 0.777 ± 0.029 |
| fsd_fp | 0.799 ± 0.021 | 0.751 ± 0.024 | 0.747 ± 0.021 | 0.777 ± 0.029 |
| fsd_gamma1 | 0.799 ± 0.021 | 0.751 ± 0.024 | 0.747 ± 0.021 | 0.777 ± 0.028 |
| fsd | 0.799 ± 0.021 | 0.751 ± 0.024 | 0.747 ± 0.021 | 0.777 ± 0.029 |
| vgae | 0.821 ± 0.005 | 0.777 ± 0.010 | 0.774 ± 0.007 | 0.802 ± 0.005 |
| seal | 0.866 ± 0.010 | 0.778 ± 0.010 | 0.751 ± 0.012 | 0.833 ± 0.015 |
| sgdiff | 0.772 ± 0.018 | 0.695 ± 0.010 | 0.698 ± 0.009 | 0.711 ± 0.031 |

### cora: 40% missing node attributes

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---|---|---|---|
| fsd_none | 0.756 ± 0.005 | 0.703 ± 0.012 | 0.702 ± 0.008 | 0.741 ± 0.006 |
| fsd_fp | 0.778 ± 0.013 | 0.714 ± 0.007 | 0.706 ± 0.017 | 0.751 ± 0.018 |
| fsd_gamma1 | 0.778 ± 0.014 | 0.714 ± 0.007 | 0.706 ± 0.016 | 0.751 ± 0.018 |
| fsd | 0.778 ± 0.016 | 0.714 ± 0.006 | 0.709 ± 0.017 | 0.753 ± 0.019 |
| vgae | 0.791 ± 0.014 | 0.738 ± 0.019 | 0.730 ± 0.015 | 0.768 ± 0.017 |
| seal | 0.868 ± 0.010 | 0.781 ± 0.018 | 0.748 ± 0.010 | 0.836 ± 0.010 |
| sgdiff | 0.785 ± 0.016 | 0.711 ± 0.008 | 0.708 ± 0.019 | 0.727 ± 0.011 |

### citeseer: complete attributes

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---|---|---|---|
| fsd_none | 0.791 ± 0.031 | 0.765 ± 0.025 | 0.768 ± 0.015 | 0.766 ± 0.030 |
| fsd_fp | 0.791 ± 0.031 | 0.765 ± 0.025 | 0.768 ± 0.015 | 0.767 ± 0.030 |
| fsd_gamma1 | 0.791 ± 0.031 | 0.765 ± 0.025 | 0.768 ± 0.015 | 0.766 ± 0.030 |
| fsd | 0.791 ± 0.031 | 0.765 ± 0.025 | 0.768 ± 0.015 | 0.766 ± 0.030 |
| vgae | 0.788 ± 0.013 | 0.752 ± 0.016 | 0.747 ± 0.011 | 0.768 ± 0.008 |
| seal | 0.861 ± 0.016 | 0.736 ± 0.022 | 0.743 ± 0.026 | 0.819 ± 0.024 |
| sgdiff | 0.783 ± 0.019 | 0.684 ± 0.026 | 0.688 ± 0.023 | 0.737 ± 0.020 |

### citeseer: 40% missing node attributes

| Model | Clean | RF 50% | DICE 50% | EA 50% |
|---|---|---|---|---|
| fsd_none | 0.728 ± 0.023 | 0.697 ± 0.026 | 0.702 ± 0.023 | 0.705 ± 0.019 |
| fsd_fp | 0.733 ± 0.028 | 0.696 ± 0.022 | 0.710 ± 0.020 | 0.708 ± 0.030 |
| fsd_gamma1 | 0.724 ± 0.015 | 0.687 ± 0.013 | 0.705 ± 0.009 | 0.699 ± 0.016 |
| fsd | 0.726 ± 0.015 | 0.690 ± 0.011 | 0.709 ± 0.005 | 0.701 ± 0.013 |
| vgae | 0.743 ± 0.012 | 0.699 ± 0.022 | 0.701 ± 0.013 | 0.727 ± 0.013 |
| seal | 0.862 ± 0.012 | 0.734 ± 0.009 | 0.747 ± 0.023 | 0.816 ± 0.028 |
| sgdiff | 0.778 ± 0.016 | 0.686 ± 0.025 | 0.695 ± 0.031 | 0.733 ± 0.002 |

## Mechanism findings

1. **Imputation is dataset-dependent.** On Cora with 40% missing rows, FSD changes clean AUC from 0.756 to 0.778 (+2.22 percentage points). On Citeseer it changes 0.728 to 0.726 (-0.21 points). The gain does not transfer uniformly across these datasets.
2. **Gamma=1.2 has no stable advantage over ordinary propagation here.** Cora's ordinary propagation reaches 0.778; FSD adds only +0.05 percentage points. On Citeseer, ordinary propagation / gamma=1 / gamma=1.2 reach 0.733 / 0.724 / 0.726. Most of the Cora improvement is already present with ordinary propagation. Complete-feature controls coincide up to small numerical differences when imputation is bypassed.
3. **Robustness rankings depend on the attack.** SEAL has the highest clean AUC in all four settings here. On complete-feature Citeseer, RF-50% gives GAE/FSD about 0.765 versus SEAL 0.736; DICE-50% gives about 0.768 versus 0.743. Both absolute attacked AUC and the drop from clean are needed to assess robustness.
4. **SGDiff's advantage is not observed under this budget against SEAL.** Its clean AUC is 0.772/0.783 on complete-feature Cora/Citeseer, versus SEAL 0.866/0.861. This concerns this repository implementation and small training budget, not the original paper's full-budget claim.
5. **Mechanism interpretation.** Propagation uses observed neighbors to fill absent features, so incorrect connections can also transmit incorrect information. Dataset-dependent results are consistent with a dependence on neighborhood quality; homophily was not separately manipulated. SEAL retains structural labels alongside attributes. The four FSD controls isolate completion choices; they do not constitute a DRNL-removal ablation for SEAL or an isolated KL ablation for VGAE.

## Evidence boundaries

These are controlled **small-budget repository comparisons**, not the original papers' full-budget leaderboard results. The three seeds change data splits, missing-feature masks and optimization randomness; their standard deviation is not a confidence interval. CUDA sparse reductions can cause small numerical differences even between mathematically identical complete-feature controls. There was no hyperparameter search, and results cover 40% row missingness rather than FSD-CAP's extreme missingness setting. The EA implementation is a local linear proxy, not an exact external attack implementation. SEAL retains the repository's GCN/DRNL/mean-max implementation.

## Reproduce

From the repository root with its runtime requirements installed:

```bash
for dataset in cora citeseer; do
  for spec in observed:0.0 missing40:0.4; do
    tag=${spec%%:*}; rate=${spec##*:}
    python scripts/run_robustness.py --dataset "$dataset" \
      --models fsd_none fsd_fp fsd_gamma1 fsd seal sgdiff \
      --profile cpu --seeds 0 1 2 --device cuda \
      --output results/line-c-fsd/cuda --tag "${tag}_cuda" \
      --overrides "data.feature_missing_rate=$rate"
    python scripts/run_robustness.py --dataset "$dataset" \
      --models vgae --profile cpu --seeds 0 1 2 --device cuda \
      --output results/line-c-fsd/cuda --tag "${tag}_vgae_cuda" \
      --overrides "data.feature_missing_rate=$rate"
  done
done
```

VGAE was run separately in the same original GPU job after its approved correction. The robustness harness reseeds each model independently. Source: `cb7e60e` plus VGAE correction `7ac8c5f`. Seven native files come from Fuyao job `49078609`; the final Citeseer 40%-missing primary-model file was rerun with identical arguments/source/environment in job `49091270` after an archive quota error and a failed handoff. Only the successful native artifact is used for that condition. Original and recovery logs and the run manifest preserve this provenance.

Validation: four FSD/masking/validation-context tests and the VGAE posterior-sampling regression passed. All eight native result files were retrieved and checked: **840/840 distinct model/seed/condition cells**, finite AUC/AP/Hit@100 values, three seeds per summary, 500 positive + 500 negative scored pairs, CUDA execution, and no skipped attacks. Transfer verification matched remote and local bytes.
