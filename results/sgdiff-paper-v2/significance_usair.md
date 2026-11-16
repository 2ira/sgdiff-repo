# Paired bootstrap: usair

2000 resamples of the 638 scored pairs per seed, pooled over
3 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| seal | del | 0.971 | 0.890 | +0.080 [+0.052, +0.110] | 0.0000 |
| seal | del_hub | 0.971 | 0.892 | +0.078 [+0.050, +0.112] | 0.0000 |
| rgcn | ins_hub | 0.855 | 0.788 | +0.066 [+0.004, +0.159] | 0.0140 |
| rgcn_novar | ins_hub | 0.854 | 0.797 | +0.057 [+0.002, +0.138] | 0.0200 |
| sgdiff | del | 0.922 | 0.869 | +0.053 [+0.022, +0.091] | 0.0000 |
| sgdiff | del_hub | 0.922 | 0.872 | +0.050 [+0.010, +0.112] | 0.0022 |
| pencil | del_hub | 0.957 | 0.907 | +0.050 [+0.019, +0.087] | 0.0002 |
| pencil | del | 0.957 | 0.907 | +0.049 [+0.028, +0.072] | 0.0000 |
| gcn | ins_hub | 0.886 | 0.837 | +0.049 [+0.025, +0.074] | 0.0000 |
| gcn | del | 0.886 | 0.838 | +0.048 [+0.028, +0.069] | 0.0000 |
| vgae | ins_hub | 0.847 | 0.801 | +0.046 [+0.015, +0.075] | 0.0018 |
| pencil | ins_hub | 0.957 | 0.916 | +0.041 [+0.026, +0.057] | 0.0000 |
| gcn | del_hub | 0.886 | 0.846 | +0.040 [+0.023, +0.059] | 0.0000 |
| seal | ins_hub | 0.971 | 0.931 | +0.040 [+0.022, +0.060] | 0.0000 |
| sage | del | 0.885 | 0.847 | +0.038 [+0.020, +0.058] | 0.0000 |
| sage | del_hub | 0.885 | 0.852 | +0.033 [+0.014, +0.053] | 0.0003 |
| sgdiff | ins_hub | 0.922 | 0.892 | +0.030 [+0.010, +0.051] | 0.0012 |
| rgcn | del_hub | 0.855 | 0.829 | +0.025 [+0.003, +0.051] | 0.0125 |
| vgae | del | 0.847 | 0.827 | +0.020 [+0.003, +0.039] | 0.0085 |
| vgae | del_hub | 0.847 | 0.827 | +0.019 [+0.001, +0.040] | 0.0205 |
| gat | del | 0.849 | 0.832 | +0.018 [-0.007, +0.046] | 0.0818 |
| rgcn_novar | del_hub | 0.854 | 0.837 | +0.017 [-0.005, +0.039] | 0.0605 |
| gat | del_hub | 0.849 | 0.832 | +0.017 [-0.013, +0.051] | 0.2392 |
| rgcn_novar | del | 0.854 | 0.838 | +0.016 [-0.010, +0.039] | 0.0953 |
| rgcn | del | 0.855 | 0.840 | +0.014 [-0.009, +0.036] | 0.1070 |
| sage | ins_hub | 0.885 | 0.893 | -0.007 [-0.036, +0.022] | 0.6892 |
| gat | ins_hub | 0.849 | 0.844 | +0.006 [-0.026, +0.033] | 0.3322 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 3
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
