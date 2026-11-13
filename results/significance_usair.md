# Paired bootstrap: usair

2000 resamples of the 638 scored pairs per seed, pooled over
4 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| pencil | ins_hub | 0.933 | 0.759 | +0.173 [+0.106, +0.231] | 0.0000 |
| seal | ins_hub | 0.958 | 0.867 | +0.091 [+0.060, +0.125] | 0.0000 |
| sgdiff | ins_hub | 0.942 | 0.866 | +0.075 [+0.047, +0.107] | 0.0000 |
| rgcn | ins_hub | 0.862 | 0.799 | +0.063 [+0.001, +0.159] | 0.0222 |
| seal | del | 0.958 | 0.897 | +0.061 [+0.036, +0.087] | 0.0000 |
| seal | del_hub | 0.958 | 0.901 | +0.057 [+0.035, +0.082] | 0.0000 |
| gcn | del_hub | 0.886 | 0.830 | +0.056 [+0.031, +0.096] | 0.0000 |
| gcn | del | 0.886 | 0.835 | +0.051 [+0.023, +0.083] | 0.0000 |
| rgcn | del_hub | 0.862 | 0.822 | +0.040 [+0.015, +0.068] | 0.0009 |
| sage | del | 0.876 | 0.838 | +0.038 [+0.014, +0.062] | 0.0008 |
| gcn | ins_hub | 0.886 | 0.848 | +0.038 [+0.003, +0.072] | 0.0173 |
| vgae | ins_hub | 0.845 | 0.807 | +0.038 [+0.002, +0.072] | 0.0194 |
| sgdiff | del | 0.942 | 0.905 | +0.036 [+0.011, +0.063] | 0.0009 |
| sage | del_hub | 0.876 | 0.841 | +0.035 [+0.013, +0.061] | 0.0001 |
| sgdiff | del_hub | 0.942 | 0.907 | +0.035 [-0.003, +0.077] | 0.0459 |
| rgcn_novar | del_hub | 0.864 | 0.830 | +0.034 [+0.011, +0.060] | 0.0020 |
| pencil | del | 0.933 | 0.902 | +0.030 [-0.009, +0.075] | 0.1206 |
| rgcn_novar | del | 0.864 | 0.835 | +0.030 [-0.001, +0.066] | 0.0311 |
| rgcn_novar | ins_hub | 0.864 | 0.835 | +0.030 [-0.005, +0.068] | 0.0511 |
| rgcn | del | 0.862 | 0.833 | +0.028 [-0.002, +0.063] | 0.0338 |
| vgae | del_hub | 0.845 | 0.820 | +0.024 [+0.006, +0.041] | 0.0056 |
| gat | del_hub | 0.861 | 0.837 | +0.024 [-0.013, +0.055] | 0.1086 |
| vgae | del | 0.845 | 0.826 | +0.018 [+0.001, +0.038] | 0.0182 |
| pencil | del_hub | 0.933 | 0.915 | +0.018 [-0.020, +0.068] | 0.3464 |
| gat | del | 0.861 | 0.847 | +0.015 [-0.026, +0.043] | 0.2156 |
| gat | ins_hub | 0.861 | 0.850 | +0.011 [-0.032, +0.051] | 0.2961 |
| sage | ins_hub | 0.876 | 0.866 | +0.010 [-0.034, +0.048] | 0.3075 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 4
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
