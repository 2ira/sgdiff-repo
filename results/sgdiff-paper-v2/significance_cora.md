# Paired bootstrap: cora

2000 resamples of the 1000 scored pairs per seed, pooled over
3 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| rgcn_novar | feat_shuffle | 0.825 | 0.590 | +0.235 [+0.188, +0.281] | 0.0000 |
| vgae | feat_shuffle | 0.816 | 0.584 | +0.232 [+0.169, +0.293] | 0.0000 |
| seal | del_hub | 0.872 | 0.643 | +0.229 [+0.193, +0.266] | 0.0000 |
| rgcn | feat_shuffle | 0.826 | 0.598 | +0.228 [+0.183, +0.273] | 0.0000 |
| sgdiff | del_hub | 0.857 | 0.642 | +0.215 [+0.176, +0.257] | 0.0000 |
| seal | del | 0.872 | 0.661 | +0.212 [+0.171, +0.250] | 0.0000 |
| gat | feat_shuffle | 0.850 | 0.654 | +0.197 [+0.148, +0.240] | 0.0000 |
| sage | feat_shuffle | 0.766 | 0.587 | +0.179 [+0.126, +0.234] | 0.0000 |
| gcn | feat_shuffle | 0.815 | 0.637 | +0.178 [+0.102, +0.237] | 0.0000 |
| pencil | del | 0.883 | 0.713 | +0.170 [+0.122, +0.212] | 0.0000 |
| pencil | del_hub | 0.883 | 0.715 | +0.168 [+0.134, +0.212] | 0.0000 |
| sgdiff | del | 0.857 | 0.711 | +0.146 [+0.100, +0.190] | 0.0000 |
| gat | del_hub | 0.850 | 0.730 | +0.121 [+0.085, +0.154] | 0.0000 |
| rgcn | del_hub | 0.826 | 0.717 | +0.109 [+0.074, +0.142] | 0.0000 |
| gat | del | 0.850 | 0.747 | +0.103 [+0.073, +0.136] | 0.0000 |
| rgcn_novar | del_hub | 0.825 | 0.727 | +0.098 [+0.069, +0.129] | 0.0000 |
| rgcn | del | 0.826 | 0.740 | +0.086 [+0.057, +0.117] | 0.0000 |
| rgcn_novar | del | 0.825 | 0.740 | +0.085 [+0.053, +0.116] | 0.0000 |
| rgcn_novar | ins_hub | 0.825 | 0.741 | +0.084 [+0.059, +0.110] | 0.0000 |
| rgcn | ins_hub | 0.826 | 0.744 | +0.082 [+0.058, +0.107] | 0.0000 |
| gcn | del_hub | 0.815 | 0.740 | +0.075 [+0.054, +0.096] | 0.0000 |
| vgae | del_hub | 0.816 | 0.749 | +0.067 [+0.045, +0.093] | 0.0000 |
| vgae | ins_hub | 0.816 | 0.756 | +0.060 [+0.040, +0.082] | 0.0000 |
| pencil | feat_shuffle | 0.883 | 0.824 | +0.059 [+0.024, +0.106] | 0.0000 |
| gcn | del | 0.815 | 0.755 | +0.059 [+0.041, +0.077] | 0.0000 |
| sage | del_hub | 0.766 | 0.708 | +0.059 [+0.035, +0.084] | 0.0000 |
| vgae | del | 0.816 | 0.761 | +0.056 [+0.036, +0.074] | 0.0000 |
| sgdiff | ins_hub | 0.857 | 0.808 | +0.049 [+0.025, +0.075] | 0.0002 |
| sage | del | 0.766 | 0.718 | +0.048 [+0.023, +0.077] | 0.0000 |
| seal | ins_hub | 0.872 | 0.828 | +0.044 [+0.018, +0.079] | 0.0002 |
| pencil | ins_hub | 0.883 | 0.844 | +0.039 [+0.020, +0.062] | 0.0000 |
| gcn | ins_hub | 0.815 | 0.779 | +0.036 [+0.019, +0.051] | 0.0000 |
| gat | ins_hub | 0.850 | 0.815 | +0.035 [+0.017, +0.054] | 0.0000 |
| sage | ins_hub | 0.766 | 0.734 | +0.033 [+0.014, +0.051] | 0.0002 |
| seal | feat_shuffle | 0.872 | 0.877 | -0.005 [-0.024, +0.014] | 0.7033 |
| sgdiff | feat_shuffle | 0.857 | 0.859 | -0.002 [-0.005, +0.001] | 0.9187 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 3
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
