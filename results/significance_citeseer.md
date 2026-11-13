# Paired bootstrap: citeseer

2000 resamples of the 1000 scored pairs per seed, pooled over
4 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| seal | feat_shuffle | 0.731 | 0.511 | +0.220 [+0.160, +0.277] | 0.0000 |
| vgae | feat_shuffle | 0.789 | 0.571 | +0.218 [+0.168, +0.273] | 0.0000 |
| gcn | feat_shuffle | 0.812 | 0.601 | +0.210 [+0.159, +0.260] | 0.0000 |
| sgdiff | del_hub | 0.744 | 0.550 | +0.194 [+0.124, +0.315] | 0.0000 |
| pencil | feat_shuffle | 0.807 | 0.615 | +0.192 [+0.111, +0.258] | 0.0000 |
| sage | feat_shuffle | 0.739 | 0.550 | +0.189 [+0.104, +0.252] | 0.0000 |
| gat | feat_shuffle | 0.736 | 0.585 | +0.151 [+0.075, +0.213] | 0.0000 |
| sgdiff | del | 0.744 | 0.595 | +0.149 [+0.071, +0.221] | 0.0000 |
| rgcn | feat_shuffle | 0.664 | 0.527 | +0.136 [+0.086, +0.186] | 0.0000 |
| rgcn_novar | feat_shuffle | 0.661 | 0.549 | +0.112 [+0.048, +0.168] | 0.0000 |
| seal | ins_hub | 0.731 | 0.634 | +0.097 [+0.056, +0.141] | 0.0000 |
| gat | del_hub | 0.736 | 0.646 | +0.090 [+0.047, +0.130] | 0.0000 |
| seal | del | 0.731 | 0.661 | +0.069 [+0.036, +0.104] | 0.0000 |
| rgcn | del_hub | 0.664 | 0.601 | +0.063 [+0.014, +0.103] | 0.0049 |
| rgcn_novar | del_hub | 0.661 | 0.606 | +0.055 [+0.010, +0.097] | 0.0071 |
| gat | del | 0.736 | 0.682 | +0.054 [+0.019, +0.089] | 0.0006 |
| rgcn_novar | ins_hub | 0.661 | 0.612 | +0.049 [+0.009, +0.083] | 0.0071 |
| rgcn | ins_hub | 0.664 | 0.618 | +0.046 [+0.012, +0.078] | 0.0044 |
| pencil | del | 0.807 | 0.762 | +0.045 [+0.021, +0.070] | 0.0000 |
| vgae | ins_hub | 0.789 | 0.747 | +0.042 [+0.021, +0.063] | 0.0000 |
| sage | del_hub | 0.739 | 0.697 | +0.042 [+0.016, +0.069] | 0.0003 |
| seal | del_hub | 0.731 | 0.690 | +0.041 [+0.020, +0.068] | 0.0000 |
| gcn | del_hub | 0.812 | 0.772 | +0.040 [+0.018, +0.063] | 0.0000 |
| vgae | del_hub | 0.789 | 0.753 | +0.036 [+0.018, +0.054] | 0.0000 |
| pencil | del_hub | 0.807 | 0.778 | +0.029 [+0.012, +0.048] | 0.0001 |
| gcn | ins_hub | 0.812 | 0.783 | +0.029 [+0.013, +0.045] | 0.0000 |
| gcn | del | 0.812 | 0.784 | +0.028 [+0.011, +0.046] | 0.0006 |
| gat | ins_hub | 0.736 | 0.710 | +0.027 [+0.004, +0.051] | 0.0090 |
| vgae | del | 0.789 | 0.765 | +0.024 [+0.009, +0.039] | 0.0010 |
| sgdiff | ins_hub | 0.744 | 0.766 | -0.022 [-0.080, +0.030] | 0.6376 |
| pencil | ins_hub | 0.807 | 0.785 | +0.022 [+0.011, +0.036] | 0.0000 |
| sage | del | 0.739 | 0.724 | +0.014 [-0.013, +0.040] | 0.1624 |
| rgcn | del | 0.664 | 0.652 | +0.012 [-0.040, +0.058] | 0.3065 |
| sage | ins_hub | 0.739 | 0.733 | +0.005 [-0.009, +0.019] | 0.2349 |
| rgcn_novar | del | 0.661 | 0.665 | -0.004 [-0.042, +0.033] | 0.5891 |
| sgdiff | feat_shuffle | 0.744 | 0.744 | +0.000 [+0.000, +0.000] | 1.0000 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 4
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
