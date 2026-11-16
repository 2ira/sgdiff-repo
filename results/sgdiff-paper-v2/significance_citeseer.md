# Paired bootstrap: citeseer

2000 resamples of the 1000 scored pairs per seed, pooled over
3 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| seal | del | 0.879 | 0.627 | +0.252 [+0.199, +0.305] | 0.0000 |
| rgcn | feat_shuffle | 0.771 | 0.549 | +0.222 [+0.172, +0.272] | 0.0000 |
| vgae | feat_shuffle | 0.772 | 0.559 | +0.213 [+0.157, +0.275] | 0.0000 |
| seal | del_hub | 0.879 | 0.672 | +0.207 [+0.169, +0.248] | 0.0000 |
| sgdiff | del_hub | 0.838 | 0.631 | +0.207 [+0.163, +0.251] | 0.0000 |
| rgcn_novar | feat_shuffle | 0.771 | 0.567 | +0.205 [+0.154, +0.258] | 0.0000 |
| gcn | feat_shuffle | 0.788 | 0.618 | +0.170 [+0.120, +0.220] | 0.0000 |
| sage | feat_shuffle | 0.733 | 0.572 | +0.161 [+0.112, +0.208] | 0.0000 |
| gat | feat_shuffle | 0.788 | 0.627 | +0.161 [+0.106, +0.221] | 0.0000 |
| sgdiff | del | 0.838 | 0.685 | +0.152 [+0.104, +0.205] | 0.0000 |
| gat | del_hub | 0.788 | 0.682 | +0.106 [+0.076, +0.139] | 0.0000 |
| rgcn_novar | del_hub | 0.771 | 0.670 | +0.102 [+0.069, +0.135] | 0.0000 |
| rgcn | del_hub | 0.771 | 0.670 | +0.101 [+0.065, +0.141] | 0.0000 |
| pencil | feat_shuffle | 0.881 | 0.782 | +0.099 [+0.070, +0.132] | 0.0000 |
| pencil | del | 0.881 | 0.788 | +0.094 [+0.060, +0.123] | 0.0000 |
| pencil | del_hub | 0.881 | 0.796 | +0.085 [+0.053, +0.113] | 0.0000 |
| rgcn_novar | ins_hub | 0.771 | 0.700 | +0.072 [+0.037, +0.109] | 0.0000 |
| gat | del | 0.788 | 0.719 | +0.070 [+0.031, +0.109] | 0.0002 |
| rgcn | ins_hub | 0.771 | 0.702 | +0.068 [+0.032, +0.104] | 0.0000 |
| sgdiff | ins_hub | 0.838 | 0.777 | +0.061 [+0.034, +0.090] | 0.0000 |
| rgcn_novar | del | 0.771 | 0.713 | +0.059 [+0.020, +0.092] | 0.0007 |
| seal | ins_hub | 0.879 | 0.822 | +0.056 [+0.029, +0.083] | 0.0000 |
| vgae | ins_hub | 0.772 | 0.722 | +0.050 [+0.025, +0.076] | 0.0000 |
| rgcn | del | 0.771 | 0.722 | +0.049 [+0.018, +0.079] | 0.0005 |
| vgae | del_hub | 0.772 | 0.727 | +0.046 [+0.028, +0.064] | 0.0000 |
| gcn | del_hub | 0.788 | 0.748 | +0.039 [+0.018, +0.061] | 0.0000 |
| sage | del_hub | 0.733 | 0.695 | +0.038 [+0.017, +0.060] | 0.0002 |
| gat | ins_hub | 0.788 | 0.755 | +0.033 [+0.003, +0.059] | 0.0150 |
| pencil | ins_hub | 0.881 | 0.850 | +0.032 [+0.010, +0.055] | 0.0003 |
| gcn | ins_hub | 0.788 | 0.757 | +0.031 [+0.015, +0.048] | 0.0000 |
| gcn | del | 0.788 | 0.761 | +0.026 [+0.010, +0.045] | 0.0013 |
| vgae | del | 0.772 | 0.747 | +0.026 [+0.010, +0.042] | 0.0012 |
| sage | del | 0.733 | 0.727 | +0.006 [-0.015, +0.027] | 0.2902 |
| sage | ins_hub | 0.733 | 0.737 | -0.004 [-0.025, +0.013] | 0.6008 |
| seal | feat_shuffle | 0.879 | 0.878 | +0.001 [-0.028, +0.031] | 0.5163 |
| sgdiff | feat_shuffle | 0.838 | 0.837 | +0.000 [-0.008, +0.010] | 0.4865 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 3
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
