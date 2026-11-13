# Paired bootstrap: cora

2000 resamples of the 1000 scored pairs per seed, pooled over
4 seeds; clean and probe are resampled together.
`p` is the one-sided bootstrap probability that the damage is ≤ 0.

| model | probe | clean AUC | probe AUC | ΔAUC [95 % CI] | p(no damage) |
|---|---|---:|---:|---|---:|
| sgdiff | del | 0.732 | 0.522 | +0.211 [+0.107, +0.301] | 0.0000 |
| gcn | feat_shuffle | 0.827 | 0.629 | +0.198 [+0.143, +0.256] | 0.0000 |
| vgae | feat_shuffle | 0.784 | 0.589 | +0.195 [+0.128, +0.250] | 0.0000 |
| rgcn | feat_shuffle | 0.773 | 0.580 | +0.192 [+0.140, +0.241] | 0.0000 |
| rgcn_novar | feat_shuffle | 0.772 | 0.587 | +0.185 [+0.115, +0.243] | 0.0000 |
| gat | feat_shuffle | 0.807 | 0.627 | +0.181 [+0.127, +0.237] | 0.0000 |
| sgdiff | del_hub | 0.732 | 0.561 | +0.172 [+0.100, +0.237] | 0.0000 |
| sage | feat_shuffle | 0.749 | 0.590 | +0.159 [+0.113, +0.205] | 0.0000 |
| seal | del_hub | 0.781 | 0.632 | +0.149 [+0.093, +0.203] | 0.0000 |
| pencil | del_hub | 0.795 | 0.650 | +0.145 [+0.075, +0.231] | 0.0000 |
| seal | del | 0.781 | 0.644 | +0.137 [+0.104, +0.168] | 0.0000 |
| pencil | del | 0.795 | 0.671 | +0.125 [+0.074, +0.209] | 0.0000 |
| gat | del | 0.807 | 0.705 | +0.102 [+0.070, +0.137] | 0.0000 |
| gat | del_hub | 0.807 | 0.712 | +0.095 [+0.063, +0.125] | 0.0000 |
| rgcn | del | 0.773 | 0.686 | +0.087 [+0.044, +0.135] | 0.0000 |
| rgcn | del_hub | 0.773 | 0.687 | +0.086 [+0.052, +0.120] | 0.0000 |
| rgcn_novar | del_hub | 0.772 | 0.690 | +0.082 [+0.050, +0.114] | 0.0000 |
| rgcn_novar | del | 0.772 | 0.691 | +0.081 [+0.045, +0.122] | 0.0000 |
| rgcn | ins_hub | 0.773 | 0.695 | +0.078 [+0.044, +0.110] | 0.0000 |
| rgcn_novar | ins_hub | 0.772 | 0.698 | +0.074 [+0.044, +0.107] | 0.0000 |
| vgae | del_hub | 0.784 | 0.722 | +0.061 [+0.040, +0.087] | 0.0000 |
| sage | del_hub | 0.749 | 0.690 | +0.060 [+0.035, +0.084] | 0.0000 |
| gcn | del_hub | 0.827 | 0.769 | +0.059 [+0.037, +0.080] | 0.0000 |
| seal | feat_shuffle | 0.781 | 0.726 | +0.055 [-0.001, +0.111] | 0.0265 |
| gcn | del | 0.827 | 0.781 | +0.046 [+0.027, +0.070] | 0.0000 |
| vgae | del | 0.784 | 0.739 | +0.045 [+0.017, +0.077] | 0.0000 |
| vgae | ins_hub | 0.784 | 0.740 | +0.044 [+0.023, +0.072] | 0.0000 |
| sgdiff | ins_hub | 0.732 | 0.776 | -0.043 [-0.088, +0.015] | 0.8644 |
| sage | del | 0.749 | 0.708 | +0.042 [+0.015, +0.072] | 0.0001 |
| pencil | feat_shuffle | 0.795 | 0.755 | +0.040 [-0.023, +0.106] | 0.1676 |
| gat | ins_hub | 0.807 | 0.772 | +0.035 [+0.015, +0.056] | 0.0003 |
| sage | ins_hub | 0.749 | 0.723 | +0.026 [+0.009, +0.045] | 0.0010 |
| gcn | ins_hub | 0.827 | 0.804 | +0.024 [+0.005, +0.042] | 0.0047 |
| seal | ins_hub | 0.781 | 0.760 | +0.022 [+0.005, +0.043] | 0.0029 |
| pencil | ins_hub | 0.795 | 0.788 | +0.008 [-0.009, +0.022] | 0.1416 |
| sgdiff | feat_shuffle | 0.732 | 0.732 | +0.000 [+0.000, +0.000] | 1.0000 |

### What this does and does not cover

* It covers **pair-sampling** uncertainty, which dominates for a fixed (model, probe).
* It does **not** cover training-seed uncertainty; the dumps are 4
  seeds, so a claim about the population of trained models still rests on the seed spreads in
  the other tables. Pooling seeds widens the interval when they disagree; it does not turn
  a pair-level claim into a model-population claim.
* Effects whose interval includes 0 should not be reported as an effect, however
  consistent their sign looks across the eight seeds of §7.6.
