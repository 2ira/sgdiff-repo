# Deep probe: citeseer

* models: sgdiff, seal, pencil, gcn, gat, sage, vgae
* seeds: [0, 1]
* variants: del, del_hub, ins_hub, feat_shuffle

## 1. Layer-wise damage (node-pair models)

`drift` = 1 − mean cosine between the layer representation on the probe graph and
on the clean graph; `probe_auc` = AUC of a linear read-out trained on the *clean*
layer and applied to the probe layer (chance ≈ 0.5).

### gcn

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.069 | 0.079 | 0.846 | 0.815 |
| del_hub | 0.053 | 0.064 | 0.832 | 0.786 |
| ins_hub | 0.035 | 0.050 | 0.840 | 0.817 |
| feat_shuffle | 0.995 | 0.970 | 0.606 | 0.689 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.338 | 0.837 | 0.736 |
| del_hub | 0.000 | 0.222 | 0.837 | 0.712 |
| ins_hub | 0.000 | 0.172 | 0.837 | 0.767 |
| feat_shuffle | 0.993 | 0.885 | 0.511 | 0.609 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.056 | 0.129 | 0.776 | 0.758 |
| del_hub | 0.038 | 0.081 | 0.762 | 0.743 |
| ins_hub | 0.017 | 0.041 | 0.776 | 0.776 |
| feat_shuffle | 0.995 | 0.968 | 0.539 | 0.582 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | feature_shuffle | ins_hub@50% |
|---|---|---|---|---|---|
| gcn | sym | 0.796 | 0.768 | 0.632 | 0.764 |
| gcn | row | 0.784 | 0.762 | 0.640 | 0.740 |
| gcn | sum | 0.791 | 0.775 | 0.728 | 0.757 |
| gcn | noloop | 0.740 | 0.676 | 0.628 | 0.727 |
| sage | sym | 0.724 | 0.719 | 0.562 | 0.730 |
| sage | row | 0.713 | 0.716 | 0.569 | 0.697 |
| sage | sum | 0.767 | 0.738 | 0.682 | 0.737 |
| sage | noloop | 0.724 | 0.719 | 0.562 | 0.730 |
| vgae | sym | 0.781 | 0.754 | 0.565 | 0.731 |
| vgae | row | 0.774 | 0.750 | 0.569 | 0.728 |
| vgae | sum | 0.784 | 0.754 | 0.674 | 0.737 |
| vgae | noloop | 0.745 | 0.676 | 0.568 | 0.688 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.841 | 0.813 | 0.732 | 1271 | 2598 |
| seal | 0.875 | 0.845 | 0.702 | 1271 | 2598 |
| pencil | 0.883 | 0.865 | 0.813 | 1271 | 2598 |
| gcn | 0.796 | 0.790 | 0.772 | 1271 | 2598 |
| gat | 0.795 | 0.781 | 0.727 | 1271 | 2598 |
| sage | 0.724 | 0.720 | 0.719 | 1271 | 2598 |
| vgae | 0.781 | 0.773 | 0.760 | 1271 | 2598 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.561, AA-only 0.561 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.678 | 0.652 | -0.026 |
| sgdiff | del + complete_aa | 0.678 | 0.611 | -0.067 |
| sgdiff | del_hub + complete_cn | 0.641 | 0.597 | -0.044 |
| sgdiff | del_hub + complete_aa | 0.641 | 0.597 | -0.044 |
| sgdiff | feat_shuffle + smooth1 | 0.842 | 0.852 | +0.010 |
| sgdiff | feat_shuffle + smooth2 | 0.842 | 0.856 | +0.014 |
| sgdiff | del + hybrid_cn0 | 0.678 | 0.678 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.678 | 0.679 | +0.002 |
| sgdiff | del + hybrid_cn0.5 | 0.678 | 0.679 | +0.002 |
| sgdiff | del + hybrid_cn1 | 0.678 | 0.679 | +0.002 |
| sgdiff | del + hybrid_cn2 | 0.678 | 0.680 | +0.002 |
| seal | del + complete_cn | 0.623 | 0.657 | +0.034 |
| seal | del + complete_aa | 0.623 | 0.642 | +0.020 |
| seal | del_hub + complete_cn | 0.666 | 0.677 | +0.011 |
| seal | del_hub + complete_aa | 0.666 | 0.677 | +0.011 |
| seal | feat_shuffle + smooth1 | 0.883 | 0.887 | +0.004 |
| seal | feat_shuffle + smooth2 | 0.883 | 0.887 | +0.004 |
| seal | del + hybrid_cn0 | 0.623 | 0.623 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.623 | 0.625 | +0.002 |
| seal | del + hybrid_cn0.5 | 0.623 | 0.625 | +0.002 |
| seal | del + hybrid_cn1 | 0.623 | 0.625 | +0.002 |
| seal | del + hybrid_cn2 | 0.623 | 0.625 | +0.002 |
| pencil | del + complete_cn | 0.780 | 0.799 | +0.019 |
| pencil | del + complete_aa | 0.780 | 0.798 | +0.018 |
| pencil | del_hub + complete_cn | 0.789 | 0.808 | +0.019 |
| pencil | del_hub + complete_aa | 0.789 | 0.808 | +0.019 |
| pencil | feat_shuffle + smooth1 | 0.787 | 0.820 | +0.033 |
| pencil | feat_shuffle + smooth2 | 0.787 | 0.830 | +0.043 |
| pencil | del + hybrid_cn0 | 0.780 | 0.780 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.780 | 0.783 | +0.003 |
| pencil | del + hybrid_cn0.5 | 0.780 | 0.783 | +0.003 |
| pencil | del + hybrid_cn1 | 0.780 | 0.783 | +0.003 |
| pencil | del + hybrid_cn2 | 0.780 | 0.783 | +0.003 |
| gcn | del + complete_cn | 0.768 | 0.754 | -0.015 |
| gcn | del + complete_aa | 0.768 | 0.758 | -0.010 |
| gcn | del_hub + complete_cn | 0.758 | 0.749 | -0.009 |
| gcn | del_hub + complete_aa | 0.758 | 0.749 | -0.009 |
| gcn | feat_shuffle + smooth1 | 0.632 | 0.654 | +0.022 |
| gcn | feat_shuffle + smooth2 | 0.632 | 0.666 | +0.033 |
| gcn | del + hybrid_cn0 | 0.768 | 0.768 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.768 | 0.780 | +0.011 |
| gcn | del + hybrid_cn0.5 | 0.768 | 0.780 | +0.012 |
| gcn | del + hybrid_cn1 | 0.768 | 0.781 | +0.012 |
| gcn | del + hybrid_cn2 | 0.768 | 0.781 | +0.012 |
| gat | del + complete_cn | 0.717 | 0.726 | +0.009 |
| gat | del + complete_aa | 0.717 | 0.725 | +0.008 |
| gat | del_hub + complete_cn | 0.684 | 0.695 | +0.011 |
| gat | del_hub + complete_aa | 0.684 | 0.695 | +0.011 |
| gat | feat_shuffle + smooth1 | 0.632 | 0.642 | +0.011 |
| gat | feat_shuffle + smooth2 | 0.632 | 0.637 | +0.005 |
| gat | del + hybrid_cn0 | 0.717 | 0.717 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.717 | 0.719 | +0.002 |
| gat | del + hybrid_cn0.5 | 0.717 | 0.719 | +0.002 |
| gat | del + hybrid_cn1 | 0.717 | 0.720 | +0.003 |
| gat | del + hybrid_cn2 | 0.717 | 0.720 | +0.003 |
| sage | del + complete_cn | 0.719 | 0.712 | -0.007 |
| sage | del + complete_aa | 0.719 | 0.716 | -0.002 |
| sage | del_hub + complete_cn | 0.688 | 0.689 | +0.001 |
| sage | del_hub + complete_aa | 0.688 | 0.689 | +0.001 |
| sage | feat_shuffle + smooth1 | 0.562 | 0.598 | +0.036 |
| sage | feat_shuffle + smooth2 | 0.562 | 0.622 | +0.060 |
| sage | del + hybrid_cn0 | 0.719 | 0.719 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.719 | 0.730 | +0.012 |
| sage | del + hybrid_cn0.5 | 0.719 | 0.732 | +0.013 |
| sage | del + hybrid_cn1 | 0.719 | 0.732 | +0.014 |
| sage | del + hybrid_cn2 | 0.719 | 0.732 | +0.014 |
| vgae | del + complete_cn | 0.754 | 0.747 | -0.007 |
| vgae | del + complete_aa | 0.754 | 0.748 | -0.006 |
| vgae | del_hub + complete_cn | 0.735 | 0.726 | -0.009 |
| vgae | del_hub + complete_aa | 0.735 | 0.726 | -0.009 |
| vgae | feat_shuffle + smooth1 | 0.565 | 0.585 | +0.020 |
| vgae | feat_shuffle + smooth2 | 0.565 | 0.604 | +0.040 |
| vgae | del + hybrid_cn0 | 0.754 | 0.754 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.754 | 0.763 | +0.009 |
| vgae | del + hybrid_cn0.5 | 0.754 | 0.764 | +0.010 |
| vgae | del + hybrid_cn1 | 0.754 | 0.764 | +0.010 |
| vgae | del + hybrid_cn2 | 0.754 | 0.764 | +0.010 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.875 | 0.887 | +0.013 |
| seal | del | 0.623 | 0.627 | +0.004 |
| seal | del_hub | 0.666 | 0.627 | -0.039 |
| seal | ins_hub | 0.823 | 0.776 | -0.048 |
| pencil | clean | 0.883 | 0.900 | +0.018 |
| pencil | del | 0.780 | 0.809 | +0.029 |
| pencil | del_hub | 0.789 | 0.816 | +0.027 |
| pencil | ins_hub | 0.851 | 0.848 | -0.003 |

