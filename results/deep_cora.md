# Deep probe: cora

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
| del | 0.092 | 0.092 | 0.839 | 0.809 |
| del_hub | 0.068 | 0.074 | 0.827 | 0.802 |
| ins_hub | 0.036 | 0.049 | 0.841 | 0.830 |
| feat_shuffle | 1.001 | 0.962 | 0.614 | 0.668 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.331 | 0.754 | 0.688 |
| del_hub | 0.000 | 0.256 | 0.754 | 0.702 |
| ins_hub | 0.000 | 0.171 | 0.754 | 0.758 |
| feat_shuffle | 0.994 | 0.882 | 0.525 | 0.617 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.065 | 0.132 | 0.744 | 0.750 |
| del_hub | 0.046 | 0.097 | 0.750 | 0.737 |
| ins_hub | 0.018 | 0.047 | 0.745 | 0.766 |
| feat_shuffle | 0.994 | 0.963 | 0.557 | 0.587 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | feature_shuffle | ins_hub@50% |
|---|---|---|---|---|---|
| gcn | sym | 0.832 | 0.790 | 0.628 | 0.806 |
| gcn | row | 0.821 | 0.777 | 0.637 | 0.783 |
| gcn | sum | 0.806 | 0.773 | 0.703 | 0.754 |
| gcn | noloop | 0.811 | 0.718 | 0.647 | 0.788 |
| sage | sym | 0.768 | 0.729 | 0.570 | 0.740 |
| sage | row | 0.755 | 0.722 | 0.565 | 0.710 |
| sage | sum | 0.781 | 0.732 | 0.671 | 0.718 |
| sage | noloop | 0.768 | 0.729 | 0.570 | 0.740 |
| vgae | sym | 0.783 | 0.746 | 0.569 | 0.731 |
| vgae | row | 0.778 | 0.736 | 0.574 | 0.724 |
| vgae | sum | 0.786 | 0.740 | 0.662 | 0.718 |
| vgae | noloop | 0.768 | 0.694 | 0.573 | 0.709 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.739 | 0.650 | 0.630 | 1992 | 2494 |
| seal | 0.780 | 0.720 | 0.707 | 1992 | 2494 |
| pencil | 0.779 | 0.727 | 0.721 | 1992 | 2494 |
| gcn | 0.832 | 0.823 | 0.796 | 1992 | 2494 |
| gat | 0.802 | 0.772 | 0.739 | 1992 | 2494 |
| sage | 0.768 | 0.753 | 0.730 | 1992 | 2494 |
| vgae | 0.783 | 0.780 | 0.747 | 1992 | 2494 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.572, AA-only 0.572 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.521 | 0.634 | +0.113 |
| sgdiff | del + complete_aa | 0.521 | 0.614 | +0.093 |
| sgdiff | del_hub + complete_cn | 0.572 | 0.610 | +0.038 |
| sgdiff | del_hub + complete_aa | 0.572 | 0.631 | +0.059 |
| sgdiff | feat_shuffle + smooth1 | 0.739 | 0.739 | +0.000 |
| sgdiff | feat_shuffle + smooth2 | 0.739 | 0.739 | +0.000 |
| sgdiff | del + hybrid_cn0 | 0.521 | 0.521 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.521 | 0.521 | +0.000 |
| sgdiff | del + hybrid_cn0.5 | 0.521 | 0.521 | +0.000 |
| sgdiff | del + hybrid_cn1 | 0.521 | 0.521 | +0.000 |
| sgdiff | del + hybrid_cn2 | 0.521 | 0.521 | +0.000 |
| seal | del + complete_cn | 0.639 | 0.645 | +0.006 |
| seal | del + complete_aa | 0.639 | 0.638 | -0.002 |
| seal | del_hub + complete_cn | 0.628 | 0.675 | +0.046 |
| seal | del_hub + complete_aa | 0.628 | 0.661 | +0.033 |
| seal | feat_shuffle + smooth1 | 0.734 | 0.736 | +0.001 |
| seal | feat_shuffle + smooth2 | 0.734 | 0.726 | -0.008 |
| seal | del + hybrid_cn0 | 0.639 | 0.639 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.639 | 0.646 | +0.006 |
| seal | del + hybrid_cn0.5 | 0.639 | 0.646 | +0.007 |
| seal | del + hybrid_cn1 | 0.639 | 0.646 | +0.007 |
| seal | del + hybrid_cn2 | 0.639 | 0.646 | +0.007 |
| pencil | del + complete_cn | 0.668 | 0.677 | +0.009 |
| pencil | del + complete_aa | 0.668 | 0.671 | +0.003 |
| pencil | del_hub + complete_cn | 0.636 | 0.675 | +0.039 |
| pencil | del_hub + complete_aa | 0.636 | 0.670 | +0.034 |
| pencil | feat_shuffle + smooth1 | 0.753 | 0.763 | +0.010 |
| pencil | feat_shuffle + smooth2 | 0.753 | 0.763 | +0.010 |
| pencil | del + hybrid_cn0 | 0.668 | 0.668 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.668 | 0.670 | +0.002 |
| pencil | del + hybrid_cn0.5 | 0.668 | 0.670 | +0.002 |
| pencil | del + hybrid_cn1 | 0.668 | 0.670 | +0.002 |
| pencil | del + hybrid_cn2 | 0.668 | 0.670 | +0.002 |
| gcn | del + complete_cn | 0.790 | 0.784 | -0.006 |
| gcn | del + complete_aa | 0.790 | 0.786 | -0.004 |
| gcn | del_hub + complete_cn | 0.777 | 0.775 | -0.002 |
| gcn | del_hub + complete_aa | 0.777 | 0.772 | -0.005 |
| gcn | feat_shuffle + smooth1 | 0.628 | 0.662 | +0.034 |
| gcn | feat_shuffle + smooth2 | 0.628 | 0.683 | +0.054 |
| gcn | del + hybrid_cn0 | 0.790 | 0.790 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.790 | 0.799 | +0.010 |
| gcn | del + hybrid_cn0.5 | 0.790 | 0.801 | +0.011 |
| gcn | del + hybrid_cn1 | 0.790 | 0.801 | +0.012 |
| gcn | del + hybrid_cn2 | 0.790 | 0.802 | +0.012 |
| gat | del + complete_cn | 0.708 | 0.716 | +0.008 |
| gat | del + complete_aa | 0.708 | 0.719 | +0.011 |
| gat | del_hub + complete_cn | 0.704 | 0.709 | +0.004 |
| gat | del_hub + complete_aa | 0.704 | 0.712 | +0.007 |
| gat | feat_shuffle + smooth1 | 0.637 | 0.663 | +0.025 |
| gat | feat_shuffle + smooth2 | 0.637 | 0.667 | +0.030 |
| gat | del + hybrid_cn0 | 0.708 | 0.708 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.708 | 0.715 | +0.007 |
| gat | del + hybrid_cn0.5 | 0.708 | 0.716 | +0.008 |
| gat | del + hybrid_cn1 | 0.708 | 0.717 | +0.009 |
| gat | del + hybrid_cn2 | 0.708 | 0.717 | +0.009 |
| sage | del + complete_cn | 0.729 | 0.727 | -0.002 |
| sage | del + complete_aa | 0.729 | 0.728 | -0.001 |
| sage | del_hub + complete_cn | 0.709 | 0.718 | +0.009 |
| sage | del_hub + complete_aa | 0.709 | 0.717 | +0.008 |
| sage | feat_shuffle + smooth1 | 0.570 | 0.602 | +0.032 |
| sage | feat_shuffle + smooth2 | 0.570 | 0.635 | +0.065 |
| sage | del + hybrid_cn0 | 0.729 | 0.729 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.729 | 0.740 | +0.011 |
| sage | del + hybrid_cn0.5 | 0.729 | 0.741 | +0.012 |
| sage | del + hybrid_cn1 | 0.729 | 0.741 | +0.012 |
| sage | del + hybrid_cn2 | 0.729 | 0.741 | +0.012 |
| vgae | del + complete_cn | 0.746 | 0.740 | -0.006 |
| vgae | del + complete_aa | 0.746 | 0.746 | -0.000 |
| vgae | del_hub + complete_cn | 0.719 | 0.721 | +0.003 |
| vgae | del_hub + complete_aa | 0.719 | 0.719 | +0.001 |
| vgae | feat_shuffle + smooth1 | 0.569 | 0.604 | +0.034 |
| vgae | feat_shuffle + smooth2 | 0.569 | 0.627 | +0.058 |
| vgae | del + hybrid_cn0 | 0.746 | 0.746 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.746 | 0.759 | +0.013 |
| vgae | del + hybrid_cn0.5 | 0.746 | 0.761 | +0.016 |
| vgae | del + hybrid_cn1 | 0.746 | 0.762 | +0.016 |
| vgae | del + hybrid_cn2 | 0.746 | 0.762 | +0.016 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.780 | 0.850 | +0.070 |
| seal | del | 0.639 | 0.661 | +0.022 |
| seal | del_hub | 0.628 | 0.636 | +0.008 |
| seal | ins_hub | 0.762 | 0.791 | +0.029 |
| pencil | clean | 0.779 | 0.845 | +0.066 |
| pencil | del | 0.668 | 0.672 | +0.004 |
| pencil | del_hub | 0.636 | 0.638 | +0.002 |
| pencil | ins_hub | 0.771 | 0.806 | +0.035 |

