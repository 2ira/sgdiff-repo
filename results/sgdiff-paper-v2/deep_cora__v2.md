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
| del | 0.104 | 0.104 | 0.804 | 0.787 |
| del_hub | 0.077 | 0.085 | 0.801 | 0.774 |
| ins_hub | 0.037 | 0.051 | 0.821 | 0.813 |
| feat_shuffle | 0.992 | 0.973 | 0.600 | 0.658 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.321 | 0.794 | 0.735 |
| del_hub | 0.000 | 0.252 | 0.794 | 0.727 |
| ins_hub | 0.000 | 0.174 | 0.794 | 0.779 |
| feat_shuffle | 0.996 | 0.935 | 0.502 | 0.653 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.076 | 0.125 | 0.707 | 0.736 |
| del_hub | 0.056 | 0.090 | 0.712 | 0.732 |
| ins_hub | 0.020 | 0.039 | 0.718 | 0.759 |
| feat_shuffle | 0.985 | 0.974 | 0.551 | 0.609 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | feature_shuffle | ins_hub@50% |
|---|---|---|---|---|---|
| gcn | sym | 0.819 | 0.763 | 0.632 | 0.785 |
| gcn | row | 0.819 | 0.762 | 0.636 | 0.780 |
| gcn | sum | 0.822 | 0.767 | 0.711 | 0.774 |
| gcn | noloop | 0.802 | 0.692 | 0.650 | 0.770 |
| sage | sym | 0.760 | 0.714 | 0.594 | 0.730 |
| sage | row | 0.752 | 0.709 | 0.607 | 0.712 |
| sage | sum | 0.779 | 0.715 | 0.717 | 0.737 |
| sage | noloop | 0.760 | 0.714 | 0.594 | 0.730 |
| vgae | sym | 0.823 | 0.768 | 0.574 | 0.767 |
| vgae | row | 0.826 | 0.765 | 0.593 | 0.782 |
| vgae | sum | 0.820 | 0.773 | 0.683 | 0.759 |
| vgae | noloop | 0.805 | 0.691 | 0.574 | 0.729 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.853 | 0.789 | 0.764 | 1992 | 2494 |
| seal | 0.876 | 0.809 | 0.752 | 1992 | 2494 |
| pencil | 0.890 | 0.843 | 0.811 | 1992 | 2494 |
| gcn | 0.819 | 0.811 | 0.776 | 1992 | 2494 |
| gat | 0.854 | 0.830 | 0.780 | 1992 | 2494 |
| sage | 0.760 | 0.751 | 0.731 | 1992 | 2494 |
| vgae | 0.823 | 0.807 | 0.788 | 1992 | 2494 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.572, AA-only 0.572 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.711 | 0.656 | -0.055 |
| sgdiff | del + complete_aa | 0.711 | 0.628 | -0.083 |
| sgdiff | del_hub + complete_cn | 0.633 | 0.654 | +0.022 |
| sgdiff | del_hub + complete_aa | 0.633 | 0.643 | +0.011 |
| sgdiff | feat_shuffle + smooth1 | 0.854 | 0.855 | +0.001 |
| sgdiff | feat_shuffle + smooth2 | 0.854 | 0.855 | +0.001 |
| sgdiff | del + hybrid_cn0 | 0.711 | 0.711 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.711 | 0.714 | +0.003 |
| sgdiff | del + hybrid_cn0.5 | 0.711 | 0.714 | +0.004 |
| sgdiff | del + hybrid_cn1 | 0.711 | 0.714 | +0.004 |
| sgdiff | del + hybrid_cn2 | 0.711 | 0.715 | +0.004 |
| seal | del + complete_cn | 0.667 | 0.687 | +0.021 |
| seal | del + complete_aa | 0.667 | 0.669 | +0.002 |
| seal | del_hub + complete_cn | 0.643 | 0.682 | +0.039 |
| seal | del_hub + complete_aa | 0.643 | 0.667 | +0.024 |
| seal | feat_shuffle + smooth1 | 0.881 | 0.885 | +0.003 |
| seal | feat_shuffle + smooth2 | 0.881 | 0.887 | +0.005 |
| seal | del + hybrid_cn0 | 0.667 | 0.667 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.667 | 0.667 | +0.000 |
| seal | del + hybrid_cn0.5 | 0.667 | 0.667 | +0.000 |
| seal | del + hybrid_cn1 | 0.667 | 0.667 | +0.000 |
| seal | del + hybrid_cn2 | 0.667 | 0.667 | +0.000 |
| pencil | del + complete_cn | 0.726 | 0.756 | +0.030 |
| pencil | del + complete_aa | 0.726 | 0.746 | +0.020 |
| pencil | del_hub + complete_cn | 0.717 | 0.752 | +0.035 |
| pencil | del_hub + complete_aa | 0.717 | 0.747 | +0.030 |
| pencil | feat_shuffle + smooth1 | 0.824 | 0.833 | +0.009 |
| pencil | feat_shuffle + smooth2 | 0.824 | 0.836 | +0.012 |
| pencil | del + hybrid_cn0 | 0.726 | 0.726 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.726 | 0.728 | +0.002 |
| pencil | del + hybrid_cn0.5 | 0.726 | 0.728 | +0.002 |
| pencil | del + hybrid_cn1 | 0.726 | 0.728 | +0.002 |
| pencil | del + hybrid_cn2 | 0.726 | 0.728 | +0.002 |
| gcn | del + complete_cn | 0.763 | 0.757 | -0.006 |
| gcn | del + complete_aa | 0.763 | 0.754 | -0.009 |
| gcn | del_hub + complete_cn | 0.751 | 0.755 | +0.003 |
| gcn | del_hub + complete_aa | 0.751 | 0.749 | -0.003 |
| gcn | feat_shuffle + smooth1 | 0.632 | 0.662 | +0.030 |
| gcn | feat_shuffle + smooth2 | 0.632 | 0.680 | +0.048 |
| gcn | del + hybrid_cn0 | 0.763 | 0.763 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.763 | 0.772 | +0.010 |
| gcn | del + hybrid_cn0.5 | 0.763 | 0.773 | +0.011 |
| gcn | del + hybrid_cn1 | 0.763 | 0.774 | +0.011 |
| gcn | del + hybrid_cn2 | 0.763 | 0.774 | +0.011 |
| gat | del + complete_cn | 0.756 | 0.752 | -0.005 |
| gat | del + complete_aa | 0.756 | 0.755 | -0.001 |
| gat | del_hub + complete_cn | 0.735 | 0.746 | +0.011 |
| gat | del_hub + complete_aa | 0.735 | 0.746 | +0.011 |
| gat | feat_shuffle + smooth1 | 0.663 | 0.710 | +0.047 |
| gat | feat_shuffle + smooth2 | 0.663 | 0.716 | +0.053 |
| gat | del + hybrid_cn0 | 0.756 | 0.756 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.756 | 0.759 | +0.003 |
| gat | del + hybrid_cn0.5 | 0.756 | 0.760 | +0.004 |
| gat | del + hybrid_cn1 | 0.756 | 0.760 | +0.004 |
| gat | del + hybrid_cn2 | 0.756 | 0.760 | +0.004 |
| sage | del + complete_cn | 0.714 | 0.711 | -0.003 |
| sage | del + complete_aa | 0.714 | 0.720 | +0.006 |
| sage | del_hub + complete_cn | 0.705 | 0.708 | +0.003 |
| sage | del_hub + complete_aa | 0.705 | 0.710 | +0.005 |
| sage | feat_shuffle + smooth1 | 0.594 | 0.639 | +0.045 |
| sage | feat_shuffle + smooth2 | 0.594 | 0.672 | +0.077 |
| sage | del + hybrid_cn0 | 0.714 | 0.714 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.714 | 0.723 | +0.009 |
| sage | del + hybrid_cn0.5 | 0.714 | 0.725 | +0.011 |
| sage | del + hybrid_cn1 | 0.714 | 0.725 | +0.011 |
| sage | del + hybrid_cn2 | 0.714 | 0.725 | +0.011 |
| vgae | del + complete_cn | 0.768 | 0.766 | -0.001 |
| vgae | del + complete_aa | 0.768 | 0.761 | -0.007 |
| vgae | del_hub + complete_cn | 0.753 | 0.754 | +0.000 |
| vgae | del_hub + complete_aa | 0.753 | 0.752 | -0.001 |
| vgae | feat_shuffle + smooth1 | 0.574 | 0.625 | +0.051 |
| vgae | feat_shuffle + smooth2 | 0.574 | 0.649 | +0.075 |
| vgae | del + hybrid_cn0 | 0.768 | 0.768 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.768 | 0.777 | +0.010 |
| vgae | del + hybrid_cn0.5 | 0.768 | 0.779 | +0.012 |
| vgae | del + hybrid_cn1 | 0.768 | 0.780 | +0.012 |
| vgae | del + hybrid_cn2 | 0.768 | 0.780 | +0.012 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.876 | 0.912 | +0.035 |
| seal | del | 0.667 | 0.672 | +0.005 |
| seal | del_hub | 0.643 | 0.643 | -0.000 |
| seal | ins_hub | 0.826 | 0.815 | -0.011 |
| pencil | clean | 0.890 | 0.902 | +0.012 |
| pencil | del | 0.726 | 0.740 | +0.014 |
| pencil | del_hub | 0.717 | 0.715 | -0.001 |
| pencil | ins_hub | 0.849 | 0.816 | -0.032 |

