# Deep probe: cora

* models: sgdiff, seal, pencil, gcn, gat, sage, vgae
* seeds: [0, 1, 2, 3]
* variants: del, del_hub, ins_hub, feat_shuffle

## 1. Layer-wise damage (node-pair models)

`drift` = 1 − mean cosine between the layer representation on the probe graph and
on the clean graph; `probe_auc` = AUC of a linear read-out trained on the *clean*
layer and applied to the probe layer (chance ≈ 0.5).

### gcn

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.093 | 0.092 | 0.831 | 0.804 |
| del_hub | 0.070 | 0.076 | 0.821 | 0.796 |
| ins_hub | 0.037 | 0.049 | 0.839 | 0.831 |
| feat_shuffle | 0.998 | 0.962 | 0.604 | 0.667 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.322 | 0.762 | 0.693 |
| del_hub | 0.000 | 0.253 | 0.762 | 0.712 |
| ins_hub | 0.000 | 0.171 | 0.762 | 0.765 |
| feat_shuffle | 0.996 | 0.896 | 0.520 | 0.640 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.065 | 0.132 | 0.751 | 0.742 |
| del_hub | 0.047 | 0.095 | 0.753 | 0.735 |
| ins_hub | 0.018 | 0.046 | 0.756 | 0.759 |
| feat_shuffle | 0.994 | 0.951 | 0.551 | 0.587 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | feature_shuffle | ins_hub@50% |
|---|---|---|---|---|---|
| gcn | sym | 0.827 | 0.781 | 0.629 | 0.804 |
| gcn | row | 0.820 | 0.771 | 0.641 | 0.781 |
| gcn | sum | 0.807 | 0.765 | 0.712 | 0.755 |
| gcn | noloop | 0.808 | 0.708 | 0.644 | 0.783 |
| sage | sym | 0.757 | 0.714 | 0.575 | 0.728 |
| sage | row | 0.752 | 0.714 | 0.574 | 0.712 |
| sage | sum | 0.768 | 0.718 | 0.684 | 0.715 |
| sage | noloop | 0.757 | 0.714 | 0.575 | 0.728 |
| vgae | sym | 0.784 | 0.739 | 0.589 | 0.740 |
| vgae | row | 0.781 | 0.732 | 0.593 | 0.735 |
| vgae | sum | 0.791 | 0.737 | 0.673 | 0.735 |
| vgae | noloop | 0.771 | 0.693 | 0.589 | 0.721 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.732 | 0.651 | 0.631 | 1984 | 2502 |
| seal | 0.781 | 0.727 | 0.706 | 1984 | 2502 |
| pencil | 0.795 | 0.740 | 0.736 | 1984 | 2502 |
| gcn | 0.827 | 0.818 | 0.794 | 1984 | 2502 |
| gat | 0.807 | 0.781 | 0.736 | 1984 | 2502 |
| sage | 0.757 | 0.746 | 0.723 | 1984 | 2502 |
| vgae | 0.784 | 0.774 | 0.743 | 1984 | 2502 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.570, AA-only 0.570 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.522 | 0.619 | +0.097 |
| sgdiff | del + complete_aa | 0.522 | 0.608 | +0.086 |
| sgdiff | del_hub + complete_cn | 0.561 | 0.596 | +0.035 |
| sgdiff | del_hub + complete_aa | 0.561 | 0.605 | +0.045 |
| sgdiff | feat_shuffle + smooth1 | 0.732 | 0.732 | +0.000 |
| sgdiff | feat_shuffle + smooth2 | 0.732 | 0.732 | +0.000 |
| sgdiff | del + hybrid_cn0 | 0.522 | 0.522 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.522 | 0.522 | +0.000 |
| sgdiff | del + hybrid_cn0.5 | 0.522 | 0.522 | +0.000 |
| sgdiff | del + hybrid_cn1 | 0.522 | 0.522 | +0.000 |
| sgdiff | del + hybrid_cn2 | 0.522 | 0.522 | +0.000 |
| seal | del + complete_cn | 0.644 | 0.650 | +0.006 |
| seal | del + complete_aa | 0.644 | 0.639 | -0.005 |
| seal | del_hub + complete_cn | 0.632 | 0.668 | +0.036 |
| seal | del_hub + complete_aa | 0.632 | 0.659 | +0.026 |
| seal | feat_shuffle + smooth1 | 0.726 | 0.725 | -0.001 |
| seal | feat_shuffle + smooth2 | 0.726 | 0.718 | -0.008 |
| seal | del + hybrid_cn0 | 0.644 | 0.644 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.644 | 0.650 | +0.006 |
| seal | del + hybrid_cn0.5 | 0.644 | 0.651 | +0.007 |
| seal | del + hybrid_cn1 | 0.644 | 0.651 | +0.007 |
| seal | del + hybrid_cn2 | 0.644 | 0.651 | +0.007 |
| pencil | del + complete_cn | 0.671 | 0.680 | +0.010 |
| pencil | del + complete_aa | 0.671 | 0.675 | +0.005 |
| pencil | del_hub + complete_cn | 0.650 | 0.690 | +0.040 |
| pencil | del_hub + complete_aa | 0.650 | 0.684 | +0.034 |
| pencil | feat_shuffle + smooth1 | 0.755 | 0.762 | +0.007 |
| pencil | feat_shuffle + smooth2 | 0.755 | 0.763 | +0.008 |
| pencil | del + hybrid_cn0 | 0.671 | 0.671 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.671 | 0.672 | +0.002 |
| pencil | del + hybrid_cn0.5 | 0.671 | 0.673 | +0.002 |
| pencil | del + hybrid_cn1 | 0.671 | 0.673 | +0.002 |
| pencil | del + hybrid_cn2 | 0.671 | 0.673 | +0.002 |
| gcn | del + complete_cn | 0.781 | 0.773 | -0.008 |
| gcn | del + complete_aa | 0.781 | 0.773 | -0.007 |
| gcn | del_hub + complete_cn | 0.768 | 0.769 | +0.001 |
| gcn | del_hub + complete_aa | 0.768 | 0.764 | -0.004 |
| gcn | feat_shuffle + smooth1 | 0.629 | 0.659 | +0.030 |
| gcn | feat_shuffle + smooth2 | 0.629 | 0.678 | +0.049 |
| gcn | del + hybrid_cn0 | 0.781 | 0.781 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.781 | 0.791 | +0.010 |
| gcn | del + hybrid_cn0.5 | 0.781 | 0.792 | +0.011 |
| gcn | del + hybrid_cn1 | 0.781 | 0.793 | +0.012 |
| gcn | del + hybrid_cn2 | 0.781 | 0.793 | +0.012 |
| gat | del + complete_cn | 0.705 | 0.711 | +0.006 |
| gat | del + complete_aa | 0.705 | 0.715 | +0.010 |
| gat | del_hub + complete_cn | 0.712 | 0.718 | +0.006 |
| gat | del_hub + complete_aa | 0.712 | 0.720 | +0.007 |
| gat | feat_shuffle + smooth1 | 0.627 | 0.660 | +0.033 |
| gat | feat_shuffle + smooth2 | 0.627 | 0.665 | +0.039 |
| gat | del + hybrid_cn0 | 0.705 | 0.705 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.705 | 0.711 | +0.006 |
| gat | del + hybrid_cn0.5 | 0.705 | 0.712 | +0.007 |
| gat | del + hybrid_cn1 | 0.705 | 0.712 | +0.007 |
| gat | del + hybrid_cn2 | 0.705 | 0.712 | +0.007 |
| sage | del + complete_cn | 0.714 | 0.711 | -0.003 |
| sage | del + complete_aa | 0.714 | 0.712 | -0.002 |
| sage | del_hub + complete_cn | 0.697 | 0.708 | +0.011 |
| sage | del_hub + complete_aa | 0.697 | 0.706 | +0.009 |
| sage | feat_shuffle + smooth1 | 0.575 | 0.602 | +0.027 |
| sage | feat_shuffle + smooth2 | 0.575 | 0.627 | +0.052 |
| sage | del + hybrid_cn0 | 0.714 | 0.714 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.714 | 0.724 | +0.010 |
| sage | del + hybrid_cn0.5 | 0.714 | 0.726 | +0.012 |
| sage | del + hybrid_cn1 | 0.714 | 0.726 | +0.012 |
| sage | del + hybrid_cn2 | 0.714 | 0.726 | +0.012 |
| vgae | del + complete_cn | 0.739 | 0.736 | -0.003 |
| vgae | del + complete_aa | 0.739 | 0.737 | -0.002 |
| vgae | del_hub + complete_cn | 0.722 | 0.731 | +0.009 |
| vgae | del_hub + complete_aa | 0.722 | 0.727 | +0.005 |
| vgae | feat_shuffle + smooth1 | 0.589 | 0.619 | +0.030 |
| vgae | feat_shuffle + smooth2 | 0.589 | 0.641 | +0.051 |
| vgae | del + hybrid_cn0 | 0.739 | 0.739 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.739 | 0.751 | +0.012 |
| vgae | del + hybrid_cn0.5 | 0.739 | 0.754 | +0.015 |
| vgae | del + hybrid_cn1 | 0.739 | 0.755 | +0.016 |
| vgae | del + hybrid_cn2 | 0.739 | 0.755 | +0.016 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.781 | 0.856 | +0.075 |
| seal | del | 0.644 | 0.648 | +0.004 |
| seal | del_hub | 0.632 | 0.640 | +0.008 |
| seal | ins_hub | 0.760 | 0.803 | +0.043 |
| pencil | clean | 0.795 | 0.860 | +0.065 |
| pencil | del | 0.671 | 0.688 | +0.017 |
| pencil | del_hub | 0.650 | 0.662 | +0.012 |
| pencil | ins_hub | 0.788 | 0.813 | +0.025 |

