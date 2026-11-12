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
| del | 0.069 | 0.078 | 0.847 | 0.827 |
| del_hub | 0.051 | 0.059 | 0.848 | 0.821 |
| ins_hub | 0.037 | 0.051 | 0.842 | 0.835 |
| feat_shuffle | 0.988 | 0.936 | 0.592 | 0.686 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.339 | 0.803 | 0.686 |
| del_hub | 0.000 | 0.229 | 0.803 | 0.664 |
| ins_hub | 0.000 | 0.195 | 0.803 | 0.712 |
| feat_shuffle | 0.990 | 0.803 | 0.497 | 0.608 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.055 | 0.139 | 0.805 | 0.778 |
| del_hub | 0.037 | 0.089 | 0.794 | 0.776 |
| ins_hub | 0.017 | 0.049 | 0.802 | 0.786 |
| feat_shuffle | 0.990 | 0.944 | 0.525 | 0.584 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | feature_shuffle | ins_hub@50% |
|---|---|---|---|---|---|
| gcn | sym | 0.815 | 0.787 | 0.609 | 0.784 |
| gcn | row | 0.803 | 0.779 | 0.598 | 0.749 |
| gcn | sum | 0.774 | 0.761 | 0.699 | 0.720 |
| gcn | noloop | 0.741 | 0.675 | 0.590 | 0.727 |
| sage | sym | 0.754 | 0.736 | 0.567 | 0.743 |
| sage | row | 0.739 | 0.733 | 0.561 | 0.712 |
| sage | sum | 0.743 | 0.732 | 0.678 | 0.703 |
| sage | noloop | 0.754 | 0.736 | 0.567 | 0.743 |
| vgae | sym | 0.781 | 0.756 | 0.571 | 0.732 |
| vgae | row | 0.775 | 0.753 | 0.570 | 0.725 |
| vgae | sum | 0.780 | 0.765 | 0.675 | 0.734 |
| vgae | noloop | 0.731 | 0.665 | 0.571 | 0.691 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.773 | 0.720 | 0.698 | 1271 | 2598 |
| seal | 0.740 | 0.725 | 0.690 | 1271 | 2598 |
| pencil | 0.816 | 0.812 | 0.792 | 1271 | 2598 |
| gcn | 0.815 | 0.812 | 0.795 | 1271 | 2598 |
| gat | 0.738 | 0.720 | 0.703 | 1271 | 2598 |
| sage | 0.754 | 0.746 | 0.749 | 1271 | 2598 |
| vgae | 0.781 | 0.778 | 0.764 | 1271 | 2598 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.561, AA-only 0.561 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.664 | 0.652 | -0.012 |
| sgdiff | del + complete_aa | 0.664 | 0.639 | -0.025 |
| sgdiff | del_hub + complete_cn | 0.550 | 0.608 | +0.058 |
| sgdiff | del_hub + complete_aa | 0.550 | 0.608 | +0.058 |
| sgdiff | feat_shuffle + smooth1 | 0.773 | 0.773 | +0.000 |
| sgdiff | feat_shuffle + smooth2 | 0.773 | 0.773 | +0.000 |
| sgdiff | del + hybrid_cn0 | 0.664 | 0.664 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.664 | 0.666 | +0.001 |
| sgdiff | del + hybrid_cn0.5 | 0.664 | 0.666 | +0.002 |
| sgdiff | del + hybrid_cn1 | 0.664 | 0.666 | +0.002 |
| sgdiff | del + hybrid_cn2 | 0.664 | 0.666 | +0.002 |
| seal | del + complete_cn | 0.654 | 0.663 | +0.009 |
| seal | del + complete_aa | 0.654 | 0.662 | +0.008 |
| seal | del_hub + complete_cn | 0.694 | 0.708 | +0.014 |
| seal | del_hub + complete_aa | 0.694 | 0.708 | +0.014 |
| seal | feat_shuffle + smooth1 | 0.535 | 0.535 | +0.001 |
| seal | feat_shuffle + smooth2 | 0.535 | 0.552 | +0.017 |
| seal | del + hybrid_cn0 | 0.654 | 0.654 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.654 | 0.666 | +0.013 |
| seal | del + hybrid_cn0.5 | 0.654 | 0.669 | +0.015 |
| seal | del + hybrid_cn1 | 0.654 | 0.669 | +0.016 |
| seal | del + hybrid_cn2 | 0.654 | 0.669 | +0.016 |
| pencil | del + complete_cn | 0.769 | 0.771 | +0.002 |
| pencil | del + complete_aa | 0.769 | 0.773 | +0.004 |
| pencil | del_hub + complete_cn | 0.789 | 0.790 | +0.001 |
| pencil | del_hub + complete_aa | 0.789 | 0.790 | +0.001 |
| pencil | feat_shuffle + smooth1 | 0.598 | 0.623 | +0.025 |
| pencil | feat_shuffle + smooth2 | 0.598 | 0.669 | +0.071 |
| pencil | del + hybrid_cn0 | 0.769 | 0.769 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.769 | 0.775 | +0.006 |
| pencil | del + hybrid_cn0.5 | 0.769 | 0.776 | +0.007 |
| pencil | del + hybrid_cn1 | 0.769 | 0.776 | +0.007 |
| pencil | del + hybrid_cn2 | 0.769 | 0.776 | +0.007 |
| gcn | del + complete_cn | 0.787 | 0.778 | -0.010 |
| gcn | del + complete_aa | 0.787 | 0.774 | -0.013 |
| gcn | del_hub + complete_cn | 0.783 | 0.778 | -0.006 |
| gcn | del_hub + complete_aa | 0.783 | 0.778 | -0.006 |
| gcn | feat_shuffle + smooth1 | 0.609 | 0.616 | +0.007 |
| gcn | feat_shuffle + smooth2 | 0.609 | 0.630 | +0.021 |
| gcn | del + hybrid_cn0 | 0.787 | 0.787 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.787 | 0.793 | +0.006 |
| gcn | del + hybrid_cn0.5 | 0.787 | 0.794 | +0.006 |
| gcn | del + hybrid_cn1 | 0.787 | 0.794 | +0.006 |
| gcn | del + hybrid_cn2 | 0.787 | 0.794 | +0.006 |
| gat | del + complete_cn | 0.681 | 0.691 | +0.010 |
| gat | del + complete_aa | 0.681 | 0.696 | +0.015 |
| gat | del_hub + complete_cn | 0.641 | 0.654 | +0.012 |
| gat | del_hub + complete_aa | 0.641 | 0.654 | +0.012 |
| gat | feat_shuffle + smooth1 | 0.579 | 0.623 | +0.045 |
| gat | feat_shuffle + smooth2 | 0.579 | 0.623 | +0.044 |
| gat | del + hybrid_cn0 | 0.681 | 0.681 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.681 | 0.684 | +0.004 |
| gat | del + hybrid_cn0.5 | 0.681 | 0.685 | +0.004 |
| gat | del + hybrid_cn1 | 0.681 | 0.685 | +0.005 |
| gat | del + hybrid_cn2 | 0.681 | 0.685 | +0.005 |
| sage | del + complete_cn | 0.736 | 0.735 | -0.001 |
| sage | del + complete_aa | 0.736 | 0.735 | -0.001 |
| sage | del_hub + complete_cn | 0.715 | 0.715 | -0.000 |
| sage | del_hub + complete_aa | 0.715 | 0.715 | -0.000 |
| sage | feat_shuffle + smooth1 | 0.567 | 0.586 | +0.019 |
| sage | feat_shuffle + smooth2 | 0.567 | 0.606 | +0.040 |
| sage | del + hybrid_cn0 | 0.736 | 0.736 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.736 | 0.742 | +0.006 |
| sage | del + hybrid_cn0.5 | 0.736 | 0.743 | +0.007 |
| sage | del + hybrid_cn1 | 0.736 | 0.743 | +0.007 |
| sage | del + hybrid_cn2 | 0.736 | 0.743 | +0.007 |
| vgae | del + complete_cn | 0.756 | 0.751 | -0.004 |
| vgae | del + complete_aa | 0.756 | 0.747 | -0.009 |
| vgae | del_hub + complete_cn | 0.745 | 0.744 | -0.002 |
| vgae | del_hub + complete_aa | 0.745 | 0.744 | -0.002 |
| vgae | feat_shuffle + smooth1 | 0.571 | 0.592 | +0.021 |
| vgae | feat_shuffle + smooth2 | 0.571 | 0.604 | +0.033 |
| vgae | del + hybrid_cn0 | 0.756 | 0.756 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.756 | 0.766 | +0.010 |
| vgae | del + hybrid_cn0.5 | 0.756 | 0.767 | +0.012 |
| vgae | del + hybrid_cn1 | 0.756 | 0.768 | +0.012 |
| vgae | del + hybrid_cn2 | 0.756 | 0.768 | +0.012 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.740 | 0.814 | +0.074 |
| seal | del | 0.654 | 0.632 | -0.021 |
| seal | del_hub | 0.694 | 0.676 | -0.018 |
| seal | ins_hub | 0.664 | 0.718 | +0.054 |
| pencil | clean | 0.816 | 0.835 | +0.019 |
| pencil | del | 0.769 | 0.757 | -0.012 |
| pencil | del_hub | 0.789 | 0.768 | -0.022 |
| pencil | ins_hub | 0.796 | 0.783 | -0.012 |

