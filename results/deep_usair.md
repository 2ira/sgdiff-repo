# Deep probe: usair

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
| del | 0.063 | 0.225 | 0.902 | 0.869 |
| del_hub | 0.055 | 0.204 | 0.898 | 0.874 |
| ins_hub | 0.126 | 0.410 | 0.905 | 0.905 |

### gat

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.000 | 0.304 | 0.896 | 0.832 |
| del_hub | 0.000 | 0.263 | 0.896 | 0.828 |
| ins_hub | 0.000 | 0.430 | 0.896 | 0.872 |

### sage

| probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|---|---|---|---|---|
| del | 0.041 | 0.175 | 0.898 | 0.875 |
| del_hub | 0.033 | 0.167 | 0.898 | 0.859 |
| ins_hub | 0.067 | 0.291 | 0.902 | 0.893 |

## 2. Same weights, different propagation operator

| model | operator | clean | del@50% | ins_hub@50% |
|---|---|---|---|---|
| gcn | sym | 0.888 | 0.825 | 0.853 |
| gcn | row | 0.813 | 0.806 | 0.874 |
| gcn | sum | 0.869 | 0.854 | 0.884 |
| gcn | noloop | 0.877 | 0.821 | 0.806 |
| sage | sym | 0.878 | 0.842 | 0.880 |
| sage | row | 0.780 | 0.792 | 0.781 |
| sage | sum | 0.871 | 0.851 | 0.887 |
| sage | noloop | 0.878 | 0.842 | 0.880 |
| vgae | sym | 0.841 | 0.824 | 0.790 |
| vgae | row | 0.820 | 0.820 | 0.875 |
| vgae | sum | 0.870 | 0.850 | 0.886 |
| vgae | noloop | 0.830 | 0.807 | 0.766 |

## 3. Which kind of edge: triadic (CN > 0) vs bridging (CN = 0)

| model | clean | del_tri | del_bridge | tri edges | bridge edges |
|---|---|---|---|---|---|
| sgdiff | 0.935 | 0.909 | 0.932 | 1724 | 83 |
| seal | 0.952 | 0.905 | 0.949 | 1724 | 83 |
| pencil | 0.922 | 0.913 | 0.920 | 1724 | 83 |
| gcn | 0.888 | 0.839 | 0.881 | 1724 | 83 |
| gat | 0.859 | 0.835 | 0.857 | 1724 | 83 |
| sage | 0.878 | 0.847 | 0.875 | 1724 | 83 |
| vgae | 0.841 | 0.819 | 0.841 | 1724 | 83 |

## 4. Repairs on the inference graph (no retraining)

Reference on the damaged graph itself: CN-only AUC 0.865, AA-only 0.871 — a repair cannot be expected to beat the prior it uses.

| model | probe | base | repaired | gain |
|---|---|---|---|---|
| sgdiff | del + complete_cn | 0.905 | 0.899 | -0.006 |
| sgdiff | del + complete_aa | 0.905 | 0.903 | -0.002 |
| sgdiff | del_hub + complete_cn | 0.906 | 0.910 | +0.004 |
| sgdiff | del_hub + complete_aa | 0.906 | 0.919 | +0.013 |
| sgdiff | del + hybrid_cn0 | 0.905 | 0.905 | +0.000 |
| sgdiff | del + hybrid_cn0.25 | 0.905 | 0.909 | +0.003 |
| sgdiff | del + hybrid_cn0.5 | 0.905 | 0.910 | +0.005 |
| sgdiff | del + hybrid_cn1 | 0.905 | 0.910 | +0.005 |
| sgdiff | del + hybrid_cn2 | 0.905 | 0.910 | +0.004 |
| seal | del + complete_cn | 0.887 | 0.873 | -0.013 |
| seal | del + complete_aa | 0.887 | 0.878 | -0.008 |
| seal | del_hub + complete_cn | 0.898 | 0.899 | +0.001 |
| seal | del_hub + complete_aa | 0.898 | 0.905 | +0.007 |
| seal | del + hybrid_cn0 | 0.887 | 0.887 | +0.000 |
| seal | del + hybrid_cn0.25 | 0.887 | 0.892 | +0.005 |
| seal | del + hybrid_cn0.5 | 0.887 | 0.895 | +0.009 |
| seal | del + hybrid_cn1 | 0.887 | 0.898 | +0.012 |
| seal | del + hybrid_cn2 | 0.887 | 0.900 | +0.014 |
| pencil | del + complete_cn | 0.914 | 0.891 | -0.023 |
| pencil | del + complete_aa | 0.914 | 0.897 | -0.017 |
| pencil | del_hub + complete_cn | 0.927 | 0.904 | -0.022 |
| pencil | del_hub + complete_aa | 0.927 | 0.909 | -0.018 |
| pencil | del + hybrid_cn0 | 0.914 | 0.914 | +0.000 |
| pencil | del + hybrid_cn0.25 | 0.914 | 0.915 | +0.001 |
| pencil | del + hybrid_cn0.5 | 0.914 | 0.916 | +0.002 |
| pencil | del + hybrid_cn1 | 0.914 | 0.916 | +0.002 |
| pencil | del + hybrid_cn2 | 0.914 | 0.916 | +0.002 |
| gcn | del + complete_cn | 0.825 | 0.802 | -0.024 |
| gcn | del + complete_aa | 0.825 | 0.809 | -0.017 |
| gcn | del_hub + complete_cn | 0.824 | 0.821 | -0.003 |
| gcn | del_hub + complete_aa | 0.824 | 0.821 | -0.003 |
| gcn | del + hybrid_cn0 | 0.825 | 0.825 | +0.000 |
| gcn | del + hybrid_cn0.25 | 0.825 | 0.841 | +0.016 |
| gcn | del + hybrid_cn0.5 | 0.825 | 0.851 | +0.025 |
| gcn | del + hybrid_cn1 | 0.825 | 0.861 | +0.035 |
| gcn | del + hybrid_cn2 | 0.825 | 0.868 | +0.042 |
| gat | del + complete_cn | 0.834 | 0.838 | +0.003 |
| gat | del + complete_aa | 0.834 | 0.821 | -0.013 |
| gat | del_hub + complete_cn | 0.822 | 0.831 | +0.009 |
| gat | del_hub + complete_aa | 0.822 | 0.816 | -0.005 |
| gat | del + hybrid_cn0 | 0.834 | 0.834 | +0.000 |
| gat | del + hybrid_cn0.25 | 0.834 | 0.855 | +0.021 |
| gat | del + hybrid_cn0.5 | 0.834 | 0.865 | +0.031 |
| gat | del + hybrid_cn1 | 0.834 | 0.875 | +0.041 |
| gat | del + hybrid_cn2 | 0.834 | 0.884 | +0.050 |
| sage | del + complete_cn | 0.842 | 0.833 | -0.008 |
| sage | del + complete_aa | 0.842 | 0.836 | -0.006 |
| sage | del_hub + complete_cn | 0.839 | 0.830 | -0.010 |
| sage | del_hub + complete_aa | 0.839 | 0.836 | -0.004 |
| sage | del + hybrid_cn0 | 0.842 | 0.842 | +0.000 |
| sage | del + hybrid_cn0.25 | 0.842 | 0.868 | +0.027 |
| sage | del + hybrid_cn0.5 | 0.842 | 0.874 | +0.032 |
| sage | del + hybrid_cn1 | 0.842 | 0.879 | +0.037 |
| sage | del + hybrid_cn2 | 0.842 | 0.879 | +0.038 |
| vgae | del + complete_cn | 0.824 | 0.810 | -0.014 |
| vgae | del + complete_aa | 0.824 | 0.813 | -0.011 |
| vgae | del_hub + complete_cn | 0.818 | 0.812 | -0.006 |
| vgae | del_hub + complete_aa | 0.818 | 0.811 | -0.008 |
| vgae | del + hybrid_cn0 | 0.824 | 0.824 | +0.000 |
| vgae | del + hybrid_cn0.25 | 0.824 | 0.840 | +0.016 |
| vgae | del + hybrid_cn0.5 | 0.824 | 0.849 | +0.025 |
| vgae | del + hybrid_cn1 | 0.824 | 0.860 | +0.036 |
| vgae | del + hybrid_cn2 | 0.824 | 0.869 | +0.045 |

## 5. Model-side repair: give the sub-graph family more evidence (k = 2)

SEAL / PENCIL are retrained on 2-hop enclosing sub-graphs (default is k = 1).
If the deletion damage is evidence starvation rather than a weight-space property,
the k = 2 model should lose less under the deletion probes.

| model | probe | k = 1 | k = 2 | gain |
|---|---|---|---|---|
| seal | clean | 0.952 | 0.951 | -0.001 |
| seal | del | 0.887 | 0.869 | -0.017 |
| seal | del_hub | 0.898 | 0.895 | -0.003 |
| seal | ins_hub | 0.856 | 0.857 | +0.001 |
| pencil | clean | 0.922 | 0.938 | +0.016 |
| pencil | del | 0.914 | 0.876 | -0.038 |
| pencil | del_hub | 0.927 | 0.875 | -0.052 |
| pencil | ins_hub | 0.738 | 0.842 | +0.104 |

## Skipped

* `feature_shuffle`: feature_shuffle needs node features
