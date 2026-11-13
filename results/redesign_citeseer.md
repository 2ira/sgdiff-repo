# Component redesign: citeseer

* seeds: [0, 1, 2, 3, 4, 5, 6, 7]
* every row retrains from scratch with one component changed; `base` is the reference
* augmented inputs are rebuilt from the graph being scored (test positives withheld)

## 1. AUC after retraining

| config | clean | del | del_hub | ins_hub | feat_shuffle |
|---|---:|---:|---:|---:|---:|
| base | 0.799 | 0.768 | 0.755 | 0.771 | 0.610 |
| depth-L1 | 0.862 | 0.837 | 0.827 | 0.831 | 0.573 |
| depth-L3 | 0.805 | 0.786 | 0.779 | 0.750 | 0.591 |
| noloop | 0.733 | 0.681 | 0.650 | 0.723 | 0.586 |
| row | 0.781 | 0.754 | 0.746 | 0.745 | 0.608 |
| +cnfeat | 0.786 | 0.758 | 0.750 | 0.761 | 0.608 |
| +smooth | 0.786 | 0.762 | 0.741 | 0.755 | 0.616 |
| +cnfeat+smooth | 0.775 | 0.754 | 0.736 | 0.751 | 0.630 |
| +edgedropout | 0.818 | 0.784 | 0.772 | 0.786 | 0.605 |

## 2. Damage = clean − probe (smaller is more robust)

| config | del | del_hub | ins_hub | feat_shuffle | Δ vs base (mean) |
|---|---:|---:|---:|---:|---:|
| base | 0.031 | 0.043 | 0.028 | 0.189 | +0.000 |
| depth-L1 | 0.025 | 0.035 | 0.031 | 0.289 | +0.022 |
| depth-L3 | 0.019 | 0.026 | 0.055 | 0.214 | +0.006 |
| noloop | 0.052 | 0.083 | 0.009 | 0.147 | -0.000 |
| row | 0.028 | 0.035 | 0.037 | 0.174 | -0.004 |
| +cnfeat | 0.028 | 0.036 | 0.025 | 0.178 | -0.006 |
| +smooth | 0.024 | 0.045 | 0.030 | 0.169 | -0.006 |
| +cnfeat+smooth | 0.021 | 0.039 | 0.023 | 0.145 | -0.016 |
| +edgedropout | 0.034 | 0.046 | 0.032 | 0.213 | +0.008 |

Reference clean AUC (`base`): 0.799.

## 3. Training cost

| config | seconds / fit |
|---|---:|
| base | 0.9 |
| depth-L1 | 0.7 |
| depth-L3 | 1.1 |
| noloop | 0.9 |
| row | 0.8 |
| +cnfeat | 1.0 |
| +smooth | 0.8 |
| +cnfeat+smooth | 1.0 |
| +edgedropout | 0.9 |
