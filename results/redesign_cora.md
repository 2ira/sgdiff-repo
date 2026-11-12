# Component redesign: cora

* seeds: [0, 1, 2, 3, 4, 5, 6, 7]
* every row retrains from scratch with one component changed; `base` is the reference
* augmented inputs are rebuilt from the graph being scored (test positives withheld)

## 1. AUC after retraining

| config | clean | del | del_hub | ins_hub | feat_shuffle |
|---|---:|---:|---:|---:|---:|
| base | 0.813 | 0.758 | 0.746 | 0.785 | 0.647 |
| depth-L1 | 0.833 | 0.789 | 0.785 | 0.813 | 0.597 |
| depth-L3 | 0.824 | 0.784 | 0.782 | 0.784 | 0.599 |
| noloop | 0.810 | 0.696 | 0.689 | 0.785 | 0.643 |
| row | 0.802 | 0.742 | 0.739 | 0.769 | 0.645 |
| +cnfeat | 0.806 | 0.744 | 0.738 | 0.778 | 0.646 |
| +smooth | 0.813 | 0.760 | 0.741 | 0.782 | 0.666 |
| +cnfeat+smooth | 0.807 | 0.744 | 0.732 | 0.776 | 0.677 |
| +edgedropout | 0.839 | 0.785 | 0.778 | 0.809 | 0.633 |

## 2. Damage = clean − probe (smaller is more robust)

| config | del | del_hub | ins_hub | feat_shuffle | Δ vs base (mean) |
|---|---:|---:|---:|---:|---:|
| base | 0.056 | 0.067 | 0.029 | 0.167 | +0.000 |
| depth-L1 | 0.044 | 0.048 | 0.021 | 0.236 | +0.008 |
| depth-L3 | 0.040 | 0.042 | 0.040 | 0.225 | +0.007 |
| noloop | 0.114 | 0.121 | 0.025 | 0.167 | +0.027 |
| row | 0.061 | 0.063 | 0.033 | 0.157 | -0.001 |
| +cnfeat | 0.062 | 0.068 | 0.028 | 0.160 | -0.000 |
| +smooth | 0.053 | 0.072 | 0.031 | 0.147 | -0.004 |
| +cnfeat+smooth | 0.063 | 0.075 | 0.031 | 0.130 | -0.005 |
| +edgedropout | 0.055 | 0.061 | 0.030 | 0.206 | +0.009 |

Reference clean AUC (`base`): 0.813.

## 3. Training cost

| config | seconds / fit |
|---|---:|
| base | 0.5 |
| depth-L1 | 0.4 |
| depth-L3 | 0.6 |
| noloop | 0.5 |
| row | 0.4 |
| +cnfeat | 0.6 |
| +smooth | 0.4 |
| +cnfeat+smooth | 0.5 |
| +edgedropout | 0.4 |
