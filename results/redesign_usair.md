# Component redesign: usair

* seeds: [0, 1, 2, 3, 4, 5, 6, 7]
* every row retrains from scratch with one component changed; `base` is the reference
* augmented inputs are rebuilt from the graph being scored (test positives withheld)

## 1. AUC after retraining

| config | clean | del | del_hub | ins_hub |
|---|---:|---:|---:|---:|
| base | 0.886 | 0.838 | 0.850 | 0.870 |
| depth-L1 | 0.871 | 0.818 | 0.834 | 0.875 |
| depth-L3 | 0.875 | 0.842 | 0.844 | 0.897 |
| noloop | 0.882 | 0.844 | 0.854 | 0.892 |
| row | 0.840 | 0.757 | 0.696 | 0.878 |
| +cnfeat | 0.894 | 0.848 | 0.854 | 0.855 |
| +smooth | 0.869 | 0.826 | 0.807 | 0.894 |
| +cnfeat+smooth | 0.890 | 0.842 | 0.844 | 0.892 |
| +edgedropout | 0.878 | 0.832 | 0.843 | 0.888 |

## 2. Damage = clean − probe (smaller is more robust)

| config | del | del_hub | ins_hub | Δ vs base (mean) |
|---|---:|---:|---:|---:|
| base | 0.049 | 0.036 | 0.016 | +0.000 |
| depth-L1 | 0.054 | 0.038 | -0.003 | -0.004 |
| depth-L3 | 0.033 | 0.031 | -0.022 | -0.020 |
| noloop | 0.038 | 0.028 | -0.010 | -0.015 |
| row | 0.083 | 0.144 | -0.038 | +0.029 |
| +cnfeat | 0.046 | 0.040 | 0.039 | +0.008 |
| +smooth | 0.043 | 0.063 | -0.025 | -0.007 |
| +cnfeat+smooth | 0.048 | 0.046 | -0.002 | -0.003 |
| +edgedropout | 0.045 | 0.035 | -0.010 | -0.010 |

Reference clean AUC (`base`): 0.886.

## 3. Training cost

| config | seconds / fit |
|---|---:|
| base | 0.3 |
| depth-L1 | 0.1 |
| depth-L3 | 0.3 |
| noloop | 0.2 |
| row | 0.1 |
| +cnfeat | 0.2 |
| +smooth | 0.1 |
| +cnfeat+smooth | 0.2 |
| +edgedropout | 0.2 |
