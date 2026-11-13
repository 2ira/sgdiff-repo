# Adaptive vs non-adaptive deletion: cora

Same model, same working pairs, same number of deleted edges; only the *choice* of edges differs.
`adaptive` rebuilds the evidence pool on the current graph and deletes the candidate that costs the most AUC.

| model | budget | random ΔAUC | hub ΔAUC | adaptive ΔAUC | extra damage vs random | deleted CN (adaptive) | bridge share |
|---|---:|---:|---:|---:|---:|---:|---:|
| seal | 5% | +0.008 | +0.003 | +0.290 | +0.282 | 0.71 | 0.57 |
| gcn | 2% | +0.002 | +0.004 | +0.037 | +0.035 | 0.79 | 0.52 |
| gcn | 5% | +0.003 | +0.005 | +0.115 | +0.113 | 0.62 | 0.62 |
| gcn | 10% | +0.013 | +0.011 | +0.218 | +0.205 | 0.51 | 0.66 |
| gcn | 20% | +0.024 | +0.029 | +0.314 | +0.290 | 0.50 | 0.66 |
| sage | 5% | +0.004 | +0.001 | +0.154 | +0.150 | 0.58 | 0.63 |
| sage | 10% | +0.004 | +0.006 | +0.282 | +0.278 | 0.46 | 0.68 |
| pencil | 5% | +0.012 | +0.004 | +0.194 | +0.182 | 0.73 | 0.58 |

`clean` AUC per model: seal 0.774, gcn 0.826, sage 0.738, pencil 0.797.
