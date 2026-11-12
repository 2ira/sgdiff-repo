# Adaptive vs non-adaptive deletion: cora

Same model, same working pairs, same number of deleted edges; only the *choice* of edges differs.
`adaptive` rebuilds the evidence pool on the current graph and deletes the candidate that costs the most AUC.

| model | budget | random ΔAUC | hub ΔAUC | adaptive ΔAUC | extra damage vs random | deleted CN (adaptive) | bridge share |
|---|---:|---:|---:|---:|---:|---:|---:|
| gcn | 5% | +0.003 | +0.005 | +0.112 | +0.109 | 0.65 | 0.61 |
| gcn | 10% | +0.013 | +0.011 | +0.220 | +0.207 | 0.52 | 0.65 |
| sage | 5% | +0.007 | -0.002 | +0.152 | +0.145 | 0.58 | 0.64 |
| sage | 10% | +0.010 | +0.004 | +0.268 | +0.258 | 0.50 | 0.66 |

`clean` AUC per model: gcn 0.826, sage 0.738.
