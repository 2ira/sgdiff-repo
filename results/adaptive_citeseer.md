# Adaptive vs non-adaptive deletion: citeseer

Same model, same working pairs, same number of deleted edges; only the *choice* of edges differs.
`adaptive` rebuilds the evidence pool on the current graph and deletes the candidate that costs the most AUC.

| model | budget | random ΔAUC | hub ΔAUC | adaptive ΔAUC | extra damage vs random | deleted CN (adaptive) | bridge share |
|---|---:|---:|---:|---:|---:|---:|---:|
| gcn | 5% | -0.000 | +0.004 | +0.085 | +0.085 | 0.65 | 0.66 |
| gcn | 10% | +0.002 | +0.005 | +0.161 | +0.159 | 0.45 | 0.73 |
| sage | 5% | +0.001 | +0.007 | +0.127 | +0.125 | 0.61 | 0.67 |
| sage | 10% | +0.001 | +0.009 | +0.217 | +0.216 | 0.43 | 0.73 |

`clean` AUC per model: gcn 0.767, sage 0.739.
