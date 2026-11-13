# Adaptive vs non-adaptive deletion: usair

Same model, same working pairs, same number of deleted edges; only the *choice* of edges differs.
`adaptive` rebuilds the evidence pool on the current graph and deletes the candidate that costs the most AUC.

| model | budget | random ΔAUC | hub ΔAUC | adaptive ΔAUC | extra damage vs random | deleted CN (adaptive) | bridge share |
|---|---:|---:|---:|---:|---:|---:|---:|
| gcn | 5% | +0.001 | +0.008 | +0.077 | +0.076 | 4.58 | 0.13 |
| gcn | 10% | +0.004 | +0.015 | +0.145 | +0.141 | 5.77 | 0.12 |
| sage | 5% | +0.001 | +0.004 | +0.080 | +0.078 | 5.30 | 0.09 |
| sage | 10% | +0.003 | +0.010 | +0.124 | +0.121 | 6.23 | 0.10 |

`clean` AUC per model: gcn 0.889, sage 0.871.
