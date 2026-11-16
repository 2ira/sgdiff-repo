# Counterfactual splice gate: cora

For every pair the k = 1 model is scored twice: once on its enclosing sub-graph, once with a
synthetic shared neighbour `w` spliced in (`u-w`, `v-w`, DRNL label 2). The gate routes to the
k = 2 model for the pairs whose score the splice raises most. Damage = clean − probe.

| model | probe | k = 1 | k = 2 (global) | `CN = 0` gate | splice gate (top 50 %) | splice − k1 | 95 % CI (paired) |
|---|---|---:|---:|---:|---:|---:|---|
| seal | clean | 0.872 | 0.906 | 0.908 | 0.903 | +0.031 | — |
| seal | del@50% | +0.212 | +0.235 | +0.235 | +0.228 | -0.017 | [-0.036, +0.003] |
| seal | del_hub@50% | +0.229 | +0.262 | +0.263 | +0.248 | -0.019 | [-0.040, +0.003] |
| seal | ins_hub@50% | +0.044 | +0.088 | +0.082 | +0.085 | -0.041 | [-0.058, -0.023] |
| pencil | clean | 0.883 | 0.897 | 0.887 | 0.885 | +0.002 | — |
| pencil | del@50% | +0.170 | +0.176 | +0.171 | +0.158 | +0.011 | [-0.011, +0.033] |
| pencil | del_hub@50% | +0.168 | +0.203 | +0.197 | +0.162 | +0.007 | [-0.017, +0.030] |
| pencil | ins_hub@50% | +0.039 | +0.084 | +0.119 | +0.112 | -0.073 | [-0.096, -0.050] |

Positive `splice − k1` on a damage row = the splice gate shrinks the damage; on the clean row it is
the ceiling it buys. The interval is a paired bootstrap over scored pairs for the damage difference;
it covers pair sampling, not training seeds.

## Gate frontier: sign is useless, magnitude is not

The splice raises the score for almost every pair, so the *sign* gate degenerates into "always k2".
The quantile gates route only the pairs with the largest rise; `CN = 0` is the §7.11 gate, shown for
reference.

| model | probe | `CN = 0` | sign(Δ) | top 50 % | top 30 % | top 10 % | always k2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seal | del@50% | +0.235 | +0.235 | +0.228 | +0.220 | +0.212 | +0.235 | +0.212 |
| seal | del_hub@50% | +0.263 | +0.262 | +0.248 | +0.238 | +0.230 | +0.262 | +0.229 |
| seal | ins_hub@50% | +0.082 | +0.088 | +0.085 | +0.075 | +0.044 | +0.088 | +0.044 |
| pencil | del@50% | +0.171 | +0.178 | +0.158 | +0.153 | +0.168 | +0.176 | +0.170 |
| pencil | del_hub@50% | +0.197 | +0.201 | +0.162 | +0.151 | +0.167 | +0.203 | +0.168 |
| pencil | ins_hub@50% | +0.119 | +0.090 | +0.112 | +0.075 | +0.049 | +0.084 | +0.039 |

## Is the gate signal informative?

`Δ` is computed on the graph being scored (deployment-time signal). `AUC(Δ → drop)` asks whether the
magnitude ranks the pairs that lose score under deletion above the ones that do not; 0.5 is chance.

| model | probe | sign gate fires | top 50 % fires | `CN = 0` share | Δ > 0 (own feats) | Δ > 0 (blank feats) | corr(own, blank) | AUC(Δ → drop) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seal | del@50% | 99.8% | 50.0% | 92.8% | 99.8% | 100.0% | 0.986 | 0.528 |
| seal | del_hub@50% | 99.9% | 50.0% | 94.1% | 99.9% | 100.0% | 0.986 | 0.531 |
| seal | ins_hub@50% | 99.6% | 50.0% | 74.7% | 99.6% | 99.8% | 0.992 | 0.527 |
| pencil | del@50% | 97.6% | 50.0% | 92.8% | 97.6% | 98.1% | 0.964 | 0.511 |
| pencil | del_hub@50% | 97.5% | 50.0% | 94.1% | 97.5% | 98.2% | 0.960 | 0.523 |
| pencil | ins_hub@50% | 94.0% | 50.0% | 74.7% | 94.0% | 95.8% | 0.979 | 0.590 |
