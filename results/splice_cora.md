# Counterfactual splice gate: cora

For every pair the k = 1 model is scored twice: once on its enclosing sub-graph, once with a
synthetic shared neighbour `w` spliced in (`u-w`, `v-w`, DRNL label 2). The gate routes to the
k = 2 model for the pairs whose score the splice raises most. Damage = clean − probe.

| model | probe | k = 1 | k = 2 (global) | `CN = 0` gate | splice gate (top 50 %) | splice − k1 | 95 % CI (paired) |
|---|---|---:|---:|---:|---:|---:|---|
| seal | clean | 0.781 | 0.856 | 0.844 | 0.816 | +0.035 | — |
| seal | del@50% | +0.137 | +0.208 | +0.204 | +0.175 | -0.038 | [-0.065, -0.012] |
| seal | del_hub@50% | +0.149 | +0.216 | +0.213 | +0.173 | -0.025 | [-0.051, +0.002] |
| seal | ins_hub@50% | +0.022 | +0.054 | +0.073 | +0.054 | -0.032 | [-0.054, -0.011] |
| pencil | clean | 0.795 | 0.860 | 0.852 | 0.827 | +0.031 | — |
| pencil | del@50% | +0.125 | +0.173 | +0.169 | +0.131 | -0.006 | [-0.029, +0.019] |
| pencil | del_hub@50% | +0.145 | +0.198 | +0.195 | +0.160 | -0.014 | [-0.040, +0.012] |
| pencil | ins_hub@50% | +0.008 | +0.047 | +0.054 | +0.036 | -0.028 | [-0.046, -0.012] |

Positive `splice − k1` on a damage row = the splice gate shrinks the damage; on the clean row it is
the ceiling it buys. The interval is a paired bootstrap over scored pairs for the damage difference;
it covers pair sampling, not training seeds.

## Gate frontier: sign is useless, magnitude is not

The splice raises the score for almost every pair, so the *sign* gate degenerates into "always k2".
The quantile gates route only the pairs with the largest rise; `CN = 0` is the §7.11 gate, shown for
reference.

| model | probe | `CN = 0` | sign(Δ) | top 50 % | top 30 % | top 10 % | always k2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seal | del@50% | +0.204 | +0.208 | +0.175 | +0.167 | +0.152 | +0.208 | +0.137 |
| seal | del_hub@50% | +0.213 | +0.216 | +0.173 | +0.163 | +0.159 | +0.216 | +0.149 |
| seal | ins_hub@50% | +0.073 | +0.055 | +0.054 | +0.041 | +0.024 | +0.054 | +0.022 |
| pencil | del@50% | +0.169 | +0.171 | +0.131 | +0.121 | +0.121 | +0.173 | +0.125 |
| pencil | del_hub@50% | +0.195 | +0.197 | +0.160 | +0.151 | +0.143 | +0.198 | +0.145 |
| pencil | ins_hub@50% | +0.054 | +0.047 | +0.036 | +0.028 | +0.014 | +0.047 | +0.008 |

## Is the gate signal informative?

`Δ` is computed on the graph being scored (deployment-time signal). `AUC(Δ → drop)` asks whether the
magnitude ranks the pairs that lose score under deletion above the ones that do not; 0.5 is chance.

| model | probe | sign gate fires | top 50 % fires | `CN = 0` share | Δ > 0 (own feats) | Δ > 0 (blank feats) | corr(own, blank) | AUC(Δ → drop) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seal | del@50% | 99.5% | 50.0% | 92.8% | 99.5% | 100.0% | 0.937 | 0.519 |
| seal | del_hub@50% | 99.6% | 50.0% | 94.2% | 99.6% | 100.0% | 0.924 | 0.507 |
| seal | ins_hub@50% | 99.4% | 50.0% | 74.9% | 99.4% | 100.0% | 0.946 | 0.445 |
| pencil | del@50% | 98.7% | 50.0% | 92.8% | 98.7% | 99.8% | 0.938 | 0.531 |
| pencil | del_hub@50% | 98.9% | 50.0% | 94.2% | 98.9% | 99.9% | 0.937 | 0.531 |
| pencil | ins_hub@50% | 96.9% | 50.0% | 74.9% | 96.9% | 99.3% | 0.964 | 0.507 |
