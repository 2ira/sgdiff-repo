# Evidence-gated routing repair: cora

Two sub-graph models are trained (k = 1 and k = 2); the router keeps the k2 score
for pairs with `CN <= 0` on the graph being scored. Damage = clean − probe.

| model | probe | k = 1 | k = 2 (global) | routed (`CN ≤ 0`) | routed − k1 | 95 % CI (paired) |
|---|---|---:|---:|---:|---:|---|
| seal | clean | 0.781 | 0.856 | 0.844 | +0.062 | — |
| seal | del@50% | +0.137 | +0.208 | +0.204 | -0.067 | [-0.097, -0.037] |
| seal | del_hub@50% | +0.149 | +0.216 | +0.213 | -0.064 | [-0.096, -0.032] |
| seal | ins_hub@50% | +0.022 | +0.054 | +0.073 | -0.051 | [-0.074, -0.028] |
| pencil | clean | 0.795 | 0.860 | 0.852 | +0.056 | — |
| pencil | del@50% | +0.125 | +0.173 | +0.169 | -0.044 | [-0.078, -0.012] |
| pencil | del_hub@50% | +0.145 | +0.198 | +0.195 | -0.049 | [-0.085, -0.013] |
| pencil | ins_hub@50% | +0.008 | +0.047 | +0.054 | -0.047 | [-0.068, -0.026] |

Positive `routed − k1` on a damage row = the router shrinks the damage; on the clean row it is the
ceiling the router buys (or costs). The interval is a paired bootstrap over the scored pairs for the
*damage difference* (clean and probe resampled jointly); it covers pair sampling, not training seeds.

## Routing frontier: is there a threshold that works?

| model | probe | route<=0 | route<=1 | route<=2 | always k2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|
| seal | del@50% | +0.204 | +0.206 | +0.207 | +0.208 | +0.137 |
| seal | del_hub@50% | +0.213 | +0.214 | +0.216 | +0.216 | +0.149 |
| seal | ins_hub@50% | +0.073 | +0.062 | +0.056 | +0.054 | +0.022 |
| pencil | del@50% | +0.169 | +0.170 | +0.172 | +0.173 | +0.125 |
| pencil | del_hub@50% | +0.195 | +0.196 | +0.198 | +0.198 | +0.145 |
| pencil | ins_hub@50% | +0.054 | +0.051 | +0.049 | +0.047 | +0.008 |

Share of scored pairs the gate fires on (higher = more routing):

| probe | `CN ≤ 0` | `CN ≤ 1` | `CN ≤ 2` |
|---|---:|---:|---:|
| clean | 77.0% | 92.5% | 98.0% |
| del@50% | 92.8% | 99.3% | 99.9% |
| del_hub@50% | 94.2% | 99.3% | 100.0% |
| ins_hub@50% | 74.9% | 92.0% | 97.6% |

## Can a CN-completion repair reach the low-evidence pairs?

Deleted edges are rebuilt with the CN prior of §7.5 (same count as removed); a `CN = 0` pair
benefits only if the completion creates a shared neighbour for *that pair*.

| probe | `CN = 0` share | recovered by completion | edges added |
|---|---:|---:|---:|
| del@50% | 92.8% | 3.7% | 2235 |
| del_hub@50% | 94.2% | 4.2% | 2240 |
