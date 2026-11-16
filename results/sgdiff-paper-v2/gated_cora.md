# Evidence-gated routing repair: cora

Two sub-graph models are trained (k = 1 and k = 2); the router keeps the k2 score
for pairs with `CN <= 0` on the graph being scored. Damage = clean − probe.

| model | probe | k = 1 | k = 2 (global) | routed (`CN ≤ 0`) | routed − k1 | 95 % CI (paired) |
|---|---|---:|---:|---:|---:|---|
| seal | clean | 0.872 | 0.906 | 0.908 | +0.035 | — |
| seal | del@50% | +0.212 | +0.235 | +0.235 | -0.023 | [-0.045, -0.002] |
| seal | del_hub@50% | +0.229 | +0.262 | +0.263 | -0.034 | [-0.058, -0.010] |
| seal | ins_hub@50% | +0.044 | +0.088 | +0.082 | -0.038 | [-0.056, -0.019] |
| pencil | clean | 0.883 | 0.897 | 0.887 | +0.004 | — |
| pencil | del@50% | +0.170 | +0.176 | +0.171 | -0.002 | [-0.032, +0.028] |
| pencil | del_hub@50% | +0.168 | +0.203 | +0.197 | -0.029 | [-0.060, +0.003] |
| pencil | ins_hub@50% | +0.039 | +0.084 | +0.119 | -0.080 | [-0.105, -0.056] |

Positive `routed − k1` on a damage row = the router shrinks the damage; on the clean row it is the
ceiling the router buys (or costs). The interval is a paired bootstrap over the scored pairs for the
*damage difference* (clean and probe resampled jointly); it covers pair sampling, not training seeds.

## Routing frontier: is there a threshold that works?

| model | probe | route<=0 | route<=1 | route<=2 | always k2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|
| seal | del@50% | +0.235 | +0.235 | +0.235 | +0.235 | +0.212 |
| seal | del_hub@50% | +0.263 | +0.263 | +0.262 | +0.262 | +0.229 |
| seal | ins_hub@50% | +0.082 | +0.086 | +0.087 | +0.088 | +0.044 |
| pencil | del@50% | +0.171 | +0.174 | +0.175 | +0.176 | +0.170 |
| pencil | del_hub@50% | +0.197 | +0.201 | +0.202 | +0.203 | +0.168 |
| pencil | ins_hub@50% | +0.119 | +0.097 | +0.089 | +0.084 | +0.039 |

Share of scored pairs the gate fires on (higher = more routing):

| probe | `CN ≤ 0` | `CN ≤ 1` | `CN ≤ 2` |
|---|---:|---:|---:|
| clean | 76.4% | 92.3% | 97.7% |
| del@50% | 92.8% | 99.2% | 99.9% |
| del_hub@50% | 94.1% | 99.2% | 99.9% |
| ins_hub@50% | 74.7% | 91.6% | 97.4% |

## Can a CN-completion repair reach the low-evidence pairs?

Deleted edges are rebuilt with the CN prior of §7.5 (same count as removed); a `CN = 0` pair
benefits only if the completion creates a shared neighbour for *that pair*.

| probe | `CN = 0` share | recovered by completion | edges added |
|---|---:|---:|---:|
| del@50% | 92.8% | 4.1% | 2233 |
| del_hub@50% | 94.1% | 4.4% | 2230 |
