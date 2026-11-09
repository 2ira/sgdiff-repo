# Method: paper → code map

Reference: *Sub-graph Based Diffusion Model for Link Prediction* (Li et al.,
LoG 2024). This document maps every equation of the paper onto the code that
implements it.

## Notation

| Paper | Code |
|-------|------|
| graph `G = (V, E)` | `sgdiff.data.GraphData` |
| enclosing sub-graph `S^k_uv = (A_uv, X_uv)` | `sgdiff.data.Subgraph` |
| link label `y_uv ∈ {0, 1}` | `SubgraphBatch.y` |
| DRNL node labels | `Subgraph.labels` (`sgdiff.data.subgraph.drnl_label`) |
| class-conditional structure model `φ_θ` | `sgdiff.models.StructureDiffusionModel` |
| class-conditional feature model `ε_θ` | `sgdiff.models.FeatureDiffusionModel` |
| fusion parameters `{η₁, η₂, δ}` | `sgdiff.models.LikelihoodFusion` |

## 1. Sub-graph construction (Sec. 2.1, 3.1)

```python
neighbors = graph.neighbors()                       # adjacency list
sub = extract_enclosing_subgraph(neighbors, u, v,
                                 hops=1,            # Table 4: k = 1
                                 max_nodes_per_hop=None,  # Table 4: ns (per dataset)
                                 max_label=16,
                                 remove_target=True)
```

* `hops` = `k` in `S^k_uv`; the node set is the union of the `k`-hop
  neighbourhoods of `u` and `v`. The paper uses `k = 1` for every dataset, with
  per-dataset neighbour sampling `ns` (Cora none, Citeseer/Pubmed 20, Router 10,
  USAir 40, NS 5).
* The edge `(u, v)` is removed **from the BFS itself** (not only from the
  induced adjacency), so neither the induced matrix nor the DRNL distances can
  reveal the answer. This is required for a leakage-free evaluation; an
  implementation that only zeroes `A[u, v]` after the BFS leaks the label
  through the labels of `u` and `v` (we hit exactly this bug while building the
  reproduction — see `docs/reproduction.md`).
* DRNL (*double-radius node labeling*):

  ```
  d = du + dv
  label = 1 + min(du, dv) + (d//2) * ((d//2) + (d%2) - 1)
  ```

  clipped to `max_label`. Nodes unreachable within `k` hops from one endpoint
  fall back to distance `k + 1`.
* Node attributes are z-scored per dataset (`SubgraphExtractor._standardize`).
  For datasets without attributes (`USAir`, `NS`, `Router`) the feature-diffusion
  stream operates on the DRNL one-hot encoding instead
  (`feature_mode="auto"`).

## 2. Structure diffusion `p(A | y)` (Sec. 3.3)

The sub-graph adjacency (upper triangle, binary) is diffused with a uniform-state
D3PM, `K = 2`:

```
Q_t      = (1 - β_t) I + (β_t / 2) 11ᵀ
Q̄_t      = ᾱ_t I + (1 - ᾱ_t)/2 11ᵀ,   ᾱ_t = Π_{s≤t}(1 - β_s)
q(A^t|A^0):  A^t = A^0 with probability ᾱ_t, otherwise flipped
```

`StructureDiffusionModel.q_sample` samples `A^t`;
`StructureDiffusionModel.posterior_state1` implements the closed form

```
q(A^{t-1} = 1 | A^t, A^0) ∝ Σ_{a ∈ {0,1}} q(A^{t-1}=1 | A^0) q(A^t | A^{t-1}=1)
```

and `compute_kl` evaluates the per-step term of Algorithm 1:

```
KL[ q(A^{t-1} | A^t, A^0) ‖ p_θ(A^{t-1} | A^t, y) ]
```

summed over the upper-triangular entries of `A` (normalised by the number of
entries when `structure.normalize_score=true`).

The denoiser follows Sec. A.3: a **transformer** over node tokens (DRNL label
embedding), edge tokens (`[A^t_ij, onehot(|l_i - l_j|)]`) and a global token
(`time_embedding(t) + class_embedding(y)`), with 8 heads and 2 layers
(`sgdiff/models/transformer_denoiser.py`). Edge tokens bias the node attention.
Training minimises the VLB **plus the sum of node and edge cross-entropy**
(`structure.aux_ce_weight`). `structure.arch="edge_gnn"` keeps a lighter
message-passing variant for ablations.

> **Schedule matters.** With `β_end = 0.5` and 20 steps, `ᾱ_20 ≈ 0.002`: most
> timesteps are pure noise and averaging the KL over them buries the class
> signal. The default schedule (`1e-4 → 0.2`) keeps `ᾱ_t` in `[1.0, 0.12]`.

## 3. Feature diffusion `p(X | A, y)` (Sec. 3.4)

For datasets without attributes (USAir, NS, Router) and for cross-dataset
transfer the paper uses **no** feature stream (`h_g = N/A` in Table 4); set
`feature.enabled: false` and the fusion freezes `eta_1 = 0`.

Standard DDPM on the node-attribute matrix:

```
X^t = √ᾱ_t X^0 + √(1 - ᾱ_t) ε,        ε ~ N(0, I)
score = -mean_t ‖ε - ε_θ(X^t, A, y)‖²
```

`FeatureDiffusionModel.compute_mse` returns the `FeatureScore` of Algorithm 1;
`log_likelihood` averages it over the sampled timesteps and negates it.

## 4. Fusion and decision rule (Eqs. 12-13, Sec. 3.5)

```python
log P(A, X | y) = η₁ · log P̂(X | A, y) + η₂ · log P̂(A | y) + δ
P(y | A, X)     = softmax_y( log P(A, X | y) + log π_y )
```

`LikelihoodFusion` stores `η₁`, `η₂`, `δ` as parameters (initialised to `1, 1, 0`)
and a uniform `log π`. Both likelihood proxies are defined as **higher = better**
(negated losses), so a positive `η` means "the model with the smaller loss wins".

## 5. Training

The paper's Algorithm 1 describes inference, not optimisation. This
reproduction uses a two-stage schedule (`train.pretrain_epochs`):

1. **Class-conditional MLE** — both diffusion models are trained on their own
   class data (`StructureDiffusionModel.loss`,
   `FeatureDiffusionModel.loss`), i.e. maximum likelihood of `p_θ(A|y)` and
   `p_θ(X|A,y)`.
2. **Joint fusion** — the fused logits of Eq. (12) are optimised with
   cross-entropy on the link label, with a small `0.1` weight on the diffusion
   losses to keep the density models calibrated
   (`SGDiff.train_logits`, `Trainer._train_epoch`).

Stage 2 is what makes the scores discriminative: a density model trained only
with MLE on 2k sub-graphs produces nearly identical `p(A|y=0)` and `p(A|y=1)`
(AUC ≈ 0.5 in our ablations), while the cross-entropy term sharpens exactly the
class-conditional difference the decision rule needs.
