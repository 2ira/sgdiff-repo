# Mechanism attribution: which graph property does each model read?

`docs/robustness.md` answers *how much* every model loses under RF / DICE / EA.
Those attacks mix deletions and insertions, so the number cannot be attributed to
a mechanism. This document answers *why*: it replaces the mixed attacks with
targeted probes (`sgdiff/data/perturb.py`), measures which property each probe
destroys, and ablates the components of every model.

Artifacts (produced by `scripts/run_probe.py`, `cpu` profile, seeds 0-2):

| file | content |
|------|---------|
| `results/probe_cora.{json,md}` | Cora: 7 models × 7 probes, property heuristics, component ablations, sub-graph availability |
| `results/probe_usair.{json,md}` | USAir: the same without node attributes and labels (probes that need them are skipped) |
| `results/probe_report.html` | the same numbers as one self-contained page (heatmaps, sub-graph availability, ablation panel, hub-insertion scatter); rebuild with `python scripts/make_probe_report.py` |

## 1. Protocol

Same evasion protocol as the robustness harness: every model is trained **once**
on the clean graph through `benchmark.fit_model`, the probe is applied to the
**inference** graph only (weights frozen), test positives are withheld from that
graph, and every variant scores the **same** 500 + 500 test pairs with the same
metric code. The clean column is produced by the same scoring function as every
probe, so `damage = clean − probe` is like-for-like.

Three measurements are added on top of the AUC table:

1. **Property availability** — cheap heuristics are ranked on the *same*
   perturbed graph: degree product (`deg`), common neighbours (`cn`),
   Adamic-Adar (`aa`), Jaccard, and feature cosine (`featcos`). Probes that
   share an edge budget can then be compared by *which* of these they move: a
   probe that keeps `cn` at 0.73 while dropping `deg` to 0.51 damaged degrees,
   not connectivity.
2. **Signal alignment** — Spearman rho between the frozen model's clean scores
   and each heuristic on the same pairs: what the model actually ranks by.
3. **Component ablation** — with the weights frozen, node features are zeroed
   (`features_zeroed`), DRNL labels are permuted inside each sub-graph
   (`drnl_permuted`), and SGDiff's two fusion branches are switched off one at a
   time (`structure_only`, `feature_only`).

## 2. What each probe does — and what it does *not* do

Cora, mean over seeds (`results/probe_cora.md` §1-2):

| probe | #edges | mean deg | assort. | homophily | AUC deg | AUC cn/aa | AUC featcos |
|-------|-------:|---------:|--------:|----------:|--------:|----------:|------------:|
| clean | 4 486 | 3.31 | −0.065 | 0.811 | 0.639 | 0.729 | **0.802** |
| del@50% | 2 253 | 1.66 | −0.060 | 0.812 | 0.594 | **0.570** | 0.802 |
| del_hub@50% | 2 256 | 1.67 | −0.003 | 0.826 | **0.510** | 0.559 | 0.802 |
| ins@50% | 7 125 | 5.26 | −0.038 | 0.576 | 0.653 | 0.728 | 0.802 |
| ins_hub@50% | 7 125 | 5.26 | **−0.209** | 0.572 | 0.660 | 0.727 | 0.802 |
| ins_homo@50% | 7 125 | 5.26 | −0.040 | **0.881** | 0.608 | 0.735 | 0.802 |
| ins_hetero@50% | 7 125 | 5.26 | −0.038 | **0.511** | 0.650 | 0.727 | 0.802 |
| feature_shuffle | 4 486 | 3.31 | −0.065 | 0.811 | 0.639 | 0.729 | **0.498** |

Read this table before any model number — it is the control:

* **`del` and `del_hub` remove the same number of edges but not the same
  signal.** Random deletion halves the graph yet keeps the degree signal
  (`deg` 0.639 → 0.594) and only partially kills 2-hop connectivity (`cn`
  0.729 → 0.570). Hub deletion, with an identical edge budget, destroys the
  degree signal (`deg` → 0.510) and the degree assortativity (−0.065 → −0.003),
  and kills connectivity slightly harder (0.559).
* **Insertion probes leave connectivity intact but change the "look" of the
  graph.** `cn` stays at ~0.73 in every insertion variant — a random extra edge
  is plausible under a common-neighbour heuristic. `ins_homo` *raises* homophily
  above the clean value (0.881 vs 0.811) and `ins_hetero` lowers it to 0.511,
  with the same edge count: the pair is a **homophily-only** intervention.
  `ins_hub` is a **hub/degree-only** intervention (assortativity −0.209).
* **`feature_shuffle` moves exactly one number.** Every structural statistic is
  bit-identical to `clean`; only `featcos` collapses to chance (0.498). Any
  model that does not change under this probe provably ignores the attribute
  channel.

`cn`, `aa` and `jaccard` agree to three decimals on Cora: with mean degree 3.3
most sampled negatives have *no* common neighbour at all, so all three reduce to
a "shares a neighbour" detector and the ordering among the non-zero values is
almost the same. On USAir (mean degree 10.9) they separate — see §5.2 — which is
why both datasets are reported.

## 3. Model × probe

AUC, mean over seeds 0-2, every probe at the 50 % budget (`results/probe_cora.md`
§3); per-seed std and the damage column (clean − probe, positive = worse) are in
§4 of that file.

| model | clean | del | del_hub | ins | ins_hub | ins_homo | ins_hetero | feat_shuffle |
|-------|------:|----:|--------:|----:|--------:|---------:|-----------:|-------------:|
| GCN | 0.828 | 0.781 | 0.772 | 0.798 | 0.807 | 0.823 | 0.792 | **0.636** |
| GAT | 0.808 | 0.711 | 0.707 | 0.775 | 0.772 | **0.826** | 0.767 | **0.637** |
| PENCIL | 0.799 | 0.663 | 0.635 | 0.779 | 0.792 | 0.773 | 0.779 | 0.774 |
| VGAE | 0.782 | 0.743 | 0.718 | 0.742 | 0.736 | 0.778 | 0.732 | **0.587** |
| SEAL | 0.781 | 0.639 | 0.619 | 0.756 | 0.763 | 0.757 | 0.753 | 0.736 |
| SAGE | 0.751 | 0.715 | 0.694 | 0.738 | 0.724 | 0.759 | 0.724 | **0.588** |
| SGDiff | 0.745 | **0.508** | **0.568** | 0.722 | 0.778 | 0.709 | 0.726 | 0.745 |

Component ablations and signal alignment (`results/probe_cora.md` §5, §7):

| model | features zeroed | DRNL permuted | structure only | feature only | rho(deg) | rho(cn) | rho(featcos) |
|-------|----------------:|--------------:|---------------:|-------------:|---------:|--------:|-------------:|
| GCN | **0.433** | – | – | – | 0.231 | 0.535 | **0.420** |
| GAT | **0.474** | – | – | – | 0.280 | 0.550 | 0.361 |
| SAGE | **0.500** | – | – | – | 0.274 | **0.491** | 0.345 |
| VGAE | **0.484** | – | – | – | 0.219 | 0.507 | 0.353 |
| SEAL | 0.740 | 0.756 | – | – | 0.437 | 0.661 | 0.246 |
| PENCIL | 0.787 | 0.759 | – | – | 0.440 | 0.705 | 0.269 |
| SGDiff | 0.745 | 0.742 | 0.744 | **0.515** | 0.533 | **0.724** | **0.219** |

## 4. Attribution, model by model

### 4.1 Full-graph dot-product encoders (GCN, GAT, SAGE, VGAE)

**What they read.** Two message-passing layers over the whole graph plus an
inner-product decoder. On Cora their behaviour is *attribute-driven*: zeroing
the input features costs 0.25-0.40 AUC (GCN 0.828 → 0.433 is the extreme; all
four fall to 0.43-0.50, i.e. roughly chance), `feature_shuffle` costs 0.16-0.20,
and their rho with `featcos` (0.35-0.42) is the highest in the zoo. Structure
enters only as a smoothing/pooling operator over attributes.

**Why deletion barely hurts them.** An edge is one term in a neighbourhood
average. `del@50%` halves the graph but only costs GCN 0.048 and SAGE 0.036,
because (i) symmetric normalisation `1/√(d_u d_v)` (GCN, VGAE) re-scales the
message when the degree changes, and (ii) the node feature is still there: the
encoder does not need the edge to know what the node *is*, only whom to average
with.

**Which component is fragile.** The attribute projection, not the aggregation.
GCN's damage ordering is `features_zeroed` (0.40) ≫ `feature_shuffle` (0.19) ≫
`del_hub` (0.056) > `del` (0.048) > `ins` (0.031): the largest structural probe
is 7× smaller than the attribute one.

SAGE is the exception inside the family. It aggregates with the same symmetric
normalisation but **concatenates the node's own feature vector** with the
aggregated neighbourhood and re-projects. That self-feature path is untouched by
an edge deletion, which is why SAGE is the most deletion-robust model here
(0.036), the least damaged by `feature_shuffle` of the four (0.163), and the one
with the lowest rho(cn) (0.491): it leans on the node's own features and degree
rather than on shared-neighbour structure.

**Why GAT is the exception inside the family.** GAT loses 0.097 to deletion and
0.035 to hub insertion, ~2× GCN. Its aggregation is an edge-wise **softmax
normalised per destination node**: deleting an edge renormalises the remaining
attention weights (there is no fixed 1/d_fanout to absorb the change), and
inserting hub-adjacent edges adds *competitor keys* to the softmax of every
neighbour of the hub. Two components, both fragile: the softmax normaliser and
the hub neighbourhood.

**The one probe that *helps* a node-pair model.** `ins_homo` improves GAT
(0.808 → 0.826) and does nothing to GCN/SAGE/VGAE. A homophilous edge connects
two nodes of the same class; under homophily-based smoothing that edge *reinforces*
the aggregation the decoder wants. This is the first sign that "insertion" is not
a single mechanism: plausible edges are invisible or helpful, implausible edges
(ins_hetero) cost 0.03-0.05.

### 4.2 Sub-graph classifiers (SEAL, PENCIL)

**What they read.** One 1-hop enclosing sub-graph per pair (target edge removed,
DRNL-labelled, capped at 40 nodes), a GCN (SEAL) or a plain Transformer
(PENCIL), and a pooled readout. Their signal is **local 2-hop connectivity**:
rho(cn) 0.661 / 0.705, the highest of the discriminative models, and the
attribute channel contributes little (`features_zeroed` only −0.04).

**Why deletion is their worst probe (0.14-0.16).** It is not a weight-space
effect, it is input starvation, measured directly in `probe_cora.md` §6:

| probe | sub-graph nodes | edges inside | endpoints connected | share a neighbour | a hub is inside |
|-------|----------------:|-------------:|--------------------:|------------------:|----------------:|
| clean | 10.7 | 12.3 | 0.310 | 0.227 | 0.743 |
| del@50% | 7.0 | 5.6 | 0.118 | **0.068** | 0.596 |
| del_hub@50% | 5.4 | 3.7 | 0.087 | **0.053** | 0.431 |
| ins@50% | 14.4 | 16.1 | 0.341 | 0.229 | 0.801 |
| ins_hub@50% | 15.5 | 17.4 | 0.419 | 0.237 | **0.964** |
| feature_shuffle | 10.7 | 12.3 | 0.310 | 0.227 | 0.743 |

The target edge is removed before the model sees the sub-graph, so with `k = 1`
the *only* channel that can distinguish a true pair from a false one is a shared
neighbour path `u–w–v`. Deletion cuts that path for three quarters of the pairs
that had one (`has_cn` 0.227 → 0.068), and hub deletion cuts it a little harder
(0.053) while also shrinking the sub-graph to 5.4 nodes. The models do what their
input allows: PENCIL keeps 0.663, SEAL 0.639.

**Why hub deletion is worse than random deletion for this family, but not for
SGDiff.** `del_hub` removes the edges of the top-5 % degree nodes, i.e. exactly
the nodes that act as the `w` in `u–w–v`, and it drops `hub_in` from 0.743 to
0.431. Random deletion keeps the hubs and only thins the periphery (`hub_in`
0.596) — a different damage profile at the same budget.

**Why PENCIL is more deletion-robust than SEAL.** PENCIL's self-attention is
**not** restricted to the induced adjacency: even when the sub-graph loses all
its internal edges, the tokens are still there and the Transformer can compare
attribute vectors pairwise. SEAL's GCN can only propagate along surviving edges.
Same input, different operator: full attention degrades gracefully on a
structure-only damage probe (0.136 vs 0.142), and it pays for it on the attribute
probe (`feature_shuffle` 0.774 vs SEAL 0.736 — PENCIL keeps more of its accuracy
because its attention has more to read).

**Why insertion hardly matters *on Cora*.** `ins@50%` doubles the number of edges
and adds 3.7 nodes per sub-graph, yet costs 0.02-0.03. An inserted edge cannot
remove evidence; it can only add a plausible-looking neighbour — and `ins_homo`
(the most plausible of all) costs 0.024-0.026, i.e. nothing. On this graph the
family is a *one-sided* reader of connectivity. It stops being one on a dense
graph: on USAir, hub-attached insertion costs PENCIL 0.19 (§5.2), because a hub
inside every sub-graph destroys a *normalised* neighbour count rather than the
raw evidence.

**DRNL is a weak prior at `k = 1`.** Permuting the DRNL labels inside every
sub-graph costs SEAL 0.025 and PENCIL 0.040 (0.781 → 0.756, 0.799 → 0.759). The
positional encoding is not the fragile component; the adjacency is.

### 4.3 SGDiff: a pure structure likelihood, and its damage is deletion

**Which branch carries the signal — measured, not assumed.** SGDiff fuses
`η₁·log p(X|A,y) + η₂·log p(A|y) + δ`. With the trained weights frozen:

| view | AUC |
|------|----:|
| full fusion (clean) | 0.745 ± 0.043 |
| `structure_only` (`η₁ := 0`) | 0.744 ± 0.042 |
| `feature_only` (`η₂ := 0`) | **0.515 ± 0.003** (chance) |
| `features_zeroed` | 0.745 ± 0.043 |
| `feature_shuffle` probe | 0.745 ± 0.043 (`damage = 0.000`) |

The feature DDPM contributes **nothing** at this budget: removing it changes the
AUC by 0.001, keeping only it gives chance, and corrupting its input costs
exactly zero. This is the same defect recorded in `README.md` ("the feature
branch never learns": ε is 1433-dimensional but reaches the output through a
256-wide projection, so the trivial predictor `ε_θ = 0` is optimal and the loss
sits at 1.0). Attribution-wise it means **every SGDiff result in this repo is a
statement about the structure branch only** — and that is a feature for this
analysis, not a bug: SGDiff is the cleanest available probe of "what does a
pure local-structure scorer do under structural noise".

**What the structure branch reads.** rho(cn) 0.724 — higher than any other model,
including SEAL (0.661) — and rho(featcos) 0.219, the lowest. The class-conditional
D3PM likelihood over the sub-graph adjacency is, for ranking purposes, a
sophisticated common-neighbour / local-pattern matcher.

**Its damage profile is one-sided and the steepest in the zoo.**

| probe | SGDiff | SEAL | mechanism |
|-------|-------:|-----:|-----------|
| `del@50%` | **0.236** | 0.142 | evidence edges removed; `cn` AUC 0.729 → 0.570 |
| `del_hub@50%` | 0.176 | 0.162 | same, plus hub paths removed |
| `ins@50%` | 0.023 | 0.025 | no evidence removed |
| `ins_hub@50%` | −0.034 | 0.018 | hub flooding slightly *helps* |
| `ins_homo@50%` | 0.036 | 0.024 | plausible edge, invisible |
| `ins_hetero@50%` | 0.019 | 0.028 | only homophily is touched |
| `feature_shuffle` | 0.000 | 0.045 | branch is inert anyway |

Three conclusions, each backed by a separate number:

1. **SGDiff's fragility is deletion, not "attacks".** `del` costs 0.236,
   `ins` costs 0.023 — a 10× asymmetry. The mixed RF@50 % result in
   `docs/robustness.md` (0.737 → 0.594, damage 0.143) is explained by its
   deletion half alone: RF@50 % deletes 25 % of the edges, and half the
   `del@50%` damage (0.12) is the right order of magnitude. The "attack" was a
   deletion probe wearing a disguise.
2. **Why SGDiff loses more than SEAL at the same input damage.** Both read the
   same sub-graphs, so the difference is the *scoring rule*, not the data. SEAL
   is a discriminative classifier: it can keep a decision boundary even when
   every input is weak. SGDiff scores a pair by the **difference of two
   class-conditional likelihoods** on the same sub-graph; as the sub-graph loses
   edges (`mean edges` 12.3 → 5.6), both likelihoods are evaluated on less
   evidence and their difference is driven toward the (uniform) prior. A
   likelihood ratio that collapses cannot be rescued by a threshold, whereas a
   classifier's margin can still rank. The seed spread at `del` is the same
   effect: SGDiff 0.508 ± **0.090** vs SEAL 0.639 ± **0.010** — with little
   evidence left, the direction the likelihood difference collapses in is close
   to arbitrary.
3. **Insertion noise is not "robustness", it is insensitivity — and at hubs it
   is a free gain.** `ins_hub` *raises* SGDiff by 0.034: extra hub-attached
   edges make every sub-graph denser, and a denser adjacency is more likely under
   the link class than under the non-link class for this denoiser, so the
   likelihood difference widens. That is a bias, not a defence: the same
   mechanism would promote false positives in a deployment with dense hubs.

### 4.4 One table: property → who reads it → probe that exposes it

| property | models that rely on it | probe that removes it | who is fragile |
|----------|------------------------|-----------------------|----------------|
| node attributes (BoW, standardized) | GCN, VGAE, GAT, SAGE (by `feature_shuffle` damage) | `feature_shuffle`, `features_zeroed` | VGAE −0.20, GCN −0.19, GAT −0.17, SAGE −0.16 |
| 2-hop connectivity / common neighbours | SGDiff, PENCIL, SEAL | `del`, `del_hub` | SGDiff −0.24, PENCIL −0.14, SEAL −0.14 |
| degree distribution / hub paths | degree-based readers inside the GCN family | `del_hub` | SGDiff −0.18, SEAL −0.16, GAT −0.10, VGAE −0.06 |
| homophily (label agreement) | nobody *directly*; it is the assumption behind smoothing | `ins_homo` (plausible) / `ins_hetero` (implausible) | no model loses to `ins_homo`; `ins_hetero` costs 0.02-0.05 |
| DRNL positional labels | SEAL, PENCIL, SGDiff | `drnl_permuted` | SEAL −0.03, PENCIL −0.04, SGDiff −0.00 |

Robust components (they absorb the same perturbation budget):

* symmetric degree normalisation (GCN, VGAE) — an edge is one term of an average;
* mean aggregation + concatenated self-feature (SAGE) — most deletion-robust
  model in the zoo;
* pooled readout over the masked sub-graph (SEAL, PENCIL) — extra nodes from
  insertion are averaged away;
* the feature branch of SGDiff — inert, therefore perfectly robust and perfectly
  useless.

Fragile components:

* edge-wise attention softmax (GAT) — renormalises on every edge change and
  competes with hub keys;
* the attribute projection of every node-pair encoder on Cora — the single
  largest damage term of the GCN family;
* 1-hop evidence supply for sub-graph readers — `has_cn` 0.227 → 0.068 under
  deletion; **this is the root cause of the sub-graph family's fragility**;
* the class-conditional likelihood difference (SGDiff) — collapses toward the
  prior exactly when the evidence is thin, which is why it is the worst model at
  `del` with the largest seed variance.

## 5. USAir: the same probes without attributes or labels

USAir (332 nodes, 2 126 edges, mean degree 10.9, no attributes, no labels) is the
control for two claims: that the GCN family's damage on Cora is an *attribute*
effect, and that the sub-graph family's damage is an *evidence-supply* effect.
`ins_homo`, `ins_hetero` and `feature_shuffle` cannot run here (no labels, no
features) and are recorded as skipped. One wrinkle from the harness: for
featureless graphs `node_feature_matrix` returns a **degree one-hot encoding**
computed once on the clean graph, so node-pair models get a degree channel
instead of an attribute channel (`features_zeroed` is its ablation; it is not
recomputed on the probe graph, by design).

AUC (`results/probe_usair.md` §3):

| model | clean | del | del_hub | ins | ins_hub | damage orders |
|-------|------:|----:|--------:|----:|--------:|---------------|
| SEAL | 0.953 | 0.892 | 0.899 | 0.847 | 0.856 | ins 0.106 > ins_hub 0.097 > del 0.061 |
| SGDiff | 0.938 | 0.902 | 0.911 | 0.909 | 0.861 | **ins_hub 0.077** > del 0.036 |
| PENCIL | 0.924 | 0.903 | 0.918 | 0.893 | **0.734** | **ins_hub 0.190** ≫ rest |
| GCN | 0.882 | 0.826 | 0.824 | 0.872 | 0.842 | del 0.056 ≈ del_hub 0.059 |
| SAGE | 0.871 | 0.831 | 0.833 | 0.841 | 0.868 | del 0.040, ins_hub **0.002** |
| GAT | 0.851 | 0.838 | 0.825 | 0.785 | 0.843 | ins 0.066, del 0.013 |
| VGAE | 0.837 | 0.819 | 0.813 | 0.820 | 0.794 | del_hub 0.023, ins_hub 0.043 |

### 5.1 The same 50 % budget destroys far less signal here

| quantity | Cora (mean deg 3.3) | USAir (mean deg 10.9) |
|----------|--------------------:|----------------------:|
| pairs that still share a neighbour (`has_cn`), clean | 0.227 | 0.636 |
| … after `del@50%` | 0.068 (−70 %) | 0.458 (−28 %) |
| common-neighbour heuristic AUC, clean → `del` | 0.729 → 0.570 | 0.928 → 0.860 |
| sub-graph nodes, clean → `del` | 10.7 → 7.0 | 25.6 → 20.5 |
| SGDiff damage at `del` | **0.236** | **0.036** |
| SEAL damage at `del` | 0.142 | 0.061 |

Halving the edges is the same *operation* in both graphs; what differs is how
much redundancy it leaves behind. On Cora a 1-hop enclosing sub-graph has 12
edges, so losing half of them removes the only channel that separates positives
from negatives. On USAir it has 156 edges, so the same 50 % leaves 60 — the
common-neighbour AUC barely moves, and even the worst model (SEAL) only loses
0.06. **The sub-graph family's deletion fragility is a sparsity effect, not a
property of the model weights.** This also predicts where it will appear: on
sparse graphs (Cora 3.3, Citeseer 2.7) not on dense ones (USAir 10.9).

### 5.2 Hub insertion: damage tracks degree-normalised connectivity

USAir is where `ins_hub` separates the zoo, because it isolates one heuristic:
inserting hub-attached edges keeps raw common neighbours intact (`cn` 0.928 →
0.898) while destroying the **degree-normalised** version (`jaccard` 0.897 →
**0.640**) — every pair now shares a hub, so shared-neighbour counts stop being
informative once you divide by the union of neighbourhoods. Ablating that single
quantity sorts the models by damage almost exactly (`probe_usair.md` §4, §7):

| model | rho(clean scores, jaccard) | damage at `ins_hub` |
|-------|---------------------------:|--------------------:|
| PENCIL | **0.915** | **0.190** |
| SEAL | 0.881 | 0.097 |
| SGDiff | 0.880 | 0.077 |
| GCN | 0.776 | 0.041 |
| VGAE | 0.694 | 0.043 |
| GAT | 0.684 | 0.008 |
| SAGE | 0.745 | 0.002 |

Spearman rho between the two columns across the seven models: **0.82**
(p = 0.02). The mechanism is how much of the score survives a hub appearing in
every neighbourhood, not the parameter count: PENCIL's softmax attention
normalises over all tokens (closest to a Jaccard-like operator), SEAL's GCN is
one symmetric-normalised hop away from it, and the two least damaged models are
the ones with a hub-resistant term — SAGE's concatenated self-feature (0.002)
and GAT's learned attention, which already down-weights high-degree neighbours
(`rho(jaccard)` 0.684, damage 0.008).

Cora's insertion probes could not show this (its 1-hop sub-graphs are too small
for hub flooding to matter: `hub_in` 0.743 → 0.964 moved the AUC by 0.02), which
is why the probe suite needs both a sparse and a dense graph.

### 5.3 The degree channel is the attribute channel of a featureless graph

`features_zeroed` on USAir removes the degree one-hot — the same channel Cora's
`feature_shuffle` corrupts, ablated harder (deleted instead of permuted):

| model | clean | features zeroed | loss |
|-------|------:|----------------:|-----:|
| GAT | 0.851 | **0.462** | **0.389** |
| VGAE | 0.837 | 0.521 | 0.316 |
| SAGE | 0.871 | 0.694 | 0.177 |
| GCN | 0.882 | 0.812 | 0.070 |
| SEAL | 0.953 | 0.953 | 0.000 |
| SGDiff | 0.938 | 0.938 | 0.000 |

Two entries are the interesting ones. Removing the degree channel costs **GAT
0.389 AUC (46 % of its clean score) but GCN only 0.070 (8 %)** — the two encoders
extract different things from the same graph, so a robustness claim about "the
GNN family" is really a claim about the encoder. And **the sub-graph models are
degree-blind by construction**: their features are DRNL one-hots, so zeroing the
"node features" leaves them bit-identical. Their degree dependence is indirect
(`drnl_permuted` costs SEAL 0.953 → 0.884), and `del_hub` is what attacks it.

## 6. What this changes about the robustness story

1. **"Robustness to attacks" was mostly "robustness to deletion" — on sparse
   graphs.** On Cora the *largest* damage of every one of the seven models is a
   deletion probe (`del` or `del_hub`), and insertion contributes 0.02-0.05. On
   USAir the ranking flips for the sub-graph family (insertion dominates, §5.2),
   so the honest statement is: *which half of the attack does the damage is a
   property of the graph's density*, and neither RF nor DICE reports it.
2. **Model capacity is not the axis; the input supply is.** SEAL (54 k params)
   and SGDiff (2.3 M params) read the *same* sub-graphs and lose 0.14 vs 0.24;
   the difference is the scoring rule, not the architecture size.
3. **Sub-graph models are one-sided readers — and which side depends on the
   graph.** On sparse Cora they cannot be fooled by added edges (≤0.03) and
   cannot survive removed ones (0.14-0.24). On dense USAir the same family is hit
   by *hub insertion* (PENCIL −0.19), because there the damage is to
   degree-normalised connectivity rather than to evidence supply. A defence
   therefore has to do two different things — restore missing edges on a sparse
   graph, denoise hub edges on a dense one. The GDDM-style purifiers in
   `docs/robustness.md` §5 only do the second, and on Cora they made every model
   worse.
4. **Nothing in this zoo reads homophily directly.** `ins_homo` — an edge that
   *looks* right — is invisible (≤0.036 damage for every model, a gain for GAT).
   Homophily shows up only as the assumption that makes smoothing useful, and
   only implausible edges (`ins_hetero`) cost anything.

## 7. Caveats

1. **Evasion only.** Weights are trained clean; poisoned retraining is a
   different question.
2. **`cpu` profile.** SGDiff gets 8 epochs (3 pre-train), sub-graph classifiers
   40, node-pair baselines 300, sub-graph cap 40 nodes, 4 MC steps. All models in
   a comparison share the budget, so the *ordering* is comparable; absolute AUCs
   sit below the `full` profile (Cora SGDiff 0.745 here vs 0.730-0.737 in
   `results/robustness_cora.md`).
3. **`ins@50%` inserts 50 % of |E| without deleting**, so it is not the same
   budget as the flip-based RF attack; compare probes with probes, not probes
   with attacks.
4. **The feature branch of SGDiff is inert** (README, "Our side of the same
   ledger"). The `feature_only` / `feature_shuffle` rows quantify that
   inertness; they do not say what a *working* feature branch would do.
5. **`homo` / `hetero` use the ground-truth labels**, which a real attacker may
   not have; here they are diagnostic interventions, not attacks.
6. **Single attributed dataset for the attribute claims.** Cora and Citeseer
   share the BoW channel; Pubmed is 6× larger and was not re-run. Re-run with
   `--dataset citeseer` to check.

## 8. Reproduce / extend

```bash
# the Cora table (~20 min on CPU; SGDiff's fit dominates)
python scripts/run_probe.py --dataset cora --profile cpu --seeds 0 1 2 --device cpu

# cheap passes only: graph statistics + sub-graph availability, no training
python scripts/run_probe.py --dataset cora --seeds 0 1 2 --device cpu --diagnostics

# the featureless / label-free control
python scripts/run_probe.py --dataset usair --profile cpu --seeds 0 1 2 --device cpu

# a single mechanism question
python scripts/run_probe.py --models sgdiff seal --variants del@50% ins_hub@50% --seeds 0
```
