# Mechanism attribution: which graph property does each model read?

`docs/robustness.md` answers *how much* every model loses under RF / DICE / EA.
Those attacks mix deletions and insertions, so the number cannot be attributed to
a mechanism. This document answers *why*: it replaces the mixed attacks with
targeted probes (`sgdiff/data/perturb.py`), measures which property each probe
destroys, and ablates the components of every model.

Artifacts (produced by `scripts/run_probe.py`, `cpu` profile; seeds 0-4 for Cora,
seeds 0-2 for Citeseer / USAir):

| file | content |
|------|---------|
| `results/probe_cora.{json,md}` | Cora: 7 models × 7 probes, property heuristics, component ablations, sub-graph availability |
| `results/probe_citeseer.{json,md}` | Citeseer: the same probe grid on a second attributed/labelled graph (stability check) |
| `results/probe_usair.{json,md}` | USAir: the same without node attributes and labels (probes that need them are skipped) |
| `results/deep_<dataset>.{json,md}` | Cora / Citeseer / USAir: layer-level linear probes + representation drift, propagation-operator substitution, triadic-vs-bridging deletion, and the repair experiments (`scripts/run_deep_probe.py`) |
| `results/redesign_<dataset>.{json,md}` | Cora / Citeseer / USAir: one-component-at-a-time retraining (depth, normalisation, self-loop, input augmentation, robust training) re-probed with the same suite (`scripts/run_redesign.py`) |
| `results/scores/<dataset>_s<seed>.npz` | per-pair scores for every (model, probe) plus per-pair structure (CN, null expectation, degrees, same-label) — `scripts/run_score_dump.py` |
| `results/operating_<dataset>.{json,md}` | operating-point metrics (AP, hit@100, TPR@1%FPR), class-wise score drops, rank shift, and clean-defined strata (§7.7) — `scripts/run_operating_point.py` |
| `results/adaptive_<dataset>.{json,md}` | greedy score-feedback deletion at a matched budget vs random / hub deletion (§7.8) — `scripts/run_adaptive_attack.py` |
| `results/calibration_<dataset>.{json,md}` | null-model calibration sweep, fitted on the clean graph (§7.9) — `scripts/run_evidence_calibration.py` |
| `results/significance_<dataset>.{json,md}` | paired bootstrap over pairs for every (model, probe) damage, 95 % intervals and one-sided p (§7.10) — `scripts/run_bootstrap_ci.py` |
| `results/gated_<dataset>.{json,md}` | evidence-routed k1/k2 sub-graph repair (§7.11) — `scripts/run_gated_repair.py` |
| `results/splice_<dataset>.{json,md}` | counterfactual splice gate: per-pair score rise from one synthetic shared neighbour, quantile frontier, zero-feature control (§7.12) — `scripts/run_splice_gate.py` |
| `results/probe_report.html` | all of the above as one self-contained page; section 00 additionally renders the reproduction layer (`reproduction.md` / `benchmark.md` / `ablation.md` / `robustness.md` numbers). Rebuild with `python scripts/make_probe_report.py` |

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

AUC, mean over seeds 0-4, every probe at the 50 % budget (`results/probe_cora.md`
§3); per-seed std and the damage column (clean − probe, positive = worse) are in
§4 of that file.

| model | clean | del | del_hub | ins | ins_hub | ins_homo | ins_hetero | feat_shuffle |
|-------|------:|----:|--------:|----:|--------:|---------:|-----------:|-------------:|
| GCN | 0.825 | 0.777 | 0.766 | 0.798 | 0.802 | 0.817 | 0.792 | **0.637** |
| GAT | 0.800 | 0.704 | 0.699 | 0.769 | 0.767 | **0.823** | 0.759 | **0.622** |
| PENCIL | 0.796 | 0.678 | 0.654 | 0.771 | 0.783 | 0.774 | 0.772 | 0.748 |
| VGAE | 0.776 | 0.722 | 0.718 | 0.742 | 0.740 | 0.782 | 0.734 | **0.597** |
| SEAL | 0.782 | 0.644 | 0.628 | 0.764 | 0.762 | 0.763 | 0.763 | 0.733 |
| SAGE | 0.760 | 0.713 | 0.702 | 0.732 | 0.729 | 0.751 | 0.726 | **0.581** |
| SGDiff | 0.729 | **0.493** | **0.565** | 0.701 | 0.779 | 0.692 | 0.704 | 0.729 |

Component ablations and signal alignment (`results/probe_cora.md` §5, §7):

| model | features zeroed | DRNL permuted | structure only | feature only | rho(deg) | rho(cn) | rho(featcos) |
|-------|----------------:|--------------:|---------------:|-------------:|---------:|--------:|-------------:|
| GCN | **0.366** | – | – | – | 0.238 | 0.536 | **0.409** |
| GAT | **0.464** | – | – | – | 0.271 | 0.532 | 0.375 |
| SAGE | **0.500** | – | – | – | 0.280 | **0.496** | 0.346 |
| VGAE | **0.496** | – | – | – | 0.212 | 0.493 | 0.368 |
| SEAL | 0.737 | 0.759 | – | – | 0.393 | 0.655 | 0.279 |
| PENCIL | 0.768 | 0.726 | – | – | 0.441 | 0.684 | 0.310 |
| SGDiff | 0.729 | 0.732 | 0.728 | **0.497** | 0.538 | **0.723** | **0.228** |

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

## 6. Stability: the same probes on three graphs

Cora and Citeseer are both attributed, labelled BoW citation graphs; USAir is a
featureless, label-free airport network with three times their mean degree. If a
conclusion is an artefact of one graph, these columns disagree.

ΔAUC (clean − probe), means over seeds (Cora 0-4, Citeseer / USAir 0-2):

| model | probe | Cora (deg 3.31) | Citeseer (deg 2.33) | USAir (deg 10.89) |
|-------|-------|----------------:|--------------------:|------------------:|
| GCN | `feature_shuffle` | 0.193 | **0.214** | – |
| VGAE | `feature_shuffle` | 0.195 | **0.209** | – |
| SEAL | `feature_shuffle` | 0.045 | **0.214** | – |
| PENCIL | `feature_shuffle` | 0.026 | **0.211** | – |
| SEAL | `del` | 0.142 | 0.074 | 0.061 |
| SGDiff | `del` | **0.236** | 0.137 | 0.036 |
| SGDiff | `del_hub` | 0.176 | **0.207** | 0.026 |
| SGDiff | `ins` | 0.023 | **0.108** | 0.028 |
| SGDiff | `ins_homo` | 0.036 | **0.101** | – |
| PENCIL | `ins_hub` | 0.008 | 0.020 | **0.190** |

Four of the earlier claims survive, three do not.

**Stable.** (i) The GCN family's attribute dependence is quantitatively the same
on both attributed graphs: `feature_shuffle` costs GCN 0.19 / 0.21, VGAE 0.20 /
0.21, GAT 0.17 / 0.14. (ii) SGDiff's feature branch is dead on *every* graph:
`feature_only` 0.515 / 0.499 / 0.500 (chance) and `structure_only` 0.744 / 0.757 /
0.938 (= clean AUC). (iii) `ins_hub` never hurts SGDiff (−0.034 / −0.015 /
+0.077 — the last only on the dense graph, see §5.2). (iv) Nothing in the zoo is
harmed by `ins_homo` on Cora *or* Citeseer.

**Not stable, and now corrected.**

1. *"The sub-graph family reads structure, not attributes"* is **only true on
   Cora**. On Citeseer the same SEAL/PENCIL lose 0.21 to `feature_shuffle`, as
   much as GCN does. The reason is visible in the property table: Citeseer's
   features are *more* informative than Cora's (`featcos` AUC 0.890 vs 0.802), so
   the encoders actually use them there.
2. *"Insertion contributes ~0.02-0.04"* is **Cora-specific**. On the sparsest
   graph (Citeseer, mean degree 2.33) *adding* 50 % of |E| costs SGDiff 0.108,
   and even a homophilous insertion costs 0.101. Deletion is no longer the
   dominant half of the attack there.
3. *"`features_zeroed` measures the attribute channel"* **understates it**.
   PENCIL on Citeseer loses 0.022 when its features are zeroed but 0.211 when
   they are shuffled: a zero input can be absorbed by the learned bias, whereas a
   *wrong* input actively misleads. Attribute corruption is worse than attribute
   removal, which is why `feature_shuffle` (not the zeroing ablation) is the
   probe that reveals the channel.

The unifying statement that does survive all three graphs: **damage tracks how
far a probe moves the model's input distribution, and which half of a
delete/insert attack does that depends on the graph.** Simple heuristics explain
the deletion side (`cn` AUC 0.729 → 0.570 on Cora, 0.667 → 0.556 on Citeseer,
0.928 → 0.860 on USAir, with damage shrinking in the same order for the sub-graph
family). They do *not* explain the insertion side on Citeseer, where every
heuristic stays flat (CN 0.667 → 0.667) while SGDiff still loses 0.108 — an
off-distribution effect of the class-conditional likelihood that no first-order
graph statistic captures.

## 7. Down to layers, operators and edge kinds

Everything in this section comes from `scripts/run_deep_probe.py`
(`results/deep_cora.{json,md}` seeds 0-1, `results/deep_citeseer.{json,md}` and
`results/deep_usair.{json,md}` seeds 0-1), `scripts/run_redesign.py` (§7.6,
seeds 0-7 on three graphs), and the per-pair analyses `scripts/run_score_dump.py`
→ `run_operating_point.py` / `run_evidence_calibration.py` / `run_adaptive_attack.py`
(§7.7-7.9, seeds 0-1). Weights are trained once per model and never updated in
§7.1-7.5; §7.6 is the retraining counterpart, §7.8 the attacker-side counterpart.

### 7.1 Which layer does the damage enter?

Two numbers per layer: the **drift** (1 − mean cosine between the layer
representation on the probe graph and on the clean graph) and a **linear probe**
trained on the *clean* layer and applied to the probe layer (how much link signal
is still linearly readable there).

| model | probe | L1 drift | L2 drift | L1 AUC | L2 AUC |
|-------|-------|---------:|---------:|-------:|-------:|
| GCN | `del` | 0.093 | 0.092 | 0.831 | **0.804** |
| GCN | `del_hub` | 0.070 | 0.076 | 0.821 | 0.796 |
| GCN | `ins_hub` | 0.037 | 0.049 | 0.839 | 0.831 |
| GCN | `feature_shuffle` | **0.998** | 0.962 | **0.604** | 0.667 |
| SAGE | `del` | 0.065 | 0.132 | 0.751 | 0.742 |
| SAGE | `feature_shuffle` | **0.994** | 0.951 | **0.551** | 0.587 |
| GAT | `del` | 0.000* | 0.322 | 0.762* | **0.693** |
| GAT | `del_hub` | 0.000* | 0.253 | 0.762* | 0.712 |
| GAT | `feature_shuffle` | **0.996** | 0.896 | **0.520** | 0.640 |

\* GAT's layer-1 hook sits on the linear projection *before* the attention
normalisation, so its L1 representation cannot move when only the adjacency
changes; L1 is informative only for the attribute probes there.

Three statements, all layer-specific:

* **The attribute damage happens before the first aggregation.** `feature_shuffle`
  moves layer 1 by a cosine distance of ≈1.0 and drops the readable signal from
  0.839 to 0.614 already at L1. The encoder is not "confused by propagation": it
  is fed an input whose per-node identity is gone. Layer 2 changes nothing
  further.
* **The structural damage enters at the second hop.** For `del`, GCN's L1 barely
  moves (drift 0.09) and its readable signal (0.839) sits within 0.01 of the
  other structural probes (`del_hub` 0.827, `ins_hub` 0.841) — i.e. L1 is
  effectively unchanged — while L2 drops to 0.809. A 50 % deletion costs almost
  nothing at 1 hop and 3 points at 2 hops — the signature of an evidence channel
  (`has_cn`), not of a weight-space perturbation.
* **GAT is the only model whose L2 is hit twice as hard** (0.754 → 0.688, −0.066
  vs GCN's −0.030) — the layer-level confirmation of the softmax-renormalisation
  mechanism from §4.1.

### 7.2 Which operator is the robustness carrier? (a hypothesis that dies)

Same trained weights, only the propagation operator is swapped during inference:

| model | operator | clean | `del` | `feature_shuffle` | `ins_hub` |
|-------|----------|------:|------:|------------------:|----------:|
| GCN | sym (as trained) | 0.827 | 0.781 | 0.629 | 0.804 |
| GCN | row-mean | 0.820 | 0.771 | 0.641 | 0.781 |
| GCN | unnormalised sum | 0.807 | **0.765** | **0.712** | 0.755 |
| GCN | no self-loops | 0.808 | **0.708** | 0.644 | 0.783 |
| SAGE | sym / noloop | 0.757 | 0.714 | 0.575 | 0.728 |
| SAGE | sum | 0.768 | 0.718 | 0.684 | 0.715 |
| VGAE | sym | 0.784 | 0.739 | 0.589 | 0.740 |
| VGAE | no self-loops | 0.771 | **0.693** | 0.589 | 0.721 |

The mechanism I assumed in §4.1 — "symmetric normalisation `1/√(d_i d_j)`
absorbs the perturbation" — is **wrong**. Under `del`, symmetric (0.790),
row-mean (0.777) and unnormalised sum (0.773) are within 0.02 of each other; the
normalisation is not what protects the model. What *does* protect it on Cora is
the **self-loop**: removing it costs GCN 0.072 (−9 %) and VGAE 0.052 on the same
weights, because the self-loop is the identity path that keeps a node's own
signal when its neighbourhood is damaged. SAGE is the control: it is trained
without self-loops, and its `noloop` and `sym` rows are identical by construction.

The same table contains the reverse effect: under `feature_shuffle`, dropping the
normalisation *helps* (GCN 0.628 → 0.703, SAGE 0.570 → 0.671, VGAE 0.569 → 0.662).
An unnormalised sum makes high-degree structure dominate the message, which
partially substitutes for the destroyed attribute signal. Robustness is therefore
operator- and probe-dependent, not a property of "GCN" — and a claim that a
specific normalisation buys robustness would have been a false positive if it had
only been inferred from the architecture.

**Cross-graph check of the self-loop claim (USAir, `results/deep_usair.md`).**
On the dense graph the same substitution gives GCN `sym` 0.825 / `noloop` 0.821 /
`sum` **0.854** under `del`. The self-loop advantage disappears (−0.004, not
−0.072) and the unnormalised sum is *better* than the trained operator. This is
consistent with §7.1 rather than with a universal operator claim: on USAir a 50 %
deletion barely changes the local evidence (CN AUC 0.928 → 0.860), so there is
nothing for a self-loop to protect; the only effect left is the `sum` operator
amplifying the still-informative degree signal. **The self-loop is the carrier of
Cora's deletion robustness, not of robustness in general.**

### 7.3 Which edges carry the evidence? (also mostly a negative result)

Deleting 50 % of the *triadic* edges (endpoints share a neighbour, i.e. the CN
evidence) versus 50 % of the *bridging* edges (CN = 0, reachability only),
same budget:

| model | clean | delete triadic | delete bridging |
|-------|------:|---------------:|----------------:|
| SGDiff | 0.732 | 0.651 | 0.631 |
| SEAL | 0.781 | 0.727 | 0.706 |
| PENCIL | 0.795 | 0.740 | 0.736 |
| GCN | 0.827 | 0.818 | 0.794 |
| GAT | 0.807 | 0.781 | 0.736 |
| SAGE | 0.757 | 0.746 | 0.723 |
| VGAE | 0.784 | 0.774 | 0.743 |

Triadic edges are not privileged: for every model, deleting 50 % of the bridging
edges is as bad or slightly worse than deleting 50 % of the triadic ones. Damage
tracks the *amount* of local evidence removed, not its closure type. The
"triangles are the evidence" reading of §4.3 is therefore not supported at this
granularity — what the sub-graph family needs is a short path, and a bridge edge
removal breaks the path for *both* endpoints, not just for one triangle.

This probe is only interpretable when both populations are comparable. On Cora
they are (1 992 triadic vs 2 494 bridging edges); on USAir the graph is so dense
that it contains 1 724 triadic and only **83** bridging edges, so the two columns
measure different budgets and the comparison is degenerate (`results/deep_usair.md`
§3: deleting half of 83 bridges changes nothing). Reported here as a null result
with its scope condition, not as a positive finding.

### 7.4 The repairs, and why they are the causal test

Every repair is applied to the *inference* graph only (no retraining, test pairs
excluded from the candidate set). The point is not the absolute gain — it is that
**the same repair helps the models the attribution predicts and does nothing for
the others.**

| repair | SGDiff | SEAL | PENCIL | GCN | GAT | SAGE | VGAE |
|--------|-------:|-----:|-------:|----:|----:|-----:|-----:|
| `del` + CN edge completion | **+0.113** | +0.006 | +0.009 | −0.006 | +0.008 | −0.002 | −0.006 |
| `del_hub` + CN edge completion | +0.038 | **+0.046** | **+0.039** | −0.002 | +0.004 | +0.009 | +0.003 |
| `feature_shuffle` + structure-smoothed features | 0.000 | −0.008 | +0.010 | **+0.054** | **+0.030** | **+0.065** | **+0.058** |
| `del` + CN prior added to the score | +0.000 | +0.007 | +0.002 | +0.012 | +0.009 | +0.012 | +0.013 |

Read the rows, not the columns:

* **Restoring the deleted edges rescues exactly the model whose damage was
  evidence starvation.** SGDiff goes 0.521 → 0.634 (+0.113). That is above the
  CN prior's own AUC on the same damaged graph (0.572), i.e. the model is not
  merely being replaced by the heuristic: it uses the repaired local structure
  better than the heuristic does. The GCN family moves by ±0.01 — they never read
  CN, so repairing it is irrelevant to them.
* **Rebuilding the corrupted attributes from the structure rescues exactly the
  attribute-driven models.** GCN +0.054, SAGE +0.065, VGAE +0.058, GAT +0.030,
  while SGDiff (dead feature branch) moves by exactly 0.000 and SEAL/PENCIL by
  ≤0.01. The same intervention, opposite effects, predicted in advance by §4 —
  this is the cleanest falsification test in the document. It also shows the
  limit of the repair: smoothing recovers 0.628 → 0.683 against a clean 0.832, so
  roughly a quarter of the attribute damage is recoverable from structure (as it
  must be: a permutation destroys information, it does not hide it).
* **Adding the CN prior to the score is the weakest repair** (+0.000 to +0.013),
  and it is *zero for SGDiff* — because SGDiff's ranking already is the CN
  ranking (ρ = 0.72, §4.3). A model cannot be helped by a heuristic it already
  implements.

### 7.5 Model-side repair: more hops raise the ceiling, not the floor

SEAL and PENCIL retrained on 2-hop enclosing sub-graphs (`k = 2` instead of the
paper's `k = 1`), Cora, seeds 0-1:

| model | probe | k = 1 | k = 2 | Δ |
|-------|-------|------:|------:|--:|
| SEAL | clean | 0.781 | **0.856** | +0.075 |
| SEAL | `del` | 0.644 | 0.648 | +0.004 |
| SEAL | `del_hub` | 0.632 | 0.640 | +0.008 |
| SEAL | `ins_hub` | 0.760 | 0.803 | +0.043 |
| PENCIL | clean | 0.795 | **0.860** | +0.065 |
| PENCIL | `del` | 0.671 | 0.688 | +0.017 |
| PENCIL | `del_hub` | 0.650 | 0.662 | +0.012 |
| PENCIL | `ins_hub` | 0.788 | 0.813 | +0.025 |

Two hops buy ~7 points of clean accuracy, but the *damage* does not shrink: SEAL's
deletion gap goes 0.141 → 0.189 and PENCIL's 0.111 → 0.173. Richer input raises
the ceiling and the floor together, so fragility is not simply "too few hops".
Consistent with §7.1: the evidence channel is restored, but the model's reliance
on it grows at the same time. A real fix has to change *how* the model uses the
evidence (calibrate against its absence), not only how much it sees.

**Citeseer replication.** §7.1-7.5 were re-run with all seven models on Citeseer
(seeds 0-1, `results/deep_citeseer.{json,md}`). Every mechanism above replicates
in sign; two magnitudes differ, in the direction the mechanism predicts:

| mechanism | statistic | Cora | Citeseer |
|---|---|---:|---:|
| attribute damage happens before L1 | GCN `feature_shuffle` L1 probe AUC | 0.614 | 0.592 |
| structural damage enters at hop 2 | GCN `del` L1 → L2 probe AUC | 0.839 → 0.809 | 0.847 → 0.827 |
| attention renormalisation costs more | GAT `del` L2 probe AUC (drop vs clean) | 0.688 (−0.066) | 0.686 (−0.117) |
| the self-loop is the carrier | GCN `del`, sym → no self-loops | 0.790 → 0.718 | 0.787 → 0.675 |
| the normalisation is not | GCN `del`, sym → row-mean | 0.790 → 0.777 | 0.787 → 0.779 |
| bridging edges ≥ triadic ones | GCN `del_tri` / `del_bridge` | 0.823 / 0.796 | 0.812 / 0.795 |
| edge completion rescues SGDiff | `del_hub` CN completion gain | +0.038 | +0.058 |
| feature smoothing rescues GCN | `feature_shuffle` smooth2 gain | +0.054 | +0.021 |
| k = 2 raises ceiling, not floor | SEAL clean / `del` | 0.780 → 0.850 / 0.639 → 0.661 | 0.740 → 0.814 / 0.654 → 0.632 |

The two weakened magnitudes are informative rather than contradictory: the
structure-smoothing repair is smaller on Citeseer (0.021 vs 0.054), and the k = 2
retrain *widens* SEAL's deletion gap there (0.086 → 0.182), which is the same
"more input ≠ less fragile" claim with a sharper sign.

### 7.6 Component-level redesign: change one part, retrain, re-probe

Everything above reads a *frozen* network. That localises the damage but cannot
say whether a component is *causally* responsible. This subsection retrains a GCN
from scratch once per configuration, with exactly one component changed
(`scripts/run_redesign.py`, `results/redesign_<dataset>.{json,md}`, seeds 0-7,
same probes, same pair budget, augmented inputs rebuilt from the graph being
scored):

| config | what changes vs `base` |
|---|---|
| `base` | 2 layers, symmetric `1/√(d_i d_j)`, self-loops, raw attributes |
| `depth-L1` / `depth-L3` | one or three propagation steps |
| `noloop` | trained with `add_self_loops = False` |
| `row` | trained with row-mean normalisation |
| `+cnfeat` | input = raw X ‖ 6 structural descriptors (degree, mean/max CN over incident edges, clustering, 2-hop reach, mean neighbour degree) |
| `+smooth` | input = structure-smoothed attributes (1 round, α = 0.5) |
| `+cnfeat+smooth` | both data-side changes |
| `+edgedropout` | robust training: 30 % of the training edges dropped per epoch |

Damage (clean − probe), means over seeds 0-7 (per-seed spread ≤ 0.03):

| config | clean AUC (Cora / Citeseer) | Cora `del` | Cora `del_hub` | Cora `feat_shuffle` | Citeseer `del` | Citeseer `del_hub` | Citeseer `feat_shuffle` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 0.813 / 0.799 | 0.056 | 0.067 | 0.167 | 0.031 | 0.043 | 0.189 |
| `depth-L1` | 0.833 / 0.862 | 0.044 | 0.048 | **0.236** | 0.025 | 0.035 | **0.289** |
| `depth-L3` | 0.824 / 0.805 | **0.040** | **0.042** | **0.225** | **0.019** | **0.026** | **0.214** |
| `noloop` | 0.810 / 0.733 | **0.114** | **0.121** | 0.167 | **0.052** | **0.083** | 0.147 |
| `row` | 0.802 / 0.781 | 0.061 | 0.063 | 0.157 | 0.028 | 0.035 | 0.174 |
| `+cnfeat` | 0.806 / 0.786 | 0.062 | 0.068 | 0.160 | 0.028 | 0.036 | 0.178 |
| `+smooth` | 0.813 / 0.786 | 0.053 | 0.072 | 0.147 | 0.024 | 0.045 | 0.169 |
| `+cnfeat+smooth` | 0.807 / 0.775 | 0.063 | 0.075 | **0.130** | 0.021 | 0.039 | **0.145** |
| `+edgedropout` | 0.839 / 0.818 | 0.055 | 0.061 | 0.206 | 0.034 | 0.046 | 0.213 |

The same grid on USAir (dense, no attributes, so no `feature_shuffle` column) is
the control for the self-loop claim — §7.2 predicted that the self-loop advantage
belongs to the sparse graphs, not to GNNs in general:

| config | USAir clean | USAir `del` | USAir `del_hub` |
|---|---:|---:|---:|
| `base` | 0.886 | 0.049 | 0.036 |
| `noloop` | 0.882 | 0.038 | **0.028** |
| `row` | 0.840 | **0.083** | **0.144** |
| `depth-L3` | 0.875 | 0.033 | 0.031 |
| `+edgedropout` | 0.878 | 0.045 | 0.035 |

On the dense graph removing the self-loop **does not hurt** (0.049 → 0.038,
`del_hub` 0.036 → 0.028) — the train-time counterpart of the inference-swap
cross-check, and a third independent confirmation that the self-loop is a
sparse-graph carrier. The dense graph has the opposite sensitivity: there the
normalisation matters, and training with row-mean doubles the deletion damage and
quadruples the `del_hub` damage.

Five readings, four of them causal confirmations of §7 and one a new trade-off:

1. **The self-loop is causal, not correlational — and only on sparse graphs.**
   Removing it *at training time* doubles the deletion damage: Cora `del` 0.056 →
   0.114, `del_hub` 0.067 → 0.121; Citeseer `del_hub` 0.043 → 0.083; USAir
   `del_hub` 0.036 → 0.028 (no penalty). On Cora the clean AUC is essentially
   unchanged (0.813 vs 0.810), so this is not a weaker model — it is the missing
   identity path. On Citeseer the clean AUC does fall (0.799 → 0.733), so the
   honest statement is: the self-loop carries a large part of the deletion
   robustness on both sparse graphs, and part of the Citeseer gap is the budget
   lost with it.
2. **The normalisation is not causal on the sparse graphs, and it is on the
   dense one.** Swapping `sym` → `row` *in training* moves the deletion damage
   0.056 → 0.061 (Cora) and 0.031 → 0.028 (Citeseer) — different signs, inside the
   seed spread, so the §7.2 negative result survives the train-time test. On USAir
   the same swap costs 0.083 / 0.144 (vs 0.049 / 0.036) — there the normalisation
   *is* the carrier, which is exactly the graph-dependence §7.2 predicted.
3. **Depth moves damage between the two channels.** At equal-or-better clean AUC,
   both a 1-step and a 3-step encoder cut the deletion damage (Cora 0.056 →
   0.044 / 0.040, Citeseer 0.031 → 0.025 / 0.019) and both raise the attribute
   damage (0.167 → 0.236 / 0.225, 0.189 → 0.289 / 0.214). More propagation means
   the representation leans more on smoothed structure and less on the per-node
   attributes, so the two robustness axes are **traded off, not jointly
   improved** — the clearest evidence that the two damages have different
   carriers. Note that the "1-step" encoder is not a smaller model here (Cora
   0.833, Citeseer 0.862 clean), so this is not a capacity artefact.
4. **Strengthening the input fixes the attribute channel, not the deletion
   channel.** `+cnfeat` — handing the model the CN/degree descriptors it was
   accused of not reading — leaves `del` where it was (0.056 → 0.062, 0.031 →
   0.028). What does work is the data-side rebuild: `+cnfeat+smooth` cuts the
   `feature_shuffle` damage from 0.167 to 0.130 (Cora) and from 0.189 to 0.145
   (Citeseer) — the only row that improves on both graphs by more than the
   per-seed spread — but it costs 0.006 / 0.024 of clean AUC. `+smooth` alone is
   roughly half that effect (0.147 / 0.169) and sits within one seed standard
   deviation. Partial fix, attribute channel only: a permutation destroys
   information, so no input transform can restore all of it.
5. **Robust training raises the ceiling and leaves the floor.** Edge dropout is
   the one configuration that clearly improves clean accuracy (Cora 0.813 →
   0.839, Citeseer 0.799 → 0.818) — and it does not shrink the deletion damage
   (0.055 / 0.034) while making the attribute probe *worse* (0.206 / 0.213).
   Same pattern as the k = 2 sub-graph retrain in §7.5, now on the GCN side:
   "train harder / see more" is not "be less fragile".

Put together, the redesign grid says the fragility is a property of *how* the
model uses the identity path and the attribute channel — not of its capacity, its
normalisation, or the richness of its input. The only intervention that moved a
damage channel in the predicted direction on both attributed graphs is the
structure-smoothed input plus structural descriptors (attribute channel), and it
buys that with clean accuracy. Nothing tested closes the structural-evidence
gap: the candidate that follows from §7.1/§7.4 is to calibrate the score against
the *absence* of evidence (score a pair relative to a null model of its expected
local evidence), which is a change in the scoring rule and is left as an untested
proposal, not a result.

### 7.7 The damage is localised — and AUC hides where

Everything so far uses one number per cell. Two measurements on the *per-pair*
scores (`scripts/run_score_dump.py` → `results/operating_<dataset>.{json,md}`,
seeds 0-1) show that "the model lost 0.05 AUC" is a statement about the *pair
mix*, not about the model degrading uniformly.

**(a) It is a partition of the pair set, not a global shift.** Take the clean
graph's common-neighbour count as a fixed stratum (grouping by the perturbed
graph's CN would condition on the outcome), and ask how much score each stratum
loses under `del@50%`. Values are mean score drops in clean-σ:

| model | pairs whose CN survived | pairs whose CN → 0 | gap |
|---|---:|---:|---:|
| `sgdiff` | -0.15 | **+2.18** | 2.33 |
| `pencil` | +0.03 | **+1.56** | 1.53 |
| `seal` | +0.05 | **+1.15** | 1.09 |
| `rgcn` | -0.50 | +0.63 | 1.14 |
| `gat` | -0.35 | +0.59 | 0.94 |
| `sage` | -0.07 | +0.43 | 0.50 |
| `gcn` | -0.12 | **-0.04** | 0.09 |
| `vgae` | -0.11 | -0.02 | 0.10 |

Citeseer gives the same ordering (`sgdiff` −0.03 / +2.16, `seal` −0.12 / +0.35,
`gat` −0.25 / +0.62, `gcn` −0.12 / −0.03). Three readings:

1. For every model that reads local evidence, the loss is **entirely** on the
   pairs whose evidence vanished; on the pairs whose CN survived the score does
   not drop at all (it often rises, because fewer competing paths exist).
2. `gcn` and `vgae` are **flat across the split** (−0.00 / −0.03). Their deletion
   damage therefore does *not* come from evidence starvation — a mechanism-level
   fact that explains §7.6: no amount of structural input (`+cnfeat`, k = 2, edge
   completion) can help a model that never reacted to the evidence in the first
   place.
3. The sub-graph family's damage is an order of magnitude larger than its AUC
   suggests at the pair level (`sgdiff` 2.11 σ on the affected half, 0.218 AUC
   overall): the average dilutes a total loss on ~30 % of the positives.

**(b) Where the loss sits: positive deflation, negative inflation, or pure
reordering.** Mean class-wise score drop under `del@50%` (Cora), with the
head-of-ranking metrics alongside:

| model | ΔAUC | ΔAP | Δhit@100 | ΔTPR@1%FPR | Δpos | Δneg | d′ (clean → probe) | τ | inversion |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `sgdiff` | +0.218 | +0.201 | **+0.080** | +0.218 | +0.69 | +0.00 | 1.19 → 0.46 | 0.32 | 0.39 |
| `seal` | +0.140 | +0.156 | +0.155 | +0.156 | +0.28 | −0.28 | 1.16 → 0.52 | 0.52 | 0.24 |
| `pencil` | +0.111 | +0.107 | **+0.045** | +0.165 | +0.54 | +0.04 | 1.20 → 0.71 | 0.54 | 0.21 |
| `gcn` | +0.042 | +0.054 | +0.065 | +0.095 | **−0.11** | −0.02 | 1.14 → 0.97 | 0.72 | 0.09 |
| `vgae` | +0.037 | +0.047 | +0.080 | +0.096 | −0.09 | +0.01 | 1.01 → 0.85 | 0.69 | 0.11 |

The four columns in the middle are the same attack measured four ways, and they
disagree in both magnitude and mechanism:

* `sgdiff` loses 0.218 AUC but only 0.080 hit@100: its head survives 2.7× better
  than its tail. The mechanism is visible in Δpos: positives fall 0.69 σ, the
  likelihood difference compresses toward the prior (d′ 1.19 → 0.46) — it is
  positive deflation, exactly what §7.4 predicted.
* `seal` is the opposite shape: Δpos = +0.28 **and** Δneg = −0.28, i.e. positives
  drop *and* negatives gain. Its damage is not "less evidence for positives" but
  "evidence that stopped discriminating" — which is why edge completion only
  half-helps it.
* `pencil` damages the average 2.5× more than the head (0.111 vs 0.045).
* `gcn` has no class-wise loss at all (both classes gain score) and the worst
  column is TPR@1%FPR (0.095): its damage is **pure reordering** (τ = 0.72,
  9 % of the correctly ordered pos/neg pairs inverted). It is a different failure
  mode from the sub-graph family, not a smaller version of it.

One cell flips sign outright: `sgdiff` under `ins_hub` *gains* 0.023 AUC while
losing **0.262 TPR@1%FPR** and 0.080 hit@100. A hub inserted next to everything
inflates the tail and destroys the head, so the ranking statistic that the
robustness literature reports says "harmless" while the deployed top-k says "the
worst attack in the table". Any conclusion of the form "insertion is not a threat"
has to name its metric.

### 7.8 The probes were a lower bound: an adaptive attacker at 1/5 of the budget

Every probe in §3-7.7 is non-adaptive: it deletes a random or degree-selected half
of the graph and never looks at the model. `scripts/run_adaptive_attack.py` runs
the missing experiment — greedy, score-feedback deletion at a **matched budget**:
at each step it rebuilds a candidate pool (the edges that carry the shared
neighbour of a scored positive) on the *current* graph, scores each candidate, and
deletes the one that costs the most AUC. 200 + 200 working pairs, seeds 0-1.

ΔAUC at equal budget (positive = damage), with the fixed 50 % probe for reference:

| dataset | model | random 5 % | hub 5 % | adaptive 5 % | adaptive 10 % | fixed probe `del@50%` | adaptive 10 % ÷ probe |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | `gcn` | +0.003 | +0.005 | +0.115 | **+0.218** | +0.042 | **5.2×** |
| Cora | `sage` | +0.004 | +0.001 | +0.154 | **+0.282** | +0.035 | **8.1×** |
| Citeseer | `gcn` | −0.000 | +0.004 | +0.085 | +0.161 | +0.027 | 6.0× |
| Citeseer | `sage` | +0.001 | +0.007 | +0.127 | +0.217 | +0.013 | **16.7×** |
| USAir | `gcn` | +0.001 | +0.008 | +0.077 | +0.145 | +0.062 | 2.3× |
| USAir | `sage` | +0.001 | +0.004 | +0.080 | +0.124 | +0.036 | 3.4× |

GCN on Cora across four budgets (seed 0): **2 % → 0.037, 5 % → 0.115, 10 % →
0.218, 20 % → 0.314** — clean AUC 0.826, so a targeted attacker with a fifth of
the edges drives GCN to 0.51, i.e. chance. The curve is near-linear up to 10 %
(≈2.2 points of AUC per percent of edges) and saturates after that; the random
probe never exceeds 0.04 at 50 %.

The sub-graph family can be attacked the same way, but every candidate costs a
sub-graph extraction, so the budget is spent instead on breadth
(`seal` and `pencil`, Cora, seed 0, 100 + 100 working pairs, 5 % budget,
8 candidates/step): `seal` random **+0.008** / hub +0.003 / adaptive **+0.290**
(36× random per edge, 2.1× its own 50 % probe of 0.140); `pencil` random **+0.012**
/ hub +0.004 / adaptive **+0.194** (16× random, 1.75× its 50 % probe of 0.111).

Four consequences, and the first one revises an earlier conclusion:

1. **A tenth of the budget beats half the graph.** Adaptive deletion at 10 % costs
   every model more than the fixed 50 % probe does, by 2.3-17×. The ordering
   "GCN is barely affected by deletion" is a statement about *random* deletion,
   not about GCN: with 5 % of the edges chosen by feedback, GCN loses
   0.077-0.115, i.e. more than the 50 % random probe ever managed.
2. **Per-edge efficiency differs by 1.5-2 orders of magnitude.** Random deletion
   buys 0.003-0.013 AUC for 5 % of the edges; targeted deletion buys 0.077-0.152.
   Any statement of the form "the attack needs 50 % of the edges" is really a
   statement about the attacker's selection rule.
3. **The lethal edge type is graph-dependent, and it is not the one the CN story
   predicts.** On the sparse graphs the greedy keeps picking edges with CN ≈ 0
   (0.43-0.65) that are mostly bridges (61-73 %), i.e. it removes the *only* path
   between a pair's neighbourhoods. On dense USAir it picks high-CN edges
   (4.6-6.2 triangles per edge, 10-13 % bridges) — the same graph-dependence of
   the carrier that §7.2 and §7.6 found by substitution.
4. **The lower bound is still a lower bound.** Candidates are sampled (24 per
   step) and the pool is restricted to edges incident to the scored positives; a
   gradient or betweenness-guided attacker with the full edge set would do at
   least this well. So the numbers in §3-7.7 should be read as *how the models
   respond to damage*, not as *how fragile they are*.
5. **Selection rule ≫ budget, and that is a measurement problem, not a modelling
   one.** At 5 % the adaptive attacker beats random deletion by 36-40× per edge on
   every model tested (node-pair and sub-graph), and at 10 % it beats the 50 %
   random probe itself. A robustness number without its attacker is therefore
   uninterpretable — which is also the honest reading of the paper's own Table 4:
   RF and DICE are random-selection attackers, so their damage is a floor.

### 7.9 Can the scoring rule be fixed? A null-model calibration attempt (negative)

§7.6 closed with an untested proposal: score a pair against the *absence* of
evidence instead of only against other pairs. First question: how much of a
model's score is structural prior at all? Ridge of the clean scores on
`[CN, log(1+Ê[CN]), d_u, d_v, d_u d_v]`, where `Ê[CN] = d_u d_v / 2m` is the
degree-preserving (configuration-model) expectation — `results/calibration_<dataset>.{json,md}`:

| model | Cora R² | Citeseer R² |
|---|---:|---:|
| `sgdiff` | **0.656** | **0.544** |
| `pencil` | 0.515 | 0.263 |
| `seal` | 0.452 | 0.216 |
| `gcn` | 0.301 | 0.195 |
| `vgae` | 0.254 | 0.148 |
| `sage` | 0.223 | 0.186 |
| `rgcn` | 0.193 | 0.075 |
| `gat` | 0.238 | 0.161 |

This is a third, independent measurement of the same channel story: two thirds of
SGDiff's score is recoverable from common-neighbour and degree structure, a
quarter of GCN's is — without touching the model, and without any probe.

Second question: does subtracting that prior repair the damaged graph? The
correction `s − λ·ŝ` is fitted on the clean graph and applied with the features of
whichever graph is scored (λ = 1 in the table):

| model | clean AUC (λ=0 → 1) | `del` AUC | `ins_hub` AUC |
|---|---|---|---:|---:|
| `gcn` | 0.827 → 0.667 | 0.781 → 0.748 | 0.804 → **0.430** |
| `sgdiff` | 0.732 → 0.607 | 0.522 → 0.422 | 0.776 → 0.674 |
| `seal` | 0.781 → 0.651 | 0.644 → 0.582 | 0.760 → 0.641 |

The correction moves along a frontier, it never lifts it. Because the clean
reference is wrecked faster than the damaged one, the *relative* damage can even
flip sign (`gcn` `del`: +0.061 → −0.092) while the absolute accuracy on the
damaged graph falls (0.790 → 0.771) — a textbook trap of measuring damage as a
difference against a moving baseline. The cheaper parametric form
`s − λ·z(Ê[CN])` is a no-op (`gcn` `del` 0.790 → 0.788 at λ = 1), because the
expectation is nearly constant across pairs on a sparse graph.

So the simple form of the §7.6 proposal is **falsified**: subtracting the prior is
not the same as knowing that evidence is missing.

**The localised version also fails, for a reason that is worth stating.** Gating
the correction on the observed evidence — `s − λ·σ·1[CN = 0]`, so that pairs which
still have a shared neighbour are untouched — makes deletion damage *worse*
(`gcn` 0.042 → 0.059, `sage` 0.035 → 0.073 at λ = 1) and only mildly helps the
insertion channel (`rgcn` 0.085 → 0.054, `vgae` 0.052 → 0.033). Ranking inside
expected-evidence strata (5 or 10 bins, i.e. "compare a pair only against pairs
with similar degree product") behaves the same way as the global correction:
*relative* damage shrinks (`gcn` 0.036 → 0.026) while absolute accuracy falls on
both graphs (`clean` 0.826 → 0.792, `del` 0.790 → 0.766).

The reason is structural, not a matter of tuning: on Cora **54 % of the true
positives already have CN = 0** (270/500) versus 99 % of the negatives (495/500).
"Evidence vanished" is therefore not an observable anomaly at the pair level —
it is the normal state of the data — so a gate built on it cannot separate the
pairs that were attacked from the pairs that never had evidence. The information
needed for that separation is exactly the prior the score already encodes, which
closes the loop: on a sparse graph, the score's structural prior is not noise to
be calibrated away, it is the only thing standing in for the missing evidence.
Any real fix has to come from the *deployment side* (an evidence-complete
extraction, or an attacker-aware retraining objective), not from a post-hoc
re-ranking of the same scores.

### 7.10 How large are these effects, really? Paired bootstrap

Every table above compares means over 3-8 seeds to a "seed spread". The right
uncertainty for a fixed (model, probe) is the *pair sample*: the damage is
estimated from ~1000 scored pairs, and clean and probe scores are computed on the
same pairs, so the resampling must be paired.
`scripts/run_bootstrap_ci.py` → `results/significance_<dataset>.{json,md}`, 2000
paired resamples per (model, probe, seed), pooled over 4 seeds per graph. Pooling
seeds is deliberate: it widens the interval when seeds disagree, so the "excludes
0" claims are established *despite* seed variation rather than because a lucky
seed was dropped.

| dataset | claim | ΔAUC [95 % CI] | p(no damage) |
|---|---|---|---:|
| Cora | `sgdiff` del | +0.211 [+0.107, +0.301] | 0.0000 |
| Cora | `gcn` feat_shuffle | +0.198 [+0.143, +0.256] | 0.0000 |
| Cora | `seal` del | +0.137 [+0.104, +0.168] | 0.0000 |
| Cora | `pencil` del | +0.125 [+0.074, +0.209] | 0.0000 |
| Cora | `gcn` del | +0.046 [+0.027, +0.070] | 0.0000 |
| Cora | `seal` feat_shuffle | +0.055 [−0.001, +0.111] | 0.0265 |
| Cora | `sgdiff` ins_hub (AUC) | −0.043 [−0.088, +0.015] | 0.8644 |
| Citeseer | `sage` del | +0.014 [−0.013, +0.040] | 0.1624 |
| USAir | `pencil` del | +0.030 [−0.009, +0.075] | 0.1206 |

Two things follow, and they cut in opposite directions:

1. **The headline effects are far outside the interval.** The 0.12-0.21 AUC
   claims sit 2-4 interval-widths above 0; no amount of extra seeds would change
   them. This also retro-justifies the 2-seed deep probes for the large effects.
2. **Most of the small cells are not distinguishable from 0**, and the report
   should stop treating them as facts: `seal`/`pencil`'s feature damage on Cora
   (0.055/0.040), `sgdiff`'s `ins_hub` AUC "gain" (−0.043, CI covers 0), `sage`
   and `rgcn`'s deletion damage on Citeseer. The `+smooth` effect of §7.6
   (0.020) is of the same size and was already flagged as not seed-robust.

The `sgdiff` `ins_hub` row is the sharpest case: its AUC difference is
statistically indistinguishable from zero while its `TPR@1%FPR` damage is 0.262.
Quoting the AUC there is not a conservative choice — it is the wrong statistic,
and the bootstrap says so.

### 7.11 A repair that follows the diagnosis: evidence routing (also negative)

The diagnosis of §7.7 suggests a conditional repair rather than a bigger model:
train the sub-graph model at both scales and route per pair — keep the k = 2 score
only where the scored graph offers no evidence, use k = 1 elsewhere.
`scripts/run_gated_repair.py` → `results/gated_cora.{json,md}`, seeds 0-1:

| model | arm | clean AUC | `del` damage | `del_hub` damage |
|---|---|---:|---:|---:|
| `seal` | k = 1 | 0.781 | +0.137 | +0.149 |
| `seal` | k = 2 (global) | **0.856** | +0.208 | +0.216 |
| `seal` | routed (`CN ≤ 0`) | 0.844 | +0.204 | +0.213 |
| `pencil` | k = 1 | 0.795 | +0.125 | +0.145 |
| `pencil` | k = 2 (global) | **0.860** | +0.173 | +0.198 |
| `pencil` | routed (`CN ≤ 0`) | 0.852 | +0.169 | +0.195 |

The router buys the k = 2 ceiling (+0.06 clean) and inherits the k = 2 gap: on
`del` it is 0.067 worse than plain k = 1, and the paired bootstrap over the scored
pairs puts that difference at **−0.067 [−0.097, −0.037]**. The interval excludes
0, so this is a regression, not pair noise — and the same holds for every probe
(`del_hub` −0.064 [−0.096, −0.032], `ins_hub` −0.051 [−0.074, −0.028]; `pencil`
−0.044 / −0.049 / −0.047, all intervals excluding 0). The 0.05-scale claims of
§7.9-7.11 hold at pair resolution, over 4 seeds.

The mechanism is visible in the gate counts: `CN = 0` fires for **77.0 % of pairs
on the clean graph and 93-94 % after deletion** — the signal is degenerate exactly
when it is needed, so "routed" collapses onto "k = 2 everywhere". Moving the
threshold does not rescue it, because the frontier is monotone in the wrong
direction — every setting is closer to "always k = 2" than to plain k = 1:

| model | probe | `CN ≤ 0` | `CN ≤ 1` | `CN ≤ 2` | always k = 2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|
| `seal` | `del` | +0.204 | +0.206 | +0.207 | +0.208 | **+0.137** |
| `seal` | `del_hub` | +0.213 | +0.214 | +0.216 | +0.216 | **+0.149** |
| `seal` | `ins_hub` | +0.073 | +0.062 | +0.056 | +0.054 | **+0.022** |
| `pencil` | `del` | +0.169 | +0.170 | +0.172 | +0.173 | **+0.125** |
| `pencil` | `del_hub` | +0.195 | +0.196 | +0.198 | +0.198 | **+0.145** |
| `pencil` | `ins_hub` | +0.054 | +0.051 | +0.049 | +0.047 | **+0.008** |

Widening the gate makes the damage converge to the k = 2 number from below; there
is no threshold where routing both keeps its clean gain and keeps the k = 1
damage. The reason is the gate share, which rises with the attack instead of
concentrating on a damaged minority: `CN ≤ 0` covers 77.0 % of pairs clean, 92.8 %
after `del`, 94.2 % after `del_hub`; `CN ≤ 1` covers 92.5 % / 99.3 % / 99.3 %.

The other fallback on the table — rebuild the deleted edges with the CN prior
(§7.5) and score with the k = 1 model — is **structurally void on exactly the
pairs it targets**. The CN prior is only defined over candidate pairs that
*already* share a neighbour, so completing the graph with the `removed` highest-CN
missing edges restores evidence for 3.7 % of the `CN = 0` pairs after `del` and
4.2 % after `del_hub`; the remaining 96 % stay at `CN = 0` by construction. A
repair that has to create a shared neighbour out of nothing cannot be built from a
prior that ranks shared neighbours:

| probe | `CN = 0` share | recovered by completion | edges added |
|---|---:|---:|---:|
| `del` | 92.8 % | **3.7 %** | 2235 |
| `del_hub` | 94.2 % | **4.2 %** | 2240 |

This is the same failure as §7.9, and it is now a pattern rather than an accident:
**every gate built on "there is no shared neighbour" fails, because on a sparse
citation graph that is the normal state of the data (54 % of positives), and
deletion only makes it more normal.** A usable gate has to be informative *under
attack*, which means comparing the pair against a counterfactual rather than
against a threshold.

### 7.12 The counterfactual splice gate: the last candidate, tested (negative)

§7.11 left one proposal untested: for each pair, splice a synthetic 2-hop path
`u-w-v` with a random `w` and gate on whether the score rises — "if it rises, the
model was starved". `scripts/run_splice_gate.py` → `results/splice_cora.{json,md}`
does exactly that: the k = 1 model scores every pair twice, once on its enclosing
sub-graph and once with a node of DRNL label 2 appended and connected to `u` and
`v`, and `Δ = spliced − plain` is the gate signal. Seeds 0-1, 1000 scored pairs,
`seal` and `pencil`:

| model | probe | `CN = 0` | sign(Δ) | top 50 % | top 30 % | top 10 % | always k = 2 | k = 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `seal` | `del` | +0.204 | +0.208 | +0.175 | +0.167 | +0.152 | +0.208 | **+0.137** |
| `seal` | `del_hub` | +0.213 | +0.216 | +0.173 | +0.163 | +0.159 | +0.216 | **+0.149** |
| `pencil` | `del` | +0.169 | +0.171 | +0.131 | +0.121 | +0.121 | +0.173 | **+0.125** |
| `pencil` | `del_hub` | +0.195 | +0.197 | +0.160 | +0.151 | +0.143 | +0.198 | **+0.145** |

Two measurements kill it, and both are the opposite of the hope in §7.11:

1. **The response is universal, so the sign gate saturates.** `Δ > 0` for
   **96.9-99.6 %** of pairs (99.3-100 % when the spliced node's features are
   zeroed; `corr(Δ_own, Δ_blank) = 0.92-0.96`). Sub-graph models answer "is there
   a shared neighbour?" with *yes* no matter whose neighbour it is — the splice
   measures path existence, not starvation. A diagnostic that fires on 99 % of
   pairs is the `CN = 0` gate with a different hat, and "routed" is again
   "always k = 2" (the `sign(Δ)` column reproduces the `CN = 0` column to within
   0.01).
2. **The magnitude does not rank the pairs that lose score.** AUC of `Δ` at
   predicting per-pair score drop under deletion is **0.445-0.531** — chance.
   So the quantile gates are not selecting starved pairs; they are selecting an
   arbitrary half/third/tenth. Tightening the quantile therefore just walks back
   along the k = 1 ↔ k = 2 dial: top 10 % routing gives `del` +0.152 for `seal`,
   better than always k = 2 (+0.208) but still worse than not routing at all
   (+0.137), and it still pays clean AUC (0.816 at top 50 % vs 0.781 for k = 1).

The pattern is now closed in both directions: gates on **state** ("is there
evidence?") fire on most pairs before the attack; gates on **response** ("would
evidence help?") fire on essentially all of them. Neither is an observable
per-pair property that separates damaged from undamaged pairs on these graphs.
What separates them is invisible from a single pair's neighbourhood — which is why
the repair that did work (§7.5, CN completion for SGDiff) changed the input supply
globally instead of trying to select pairs.

## 8. What this changes about the robustness story

1. **"Robustness to attacks" was mostly "robustness to deletion" — on Cora.**
   There the *largest* damage of every model is a deletion probe and insertion
   contributes 0.02-0.05. On Citeseer (sparsest) insertion costs SGDiff 0.108, on
   USAir (densest) hub insertion costs PENCIL 0.190. Which half of an attack does
   the damage is a property of the graph, and neither RF nor DICE reports it.
2. **Model capacity is not the axis; the scoring rule and the input supply are.**
   SEAL (54 k params) and SGDiff (2.3 M params) read the *same* sub-graphs and
   lose 0.14 vs 0.24 on Cora; the difference is the scoring rule. And the
   component that decides whether a model survives is which *channel* it reads
   (attributes vs local evidence), not its size.
3. **The robustness carrier inside a GCN is the self-loop, not the
   normalisation** (§7.2, §7.6). Under *random* deletion, symmetric, row-mean and
   unnormalised propagation are within 0.02; removing the self-loop costs 0.07,
   and retraining without it doubles the damage. Claims that attribute robustness
   to `1/√(d_i d_j)` would have been a false positive. This is a statement about
   random deletion: §7.8 shows that targeted deletion does reach GCN.
4. **Repairs are channel-specific, and that is the causal evidence.** Restoring
   deleted edges rescues SGDiff (+0.113) and does nothing for GCN (±0.01);
   rebuilding corrupted attributes from structure rescues GCN/SAGE/VGAE
   (+0.05-0.07) and does exactly nothing for SGDiff (0.000). Same intervention,
   opposite effects, predicted by the attribution.
5. **More input is not a fix.** Giving the sub-graph family 2-hop sub-graphs
   raises clean AUC by 0.07 *and* leaves the deletion gap unchanged (0.141 →
   0.189 for SEAL). Fragility tracks how the model uses evidence, not how much it
   is given.
6. **Nothing in this zoo reads homophily directly.** `ins_homo` — an edge that
   *looks* right — is invisible on Cora and Citeseer (≤0.036 for every model, a
   gain for GAT). Homophily shows up only as the assumption that makes smoothing
   useful.
7. **The two damage channels are separable, and only one of them is repairable
   from the input side** (§7.6). Retraining confirms the self-loop carries the
   deletion robustness (removing it doubles the damage) and that normalisation
   does not, on the sparse graphs. But no input augmentation closes the deletion
   gap: `+cnfeat` leaves it where it was, and the structure-smoothed input that
   does cut the attribute damage (0.167 → 0.130 Cora, 0.189 → 0.145 Citeseer)
   pays 0.006-0.024 of clean AUC for it. Depth only trades one channel for the
   other. Fragility here is about how evidence is turned into a score, not about
   capacity or input supply.
8. **AUC is not a sufficient statistic, and for one cell it has the wrong sign**
   (§7.7). `sgdiff` loses 0.218 AUC but only 0.080 hit@100 under `del` (the head
   survives 2.7× better than the tail); `pencil` damages the average 2.5× more
   than the head; `gcn` loses almost nothing on the margin and fails by
   reordering (τ = 0.72). And `sgdiff` under `ins_hub` *gains* 0.023 AUC while
   losing 0.262 TPR@1%FPR — so "insertion is harmless" is a metric artefact, not
   a property of the model.
9. **The damage is a partition of the pair set, not a uniform degradation**
   (§7.7a). For every evidence-reading model the loss is concentrated on the pairs
   whose evidence was deleted (2.11 σ for `sgdiff`, 1.10 for `seal`) and is
   approximately zero — often positive — on the pairs whose CN survived. `gcn`
   and `vgae` are flat across that split, which is *why* structural repairs cannot
   help them (§7.6).
10. **Every "the model is fragile / insensitive to X" claim above is a statement
    about a random attacker** (§7.8). With score feedback at 10 % of the budget,
    GCN loses 0.145-0.220 and SAGE 0.124-0.268 — 2.3-17× the fixed 50 % probe.
    The correct reading of §3-7.7 is "how the model responds to a given damage",
    not "how fragile the model is".
11. **The proposed repairs fail for one identifiable reason, not four.**
    (§7.9). Subtracting a degree-preserving null-model prior, its parametric
    variant, a gate that only touches pairs with `CN = 0`, and ranking inside
    expected-evidence strata all move the clean/probe trade-off along a frontier
    instead of lifting it; the gate makes deletion *worse*. The diagnosis is the
    contribution: on Cora 54 % of the positives already have `CN = 0`, so "the
    evidence vanished" is not an observable per-pair anomaly, and no post-hoc
    re-ranking of the same scores can recover it. The fix has to move to the
    deployment side (evidence-complete extraction) or into the training objective.
12. **A conditional version of that repair fails the same way** (§7.11). Routing
    between a k = 1 and a k = 2 sub-graph model on `CN = 0` buys the k = 2 ceiling
    (+0.06 clean) and inherits the k = 2 gap (+0.204 vs +0.137 on `del`; paired CI
    of the difference −0.067 [−0.097, −0.037]), because the gate fires on 77.0 % of
    pairs even *before* the attack and 93-94 % after it. Sweeping the trigger
    (`CN ≤ 0, 1, 2`) produces no operating point that beats plain k = 1, and the
    alternative fallback — CN edge completion — recovers evidence for 3.7-4.3 % of
    the low-evidence pairs it is supposed to rescue.
13. **The counterfactual gate that §7.11 proposed also fails, and for the mirror
    reason** (§7.12). Splicing one synthetic shared neighbour raises the k = 1
    score for 96.9-99.6 % of pairs — with zeroed features, 99.3-100 %, and
    `corr(Δ_own, Δ_blank) = 0.92-0.96` — so "the score rises" is universal and
    cannot identify starved pairs. The magnitude ranks per-pair damage at chance
    (AUC 0.445-0.531), so quantile gates just interpolate between k = 1 and
    "always k = 2"; the tightest (top 10 %) still loses to not routing at all
    (+0.152 vs +0.137 on `del`, `seal`). Gates on *state* fire too often; gates on
    *response* fire on everyone. Selection is the wrong lever.
14. **Statistically, only the large cells are established** (§7.10). Paired
    bootstrap over pairs, pooled over 4 seeds, puts the 0.12-0.21 AUC claims 2-4
    interval-widths above zero and leaves most cells ≤0.05 — including `sgdiff`'s
    `ins_hub` AUC "gain" (−0.043, CI covers 0, p = 0.86) — indistinguishable from
    noise, while the same
    cell's `TPR@1%FPR` damage is 0.262. Small AUC differences in this report
    should be read as directions, not quantities.

## 9. Caveats

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
6. **Two attributed datasets, one feature type.** Cora and Citeseer both use the
   standardised BoW channel, so the attribute claims cannot separate "BoW" from
   "attributes". Pubmed (6× larger) is prepared but was not re-run; a second graph
   with a different feature type is the missing control.
7. **The layer probe is a hook on the linear block.** For GCN/SAGE the hook
   captures the aggregated representation; for GAT it captures the projection
   *before* attention, so GAT's L1 cannot move under structural probes (marked
   with an asterisk, and not used to support any claim). VGAE has a single
   propagation block and is excluded from the layer table.
8. **Deep-probe runs use two seeds** (Cora / Citeseer / USAir) because each seed
   costs a full SGDiff fit (~5-8 min on CPU). The effects the section relies on are
   0.03-0.11 AUC — an order of magnitude above the seed spread at that budget
   (≤0.02 for everything except SGDiff's deletion column, ±0.09), but the exact
   decimals are not seed-converged.
9. **The `k = 2` repair changes the paper's setting** (Table 4 uses `k = 1` for
   every dataset). It is reported as a repair experiment, not as a proposed
   configuration.
10. **§7.6 retrains a GCN only.** The redesign grid holds the architecture family
    fixed (the one §7 localises the mechanism in), so the numbers say nothing
    about whether a different aggregator (attention, sub-graph classifier) would
    respond differently to the same component change. Each row is ≈1 s/fit, so the
    grid covers three graphs × eight seeds (per-seed spread ≤ 0.03 on the damage
    columns): the self-loop effect (0.05-0.09) and the `+cnfeat+smooth` effect
    (0.037-0.044) sit above it, the `+smooth`-only effect (0.020) does not.
11. **§7.7-7.9 use two seeds and a 200+200 / 500+500 pair budget.** These are
    per-pair analyses, not mean-AUC claims, so the seed cost was spent on breadth
    (three graphs, nine models) rather than on replicates. The strata counts are
    in the tables; on Cora the CN ≥ 1 stratum has ~157 positives but only 3-4
    negatives, so the per-stratum *AUC* is reported only where both classes reach
    20 and the score-drop columns carry the argument instead.
12. **The adaptive attacker of §7.8 is greedy and sampled**, not optimal: 24
    candidates per step, drawn from edges incident to the scored positives, one
    edge per step. It is a lower bound on what an attacker could do, and it is
    compared against random and hub deletion at the same budget on the same pairs.
    It covers GCN/SAGE at two budgets on three graphs, plus `seal`/`pencil` at a
    single 5 % budget and one seed (the sub-graph cost is the extractor rebuild).
13. **The bootstrap of §7.10 covers pair sampling, not training seeds.** The dumps
    are 4 seeds per graph and the intervals pool them (which widens the interval
    when seeds disagree), but a claim about the *population* of trained models
    still rests on the seed spreads in the other tables; the intervals here are the
    right ones for "what does this probe do to this model", not for "what would
    another seed do". The same applies to the gated intervals of §7.11: each seed is resampled
    over its own pairs and the reported bounds are the seed means, so they tighten
    the "does the router spend damage" claim, not the "would a retrained router
    do the same" claim.
14. **The routing frontier of §7.11 is swept on one graph.** `CN ≤ τ` for
    `τ = 0, 1, 2` plus "always k = 2" is measured on Cora with `seal` and `pencil`
    (two seeds); the completion reach is a structural count, so it transfers to any
    graph where the CN prior is the ranking rule, but the AUC frontier does not.
15. **The splice of §7.12 is one random neighbour, one draw per pair.** Each pair
    gets a single `w` (features from the graph, plus a zero-feature control), so the
    numbers are a point estimate of "one more evidence path", not an expectation
    over splices; the sign saturation and the chance-level AUC are large enough
    effects that the draw variance cannot explain them, but a distribution over
    `w` would be the stronger experiment. Cora only, `seal`/`pencil` only.

## 10. Reproduce / extend

```bash
# the Cora table (~20 min on CPU; SGDiff's fit dominates)
python scripts/run_probe.py --dataset cora --profile cpu --seeds 0 1 2 --device cpu

# the stability check: the same grid on two other graphs
python scripts/run_probe.py --dataset citeseer --profile cpu --seeds 0 1 2 --device cpu
python scripts/run_probe.py --dataset usair --profile cpu --seeds 0 1 2 --device cpu

# cheap passes only: graph statistics + sub-graph availability, no training
python scripts/run_probe.py --dataset cora --seeds 0 1 2 --device cpu --diagnostics

# layer-level attribution, operator substitution, edge kinds, repairs (~13 min/2 seeds)
python scripts/run_deep_probe.py --dataset cora --profile cpu --seeds 0 1 --device cpu
python scripts/run_deep_probe.py --dataset citeseer --profile cpu --seeds 0 1 --device cpu
python scripts/run_deep_probe.py --dataset usair --profile cpu --seeds 0 1 --device cpu

# per-pair score dumps (~7 min/seed: SGDiff's fit dominates) then the analyses
python scripts/run_score_dump.py --dataset cora --seeds 0 1
python scripts/run_score_dump.py --dataset citeseer --seeds 0 1
python scripts/run_score_dump.py --dataset usair --seeds 0 1
python scripts/run_operating_point.py --dataset cora citeseer usair
python scripts/run_evidence_calibration.py --dataset cora citeseer usair
python scripts/run_adaptive_attack.py --dataset cora --models gcn sage --budgets 0.05 0.10
python scripts/run_adaptive_attack.py --dataset cora --models seal pencil --budgets 0.05 --pool 8 --pairs 100
python scripts/run_score_dump.py --dataset cora citeseer usair --seeds 0 1 2 3
python scripts/run_bootstrap_ci.py --dataset cora citeseer usair      # pools the 4 seeds
python scripts/run_gated_repair.py --dataset cora --models seal pencil --seeds 0 1 2 3
python scripts/run_splice_gate.py --dataset cora --models seal pencil --seeds 0 1 2 3

# component-level redesign: retrain one-part-changed GCNs and re-probe (~40 s / 8 seeds)
python scripts/run_redesign.py --dataset cora --seeds 0 1 2 3 4 5 6 7
python scripts/run_redesign.py --dataset citeseer --seeds 0 1 2 3 4 5 6 7
python scripts/run_redesign.py --dataset usair --seeds 0 1 2 3 4 5 6 7

# rebuild the HTML report
python scripts/make_probe_report.py

# a single mechanism question
python scripts/run_probe.py --models sgdiff seal --variants del@50% ins_hub@50% --seeds 0
```
