# Related work: link prediction after SGDiff (surveyed 2026-11-10)

This note positions the reproduced paper, *Sub-graph Based Diffusion Model for
Link Prediction* (SGDiff, arXiv:2409.08487, LoG 2024 / PMLR 269), against work
published after it, with an emphasis on **diffusion-based link prediction**.
It also records which follow-up directions the repository already implements
(`docs/benchmark.md`) and which are queued.

Sources: arXiv API (metadata + abstracts, sorted by submission date), OpenAlex
and Semantic Scholar (citation edges), publisher / venue pages. arXiv IDs and
dates are copied from the API responses; venues marked *(unverified)* were only
reported by third-party indexes and should be checked against the camera-ready
before citing.

## 1. What SGDiff claims

| id | claim | where it is tested |
|----|-------|--------------------|
| C1 | structure diffusion transfers across datasets **without retraining** | Sec. 4.2, Table 1 |
| C2 | generative modelling generalises better with **1% of the labels** | Sec. 4.3, Fig. 2 |
| C3 | likelihood-based scoring is **robust to structural attacks** | Sec. 4.4 |
| C4 | sub-graph modelling gives **memory control** and fixes the automorphic-node problem of global generative models | Sec. 2-3 |

The method decomposes `p(y|A, X)` with Bayes' rule into a **discrete (D3PM)
diffusion over the sub-graph adjacency** and a **Gaussian diffusion over node
features**, then fuses the two class-conditional ELBOs with learnable
`{eta_1, eta_2, delta}`.

## 2. Direct successors

**None found.** Citation indexes report ~0 citing works and no paper in the
arXiv corpus extends the sub-graph-diffusion-likelihood formulation: a query for
`(subgraph OR sub-graph) AND diffusion AND "link prediction"` returns exactly
five hits (SGDiff, SaGess, S3GRL, ANAE, FSD-CAP) and none is a successor. Treat
this as "no established follow-up yet" rather than "nobody cites it" — preprint
citation counts lag 6-12 months and several 2026 papers below are not indexed.

Practical consequence: the reproduction has room to be **the** reference
implementation, but every advantage claim now has a 2026 competitor that must be
answered (Sec. 9).

## 3. Line A — diffusion as a discriminative predictor (same paradigm as SGDiff)

| work | id / date | idea | relation |
|------|-----------|------|----------|
| **GCCM: Enhancing Generative Graph Prediction via Contrastive Consistency Model** | 2605.05689, 2026-05 | diagnoses a **shortcut** in conditional diffusion graph predictors: consistency training makes the model return (nearly) the same prediction at every noise level, so the "generative" predictor collapses to a deterministic regressor. Fixes it with batch-wise **negative pairs** in the consistency objective plus input feature perturbation. | The most important read. SGDiff conditions the denoiser on `y` by concatenating it into every node/edge feature; if the condition leaks, `log p(A|y)` degenerates into a classifier and C3/C2 lose their explanation. |
| **DiffusionCom: Structure-Aware Multimodal Diffusion Model for MKGC** | 2504.06543, 2025-04 | replaces conditional-likelihood scoring with a generative view of `p((h,r),(t))`, trained with a **generative + discriminative** loss pair. | shows the fusion step (SGDiff Eq. 12-13) can be replaced by joint training |
| **DDiff** | 2410.18492, 2024-10 | denoising module bolted onto an information-diffusion encoder to resist noise. Not a generative link predictor. | useful as an "anti-noise" component, not as a baseline; the authors flag an error in their Eq. 20 |

## 4. Line B — discrete / masked diffusion, set-level link prediction

The most active 2026 branch moves from "score one node pair" to "generate the
missing relational facts at once". All of these live on knowledge graphs.

| work | id / date | idea |
|------|-----------|------|
| **KREPE** | 2605.24064, 2026-05 *(ICML 2026, unverified)* | masked discrete diffusion over hyper-relational facts; introduces *fact generation* and unifies it with link prediction in one training objective |
| **DiffTSP** | 2604.18344, 2026-04 | conditional discrete diffusion for triple **set** prediction: skips candidate-pair filtering and denoises masked relation states conditioned on supporting sub-graphs (Relational Context Encoder + Relational Graph Diffusion Transformer) |
| **RADD** | 2604.25693, 2026-04 | retrieval-anchored discrete diffusion for multi-modal KGC; retrieval distribution as the probability anchor, coverage-aware inference escapes the candidate set |
| **Observation-anchored latent diffusion** | 2610.39114, 2026-10 | latent-space version of the same "local recovery -> unified generation" move |
| **TAG-DLM** | 2606.31166, 2026-06 | masked **diffusion language model** with topology-aware attention masks over a linearised neighbourhood; link prediction by prompt, **no target-domain fine-tuning** |

TAG-DLM is the direct competitor to C1: it claims cross-dataset transfer through
prompting rather than through a dataset-independent structural prior.

## 5. Line C — diffusion on graph structure (non-KG)

| work | id / date | idea | relation to C1-C4 |
|------|-----------|------|-------------------|
| **DiffGraph** | 2501.02313, WSDM 2025 | latent-space diffusion on heterogeneous graphs with cross-view denoising and semantic transition | robustness-by-denoising claim, heterogeneous setting |
| **DiffHGLP** | Scientific Reports 2026, `s41598-026-74179-y` | diffusion-based generative fusion + relation-aware (attention) guidance for heterogeneous link prediction under noise | closest to C3 outside citation graphs |
| **FSD-CAP** | 2601.18938, 2026-01 *(ICLR, unverified)* | two-stage feature imputation: distance-guided sub-graph expansion + **fractional diffusion** operator; reports link-prediction AUC at 99.5% missing features | direct alternative to SGDiff's feature stream; also a stress test for C2 |
| **SDG: Sequence Diffusion Model for Temporal Link Prediction** | 2601.23233, 2026-01 | sequence-level diffusion over the interaction history of a continuous-time dynamic graph | SGDiff is static-only; this is the temporal extension |
| **GGDiff** | 2025-05 | unified guidance (gradient / control / zeroth-order) for conditional graph generation, with link prediction among the tasks | "guidance" is the natural way to inject `y` without leaking it |

## 6. Line D — non-diffusion generative / energy competitors

| work | id / date | why it matters |
|------|-----------|----------------|
| **CPD-PIFM: Feasible Flow Matching for Graph Reconstruction** | 2609.32980, 2026-09 | flow matching instead of DDPM, with **constraints** (degree, triangle counts, density) enforced during sampling by evolving Lagrange multipliers; evaluated on three link-prediction benchmarks, +11-26 points feasibility. This is the cheapest credible replacement for the DDPM backbone. |
| **FMS: Flow-Modulated Scoring for Semantic-Aware KGC** | 2506.23137, 2025-06 | conditional flow matching used as a *dynamic modulation* of a static scoring function - the KG analogue of "diffusion likelihood as a score" |
| **Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization** | 2610.36204, NeurIPS 2026 | energy-based, three decoupled branches (spectral anchor, mixed-curvature residual, hyperbolic vector field) + N-pair ranking loss; near-doubles P@10% on sparse graphs. Not diffusion, but occupies the same "probabilistic scoring" niche and sets the current sparse-graph bar. |
| **DMNS: Diffusion-based Negative Sampling** | 2306.xxxx / WWW 2024 | diffusion generates negatives with controllable hardness. Orthogonal to SGDiff and directly composable with this codebase's negative sampling. |

## 7. Line E — discriminative SOTA that the baseline table now needs

| work | id / date | idea | in this repo |
|------|-----------|------|--------------|
| **PENCIL: Plain Transformers are Surprisingly Powerful Link Predictors** | 2602.01553, ICML 2026 | encoder-only Transformer over the sampled sub-graph, no hand-crafted heuristics, no node-ID embeddings | **implemented** (`models/baselines/pencil.py`) |
| **SP4LP** | 2025-07 (rev. 2026-11) | GNN encoder + sequence model over the **shortest path** between the endpoints | planned |
| **EO-GNN / Edge-Level Automorphism** | 2026-09-28 | edge orbits + EAR metric, automorphism-aware dropout and orbit-biased sub-graph aggregation | planned: directly re-tests SGDiff's automorphic-node argument (C4) |
| **AtlasULP** | 2026-11-05 | relation-atlas (global + local) descriptors for universal link prediction with domain-guided prompting | planned: competitor to C1 |
| **S3GRL / ELPH / BUDDY** | 2023 | simplified sub-graph message passing, sub-graph sketching, precomputed structure features | S3GRL-style speed reference only |

## 7b. Attack-robustness baselines (for RQ3 comparisons)

None of the 2025-2026 entries above is an *attack-robust* link predictor: DiffHGLP
and DiffGraph are heterogeneous-graph models, FSD-CAP/GGDIFF address feature
missingness and conditional guidance, GCCM addresses the conditioning shortcut.
The robustness literature the paper's Sec. 4.4 competes with is the GNN-defence
line, so two canonical entries are tracked and measured here:

| work | idea | status in this repo |
|------|------|---------------------|
| **RGCN** (Zhu et al., KDD 2019) | Gaussian latent per node, reparameterised sampling, variance penalty | **implemented** (`models/baselines/robust.py`), `rgcn` + `rgcn_novar` ablation |
| **GCN-Jaccard** (Wu et al., 2019, arXiv:1903.01610) | drop edges whose Jaccard similarity is below a threshold before message passing | rejected on measurement: 47% of Cora's edges fall below 0.01, the clean graph would be destroyed |

Measured verdict (`robustness.md` §5, Cora, 3 seeds): RGCN is **not** more robust
than a plain GCN here — −7.2% vs −5.4% AUC under RF-50%, and 7.8 points worse in
absolute terms; the variance penalty changes nothing (`rgcn` ≈ `rgcn_novar`).
SGDiff stays the least robust model of the comparison (−14.6%).

## 8. Surveys

* Shou et al., *Graph diffusion models: a comprehensive survey of methods and applications*, Computer Science Review, 2026 (`10.1016/j.cosrev.2025.100854`).
* *Diffusion-based graph generative methods* (systematises DDPM / SDE / discrete diffusion on graphs).
* *Generative Diffusion Models on Graphs: Methods and Applications* (SMLD / DDPM / SGM taxonomy).

Use them for framing only - none of them discusses likelihood-as-classifier for
link prediction in depth, which is precisely SGDiff's niche.

## 9. Claim-by-claim threat table

| claim | strongest 2026 challenger | experiment that decides it | code |
|-------|---------------------------|----------------------------|------|
| C1 transfer w/o retraining | TAG-DLM (prompt transfer), AtlasULP | same `--transfer` grid, add TAG-DLM / AtlasULP numbers | `scripts/run_experiments.py --transfer` |
| C2 1% labels | GCCM (shortcut diagnosis), FSD-CAP (feature scarcity) | low-data run **plus** a condition-shuffle ablation (below) | `configs/lowdata_cora_1pct.yaml` |
| C3 adversarial robustness | DiffHGLP, DiffGraph | `--attack` with RF / DICE, add a diffusion baseline | `scripts/run_experiments.py --attack` |
| C4 automorphism / memory | EO-GNN, PENCIL, S3GRL | EO-GNN baseline + sub-graph/s throughput at fixed k | `models/baselines/pencil.py`, `docs/benchmark.md` |

**Condition-shuffle ablation (implemented, with results).** `scripts/run_ablation.py`
(see [`ablation.md`](ablation.md)) scores one checkpoint with a perturbed
condition, permuted DRNL labels, a single diffusion step and a single likelihood
stream, then retrains once with a randomized stage-1 condition. A large AUC drop
for `same_condition` means the likelihood genuinely depends on `y`; a flat AUC
across `single_t_*` is the GCCM shortcut - the "diffusion depth" would be
decorative and the score a deterministic regressor in disguise.

Measured on this reproduction (Cora, full training split):

* `same_condition` 0.7179 -> 0.4951: the condition is really used, so C2/C3 are
  not carried by leakage.
* `single_t_*` spread 0.013 with the cheapest step the best: the GCCM-shortcut
  reading applies to the *depth*, not to the conditioning.
* `label_shuffle` 0.7288 vs 0.7179: no effect, so C4's automorphic-node argument
  is not supported at `k = 1`.
* feature stream variance share 0.000 while `eta_1 = 0.61`: the structure term
  carries the whole score, which is the empirical counterpart of the EO-GNN /
  PENCIL challenge to the sub-graph likelihood story.

## 10. Reading order

1. GCCM (2605.05689) - the failure mode of this paradigm.
2. CPD-PIFM (2609.32980) - the cheap generative alternative.
3. PENCIL (2602.01553) - the baseline your table is missing.
4. KREPE / DiffTSP (2605.24064 / 2604.18344) - where the field moved.
5. FSD-CAP (2601.18938), SDG (2601.23233) - the two cheapest extensions of SGDiff itself.

BibTeX for every entry above is in [`references.bib`](references.bib).
