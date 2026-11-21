# SGDiff — Sub-graph Based Diffusion Model for Link Prediction

An independent, self-contained reproduction of

> **Sub-graph Based Diffusion Model for Link Prediction**
> Hang Li, Wei Jin, Geri Skenderi, Harry Shomer, Wenzhuo Tang, Wenqi Fan, Jiliang Tang
> *Proceedings of the Third Learning on Graphs Conference (LoG 2024), PMLR 269*

The official repository linked from the paper (<https://github.com/hzlihang99/SGDiff>)
only contains a two-line README, so this project re-implements the method from
the paper text: enclosing sub-graphs with DRNL labels, a **discrete (D3PM)
diffusion model over the sub-graph adjacency**, a **Gaussian diffusion model over
node features**, and the **Bayesian fusion** of both class-conditional
likelihoods (Eqs. 12-13).

**Method status (2026-11-15):** `sgdiff-paper-v2` (the corrected implementation:
feature GCN per Eq. 11, categorical node/edge diffusion, shared-noise scoring,
two-stage training) has been re-run end to end on GPU with the `paper` profile
(Table 4 exactly: full train/test splits, uncapped sub-graphs, all MC steps,
10 seeds) on all five datasets, plus the full RQ1 transfer / RQ2 low-data /
RQ3 attack + robustness grids.
In-domain numbers (AUC, 10-seed mean ± std): Cora **0.862 ± 0.008**, Citeseer
**0.833 ± 0.011**, Pubmed **0.902 ± 0.002**, USAir **0.868 ± 0.087**, NS
**0.965 ± 0.007** — see [`docs/reproduction.md`](docs/reproduction.md).
Results live in `results/sgdiff-paper-v2/`, checkpoints in `runs/sgdiff-paper-v2/`.
Historical reports remain visible but are explicitly marked as old-implementation
results and are not paper-faithful evidence.

The core SGDiff implementation uses plain PyTorch. Optional author-model
experiments document their additional PyG/DGL dependencies separately.
CPU is sufficient for equation/regression tests; uncapped Table 4 training should
be run on a GPU. Corrected grid outputs go to `results/sgdiff-paper-v2/` and
checkpoints to `runs/sgdiff-paper-v2/`.

```
pip install -e ".[dev]"
python -m sgdiff prepare --datasets cora usair ns
python -m sgdiff train --config configs/sgdiff_cora.yaml
```

## Alignment with the paper

The implementation follows the PDF (`li25a.pdf`) parameter by parameter; the
full mapping lives in [`docs/alignment.md`](docs/alignment.md).

| paper (Table 4 / Sec. A.3) | this repo |
|----------------------------|-----------|
| hops `k = 1` for every dataset | `subgraph.hops: 1` |
| per-hop sampling `ns` = −1 / 20 / 20 / 10 / 40 / 5 | `subgraph.max_nodes_per_hop` per dataset |
| structure denoiser = **transformer**, 8 heads, 2 layers | `structure.arch: transformer` |
| attention widths 256 / 64 / 64, MLP widths 256 / 128 / 128 | `structure.node_attn`, `edge_attn`, `global_attn`, `*_mlp` |
| structure diffusion steps 20 / 20 / 10 / 5 / 10 / 20 | `structure.num_steps` per dataset |
| feature GCN 256 / 256 / 64, 2 layers, steps 100 / 100 / 50 | `feature.hidden_dim`, `num_layers`, `num_steps` |
| no feature stream for USAir / NS / Router | `feature.enabled: false` |
| train with VLB + node/edge cross-entropy | `structure.aux_ce_weight: 1.0` |
| prior `p(y) = 0.5` | `LikelihoodFusion.log_prior` (uniform) |
| 80 / 5 / 15 split, equal negatives, AUC/AP/Hit@100, 10 seeds | `data.*`, `scripts/run_experiments.py --seeds` |

## Repository layout

The optional [NCNC / LPFormer / Refined-GAE study](experiments/new_link_models/README.md)
adds pinned author-model adapters, protected-target robustness experiments and
retrained component ablations under a documented shared protocol.
Read the [complete measured report (Chinese)](experiments/new_link_models/benchmarks/report.md):
all 14 conditions, component variants, metrics and 756 seed-level evaluations,
including random deletion, degree-weighted deletion and insertion.
The [main input/attack supplement, including PubMed](experiments/new_link_models/benchmarks/repository_protocol/report.md)
adds **81 fresh trainings and 729 evaluations**, with exact shared graph/split
snapshots and checkpoint replay. The protected-target study remains separately archived.

```
sgdiff-repro/
├── sgdiff/                     # installable package (the library)
│   ├── api.py                  # high-level Python API (train/predict/evaluate)
│   ├── config.py               # typed YAML configuration (ExperimentConfig)
│   ├── cli/                    # `sgdiff` command line: main.py + commands.py
│   ├── data/
│   │   ├── datasets.py         # loaders + caches (Planetoid, KONECT, USAir)
│   │   ├── download.py         # mirror-aware downloader
│   │   ├── graph.py            # Graph container, edge dropping
│   │   ├── splits.py           # 80/5/15 link split, equal negatives
│   │   ├── subgraph.py         # enclosing sub-graph + DRNL labels
│   │   └── attack.py           # random flip / insert / delete, DICE, EA
│   ├── models/
│   │   ├── sgdiff.py           # SGDiff = structure + feature + fusion
│   │   ├── structure_diffusion.py   # D3PM over the adjacency (VLB + CE)
│   │   ├── feature_diffusion.py     # DDPM over node features
│   │   ├── transformer_denoiser.py  # paper's denoiser (App. A.3)
│   │   ├── fusion.py           # Bayesian fusion, Eq. 12-13
│   │   ├── layers.py, schedules.py
│   │   └── baselines/          # GCN / GAT / SAGE / VGAE / SEAL / PENCIL / RGCN
│   ├── engine/
│   │   ├── factory.py          # build model/extractor, prepare datasets
│   │   ├── batch.py            # collate sub-graph pairs
│   │   ├── trainer.py          # two-stage loop, checkpoints, evaluation
│   │   ├── evaluator.py        # score_pairs, AUC/AP/Hit@100
│   │   ├── baseline_trainer.py # supervised training for the baselines
│   │   ├── benchmark.py        # data-identical SGDiff-vs-baselines study
│   │   └── ablation.py         # likelihood audit (condition / DRNL / streams)
│   └── utils/                  # metrics, seeding, logging, json IO
├── configs/                    # one YAML per dataset + smoke/transfer/lowdata/attack
├── data/                       # raw / processed data (git-ignored) + README
├── scripts/
│   ├── run_experiments.py      # paper grid + RQ1/RQ2/RQ3 driver (`--profile`)
│   ├── run_baselines.py        # GCN/GAT/SAGE/VGAE/SEAL/PENCIL on the same split
│   ├── run_benchmark.py        # SGDiff vs. everything, one table, fingerprints
│   ├── run_ablation.py         # is the likelihood really class-conditional?
│   ├── collect_results.py      # markdown table / `--merge` mean-std summaries
│   └── run_all_experiments.sh  # RQ1+RQ2+RQ3 end to end (PROFILE, SEEDS)
├── examples/                   # runnable, documented examples (01…04)
├── tests/                      # pytest suite (data, diffusion math, e2e)
├── docs/
│   ├── method.md               # paper equations -> code
│   ├── alignment.md            # Table 3/4 parameter-by-parameter mapping
│   ├── api.md                  # Python + CLI reference
│   ├── reproduction.md         # measured numbers, gaps, known deviations
│   ├── benchmark.md, ablation.md
│   ├── robustness.md           # RQ3: RF / DICE / EA for every model
│   ├── attribution.md          # which property each model reads, and what breaks it
│   └── related_work.md, references.bib
├── checkpoints/                # trained weights (git-ignored)
├── runs/                       # per-run output: metrics.jsonl, best.pt (git-ignored)
└── results/                    # metric JSON/log produced by the runs
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # torch, numpy, scipy, pyyaml, tqdm, pytest
```

Python ≥ 3.9 (tested on 3.14), PyTorch ≥ 2.0. A GPU is optional;
`train.device: auto` picks CUDA / MPS / CPU.

## Data

`python -m sgdiff prepare` downloads and caches every benchmark of the paper:

| dataset | nodes | undirected edges | features | source |
|---------|------:|-----------------:|:--------:|--------|
| Cora | 2708 | 5278 | ✔ | Planetoid |
| Citeseer | 3327 | 4552 | ✔ | Planetoid |
| Pubmed | 19717 | 44324 | ✔ | Planetoid |
| USAir | 332 | 2126 | ✘ | networkrepository (USAir97) |
| NS | 1589 | 2742 | ✘ | KONECT (dimacs10-netscience) |
| Router | 5022 | 12516 | ✘ | SEAL bundle — manual, see `data/README.md` |

Downloads are mirror-aware (`ghproxy.net` fallback) and idempotent; processed
graphs are cached as `.npz`. See `data/README.md` for formats and how to plug in
`Router` or your own graph.

## Interfaces

### Command line (`python -m sgdiff …`)

| command | purpose | key options |
|---------|---------|-------------|
| `prepare` | download + preprocess a graph, cache it as `.npz` | `--datasets cora citeseer …`, `--root`, `--no-download` |
| `train` | train SGDiff on one dataset (writes `runs/<name>/best.pt` + `metrics.jsonl`) | `--config`, `--set KEY=VALUE…`, `--run-dir`, `--max-steps` |
| `evaluate` | score a checkpoint, optionally on an attacked graph | `--checkpoint`, `--dataset`, `--split {train,val,test}`, `--max-pairs`, `--attack {none,random,dice,ea}`, `--attack-mode {flip,insert,delete}`, `--ratio`, `--device`, `--output` |
| `transfer` | zero-shot cross-dataset scoring | `--checkpoint`, `--targets …`, `--root`, `--max-pairs`, `--device`, `--output` |

Every YAML key is reachable from the command line:

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

### Budget profiles

`scripts/run_experiments.py --profile` selects how much of the paper's compute
budget is spent. All three keep the architecture, split and scoring rule; they
differ only in the knobs below.

| profile | training pairs | epochs (pre-train) | MC steps | sub-graph cap | hardware |
|---------|---------------:|-------------------:|---------:|--------------:|----------|
| `paper` | full split | 50 (10) | all `ds_t` | none | GPU (hub 1-hop graphs exceed 150 nodes) |
| `full`  | full split | 20 (4) | 8 | 60 nodes | laptop CPU, 10-30 min/dataset-seed (Pubmed ≈50 min with the 8 000-pair cap) |
| `cpu`   | 2 000 | 8 (3) | 4 | 40 nodes | laptop CPU, 2-10 min/dataset-seed |

### Experiment scripts

| script | purpose | example |
|--------|---------|---------|
| `run_experiments.py` | in-domain grid + RQ1/RQ2/RQ3 | `--datasets cora citeseer usair ns pubmed --seeds 0 1 2 --profile full` |
| — (same script) | RQ1 single-source transfer | `--transfer cora --targets citeseer pubmed usair ns` |
| — (same script) | RQ1 full matrix (every source trained once, both `ns` variants, full test set) | `--transfer-matrix --datasets cora citeseer pubmed usair ns --targets cora citeseer pubmed usair ns` |
| — (same script) | RQ2 1 % sub-graphs | `--lowdata cora --seeds 0 1 2` |
| — (same script) | RQ3 attacks on SGDiff (RF / DICE / EA) | `--attack cora --ratios 0.1 0.25 0.5 --attack-methods rf dice ea` |
| `run_baselines.py` | GCN/GAT/SAGE/VGAE/SEAL/PENCIL on the same split | `--dataset cora --baselines gcn gat sage vgae --epochs 200` |
| `run_robustness.py` | RQ3 for **every** model: RF / DICE / EA at 3 budgets, clean column included | `--dataset cora --profile cpu --seeds 0 1 2` |
| `run_benchmark.py` | SGDiff vs. all baselines on identical data (split + sub-graph fingerprints) | `--dataset cora --profile cpu --seeds 0` |
| `run_ablation.py` | does the likelihood actually depend on the link condition? | `--dataset cora --profile cpu --seed 0` |
| `run_probe.py` | **why** each model is robust or fragile: targeted deletion / insertion / feature-shuffle probes, property heuristics, component ablations | `--dataset cora --profile cpu --seeds 0 1 2` |
| `run_deep_probe.py` | **where** the damage happens: layer-wise linear probes, propagation-operator substitution, triadic-vs-bridge deletion, and repair experiments | `--dataset cora --profile cpu --seeds 0 1` |
| `run_redesign.py` | **is the component causal**: retrain a GCN with one part changed (depth / normalisation / self-loop / input augmentation / robust training) and re-probe it | `--dataset cora --seeds 0 1 2 3 4 5 6 7` |
| `run_score_dump.py` | dumps **per-pair scores** plus per-pair structure for every (model, probe); raw material for the three analyses below | `--dataset cora --seeds 0 1` |
| `run_operating_point.py` | **where** the damage lives: AP / hit@100 / TPR@1%FPR, class-wise score drops, rank shift, and clean-defined strata (CN, degree, evidence lost) | `--dataset cora citeseer usair` |
| `run_adaptive_attack.py` | is the measured damage a **lower bound**: greedy score-feedback edge deletion at a matched budget vs random and hub deletion | `--dataset cora --models gcn sage --budgets 0.05 0.10` |
| `run_evidence_calibration.py` | can the **scoring rule** be fixed without retraining: subtract a degree-preserving null-model prior, λ sweep | `--dataset cora citeseer` |
| `run_bootstrap_ci.py` | paired bootstrap over scored pairs: 95 % intervals and a one-sided p for every probe damage | `--dataset cora citeseer usair` |
| `run_gated_repair.py` | evidence-routed repair: k=1 / k=2 sub-graph models routed per pair, with a `CN ≤ τ` threshold sweep, paired-bootstrap intervals on every arm difference, and a count of how many low-evidence pairs a CN-completion fallback can actually reach | `--dataset cora --models seal pencil --seeds 0 1` |
| `run_splice_gate.py` | counterfactual splice gate: re-score each pair with one synthetic shared neighbour spliced in and gate on the score rise (with a zero-feature control and a quantile frontier) | `--dataset cora --models seal pencil --seeds 0 1` |
| `make_probe_report.py` | one self-contained HTML report for the whole repository (no CDN): section 00 is the reproduction layer, sections 01-15 the probe/attribution chain; every optional section appears when its `results/*.json` exists | (after `run_probe.py` and whichever analyses you ran) |
| `collect_results.py` | markdown table of `results/*.json`; `--merge <profile>` rebuilds the mean/std summary | `--merge full` |
| `run_all_experiments.sh` | RQ1+RQ2+RQ3 end to end | `bash scripts/run_all_experiments.sh cpu "0 1 2"` |

`--merge` exists because `run_experiments.py` rewrites
`results/summary_indomain_<profile>.json` on every invocation: running the
datasets in batches would otherwise leave only the last batch in the summary.

## Quickstart

```bash
# 1. fast smoke test (synthetic graph, ~30 s on CPU)
python -m sgdiff train --config configs/smoke_synthetic.yaml

# 2. train SGDiff on Cora.  The plain config is the full Table-4 run (GPU);
#    the --set flags below reproduce the `cpu` profile, and the run lands in
#    runs/cora-<timestamp>/ (metrics.jsonl + best.pt).
python -m sgdiff train --config configs/sgdiff_cora.yaml --run-dir runs \
  --set data.max_train_pairs=2000 data.max_eval_pairs=500 train.epochs=8 \
        train.pretrain_epochs=3 structure.num_mc_samples=4 \
        feature.num_mc_samples=4 subgraph.max_nodes_total=40

# 2b. the `full` profile used for the results table below (no --set soup needed)
python scripts/run_experiments.py --datasets cora --seeds 0 1 2 --profile full

# 3. evaluate the best checkpoint
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --split test --device cpu

# 4. zero-shot transfer to another dataset
python -m sgdiff transfer --checkpoint runs/cora-*/best.pt --targets citeseer usair

# 5. robustness under structural attacks (random / dice / ea)
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack random --ratio 0.1
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack ea --ratio 0.1

# 6. baselines on the same splits
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal pencil

# 7. the paper's experiment grid (5 datasets x 3 seeds) + RQ1/RQ2/RQ3
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed --seeds 0 1 2 --profile full
python scripts/collect_results.py --merge full      # mean/std summary per dataset
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 --attack-methods rf dice ea

# 7b. RQ3 for every model (SGDiff + all baselines) under the same three attacks
python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2

# 8. data-identical benchmark: SGDiff vs. every baseline, one split, one table
python scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0

# 9. GCCM shortcut check: does the likelihood really depend on the link condition?
python scripts/run_ablation.py --dataset cora --profile cpu --seed 0
python scripts/run_ablation.py --dataset cora --profile full --seed 0 --no-retrain --tag full

# 10. mechanism attribution: targeted probes + component ablations
python scripts/run_probe.py --dataset cora --profile cpu --seeds 0 1 2 --device cpu
python scripts/run_probe.py --dataset citeseer --profile cpu --seeds 0 1 2 --device cpu
python scripts/run_deep_probe.py --dataset cora --profile cpu --seeds 0 1 --device cpu
python scripts/run_deep_probe.py --dataset citeseer --profile cpu --seeds 0 1 --device cpu
python scripts/run_redesign.py --dataset cora --seeds 0 1 2 3 4 5 6 7   # component-level retraining
python scripts/run_redesign.py --dataset citeseer --seeds 0 1 2 3 4 5 6 7
python scripts/run_score_dump.py --dataset cora citeseer usair --seeds 0 1 2 3   # ~10 min/seed (SGDiff)
python scripts/run_operating_point.py --dataset cora citeseer usair
python scripts/run_adaptive_attack.py --dataset cora --models gcn sage --budgets 0.05 0.10
python scripts/run_evidence_calibration.py --dataset cora citeseer
python scripts/run_bootstrap_ci.py --dataset cora citeseer usair
python scripts/run_gated_repair.py --dataset cora --models seal pencil --seeds 0 1 2 3
python scripts/run_splice_gate.py --dataset cora --models seal pencil --seeds 0 1 2 3
python scripts/make_probe_report.py   # -> results/probe_report.html (self-contained)
```

The benchmark writes `results/benchmark_<dataset>.{json,md}` and records
split / sub-graph fingerprints so the "same data" claim is verifiable; see
[`docs/benchmark.md`](docs/benchmark.md). The ablation writes
`results/ablation_<dataset>.{json,md}` and answers whether the likelihood is
really conditioned on the link; see [`docs/ablation.md`](docs/ablation.md).
The 2025-2026 literature that the comparison is built around (GCCM, PENCIL,
CPD-PIFM, KREPE, FSD-CAP, ...) is mapped in
[`docs/related_work.md`](docs/related_work.md) with BibTeX in
[`docs/references.bib`](docs/references.bib).

All hyper-parameters are YAML keys that can be overridden on the command line:

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

## Measured results

One-page synthesis of every claim and every measured number:
[`docs/findings.md`](docs/findings.md). Everything below is `sgdiff-paper-v2`
at the `paper` profile (Table 4 exactly: full splits, uncapped sub-graphs, all
Monte-Carlo steps), 8x NVIDIA H20, 2026-11-14/15. Per-seed sources:
`results/sgdiff-paper-v2/{dataset}_seed{K}_paper.json`.

In-domain link prediction (10 seeds, mean ± std). These are **our** numbers;
the paper's Table 2 has no clean-AUC column to compare against (its
0.884 / 0.871 / 0.951 are SGDiff under RF@25% attack, not clean references):

| dataset | seeds | AUC | AP | Hit@100 |
|---------|------:|----:|---:|--------:|
| Cora | 0-9 | **0.862 ± 0.008** | 0.875 | 0.953 |
| Citeseer | 0-9 | **0.833 ± 0.011** | 0.858 | 0.980 |
| Pubmed | 0-9 | **0.902 ± 0.002** | 0.900 | 0.988 |
| USAir | 0-9 | **0.868 ± 0.087** | 0.883 | 0.962 |
| NS | 0-9 | **0.965 ± 0.007** | 0.974 | 1.000 |

Baselines on the identical Cora split
(`results/sgdiff-paper-v2/baselines_cora.json` + the 3-seed robustness runs):
GCN 0.823 / GAT 0.849 / SAGE 0.743 / VGAE 0.824 / SEAL 0.888 / PENCIL 0.883.
SEAL and PENCIL stay ahead of SGDiff v2 (0.858) on clean AUC; SGDiff leads the
Hit@100 ranking metric and is the most attack-resistant generative scorer
(see [`docs/robustness.md`](docs/robustness.md)).

### Research questions (paper profile)

| RQ | experiment | result |
|----|------------|--------|
| RQ1 | 5 sources × 5 targets (Router unavailable), 10 seeds, full test set, no node features, `source_ns` | e.g. citeseer→cora **0.834**, citeseer→ns **0.957**, usair→pubmed **0.799** (`results/sgdiff-paper-v2/transfer_*.json`) |
| RQ2 | Cora with 1 % of the training sub-graphs, 10 seeds | AUC **0.711 ± 0.092** |
| RQ3 | Cora, SGDiff under RF / DICE / EA @10/25/50 %, 3 seeds | EA@25% **0.834** (clean 0.858); worst case DICE@50% 0.751 |
| RQ3 | Cora, all 7 models × RF / DICE / EA, 3 seeds | `results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.{json,md}`, [`docs/robustness.md`](docs/robustness.md) |

Full matrices, per-seed std, the exact protocol and every known deviation:
[`docs/reproduction.md`](docs/reproduction.md) §0.

### What has been run here

Every experiment in one table — what was run, where the analysis lives — is in
[`docs/experiments.md`](docs/experiments.md). The short version:

| experiment | status | artifact |
|------------|--------|----------|
| in-domain grid, `paper` | done: all 5 datasets × seeds 0-9 | `results/sgdiff-paper-v2/{ds}_seed{K}_paper.json` |
| RQ1 transfer | done: 5 sources × 10 seeds, `source_ns` | `results/sgdiff-paper-v2/transfer_{src}_seed{K}_paper.json` |
| RQ2 low-data | done: Cora × 10 seeds | `results/sgdiff-paper-v2/lowdata_cora_seed{K}_paper.json` |
| RQ3 attack (SGDiff) | done: 3 seeds | `results/sgdiff-paper-v2/attack_cora_seed{K}_paper.json` |
| RQ3 robustness (7 models) | done: 3 seeds | `results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.{json,md}` |
| baselines (Cora) | done | `results/sgdiff-paper-v2/baselines_cora.json` |
| ablation / probe chain | not rerun under v2 (old artifacts deleted 2026-11-15) | `docs/ablation.md`, `docs/attribution.md` |

The runs were dispatched by `scripts/repro_queue.sh` (in-domain) and
`scripts/rq_queue.sh` (RQ grids), which parallelise the paper profile across
8 GPUs with flock-protected job lists.

## Where the paper is under-specified

Chasing fidelity exposed places where the paper does not pin the protocol
down. They are listed because they make "reproduced / not reproduced" partially
undecidable — none of them is a claim that the paper is wrong.

1. **No reference implementation.** The official repository linked from the
   paper is a two-line README, so every detail has to be inferred from prose.
2. **Optimisation is unreported.** Batch size, learning rate, the number of
   epochs, the two-stage schedule and how `{η₁, η₂, δ}` are fitted appear
   nowhere; we chose them (`docs/alignment.md` §4) and they move Cora by several
   points.
3. **No variance, no evaluation-set size.** Sec. 4.1 mentions 10 seeds, but the
   tables report one number each and never state how many test pairs or how the
   negatives were sampled.
4. **RQ1 does not say whether the diagonal is included.** "Test the trained model
   over all six graphs" mixes genuine transfer rows with the in-domain cell
   (source = target). We exclude the diagonal and report it separately.
5. **RQ1's feature handling is not neutral.** SGDIFF/SEAL get no node features,
   VGAE gets "randomly projected + row-normalised" features, and featureless
   graphs get Gaussian random vectors — that tests the structure branch, and a
   random projection cannot align feature semantics across datasets.
6. **"Without re-training" still assumes target knowledge.** Table 4 fixes `ns`
   per dataset. If the target's `ns` is used at test time the transfer is not
   purely zero-shot; the paper does not say which it uses. Our fresh matrix
   uses the strict `source_ns` reading.
7. **Source/Target averages are unweighted.** Averaging over targets of very
   different difficulty yields a number that depends on which datasets happen
   to be in the set; the full matrix is the honest artefact.

### Implementation notes

* **`k = 1` is the paper's setting, not a deviation.** Table 4 sets it for
  every dataset; it explains why 1-hop SEAL is a strong baseline here.
* **The KL term is computed in log-space.** `log((1 - pq) / (1 - sigmoid(z)))`
  is `inf` in float32 once `1 - (1 - 1e-8)` rounds to `0`; a non-finite-step
  guard prevents NaN weights.
* **Negatives are drawn per split without a shared `seen` set**, so a test
  negative can reappear as a train negative (≤ 3.8 % of test negatives on the
  small graphs). The overlap biases AUC upward by a small amount; it is
  recorded and left alone for reproducibility.

## Python API

```python
from sgdiff import api

model, summary, cfg = api.train(config_path="configs/sgdiff_cora.yaml",
                                overrides=["train.epochs=20"])
print(summary["test"])                      # {'auc': ..., 'ap': ..., 'hit@100': ...}

graph = api.load_graph("cora")
scores = api.predict(model, graph, [(0, 12), (7, 99)], cfg)   # higher = more likely
metrics = api.evaluate(model, graph, pos_pairs, neg_pairs, cfg)
```

Full reference: `docs/api.md`. How the paper's equations map onto the code:
`docs/method.md`.

## Tests

```bash
python -m pytest tests/ -q     # 52 tests: metrics, sub-graphs, diffusion math, e2e
```

The suite covers the pieces that are easy to get subtly wrong: DRNL/edge hiding,
the closed-form discrete posterior, schedule invariants, the fusion algebra, a
two-epoch end-to-end run with checkpoint round-tripping, the data-identical
benchmark fingerprints, and the likelihood ablation.

## Citation

If you use this code, please cite the original paper (see `CITATION.cff`):

```bibtex
@inproceedings{li2024sgdiff,
  title     = {Sub-graph Based Diffusion Model for Link Prediction},
  author    = {Li, Hang and Jin, Wei and Skenderi, Geri and Shomer, Harry and
               Tang, Wenzhuo and Fan, Wenqi and Tang, Jiliang},
  booktitle = {Proceedings of the Third Learning on Graphs Conference (LoG)},
  volume    = {269},
  year      = {2024}
}
```

## License

MIT (`LICENSE`). This is an independent re-implementation; the datasets keep
their original licences (see `data/README.md`).
## GCN ablation integration

The [GCN input ablations and cross-dataset transfer experiments](experiments/gcn_ablation/README.md) reuse this repository's GCN encoder and pair decoder while preserving their original experimental protocol. See the [overlap and protocol review](docs/gcn_ablation_integration.md) before comparing their archived results with the existing benchmark.
