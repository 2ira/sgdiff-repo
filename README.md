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

Everything runs on CPU with plain PyTorch — no PyTorch Geometric, no DGL.

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
│   │   └── baselines/          # GCN / GAT / SAGE / VGAE / SEAL / PENCIL
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

In-domain link prediction on this machine, `full` profile, mean ± std over
seeds. Source `results/summary_indomain_full.json` (rebuild with
`python scripts/collect_results.py --merge full`). These are **our** numbers,
not the paper's:

| dataset | seeds | AUC | AP | Hit@100 | paper (clean AUC) |
|---------|------:|----:|---:|--------:|------------------:|
| Cora | 0/1/2 | 0.730 ± 0.030 | 0.794 | 0.973 | 0.884 |
| Citeseer | 0/1/2 | 0.778 ± 0.016 | 0.814 | 0.937 | 0.871 |
| Pubmed | 0 | 0.827 | 0.817 | 0.900 | 0.951 |
| USAir | 0/1/2 | 0.929 ± 0.002 | 0.924 | 0.967 | ~0.95 |
| NS | 0/1/2 | 0.985 ± 0.002 | 0.987 | 1.000 | ~0.97 |

Attribute-free graphs (USAir, NS) match or beat the paper; every dataset with
node attributes stays below it (Cora −15, Citeseer −9, Pubmed −12). The
discriminating factor is therefore the feature stream, not the split, the
sub-graph extraction or `k = 1` — Table 4 sets `k = 1` and we match it, and SEAL
reaches 0.855 on the *same* 1-hop sub-graphs, so the information is present in
our inputs. Remaining deviations, in order of impact: the 60-node sub-graph cap
that replaces the paper's unbounded 1-hop neighbourhoods, the 20-epoch schedule
(Cora validation AUC is still climbing at epoch 19), and 8 of 20 Monte-Carlo
steps at inference. Pubmed adds one of its own: 8 000 of ~62 000 training pairs,
because its full split costs ~6 h per seed here.

Moving from `cpu` to `full` is worth +4.0 AUC on Citeseer and +2.1 on NS,
nothing measurable on USAir, and −2.1 on Cora, where the seed spread
(0.690 / 0.736 / 0.763) is larger than the gap between profiles.
`results/summary_indomain_cpu.json` keeps the cheaper profile for reference.

Baselines on the identical Cora split (`results/baselines_cora_full.json`):

| model | AUC | AP |
|-------|----:|----:|
| GCN | 0.811 | 0.826 |
| GAT | 0.846 | 0.858 |
| SAGE | 0.749 | 0.758 |
| VGAE (`kl_weight=1e-3`) | 0.798 | 0.810 |
| SEAL (1-hop sub-graphs, 20 epochs, `results/baselines_cora_seal.json`) | 0.855 | 0.885 |

SEAL wins on this 1-hop split (0.855 vs SGDiff 0.730): on the paper's `k = 1`
sub-graphs a plain sub-graph GCN classifier is a stronger baseline than the
diffusion model at our budget. SEAL's own protocol (3 000 sub-graphs, 2 hops) is
**not** reproduced here.

### Research questions (measured with the `cpu` profile)

RQ1-RQ3 need one fresh training run per seed/ratio arm, so they keep the cheap
budget; every arm inside a comparison shares it, so the trends are comparable.

| RQ | experiment | result |
|----|------------|--------|
| RQ1 | 5 sources × 5 targets (Router unavailable), 3 seeds, full test set, no node features, diagonal excluded | source avg **0.71-0.84**, target avg **0.67-0.92** (`results/transfer_matrix_cpu.json`) |
| RQ2 | Cora with 1 % of the training sub-graphs (40 epochs) | AUC 0.725 ± 0.058 |
| RQ3 | Cora, SGDiff under random flip 10 % / 25 % / 50 % | AUC 0.692 / 0.655 / 0.579 |
| RQ3 | Cora, SGDiff under DICE 10 % / 25 % / 50 % | AUC 0.685 / 0.665 / 0.578 |
| RQ3 | Cora, SGDiff under EA 10 % / 25 % / 50 % | AUC 0.702 / 0.695 / 0.684 |
| RQ3 | Cora, all 7 models × RF / DICE / EA, 3 seeds (clean 0.737 for SGDiff, 0.828 for GCN) | `results/robustness_cora.{json,md}`, [`docs/robustness.md`](docs/robustness.md) |

RQ1, in one paragraph: transfer never matches the in-domain model — it costs
1-13 points (NS as a target 0.924 vs 0.964; Citeseer 0.673 vs 0.738; Pubmed
0.682-0.738 vs 0.817) — and one cell collapses to chance (Pubmed → NS
`0.494 ± 0.038`). Better in-domain sources transfer *worse* (source averages:
Citeseer 0.838 > Cora 0.824 > USAir 0.759 > Pubmed 0.742 > NS 0.713, close to the
reverse of the in-domain ranking), i.e. small accurate sources overfit their own
degree distribution. Using the target's Table 4 `ns` instead of the source's helps
low-degree targets (USAir +2.6) and hurts Pubmed (−5.6), so "without re-training"
still hides a per-dataset knob. Full matrix and both variants:
`docs/reproduction.md` §4.

`docs/reproduction.md` has the std, the full analysis, the exact protocol and
every known deviation from the paper.

### What has been run here

| experiment | status | artifact |
|------------|--------|----------|
| in-domain grid, `full` | done: Cora/Citeseer/USAir/NS × seeds 0-2, Pubmed × seed 0 | `results/summary_indomain_full.json` |
| in-domain grid, `cpu` | done: all 5 datasets × seeds 0-2 | `results/summary_indomain_cpu.json` |
| in-domain grid, `paper` | not run — needs a GPU (unbounded 1-hop attention) | — |
| baselines (Cora) | done: GCN/GAT/SAGE/VGAE 200 epochs, SEAL 20 epochs, PENCIL in the benchmark | `results/baselines_cora_full.json`, `results/baselines_cora_seal.json` |
| benchmark (identical data) | done: `results/benchmark_*.{json,md}` | `docs/benchmark.md` |
| ablation (likelihood audit) | done: `results/ablation_*.{json,md}` | `docs/ablation.md` |
| RQ1 single-source | done (`cpu`, seed 0, 500 pairs/class) — superseded by the matrix below | `results/transfer_cora_seed0_cpu.json` |
| RQ1 matrix | 5 sources × 5 targets (Router unavailable → 5 of the paper's 6), 3 seeds, full test set, `source_ns` + `target_ns` | `results/transfer_matrix_cpu.json` |
| RQ2 / RQ3 | done, 3 seeds (`cpu`); RQ3 = SGDiff + all 7 models under RF / DICE / EA | `results/summary_lowdata_cora.json`, `results/summary_attack_cora.json`, `results/robustness_cora.{json,md}` |

## Where the paper is under-specified

Our reproduction disagrees with the paper on the three *attributed* graphs
(Cora −15, Citeseer −9, Pubmed −12 AUC points) while matching on the two
featureless ones. Chasing that exposed places where the paper, or its released
artefact, does not pin the protocol down. They are listed here because they make
"reproduced / not reproduced" undecidable rather than merely hard — none of them
is a claim that the paper is wrong.

1. **No reference implementation.** The official repository linked from the
   paper is a two-line README, so every detail below has to be inferred from
   prose.
2. **Optimisation is unreported.** Batch size, learning rate, the number of
   epochs, the two-stage schedule and how `{η₁, η₂, δ}` are fitted appear
   nowhere; we chose them (`docs/alignment.md` §4) and they move Cora by several
   points.
3. **No variance, no evaluation-set size.** Sec. 4.1 mentions 10 seeds, but the
   tables report one number each and never state how many test pairs or how the
   negatives were sampled. Without that, a 2-3 point gap can be called neither a
   reproduction failure nor a success.
4. **RQ1 does not say whether the diagonal is included.** "Test the trained model
   over all six graphs" mixes genuine transfer rows with the in-domain cell
   (source = target). Source/Target averages differ depending on that choice; we
   exclude the diagonal and report it separately.
5. **RQ1's feature handling is not neutral.** SGDIFF/SEAL get no node features,
   VGAE gets "randomly projected + row-normalised" features, and featureless
   graphs get Gaussian random vectors. That design tests whether the *structure*
   branch transfers, not the feature branch — and a random projection cannot
   align feature semantics across datasets, so VGAE's cross-dataset score is
   largely noise and says nothing about its encoder.
6. **"Without re-training" still assumes target knowledge.** Table 4 fixes `ns`
   per dataset. If the target's `ns` is used at test time the transfer is not
   purely zero-shot; the paper does not say which it uses. We report both:
   `results/transfer_matrix_cpu.json` (`source_ns` vs `target_ns`).
7. **Source/Target averages are unweighted.** Averaging over targets of very
   different difficulty (in our runs per-target AUC spans 0.80-0.99) yields a
   number that depends on which datasets happen to be in the set; the full matrix
   is the honest artefact.

### Our side of the same ledger

* **The feature branch never learns.** In every attributed run the DDPM loss sat
  at `1.0000 ± 0.0003` for all 20 epochs, i.e. the denoiser outputs ≈ 0 and the
  feature likelihood is a constant. Cause: `eps` is 1433-dimensional for Cora but
  reaches the output only through a `1433 → 256` projection, so it cannot be
  reconstructed and the trivial predictor is optimal. Adding a `sqrt(1-ā)·x_t`
  skip makes the loss fall (0.74 → 0.69) but Cora AUC *drops* 0.736 → 0.704: the
  branch is not the lever either, it is simply uninformative at this budget. The
  numbers below are measured with the branch present but inert.
* **`k = 1` is the paper's setting, not a deviation.** It explains why we sit
  below SEAL's 2-hop numbers, not why we sit below the paper, which also uses
  `k = 1`.
* **The KL term used to produce `inf`.** `log((1 - pq) / (1 - sigmoid(z)))` is
  `inf` in float32 once `1 - (1 - 1e-8)` rounds to `0`; NS seed 1 turned into NaN
  weights and scored 0.223. Fixed in log-space, with a non-finite-step guard; the
  re-run scores 0.985.
* **Negatives are drawn three times without a shared `seen` set**, so a test
  negative can reappear as a train negative (Cora 1/792, USAir 12/319). The
  overlap is ≤ 3.8 % of the test negatives and it biases AUC *upward*, so it does
  not explain the gap; it is recorded and left alone for reproducibility.

See the bug journal in `docs/reproduction.md` for the full list.

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
