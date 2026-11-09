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
│   ├── api.py                  # high-level Python API
│   ├── config.py               # typed YAML configuration
│   ├── data/                   # datasets, splits, sub-graphs, attacks
│   ├── models/                 # SGDiff + baselines
│   │   └── baselines/          # GCN / GAT / SAGE / VGAE / SEAL
│   ├── engine/                 # batching, trainer, evaluator
│   ├── cli/                    # `sgdiff` command line
│   └── utils/                  # metrics, seeding, logging, IO
├── configs/                    # experiment configurations (YAML)
├── data/                       # raw / processed data (git-ignored) + docs
├── scripts/                    # dataset preparation, baselines, full study
├── examples/                   # runnable, documented examples
├── tests/                      # pytest suite (data, diffusion math, e2e)
├── docs/                       # method map, API reference, reproduction notes
├── checkpoints/                # trained weights (git-ignored)
└── results/                    # metric files produced by the runs
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

## Quickstart

```bash
# 1. fast smoke test (synthetic graph, ~30 s on CPU)
python -m sgdiff train --config configs/smoke_synthetic.yaml

# 2. train SGDiff on Cora.  The plain config is the full Table-4 run (GPU);
#    the --set flags below are the CPU profile used for the numbers in this README.
python -m sgdiff train --config configs/sgdiff_cora.yaml --run-dir runs \
  --set data.max_train_pairs=2000 data.max_eval_pairs=500 train.epochs=8 \
        train.pretrain_epochs=3 structure.num_mc_samples=4 \
        feature.num_mc_samples=4 subgraph.max_nodes_total=40

# 3. evaluate the best checkpoint
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --split test --device cpu

# 4. zero-shot transfer to another dataset
python -m sgdiff transfer --checkpoint runs/cora-*/best.pt --targets citeseer usair

# 5. robustness under structural attacks
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack random --ratio 0.1

# 6. baselines on the same splits
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal

# 7. the paper's experiment grid (5 datasets x 3 seeds) + RQ1/RQ2/RQ3
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed --seeds 0 1 2
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5
```

All hyper-parameters are YAML keys that can be overridden on the command line:

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.max_train_pairs=2000 structure.num_steps=10 train.epochs=9
```

## Reproducing the research questions

| RQ | what | how |
|----|------|-----|
| RQ1 | cross-dataset transfer without retraining | `sgdiff transfer` / `configs/transfer_pubmed_to_cora.yaml` |
| RQ2 | generalisation with 1% training data | `configs/lowdata_cora_1pct.yaml` (`data.train_ratio: 0.01`) |
| RQ3 | robustness against graph structure attacks | `sgdiff evaluate --attack random|dice --ratio 0.05…0.2` |
| — | baselines GCN/GAT/SAGE/VGAE/SEAL | `scripts/run_baselines.py` |

`bash scripts/run_all_experiments.sh cora 20` runs the whole grid and writes
JSON/log files into `results/`.

Reference numbers from the CPU budget profile (3 seeds, mean; see
`docs/reproduction.md` for std and the full analysis) — **not** the paper's numbers:

| experiment | AUC | AP | Hit@100 |
|------------|----:|---:|--------:|
| Cora | 0.751 | 0.809 | 0.960 |
| Citeseer | 0.738 | 0.787 | 0.903 |
| Pubmed | 0.816 (2 seeds) | 0.813 | 0.810 |
| USAir | 0.929 | 0.923 | 0.960 |
| NS | 0.964 | 0.973 | 1.000 |

Attribute-free graphs (USAir, NS) land on the paper's numbers; the citation
graphs are 10-13 AUC points below because the CPU profile uses 2 000 training
pairs, 8 epochs and 4 Monte-Carlo steps. The single largest factor is `k = 1`
(Table 4): with `--set subgraph.hops=2` and the same budget Cora reaches 0.86.

Research questions (same budget, `results/`):

| RQ | experiment | AUC |
|----|------------|----:|
| RQ1 | Cora → Pubmed / NS / USAir / Citeseer (zero-shot, no features) | 0.830 / 0.947 / 0.800 / 0.733 |
| RQ2 | Cora with 1% of the training sub-graphs | 0.725 ± 0.058 |
| RQ3 | Cora under random flip 10% / 25% / 50% | 0.698 / 0.651 / 0.577 |
| RQ3 | Cora under DICE 10% / 25% / 50% | 0.692 / 0.643 / 0.584 |

`docs/reproduction.md` lists the full table (including baselines), the exact
protocol and every known deviation from the paper.

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
python -m pytest tests/ -q     # 31 tests: metrics, sub-graphs, diffusion math, e2e
```

The suite covers the pieces that are easy to get subtly wrong: DRNL/edge hiding,
the closed-form discrete posterior, schedule invariants, the fusion algebra, and
a two-epoch end-to-end run with checkpoint round-tripping.

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
