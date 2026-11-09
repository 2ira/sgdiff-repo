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

# 2. train SGDiff on Cora
python -m sgdiff train --config configs/sgdiff_cora.yaml --run-dir runs

# 3. evaluate the best checkpoint
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --split test --device cpu

# 4. zero-shot transfer to another dataset
python -m sgdiff transfer --checkpoint runs/cora-*/best.pt --targets citeseer usair

# 5. robustness under structural attacks
python -m sgdiff evaluate --checkpoint runs/cora-*/best.pt --attack random --ratio 0.1

# 6. baselines on the same splits
python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal
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

Reference numbers from a CPU-only, deliberately small budget (2 000 training
pairs, 9 epochs) — **not** the paper's numbers:

| experiment | AUC | AP | Hit@100 |
|------------|----:|---:|--------:|
| Cora | 0.860 | 0.885 | 0.96 |
| Citeseer | 0.805 | 0.854 | 0.97 |
| Cora → Citeseer (zero-shot) | 0.834 | 0.857 | 0.73 |
| Cora → USAir (zero-shot) | 0.674 | 0.619 | 0.62 |
| Cora under random attack 10% | 0.889 | 0.914 | — |

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
