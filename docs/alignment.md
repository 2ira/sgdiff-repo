# Alignment with the paper

Everything on this page is taken from the PDF in the repository root
(`li25a.pdf`): Table 3 (data), Table 4 (hyper-parameters), Sec. 4.1
(protocol), Sec. A.2-A.4 (appendix). Each row names the config key that
reproduces it, so alignment can be checked mechanically.

## 1. Data (Table 3)

| paper | value | reproduction | config / code |
|-------|-------|--------------|---------------|
| datasets | Cora, Citeseer, Pubmed, Router, USAir, NS | same names | `data.name` |
| Cora | 2 708 nodes / 10 556 directed links | 2 708 / 5 278 undirected (= 10 556 directed) | `sgdiff prepare` |
| Citeseer | 3 327 / 9 228 | 3 327 / 4 552 (9 104 directed) | Planetoid graph dict |
| Pubmed | 19 717 / 88 651 | 19 717 / 44 324 | Planetoid graph dict |
| USAir | 332 / 4 252 | 332 / 2 126 (= 4 252 directed) | networkrepository |
| NS | 1 589 / 5 484 | 1 589 / 2 742 (= 5 484 directed) | KONECT |
| Router | 5 022 / 12 516 | manual, see `data/README.md` | `data/raw/router/` |
| attributes | ✔ Cora/Citeseer/Pubmed, ✘ others | same (`feature_mode: auto`) | `sgdiff/data/subgraph.py` |

The paper's "Edge Number" column counts directed entries. Citeseer differs by
62 undirected links because Planetoid's pickled graph contains 248 self-loops
and 56 asymmetric pairs, which are dropped/ symmetrised here. Pubmed differs by
1.5 links for the same reason.

## 2. Model and diffusion hyper-parameters (Table 4)

| symbol | meaning | paper | config key | value |
|--------|---------|-------|------------|-------|
| `k` | hops of the enclosing sub-graph | 1 (all) | `subgraph.hops` | 1 |
| `ns` | max nodes sampled per hop | Cora −1, Citeseer 20, Pubmed 20, Router 10, USAir 40, NS 5 | `subgraph.max_nodes_per_hop` | `null`, 20, 20, 10, 40, 5 |
| `ha_x` | attention hidden, node | 256 | `structure.node_attn` | 256 |
| `ha_e` | attention hidden, edge | 64 | `structure.edge_attn` | 64 |
| `ha_y` | attention hidden, global condition | 64 | `structure.global_attn` | 64 |
| `hm_x` | MLP hidden, node | 256 | `structure.node_mlp` | 256 |
| `hm_e` | MLP hidden, edge | 128 | `structure.edge_mlp` | 128 |
| `hm_y` | MLP hidden, condition | 128 | `structure.global_mlp` | 128 |
| heads | attention heads | 8 | `structure.heads` | 8 |
| `l_t` | transformer layers | 2 | `structure.num_layers` | 2 |
| `ds_t` | structure diffusion steps | 20/20/10/5/10/20 | `structure.num_steps` | per dataset config |
| `h_g` | feature GCN hidden | 256/256/64/N-A | `feature.hidden_dim` | per dataset config |
| `l_g` | feature GCN layers | 2 | `feature.num_layers` | 2 |
| `ds_g` | feature diffusion steps | 100/100/50/N-A | `feature.num_steps` | per dataset config |
| `p(y)` | class prior | 0.5 | `LikelihoodFusion.log_prior` | uniform |
| `{η1, η2, δ}` | fusion parameters | learned | `LikelihoodFusion` | `eta1`, `eta2`, `delta` |

`feature.enabled: false` for Router / USAir / NS matches the "N/A" cells: those
graphs have no attributes, so SGDIFF degenerates to the structure branch
(`η1 ≡ 0`).

Two implementation details of the appendix are reproduced as well:

* the denoiser is a **transformer** over node/edge/global tokens
  (`sgdiff/models/transformer_denoiser.py`), not a message-passing GNN;
* training uses the VLB **plus the sum of node and edge cross-entropy**
  (`structure.aux_ce_weight = 1.0`, `StructureDiffusionModel.loss`).

## 3. Protocol (Sec. 4.1, A.2-A.4)

| item | paper | code |
|------|-------|------|
| split | 80% / 5% / 15% of existing links | `data.val_ratio=0.05`, `data.test_ratio=0.15` |
| negatives | same number of unconnected pairs | `data.neg_ratio=1.0` |
| metrics | AUC, AP, Hit@100 | `sgdiff/utils/metrics.py` |
| repeats | 10 seeds, mean reported | `scripts/run_experiments.py --seeds 0 … 9` |
| sub-graph source | train/val/test graphs exclude the evaluation links | `engine/factory.py::prepare_datasets` |
| RQ1 transfer | train on one dataset **without node features**, test on all others | `--transfer <src>` (forces `feature.enabled=false`) |
| RQ2 1% | shrink the number of training sub-graphs, keep the graph intact | `data.train_ratio=0.01` |
| RQ3 attacks | random flip (RF) 25%/50%, embedding attack, DICE 0.1-0.5 | `--attack`, `random_perturbation`, `dice_attack` |

## 4. Known deviations

1. **Optimisation hyper-parameters are not in the paper.** Batch size (32),
   Adam learning rates (`1e-3`, fusion `1e-2`), the 50-epoch budget and the
   two-stage schedule are our choices; every one of them is a config key.
2. **Monte-Carlo timesteps.** Algorithm 1 averages over all `ds_t` steps
   (`num_mc_samples: null`). The CPU profile uses 4 samples for speed; this is
   the only change that affects the score definition.
3. **`subgraph.max_nodes_total`.** The paper keeps whole 1-hop neighbourhoods
   (Cora hub nodes have degree > 150), which is intractable on a laptop CPU. The
   CPU profile caps the sub-graph at 40 nodes (`null` = unbounded, paper
   behaviour).
4. **Embedding attack (EA)** requires DeepRobust and is not implemented; RF and
   DICE share its interface.
5. **Router** is not redistributable; see `data/README.md`.
6. **NeoGNN** baseline is not re-implemented.

## 5. How to run

```bash
# paper settings (Table 4, full data, all MC steps) -- needs a GPU
python scripts/run_experiments.py --datasets cora --seeds 0 1 2 --profile paper

# CPU profile: same protocol/architecture, smaller budget
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed \
                                  --seeds 0 1 2 --profile cpu
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.25 0.5
```

Results land in `results/` (`*_seed*.json` per run, `summary_*.json` with
mean/std across seeds).
