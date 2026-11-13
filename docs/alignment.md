# Alignment with the paper

The source is the workspace PDF `../li25a.pdf`: Table 3 (data), Table 4
(hyper-parameters), Sec. 4.1 (protocol), Sec. A.2-A.4 (appendix).

**2026-11-13 correction (`sgdiff-paper-v2`):** earlier alignment claims were too
strong. Matching Table 4 dimensions did not imply matching the method. The
categorical node/edge diffusion, posterior parameterization, feature GCN,
shared-noise scoring and training stages have been audited against the equations.
The author's linked repository still only contains a README, so this is an
independent equation-based implementation, not verified source-code parity.
Old result files have not been regenerated and are not paper-faithful evidence.

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

Corrected equation-level mapping:

* Eqs. (2)-(3): categorical noising of DRNL nodes **and** undirected edges,
  clean-S0 prediction, and posterior computed from the transition matrices.
* Appendix A.3: transformer node/edge/global denoiser trained with **node + edge
  cross-entropy** (`structure.loss_mode=ce`). The old claim that A.3 requires
  VLB + CE was incorrect; KL is used for likelihood scoring in Algorithm 1.
* Eqs. (10)-(11): Gaussian noising and the GCN
  `A_hat ReLU(A_hat [X_t, one_hot(y)] W0) W1`. There is no DRNL embedding,
  LayerNorm, residual stack, extra output MLP or ad-hoc identity skip in this
  feature denoiser. `feature.time_dim=0` follows the printed equation.
* Algorithm 1: both class conditions see the **same** noisy graph/attributes at
  each timestep. Fusion uses exactly the inference scoring path, not an
  unrelated one-step training surrogate. Defaults sum node/undirected-edge KL
  and feature squared errors before the timestep average; size-normalized scores
  are explicit `normalize_score=true` variants. Training epsilon MSE averages
  valid elements, for interpretable convergence diagnostics.
  Structural scores implement Algorithm 1's negative-average-posterior-KL
  proxy, not a fully normalized log density: Eq. (4)'s unconditional size prior
  and terminal KL are class-common and omitted because they cancel in log odds
  under the paper's shared fusion weights. No invented class-specific size prior
  is fitted. `t=1` posterior KL already reduces to reconstruction NLL.
* Eq. (13): diffusion pretraining and frozen-denoiser fusion calibration are
  separate phases, with independent validation loss and checkpoints. The paper
  does not specify the optimizer, convergence schedule, or whether later joint
  fine-tuning was used; `train.freeze_diffusion=false` is an explicit variant.
* Validation excludes all validation and test positives; test includes training
  and validation edges but excludes test positives. Negatives are disjoint
  across splits. DRNL uses distances with the opposite endpoint removed.

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
   phase budgets (50 diffusion / 20 fusion epochs) are reproducible choices,
   not reported paper settings. `train.epochs` now counts diffusion epochs;
   `train.finetune_epochs` separately counts fusion epochs. The optimizers use
   constant Adam learning rates; inactive optimizers/schedulers do not advance.
2. **Monte-Carlo timesteps.** Algorithm 1 averages all diffusion timesteps
   (`num_mc_samples: null`). `paper` uses full steps for both training fusion
   and inference; `full` uses 8 and `cpu` 4. These smaller profiles are debugging
   variants, not exact paper metrics. Diffusion beta schedules, exact feature
   preprocessing and tie handling for Hit@100 are not fully specified: cosine
   categorical / linear Gaussian / raw attributes are explicitly recorded
   choices, not claims of author-code parity.
3. **`subgraph.max_nodes_total`.** The paper keeps whole 1-hop neighbourhoods
   (Cora hub nodes have degree > 150), which is intractable on a laptop CPU.
   `full` caps the sub-graph at 60 nodes and `cpu` at 40 (`null` = unbounded,
   paper behaviour).
4. **Embedding attack (EA)** uses a local linear surrogate, not the author's
   DeepRobust implementation. This does not establish Table 2 EA parity.
5. **Router** is not redistributable; see `data/README.md`.
6. **NeoGNN** baseline is not re-implemented.

## 5. How to run

```bash
# paper settings (Table 4, full data, all MC steps, no sub-graph cap) -- needs a GPU
python scripts/run_experiments.py --datasets cora --seeds 0 1 2 --profile paper

# `full`: Table-4 data and hyper-parameters, longer schedule, 8 MC steps and a
# 60-node cap so it also finishes on a laptop CPU
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed \
                                  --seeds 0 1 2 --profile full

# `cpu`: same protocol/architecture, smallest budget (quick sanity runs)
python scripts/run_experiments.py --datasets cora citeseer usair ns pubmed \
                                  --seeds 0 1 2 --profile cpu
python scripts/run_experiments.py --transfer cora --targets citeseer pubmed usair ns
python scripts/run_experiments.py --lowdata cora --seeds 0 1 2
python scripts/run_experiments.py --attack cora --ratios 0.25 0.5
```

Corrected grid results land in `results/sgdiff-paper-v2/` (`*_seed*.json` per run,
`summary_*.json` with mean/std across seeds), and checkpoints in
`runs/sgdiff-paper-v2/`. Never merge them with the earlier results.

Start with one seed on your GPU, then expand to the paper's ten repeats:

```bash
python scripts/run_experiments.py --datasets cora --seeds 0 --profile paper \
  --set train.device=cuda train.batch_size=8 train.eval_batch_size=8
python scripts/run_experiments.py --datasets citeseer pubmed usair ns \
  --seeds 0 1 2 3 4 5 6 7 8 9 --profile paper \
  --set train.device=cuda train.batch_size=8 train.eval_batch_size=8
```

Batch size may be reduced for memory without changing extraction or diffusion
steps. Do not silently enable a node cap or subsample timesteps and call it
`paper`. Epoch counts above remain chosen budgets; inspect separate diffusion
validation loss and fusion validation CE for convergence.

Each run saves `best_pretrain.pt`, `best_finetune.pt`, `best.pt`, `last.pt`,
`resolved_config.json` and `metrics.jsonl`. Resume at an epoch boundary with
optimizer and RNG states, keeping all settings except device/run location/workers
identical (use the same batch-size overrides as the initial run):

```bash
python scripts/run_experiments.py --datasets cora --seeds 0 --profile paper \
  --set train.device=cuda train.batch_size=8 train.eval_batch_size=8 \
  train.resume=runs/sgdiff-paper-v2/cora_seed0_paper/last.pt
```

Checkpoints load with `weights_only=True`; legacy checkpoints are rejected rather
than silently interpreted as the corrected architecture. Moving devices may
change numerical trajectories; exact bitwise equivalence is only tested on CPU.
