# NCNC, LPFormer and Refined-GAE

**Full Chinese research report:** [PDF](paper_report/report.pdf), [LaTeX source bundle](paper_report/report_sources.zip), [editable TeX](paper_report/report.tex), [BibTeX](paper_report/references.bib), and [model introductions](paper_report/MODEL_INTRO.zh-CN.md). Includes both completed protocol cohorts, original-paper figure excerpts, experiment diagrams, all 1485 evaluations and full numerical appendices.

Three external author architectures under one **new, protected-target evasion
protocol**. This experiment is independent of the older baseline trainer. It
does not overwrite SGDiff v2 results, and its numbers must not be presented as
reproductions of the original models' paper tables.

**EA comparability:** the new EA changes the attack input graph, budget
denominator, surrogate sampling and deletion/insertion allocation relative to
the existing repository implementation. All 18 measured new EA graphs contain
insertions only. These are substantive protocol differences, not just random
seeds, and the two EA result tables must not be directly combined. A separate
[main input/attack supplement including PubMed](benchmarks/repository_protocol/report.md)
is now complete:81 fresh trainings and729 evaluations, with shared graph/split
snapshots and81 independently reloaded checkpoints. It retains explicit author
optimization choices and does not assume identity with historical attack caches.
[Download the exact 18 snapshots, six clean graphs and six original splits](benchmarks/ea_graphs/README.zh-CN.md),
with a detailed comparison, file hashes and a NumPy-only verification script.

For model mechanisms and the meaning of each ablation, see the
[Chinese model guide](MODEL_GUIDE.zh-CN.md).

## Sources and scope

| Model | Author source | Pinned commit | Mechanism |
|---|---|---|---|
| NCNC | [GraphPKU/NeuralCommonNeighbor](https://github.com/GraphPKU/NeuralCommonNeighbor) | `11d597013750da17ce7468e344bec756a7af39a4` | Neural common neighbors plus incomplete-neighborhood completion |
| LPFormer | [HarryShomer/LPFormer](https://github.com/HarryShomer/LPFormer) | `3ca80aba3bd1ba4903ec5c042608e4a034110c9f` | Pairwise attention, PPR position encoding and neighborhood selection |
| Refined-GAE | [GraphPKU/Refined-GAE](https://github.com/GraphPKU/Refined-GAE) | `5bcf00ee42134a31995bbc7bcc5359810aa99b6c` | Learned weighted aggregation of propagation depths and refined decoder |

Author repositories are downloaded separately; their files and licenses remain
external. `sources.lock.json` verifies every pinned Python source file before
model construction. The adapters call the original model classes. Refined-GAE's
DGL GraphConv backend is replaced with the same symmetrically normalized sparse
operator, retaining the original parameters and aggregation/projection order;
tests compare outputs and gradients against DGL in double precision. LPFormer's
original head parameters are used with pre-sigmoid logits for stable BCE.
PPR uses an array implementation of the author's Andersen push. It preserves
the reference's LIFO queue order, thresholds, floating-point update order and
first-visit ordering while replacing linear queue-membership scans with boolean
arrays. Tests require exact equality with the pinned reference at eps 1e-5 and
1e-7, including isolated vertices; the approximation tolerance is unchanged.

## Shared protocol

- Canonical Planetoid Cora and Citeseer, pinned data revision recorded by
  `prepare.py`. Raw float32 node attributes. Class labels are attacker-only
  inputs for DICE, never model inputs or model-selection labels.
- Seeds 0, 1, 2. Positive train/validation/test ratio 80/5/15, fixed balanced
  negatives disjoint across all three splits and absent from the full graph.
- Remove each positive minibatch from message passing and local neighborhoods.
  All models use **512 positive pairs per minibatch**, retaining visible training
  edges in both datasets. The author's Refined-GAE Citeseer batch size 4096 would
  mask all 3641 training positives, leaving only self-loops; using a common batch
  avoids this degenerate context and aligns target-removal fractions across models.
  `settings` retains the original batch sizes as `author_batch_size` provenance.
  LPFormer follows the author's cached full-training-graph PPR during training;
  this PPR cache can retain information about training targets. Validation and
  test targets are never used to build the training PPR.
- Select by clean validation AUC, checking every 5 epochs, at most 100 epochs,
  patience 30 epochs. Validation context contains **training edges only**.
  Test context contains training and validation positives. Test is never used
  for checkpoint/hyperparameter selection.
- Dataset-specific model dimensions, learning rates, dropout, optimizer and
  PPR thresholds follow the author command examples; see `settings()` and
  `adapters.py`. Common batch size 512, balanced fixed negatives, target masking, shared split,
  clipping at 1 and clean-AUC stopping are deliberate shared-protocol changes.
  No hyperparameter search is performed in this first comparison.
- Frozen evasion: RF, DICE and EA at edit fractions 0.1/0.25/0.5 of the **visible
  inference graph's undirected edge count**. Protect all positive and negative
  test candidate pairs, plus validation negatives. Protecting the union does
  not expose candidate labels. Every model uses exactly the same cached attack.
- RF spends half its budget on deletion and the remainder on insertion. DICE
  deletes same-class edges and inserts cross-class edges, using known labels.
  EA is a 60-step linear-surrogate transfer attack on the visible graph, with
  symmetric one-shot gradient ranking and protected candidates; **it is not
  DeepRobust or a white-box attack against any of the three evaluated models**.
  Report realized edit counts and attack graph hashes, not just requested rates.
- LPFormer recomputes author PPR on each distinct attacked inference graph.
  Shared caches use exact edge hashes, atomic writes and process locks.
- AUC/AP are primary. `hits100` is strict global-negative-pool Hits@100;
  `precision100` separately reports pooled precision at 100, averaging ties at
  the cutoff. This avoids the repository's legacy Hit@100 naming ambiguity.

This is a transductive within-dataset study. These architectures have
dataset-specific feature input dimensions; zero-shot cross-dataset transfer
needs an additional agreed feature-alignment protocol and is outside this new
54-configuration matrix. The earlier GCN transfer study remains separate.

## Ablations and probes

Each component variant is trained independently with the same seed and data.

| Model | Retrained variants | What remains |
|---|---|---|
| NCNC | `no_completion`, `no_cn` | Author depth-0 NCN; author CN0 endpoint predictor respectively |
| LPFormer | `uniform_attention`, `no_ppr_encoding` | Same projections and candidates; PE ablation retains PPR selection/counts |
| Refined-GAE | `no_norm`, `dot_decoder` | Norm removal retains dropout; dot decoder retains the encoder |

Four frozen probes run alongside every trained variant: random deletion 50%,
degree-weighted deletion 50% (probability proportional to endpoint degree sum),
insertion 50%, and one shared permutation of node feature rows. These measure
sensitivity and do not replace component retraining. No robustness ordering is
assumed in advance. Compare attacked absolute performance and paired clean-to-
attack drops; lower clean performance can mechanically make a drop look smaller.

The full matrix is 2 datasets × 3 models × 3 seeds × 3 components = **54 training
configurations**, each with 14 evaluations = **756 evaluations**.

## Environment and commands

Use a dedicated Python 3.10 environment with PyTorch and matching PyG extension
wheels, `torch-geometric==2.5.3`, `torch-scatter==2.1.2`,
`torch-sparse==0.6.18`, `dgl==1.1.3` (CPU is sufficient), `numba`, `numpy`,
`scipy`, `scikit-learn`, `matplotlib`, and the repository's regular dependencies.
Run from the repository root (`experiments` is a source-tree namespace).

On an NVIDIA PyTorch image with CXX11 ABI=1, standard PyG ABI=0 wheels do not
load. Build the two extensions against that installed PyTorch instead:

```bash
MAX_JOBS=8 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 \
  python -m pip install --no-build-isolation --no-binary torch-scatter,torch-sparse \
  torch-scatter==2.1.2 torch-sparse==0.6.18
```

`9.0` targets the local H100; change it for another GPU. The NVIDIA base image
used locally also needs `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for its
preinstalled ONNX/protobuf combination. This is environment-specific.

```bash
export DGLBACKEND=pytorch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python -m experiments.new_link_models.sources --root /path/to/author-sources
python -m experiments.new_link_models.prepare --root /path/to/data \
  --existing-data /path/to/existing/gcn_ablation/data
NEW_LINK_SOURCE_ROOT=/path/to/author-sources PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/test_new_link_models.py

# Short functional check; keep its output separate from the formal experiment.
python -m experiments.new_link_models.run --model ncnc --dataset cora \
  --epochs 3 --eval-every 1 --device cuda \
  --data-root /path/to/data --source-root /path/to/author-sources --output /path/to/smoke

python -m experiments.new_link_models.suite --data-root /path/to/data \
  --source-root /path/to/author-sources --output /path/to/formal --jobs 1
python -m experiments.new_link_models.summarize --output /path/to/formal
python -m experiments.new_link_models.audit --output /path/to/formal \
  --data-root /path/to/data --source-root /path/to/author-sources
python -m experiments.new_link_models.diagnostics --output /path/to/formal \
  --data-root /path/to/data
python -m experiments.new_link_models.replay --output /path/to/formal \
  --data-root /path/to/data --source-root /path/to/author-sources
python -m experiments.new_link_models.introspection --output /path/to/formal \
  --data-root /path/to/data --source-root /path/to/author-sources
```

`prepare` can reuse the pinned canonical processed graphs and attributes from
the existing GCN study. Omit `--existing-data` to download and process Cora and
Citeseer directly from the pinned Planetoid revision.

The suite is also the entry point for an ACP GPU job: mount the repository,
data and pinned author directories, install the matching environment, and run
the same command. `run_gpu_job.sh` provides a queue-independent payload:

```bash
NEW_LINK_DATA_ROOT=/shared/data NEW_LINK_SOURCE_ROOT=/shared/authors \
NEW_LINK_OUTPUT_ROOT=/shared/experiment \
  bash experiments/new_link_models/run_gpu_job.sh --seeds 0
```

Separate jobs can use disjoint seeds 0/1/2 with one shared output directory;
never schedule the same configuration concurrently. Run collection/audit after
all shards finish. No provider-specific queue schema is assumed. A 40GB local
GPU is used first; `--jobs 1` limits concurrent memory use. Finished runs resume
without retraining. Configuration/code mismatches fail instead of mixing runs.
Source hashes, data/split hashes, settings, best epoch, predictions, model
weights, peak PyTorch-allocated GPU memory and exact attack statistics remain
in the external output directory. PyTorch sparse CUDA kernels may have small
floating-point nondeterminism despite fixed seeds; seed counts are independent
training trials, not bitwise-repeatability claims.

## Results

The [main input/attack supplement](benchmarks/repository_protocol/report.md) covers
Cora/Citeseer/PubMed and all component variants:81 new trainings,729 evaluations,
81 shared graph snapshots and9 splits. PubMed EA follows main's4000-node limit
and is skipped; clean/RF/DICE are measured. See the
[protocol, author PubMed recipes and commands](REPOSITORY_PROTOCOL.zh-CN.md).
The protected-target archive described below is preserved separately.

The completed 54-configuration / 756-evaluation study is archived in
[benchmarks/report.md](benchmarks/report.md), with a
[Chinese interpretation](benchmarks/FINDINGS.zh-CN.md). It includes three seeds
for each full model and each retrained component on Cora and Citeseer.
The Markdown report expands all 14 conditions, every component and all metrics,
with 252 aggregate rows and all 756 seed-level evaluations. To regenerate only
the report from the measured archive, without model dependencies or training:

```bash
python -m experiments.new_link_models.reporting --output experiments/new_link_models/benchmarks
```

`summarize` recomputes all metrics from saved predictions, verifies that attack
graphs match across models/components, and writes per-seed CSVs, mean/sample-SD
tables, paired component deltas, standalone PNG/PDF robustness plots and an audit.
`diagnostics` adds candidate-stratified results and score/attribute correlations.
`replay` reloads checkpoints and reproduces clean, DICE-50% and shuffled-feature
predictions by default. `introspection` exports learned depth/branch weights and
LPFormer attention entropy, checking that uniform attention and zero-PE behave
as intended on real held-out candidates. Read the report together with this protocol.
Large datasets, PPR caches and trained checkpoints are not committed to Git.
