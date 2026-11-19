# Course baselines, SP4LP and observation missingness

This contribution starts at `a2bcdd7` (SGDiff v2). It does not change any SGDiff model, training, factory, split, extractor or metric implementation. The shared benchmark and attack dispatch add a `course` kind; existing kinds retain their original behavior.

## Methods and installation

```bash
pip install -e '.[dev,investigation]'
python scripts/run_benchmark.py --list-models
```

New registry names: `cn`, `aa`, `pa`, `feature_dot`, `feature_cosine`, `fusion_lr`, `katz`, `spectral`, `deepwalk`, `node2vec`, `sp4lp`, `sp4lp_endpoints`, `sp4lp_featureless`.

CN counts common neighbors; AA downweights high-degree common neighbors; PA multiplies endpoint degrees. Attribute dot/cosine use raw attributes (constant scores when absent). LR fits standardized log1p(CN,AA,PA), signed-log1p(dot), and cosine (signed-log equals log1p for the nonnegative citation features) on clean training queries only, C=1. Its parameter count includes its coefficients/intercept. SP4LP uses the authors' MIT source pinned in `sgdiff/vendor/sp4lp/PROVENANCE.md`, width64 and exact BFS with a32-node path cap. Query edges are hidden from paths. Both ablations are independently retrained. The GNN still aggregates the train graph containing train-positive edges, as with conventional full-graph link prediction.

The course extensions cover three complementary non-neural structure views. `katz` uses the five-hop truncation of the Week-6 all-walk similarity with attenuation `0.85 / max_degree`; since the adjacency spectral radius is at most the maximum degree, this guarantees convergence of the corresponding infinite series. The aligned component study adds `katz_h2`, `katz_h3`, and `katz_decay50`: the first two isolate the value of longer walks, while the last retains five hops but changes the attenuation numerator from `0.85` to `0.50`. `spectral` uses 32 low-frequency normalised-Laplacian coordinates and cosine link scores, following Week 9. `deepwalk` and `node2vec` use 64-dimensional skip-gram embeddings from five 20-step walks per node; Node2Vec fixes `p=1, q=2` for the local/BFS-like setting discussed in Week 8, while DeepWalk is the uniform `p=q=1` case. Both use a single worker and fixed per-run seed. These transductive representations are recomputed on the observed inference graph, as Katz and spectral scores are, and never use edge labels or raw node attributes.

SP4LP is an optional dependency: basic benchmark imports do not import PyG. Its shared-harness trainer uses lr.001, the requested node-pair epoch budget, and clean validation AUROC every5 epochs (or final epoch), patience6 validation checks. Selected weights are frozen under attack. Validation and test use the actual query arrays and context graphs from `prepare_datasets`. Pure feature heuristics intentionally recompute from the observed raw attributes; SP4LP uses the clean standardized feature matrix shared with node models in the attack harness. Path inputs are rebuilt on every attacked/purified graph.

## Existing benchmark and attack entry points

```bash
python scripts/run_benchmark.py --dataset synthetic --profile smoke \
  --models cn aa pa katz spectral deepwalk node2vec feature_dot feature_cosine fusion_lr sp4lp sp4lp_endpoints sp4lp_featureless \
  --node-pair-epochs 2 --device cpu --output results/course_sp4lp_smoke
python scripts/run_robustness.py --dataset synthetic --profile smoke \
  --models cn aa pa katz spectral deepwalk node2vec feature_dot feature_cosine fusion_lr sp4lp sp4lp_endpoints sp4lp_featureless \
  --attacks rf dice --ratios .1 --node-pair-epochs 2 --max-pairs 20 --device cpu \
  --output results/course_sp4lp_smoke --tag integration
```

The dispatch supports the existing EA and purifier entry points as well. RF/DICE smoke runs are integration checks, not scientific model rankings. Legacy pooled `hit@100` retains upstream semantics and tie behavior; AUROC/AP are primary.

## Aligned Part 2 study

The completed SP4LP/Katz component and robustness study is documented in
[`part2_sp4lp_katz_aligned.md`](part2_sp4lp_katz_aligned.md). Its four native
CPU result grids and generated analysis tables live under
`results/part2-aligned/cpu/`. Run `python scripts/analyze_part2_aligned.py` to
revalidate all 840 model/seed/condition cells and regenerate the report.

## Separate observation-missingness entry point

```bash
python scripts/run_observation_robustness.py --datasets synthetic --profile smoke --seeds 0 \
  --conditions clean random_delete_40 degree_delete_40 feature_mask_40 joint_degree_40 \
  --output runs/course_observation_smoke
python scripts/run_observation_robustness.py --datasets cora pubmed --profile local --seeds 0 1 2
python scripts/run_isolation_control.py --root results/observation_robustness_v2/local
```

The new module is `sgdiff.engine.observation_robustness`; it does not replace `sgdiff.engine.robustness`. It intentionally omits SGDiff training/scoring; the existing v2 harness owns that method. Supported models are six course/classic configurations, GCN, VGAE, SEAL-style and three SP4LP configurations. CPU training is used. Optional `--sg-device` remains a compatibility argument for historical manifests and has no SGDiff training effect.

This protocol uses train-only context for all phases, fixed actual queries, frozen clean feature standardization, and new output root `results/observation_robustness_v2`. Model/ratio/feature conditions share the same degraded graph. Structural inputs and paths are recomputed. Low-degree deletion weight is1/(min(d_u,d_v)+1), calculated once on clean context, weighted without replacement with exact deletion count matched to random. Whole-node masking zeroes exactly20/40% of raw attribute rows. Insertions forbid held-out query edges. Predictions, split/query arrays, fingerprints and diagnostic information are saved; checkpoint resume requires identical protocol metadata. Algorithm changes require a new output root.

This separate protocol differs from the upstream attack benchmark's test context. Its outputs must not be combined with `results/sgdiff-paper-v2/` as if obtained under one protocol. The no-new-isolates condition is exploratory and changes other structural properties as well. To compare v2 SGDiff under this protocol, its owner should add a reviewed v2 scoring adapter and regenerate matched results; this branch deliberately does not alter SGDiff.

## Historical results and verification

`experiments/course_sp4lp/historical_2026-11-11/` contains1,152 non-SGDiff metric rows, six saved query/split manifests and selected figures. They are explicitly historical, not new v2 results. Checkpoints, virtual environments and dataset caches are excluded. Synthetic smoke outputs in `results/course_sp4lp_smoke/` are for interface validation only.

Verification details and upstream-environment exceptions are recorded in `experiments/course_sp4lp/VALIDATION.md`.
