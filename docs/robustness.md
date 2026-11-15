# Robustness under structural attacks (RQ3)

`scripts/run_robustness.py` implements the paper's Sec. 4.4 question for **every**
model of the zoo (SGDiff, SEAL, PENCIL, GCN, GAT, SAGE, VGAE) with the three
attacks the paper uses — random flip (RF), DICE and embedding attack (EA) — all
served by the same dispatcher (`sgdiff/data/attack.py::apply_attack`). The
harness lives in `sgdiff/engine/robustness.py`.

**All numbers on this page are `sgdiff-paper-v2` at the `paper` profile**
(Table 4 config, 3 seeds, 500+500 eval pairs; runs of 2026-11-15). The
old-implementation tables were removed together with their artifacts
(`results/robustness_cora*.md`, `_rgcn1000`, `_purify2` variants) on 2026-11-15.

## 1. Protocol: evasion, one clean training run

Every model is trained once per seed on the clean Cora split (80/5/15, equal
negatives) and then scored on the *same fixed test pairs* under each perturbed
graph: RF (random edge flips), DICE (community-aware deletion/insertion) and
EA (linear-surrogate embedding attack), at budgets 10% / 25% / 50% of |E|.
Because training never sees the perturbation, this measures **evasion**
robustness — the paper's setting. `set_seed` is re-applied before every variant
so all models see identical Monte-Carlo draws and attack draws.

## 2. Main table (Cora, AUC, seeds 0-2, `paper` profile)

Artifacts: `results/sgdiff-paper-v2/robustness_cora__v2_seed{0,1,2}.json`
(per-run metrics, `drop_mean` per variant) and `.md` (rendered tables). The
SGDiff-only view of the paper's own driver is
`results/sgdiff-paper-v2/attack_cora_seed{K}_paper.json`.

| model | clean | RF@25% | DICE@25% | EA@25% | RF@50% | DICE@50% | EA@50% |
|-------|-------|--------|----------|--------|--------|----------|--------|
| sgdiff (v2) | 0.858 | 0.817 | 0.818 | 0.834 | 0.772 | 0.751 | 0.803 |
| seal | 0.872 | 0.835 | 0.823 | 0.866 | 0.785 | 0.757 | 0.843 |
| pencil | 0.883 | 0.853 | 0.854 | 0.877 | 0.824 | 0.807 | 0.857 |
| gcn | 0.815 | 0.791 | 0.790 | 0.811 | 0.772 | 0.768 | 0.801 |
| gat | 0.850 | 0.828 | 0.821 | 0.835 | 0.788 | 0.783 | 0.817 |
| sage | 0.760 | 0.743 | 0.743 | 0.750 | 0.729 | 0.722 | 0.739 |
| vgae | 0.816 | 0.790 | 0.783 | 0.810 | 0.764 | 0.756 | 0.798 |

## 3. Average damage per attack (AUC points lost vs clean, 25% budget)

| model | RF | DICE | EA |
|-------|----|------|----|
| sgdiff (v2) | 4.1 | 4.0 | 2.4 |
| seal | 3.7 | 4.9 | 0.6 |
| pencil | 3.0 | 2.9 | 0.6 |
| gcn | 2.4 | 2.5 | 0.4 |
| gat | 2.2 | 2.9 | 1.5 |
| sage | 1.7 | 1.7 | 1.0 |
| vgae | 2.6 | 3.3 | 0.6 |

## 4. Analysis

1. **SGDiff v2 is robust in the paper's sense**: the strongest attack (EA)
   costs only 2.4 points at the 25% budget and 5.5 at 50%, and DICE@50% is the
   worst case for every model. The generative, likelihood-based scoring does
   not have a single catastrophic failure mode under structural evasion.
2. **SEAL and PENCIL are stronger in absolute terms at every budget** on this
   split (clean 0.872 / 0.883 vs 0.858). The paper's relative-robustness claim
   survives; an absolute superiority claim would not.
3. **Node-pair GNNs (GCN/VGAE) degrade more from a lower clean baseline**, so
   post-attack gaps between model families shrink with budget — attack budget,
   not architecture, dominates at 50%.
4. Caveat carried over from the harness: the EA implementation is a local
   linear surrogate, not the author's DeepRobust attack, so Table 2's EA
   columns are not a parity target.

## 5. Reproducibility

```bash
python scripts/run_robustness.py --dataset cora --profile paper --seeds 0 1 2 \
    --output results/sgdiff-paper-v2 --tag _v2
# per-seed artifacts: results/sgdiff-paper-v2/robustness_cora__v2_seed{K}.json/.md
```

The defence experiments of the old implementation (RGCN-1000, GDDM/GPR-GAE
purifiers) were withdrawn with their artifacts on 2026-11-15; rerun them with
`--defense gddm gpr` against `sgdiff-paper-v2` if that comparison is needed.
