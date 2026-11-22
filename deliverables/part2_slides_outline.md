# Part 2 slide outline

Audience: course-project group and instructors. The deck is an independent Part 2 module that can be merged into the group presentation.

## Slide 1 — Robust link prediction baselines

- Common Neighbors, GCN, and SGDiff
- Cora, CiteSeer, and PubMed
- Presenter: Jiawei Li

## Slide 2 — Problem and model spectrum

- Task: rank held-out citation links from an observed graph and node attributes.
- CN uses one-hop overlap and has no trainable parameters.
- GCN learns endpoint embeddings by neighborhood aggregation.
- SGDiff compares edge and non-edge hypotheses with structure and feature diffusion.
- Visual: a single left-to-right spectrum from local count to message passing to conditional diffusion.

## Slide 3 — CN and GCN mechanisms

- CN score: common-neighbor intersection size.
- GCN: two 64-dimensional graph-convolution layers, dropout 0.5, dot-product decoder.
- GCN ablations: no neighbors (width-matched MLP) and no attributes (constant node input).
- Team reference: compare CN with Junxiong's full NCNC result on matched Cora/CiteSeer graphs.
- Citation footer: Liben-Nowell and Kleinberg (2007); Kipf and Welling (2017, 2016).

## Slide 4 — SGDiff and stream readouts

- One-hop enclosing subgraph for each candidate pair.
- Structure diffusion and feature diffusion produce class-conditional evidence.
- Learned fusion combines the two streams.
- Readouts: full, structure-only, and feature-only from the same trained checkpoint and diffusion evidence.
- Explicit caveat: five fusion epochs on the local RTX 3060; full training split and all diffusion timesteps.
- Citation footer: Li et al., LoG 2024 / PMLR 269 (2025).

## Slide 5 — Aligned evaluation protocol

- Fixed 80/5/15 positive-edge split and disjoint balanced negatives for seeds 0, 1, and 2.
- Test context uses training and validation positive edges; all validation/test candidates are protected.
- Fourteen conditions: clean; RF, DICE, and EA at 10/25/50%; random deletion, hub deletion, and insertion at 50%; feature shuffle.
- Metrics: AUROC, AP, Hits@100, Precision@100.
- Fairness proof: one pinned evaluator and exact attacked-graph SHA-256 matches.

## Slide 6 — Clean performance

- Native grouped chart: clean AUROC for CN, GCN, and SGDiff across all three datasets.
- Editable table: SGDiff AUROC and its difference from GCN.
- Takeaway: SGDiff leads narrowly on Cora/CiteSeer, while GCN leads on PubMed.

## Slide 7 — Robustness by attack type

- Editable table: RF, DICE, EA, and hub deletion at 50%, plus feature shuffle.
- GCN leads most structural conditions; SGDiff is much more stable under feature shuffle.
- SGDiff has the higher worst-case AUROC across all 13 perturbations on all three datasets.

## Slide 8 — Component ablations

- GCN: full versus no-neighbors versus no-attributes.
- SGDiff: fusion versus structure-only versus feature-only.
- Native grouped charts compare clean AUROC for the three GCN configurations and three SGDiff readouts.
- GCN loses most when attributes are removed; structure is SGDiff's stronger single stream.

## Slide 9 — Conclusions and limitations

- Editable CN-versus-NCNC table on matched Cora and CiteSeer graphs.
- Distinguish CN's exact feature-shuffle invariance from NCNC's much stronger clean and structural-attack ranking.
- State the finalized three-dataset, 882-cell scope.

## Speaker-note requirements

- Each slide receives a 45–75 second plain-language narration.
- Every external method or dataset claim carries a citation in the footer or speaker notes.
- Numerical claims come from `results/part2-peer-cn-gcn-sgdiff-all/per_seed.csv` or the pinned full-NCNC reference extract.
