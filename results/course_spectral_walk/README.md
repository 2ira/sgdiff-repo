# Classical spectral and random-walk baselines

These results use the same Cora split, hidden test edges, and candidate pairs for every model.  The representations are rebuilt from each observed inference graph and do not use node labels, raw node attributes, or held-out edge labels.

## Methods

- Katz: five-hop walk sum, `beta = 0.85 / max_degree`.
- Spectral: 32 low-frequency normalised-Laplacian coordinates, cosine score.
- DeepWalk: 64 dimensions, 5 walks/node, walk length 20, window 5, 5 skip-gram epochs.
- Node2Vec: the same budget as DeepWalk, with `p=1, q=2`.
- CN, Adamic-Adar, and preferential attachment are the existing zero-parameter references.

All random-walk runs use one worker and the experiment seed.  Clean results use seeds 0-4; robustness results use seeds 0-2 and the same attacked graph for all models within each seed/condition.

## Clean Cora AUC

| model | mean | SD |
|---|---:|---:|
| Katz | 0.8429 | 0.0178 |
| Spectral | 0.8233 | 0.0285 |
| Node2Vec | 0.7998 | 0.0209 |
| DeepWalk | 0.7979 | 0.0233 |
| Adamic-Adar | 0.7146 | 0.0101 |
| Common Neighbours | 0.7137 | 0.0106 |
| Preferential Attachment | 0.6352 | 0.0225 |

## Robustness summary

Katz remains strongest under both random flips (RF) and DICE.  At the 50% edge-perturbation budget its AUC is `0.768 +/- 0.008` under RF and `0.755 +/- 0.013` under DICE.  Spectral is next (`0.738`/`0.729`), followed by Node2Vec (`0.719`/`0.715`) and DeepWalk (`0.712`/`0.722`).  CN and Adamic-Adar fall to about `0.64-0.65`.  Preferential attachment changes little but starts from a substantially weaker clean score.

Raw per-seed records and complete tables are in the adjacent JSON and Markdown files.
