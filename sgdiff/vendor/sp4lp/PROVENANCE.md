# Vendored SP4LP source

Source: <https://github.com/francescoferrini/sp4lp>

Pinned revision: `1b664cbb91eb0b88f7c99dca20754664f92632d0`.

`models.py`, `paths.py`, and `batching.py` are copied without modification from
`benchmarking/sp4lp/`. The source repository's MIT license is retained here.
Paper: <https://arxiv.org/abs/2507.07138v2>.

The adapter lives in `sgdiff/models/baselines/sp4lp.py`. It selects the authors'
paper-style GCN and Transformer rather than their enhanced options. Width is
64 rather than the released Cora configuration's 256. It uses this project's
saved 80/5/15 splits and random negatives rather than HeaRT splits/negatives.
The historical local investigation used up to 200 epochs. The shared-harness adapter honors the requested node-pair epoch budget, with clean-validation early stopping.
The first BFS shortest path is exact before a maximum 32-node token budget;
longer paths use the authors' endpoint-preserving truncation.

The `sp4lp_endpoints` and `sp4lp_featureless` configurations are independently
trained investigations. They are not results from the SP4LP paper.
