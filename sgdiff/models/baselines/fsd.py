"""Label-free FSD stage-1 adaptation + the shared GAE encoder/decoder.

Reference: ssjcode/FSD-CAP a601613, fsdcap.py::FsdCap.stage1 (structural mask).
The author's elementwise power of symmetric-normalized adjacency is retained;
this is not a spectral matrix fractional power. CAP is deliberately excluded:
the official CAP pre-classifier uses additional node class labels.
"""

import numpy as np
import scipy.sparse as sp
import torch

from ...data.feature_mask import observed_nodes
from .node_pair import NodePairLinkPredictor


def propagate_features(x, edge_index, observed, method="fsd", gamma=1.2,
                       iterations=100, lamda=0.1):
    """FSD author stage-1, plain propagation, or zero-fill control (CPU CSR)."""
    source = np.asarray(x, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    out = source.copy()
    out[~observed] = 0
    if method == "none" or observed.all():
        return out
    n = len(out)
    row, col = np.asarray(edge_index)
    adj = sp.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n))

    def operator(edges, exponent):
        r, c = edges.nonzero()
        deg = np.asarray(edges.sum(1)).ravel()
        # r/c index only incident nodes, whose degrees are positive.
        values = (deg[r] ** -0.5 * deg[c] ** -0.5) ** exponent
        weights = np.bincount(r, weights=values, minlength=n)
        values = values / weights[r]
        return sp.csr_matrix((values, (r, c)), shape=(n, n), dtype=np.float32)

    if method == "fp":
        transition = operator(adj, 1.0)
        for _ in range(iterations):
            out = transition @ out
            out[observed] = source[observed]
        return out

    previous = out.copy()
    visited = observed.copy()
    hop, old_unknown = 1, -1
    while True:
        selected = visited.copy()
        for _ in range(hop):
            selected |= np.asarray(adj @ selected.astype(np.float32)).ravel() > 0
        unknown = int((~visited).sum())
        if unknown == old_unknown:
            break
        old_unknown = unknown
        r, c = adj.nonzero()
        keep = selected[r] & selected[c]
        sub_adj = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                                (r[keep], c[keep])), shape=(n, n))
        transition = operator(sub_adj, gamma)
        for _ in range(iterations):
            out = transition @ out
            out[selected] += lamda * previous[selected]
            out[observed] = source[observed]
        previous[selected] = out[selected]
        previous[selected] /= np.linalg.norm(previous[selected])
        visited = selected
        hop += 1
    return out


class FSDGAE(NodePairLinkPredictor):
    """Same encode/score_pairs contract; only the parameter-free input varies."""

    def __init__(self, in_dim, missing_rate=0.0, seed=0, method="fsd", gamma=1.2):
        super().__init__(in_dim, hidden_dim=32, out_dim=16, num_layers=2, dropout=0.0)
        self.missing_rate, self.mask_seed = missing_rate, seed
        self.method, self.gamma = method, gamma
        # Retain tensor identities alongside results: no pointer-reuse ambiguity.
        # Two contexts cover the fixed train/validation inputs without repeating
        # 100 non-learned propagation iterations every optimizer step.
        self._feature_contexts = []

    def encode(self, x, edge_index, num_nodes):
        for old_x, old_edges, filled in self._feature_contexts:
            if old_x is x and old_edges is edge_index:
                return super().encode(filled, edge_index, num_nodes)
        filled = propagate_features(
            x.detach().cpu().numpy(), edge_index.detach().cpu().numpy(),
            observed_nodes(num_nodes, self.missing_rate, self.mask_seed),
            method=self.method, gamma=self.gamma,
        )
        filled = torch.as_tensor(filled, device=x.device, dtype=x.dtype)
        self._feature_contexts = (self._feature_contexts + [(x, edge_index, filled)])[-2:]
        return super().encode(filled, edge_index, num_nodes)
