"""Adapter of the authors' SP4LP implementation to this repository.

The unmodified model, batching and path code are vendored under
sgdiff/vendor/sp4lp; see its provenance file and MIT license.
"""
import numpy as np
import torch
from torch import nn

from ...vendor.sp4lp.models import build_sp4lp
from ...vendor.sp4lp.paths import build_adj, compute_paths, PathStore
from ...vendor.sp4lp.batching import make_path_batch


def path_store(graph, pairs, max_len=32):
    """Exact BFS; hide each query's own edge, including training positives."""
    paths, reachable = compute_paths(
        build_adj(graph.edges, graph.num_nodes), pairs,
        mask_target_edge=True, show_progress=False,
    )
    return PathStore.from_dicts(paths, reachable, max_len=max_len)


class SP4LPBaseline(nn.Module):
    """Authors' paper-style GCN + Transformer, using local-budget width 64.

    ``endpoints_only`` is a separately trained ablation with the same GCN and
    an endpoint MLP. It does not evaluate a full checkpoint with its path off.
    """
    def __init__(self, in_dim, hidden_dim=64, endpoints_only=False, featureless=False):
        super().__init__()
        self.endpoints_only = endpoints_only
        self.featureless = featureless
        self.settings = dict(
            input_dim=1 if featureless else in_dim, gnn_type="gcn", gnn_layers=1,
            hidden_dim=hidden_dim, dropout=0.1, encoder_style="paper",
            seq_model="transformer_paper", seq_layers=1, seq_heads=4,
            path_hidden_dim=128, max_path_len=32, use_distance_encoding=False,
            use_endpoint_flag=False, symmetrise=True, predictor_hidden=hidden_dim,
            predictor_layers=3, pair_op="hadamard", use_len_feature=False,
            fusion="concat",
        )
        self.model = build_sp4lp(self.settings)
        if endpoints_only:
            self.endpoint_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            # Exclude unused path parameters from checkpoints/parameter counts.
            self.model.tokenizer = nn.Identity()
            self.model.phi = nn.Identity()
            self.model.predictor = nn.Identity()
        self.store = None

    def set_context(self, graph, pairs):
        self.store = None if self.endpoints_only else path_store(graph, pairs)

    def encode(self, x, edge_index, num_nodes=None):
        if self.featureless:
            x = torch.ones((len(x), 1), device=x.device, dtype=x.dtype)
        return self.model.encode(x, edge_index)

    def score_pairs(self, z, pairs):
        if self.endpoints_only:
            return self.endpoint_head(z[pairs[:, 0]] * z[pairs[:, 1]]).squeeze(-1)
        if self.store is None:
            raise RuntimeError("set_context(graph, pairs) must precede scoring")
        batch = make_path_batch(self.store, pairs.detach().cpu().numpy(), z.device)
        return self.model.score(z, pairs, batch)
