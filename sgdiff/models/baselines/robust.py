"""Robust-by-design baselines for the adversarial-robustness comparison.

None of the 2025-2026 papers tracked in ``docs/related_work.md`` offers an
attack-robust *link predictor* that ports to this pipeline: DiffHGLP and
DiffGraph are heterogeneous-graph models, FSD-CAP and GGDIFF address feature
missingness / conditional guidance, and GCCM addresses the conditioning
shortcut.  The robustness literature that the paper's RQ3 competes with is the
GNN-defence line, so this module reproduces its canonical representative.

``RGCN`` (Zhu et al., *Robust Graph Convolutional Networks Against Adversarial
Attacks*, KDD 2019) keeps a Gaussian latent per node: the encoder predicts
``(mu, sigma)``, samples ``z = mu + sigma * eps`` while training, and pays a
variance penalty so that uncertain (attack-prone) nodes stay low-variance.  At
inference it uses ``mu``, which keeps AUC deterministic.

``var_weight=0`` gives the ablation (sampling without the penalty) that isolates
whether the robustness comes from the penalty or from the noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .node_pair import _scatter_softmax


class _AttentionLayer(nn.Module):
    """Edge-attention aggregation normalised over the target node's neighbours."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.att = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        wh = self.lin(h)
        pair = torch.cat([wh[row], wh[col]], dim=-1)
        score = F.leaky_relu(self.att(pair), negative_slope=0.2).squeeze(-1)
        alpha = _scatter_softmax(score, col, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.zeros(num_nodes, wh.shape[1], device=h.device, dtype=h.dtype)
        out.index_add_(0, col, wh[row] * alpha.unsqueeze(-1))
        return F.relu(out)


class RGCN(nn.Module):
    """Bayesian robust GCN with a dot-product link decoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        var_weight: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 1)
        self.layers = nn.ModuleList(
            [_AttentionLayer(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)]
        )
        self.mu_head = nn.Linear(hidden_dim, out_dim)
        self.sigma_head = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.var_weight = float(var_weight)
        self.last_sigma: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        h = x
        for index, layer in enumerate(self.layers):
            h = layer(h, edge_index, num_nodes)
            if index < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4
        if self.training:  # reparameterised sample; deterministic at inference
            z = mu + sigma * torch.randn_like(mu)
        else:
            z = mu
        self.last_sigma = sigma
        return z, sigma

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        return self.forward(x, edge_index, num_nodes)[0]

    def score_pairs(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)
