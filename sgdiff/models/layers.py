"""Dense / sparse building blocks used by SGDiff and the baselines.

Sub-graphs are small (tens of nodes), so every encoder operates on the padded
dense mini-batch tensor ``(B, n, n)``; the full-graph baselines use the sparse
helpers at the bottom of this module.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


class MLP(nn.Module):
    """A ReLU MLP with dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionEmbedding(nn.Module):
    """Transformer-style timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        device = steps.device
        half = max(self.dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = steps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


def apply_node_mask(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero out padded nodes."""
    if mask is None:
        return h
    return h * mask.unsqueeze(-1).to(h.dtype)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scores = scores.masked_fill(~mask, NEG_INF)
    weights = torch.softmax(scores, dim=dim)
    return weights * mask.to(weights.dtype)


class DenseGCNLayer(nn.Module):
    """GCN layer: ``H' = A_norm @ H @ W`` (self-loops added)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n = adj.shape[-1]
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        a = adj + eye
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            a = a * pair_mask.to(a.dtype)
        deg = a.sum(-1, keepdim=True).clamp(min=1.0)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = torch.matmul(a / deg, self.linear(h))
        return apply_node_mask(out, mask)


class DenseSAGELayer(nn.Module):
    """GraphSAGE layer (mean aggregator + concatenation)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.dropout = dropout

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(h.dtype)
        else:
            pair_mask = torch.ones_like(adj)
        a = adj * pair_mask
        deg = a.sum(-1, keepdim=True).clamp(min=1.0)
        agg = torch.matmul(a / deg, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.linear(torch.cat([h, agg], dim=-1))
        return apply_node_mask(F.relu(out), mask)


class DenseGATLayer(nn.Module):
    """Multi-head graph attention over the dense adjacency."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.per_head = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.per_head))
        self.att_dst = nn.Parameter(torch.empty(heads, self.per_head))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.dropout = dropout

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, n, _ = h.shape
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.lin(h).view(b, n, self.heads, self.per_head)
        scores = torch.einsum("bnhd,hd->bnh", z, self.att_src)
        scores = scores.unsqueeze(2) + torch.einsum("bmhd,hd->bmh", z, self.att_dst).unsqueeze(1)

        pair_mask = adj > 0
        if mask is not None:
            pair_mask = pair_mask & mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask = pair_mask.unsqueeze(-1).expand(-1, -1, -1, self.heads)
        weights = masked_softmax(scores, pair_mask, dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", weights, z).reshape(b, n, -1)
        return apply_node_mask(out, mask)


class EdgeMessageLayer(nn.Module):
    """Edge-featured message passing on a dense (complete) graph.

    ``e_ij <- f_e([e_ij, h_i, h_j])`` and ``h_i <- f_h([h_i, mean_j e_ij])``.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, edge_dim, 2, dropout)
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim, 2, dropout)
        self.norm_e = nn.LayerNorm(edge_dim)
        self.norm_h = nn.LayerNorm(node_dim)

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        e = self.norm_e(e + self.edge_mlp(torch.cat([e, h_i, h_j], dim=-1)))

        pair_mask = torch.ones_like(e[..., :1])
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(e.dtype)
        agg = (e * pair_mask).sum(dim=2) / pair_mask.sum(dim=2).clamp(min=1.0)
        h = self.norm_h(h + self.node_mlp(torch.cat([h, agg], dim=-1)))
        return apply_node_mask(h, mask), e


def mean_readout(h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked mean pooling over nodes."""
    if mask is None:
        return h.mean(dim=1)
    weights = mask.unsqueeze(-1).to(h.dtype)
    return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


# --------------------------------------------------------------------------- #
# Sparse helpers for full-graph baselines
# --------------------------------------------------------------------------- #
def sparse_normalize(edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True):
    """Return ``(row, col, weight)`` of a symmetrically normalised adjacency."""
    row, col = edge_index[0], edge_index[1]
    if add_self_loops:
        loop = torch.arange(num_nodes, device=row.device)
        row = torch.cat([row, loop])
        col = torch.cat([col, loop])
    ones = torch.ones(row.numel(), device=row.device, dtype=torch.float32)
    deg = torch.zeros(num_nodes, device=row.device).index_add_(0, row, ones).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    weights = deg_inv_sqrt[row] * ones * deg_inv_sqrt[col]
    return row, col, weights


def spmm(
    row: torch.Tensor,
    col: torch.Tensor,
    weights: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sparse-dense product for an adjacency given as COO arrays."""
    out = torch.zeros(num_nodes, x.shape[1], device=x.device, dtype=x.dtype)
    return out.index_add_(0, row, x[col] * weights.unsqueeze(-1))
