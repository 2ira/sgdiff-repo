"""Full-graph baselines: GCN / GAT / SAGE with a dot-product decoder and VGAE.

These reproduce the "encode the whole graph, score node pairs" family of
baselines from the paper (GCN, GAT, SAGE, VGAE).  Message passing is sparse, so
the models scale to Pubmed-sized graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import sparse_normalize, spmm


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over the edges that share the same target node (``scores``: ``(E,)`` or ``(E, H)``)."""
    two_d = scores.dim() == 2
    if two_d:
        index_2d = index.unsqueeze(-1).expand_as(scores)
        max_per_node = torch.full(
            (num_nodes, scores.shape[1]), float("-inf"), device=scores.device, dtype=scores.dtype
        )
        max_per_node = max_per_node.scatter_reduce(0, index_2d, scores, reduce="amax", include_self=True)
        numerator = (scores - max_per_node[index]).exp()
        denom = torch.zeros(
            num_nodes, scores.shape[1], device=scores.device, dtype=scores.dtype
        ).index_add_(0, index, numerator)
        return numerator / denom[index].clamp(min=1e-16)

    max_per_node = torch.full((num_nodes,), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per_node = max_per_node.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    numerator = (scores - max_per_node[index]).exp()
    denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype).index_add_(0, index, numerator)
    return numerator / denom[index].clamp(min=1e-16)


class _SparseEncoder(nn.Module):
    """Shared skeleton: ``num_layers`` message-passing blocks."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 1) + [out_dim]
        self.dims = dims

    def _message(self, x, row, col, weights, num_nodes):
        return spmm(row, col, weights, x, num_nodes)


class GCNEncoder(_SparseEncoder):
    """GCN (Kipf & Welling) with symmetric normalisation."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.lins = nn.ModuleList(
            [nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        )

    def forward_preaggregated(self, x, normalized_adjacency=None):
        """Encode cached ``A_norm X`` (possibly sparse), or ``X`` for an MLP.

        The first aggregation is supplied by the caller. Later layers use the
        same normalized adjacency; passing None disables all propagation.
        This avoids materializing an N-by-N identity feature matrix. Biases
        are added after aggregation, as in ``forward``. No input dropout is
        applied, so caching the first aggregation is exact.
        """
        for i, lin in enumerate(self.lins):
            x = torch.sparse.mm(x, lin.weight.T) if x.is_sparse else x @ lin.weight.T
            if i and normalized_adjacency is not None:
                x = normalized_adjacency @ x
            if lin.bias is not None:
                x = x + lin.bias
            if i < len(self.lins) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, num_nodes):
        row, col, weights = sparse_normalize(edge_index, num_nodes)
        for i, lin in enumerate(self.lins):
            x = self._message(x, row, col, weights, num_nodes)
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class SAGEEncoder(_SparseEncoder):
    """GraphSAGE with a mean aggregator."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.lins = nn.ModuleList(
            [nn.Linear(2 * self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        )

    def forward(self, x, edge_index, num_nodes):
        row, col, weights = sparse_normalize(edge_index, num_nodes, add_self_loops=False)
        for i, lin in enumerate(self.lins):
            agg = self._message(x, row, col, weights, num_nodes)
            x = lin(torch.cat([x, agg], dim=-1))
            if i < len(self.lins) - 1:
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GATEncoder(_SparseEncoder):
    """Multi-head GAT with an edge-wise softmax (sparse implementation)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.heads = heads
        self.head_dim = max(hidden_dim // heads, 1)
        width = self.heads * self.head_dim
        self.lins = nn.ModuleList(
            [nn.Linear(in_dim, width)] + [nn.Linear(width, width) for _ in range(num_layers - 1)]
        )
        self.att_src = nn.ParameterList(
            [nn.Parameter(torch.zeros(heads, self.head_dim)) for _ in range(num_layers)]
        )
        self.att_dst = nn.ParameterList(
            [nn.Parameter(torch.zeros(heads, self.head_dim)) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(width, out_dim)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index[0], edge_index[1]
        for i, lin in enumerate(self.lins):
            z = lin(F.dropout(x, p=self.dropout, training=self.training)).view(
                num_nodes, self.heads, self.head_dim
            )
            e = (z[row] * self.att_src[i]).sum(-1) + (z[col] * self.att_dst[i]).sum(-1)
            alpha = _scatter_softmax(e, col, num_nodes)
            out = torch.zeros(num_nodes, self.heads, self.head_dim, device=z.device, dtype=z.dtype)
            out.index_add_(0, col, z[row] * alpha.unsqueeze(-1))
            x = out.reshape(num_nodes, -1)
            if i < len(self.lins) - 1:
                x = F.relu(x)
        return self.proj(x)


_ENCODERS = {"gcn": GCNEncoder, "sage": SAGEEncoder, "gat": GATEncoder}


class NodePairLinkPredictor(nn.Module):
    """``score(u, v) = <z_u, z_v>`` over full-graph GNN embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        arch: str = "gcn",
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if arch not in _ENCODERS:
            raise ValueError(f"unknown arch '{arch}', expected {sorted(_ENCODERS)}")
        self.arch = arch
        self.encoder = _ENCODERS[arch](
            in_dim, hidden_dim, out_dim, num_layers=num_layers, heads=heads, dropout=dropout
        ) if arch == "gat" else _ENCODERS[arch](
            in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout
        )

    def encode(self, x, edge_index, num_nodes):
        return self.encoder(x, edge_index, num_nodes)

    def score_pairs(self, z, pairs):
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)

    def forward(self, x, edge_index, pairs):
        return self.score_pairs(self.encode(x, edge_index, x.shape[0]), pairs)


class VGAE(nn.Module):
    """Variational graph auto-encoder with an inner-product decoder (Kipf & Welling)."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, latent_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.gcn = nn.Linear(in_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.dropout = dropout

    def encode(self, x, edge_index, num_nodes):
        row, col, weights = sparse_normalize(edge_index, num_nodes)
        h = spmm(row, col, weights, x, num_nodes)
        h = F.relu(self.gcn(F.dropout(h, p=self.dropout, training=self.training)))
        h = spmm(row, col, weights, h, num_nodes)
        return self.mean(h), self.log_std(h)

    def forward(self, x, edge_index, num_nodes):
        mean, log_std = self.encode(x, edge_index, num_nodes)
        std = torch.exp(0.5 * log_std)
        z = mean + std * torch.randn_like(std)
        kl = -0.5 * (1 + 2 * log_std - mean.pow(2) - log_std.exp().pow(2)).sum(dim=1).mean()
        return z, kl

    def decode(self, z, pairs):
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)
