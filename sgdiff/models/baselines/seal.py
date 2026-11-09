"""SEAL-style sub-graph GNN baseline (Zhang & Chen, 2018).

A GCN encoder over the enclosing sub-graph with DRNL node labels and a
sort/mean read-out, trained as a binary classifier.  It is the strongest
discriminative sub-graph baseline in the paper and shares the data pipeline of
SGDiff, which makes it a fair comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import DenseGCNLayer, MLP, apply_node_mask, mean_readout


class SEALBaseline(nn.Module):
    """Sub-graph classifier: ``P(y | A, X)`` with a GCN encoder."""

    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 17,
        hidden_dim: int = 32,
        num_layers: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_labels, hidden_dim)
        self.input_proj = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DenseGCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.head = MLP(2 * hidden_dim, hidden_dim, 2, 2, dropout)

    def forward(self, batch) -> torch.Tensor:
        h = torch.cat([batch.x, self.label_emb(batch.labels)], dim=-1)
        h = self.input_proj(h)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + torch.relu(layer(h, batch.adj, batch.mask)))
            h = apply_node_mask(h, batch.mask)
        pooled = torch.cat([mean_readout(h, batch.mask), _masked_max(h, batch.mask)], dim=-1)
        return self.head(pooled)

    def predict_logit(self, batch) -> torch.Tensor:
        logits = self(batch)
        return logits[:, 1] - logits[:, 0]

    def predict_proba(self, batch) -> torch.Tensor:
        return torch.softmax(self(batch), dim=-1)[:, 1]


def _masked_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
    return out.max(dim=1).values
