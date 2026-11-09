"""Transformer denoiser for the discrete structure diffusion (paper Sec. A.3).

The paper trains a *transformer* `phi_theta` over the enclosing sub-graph:
node tokens carry the DRNL label embedding, edge tokens carry the noisy
adjacency, and a global token carries `time + class`.  Sizes follow Table 4 of
the appendix:

===============  ==================================================
attention        node 256, edge 64, condition 64, 8 heads, 2 layers
MLP              node 256, edge 128, condition 128
===============  ==================================================

Edges bias the node attention (DiGress-style), which is what lets the model see
"which pairs are connected" while denoising.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, SinusoidalPositionEmbedding

LABEL_DIFF_BUCKETS = 8


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        cond_dim: int,
        heads: int,
        node_mlp_dim: int,
        edge_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert node_dim % heads == 0, "node_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = node_dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(node_dim, 3 * node_dim)
        self.out = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_mlp = MLP(node_dim, node_mlp_dim, node_dim, 2, dropout)

        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_mlp = MLP(2 * node_dim + edge_dim, edge_mlp_dim, edge_dim, 2, dropout)
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.film_node = nn.Linear(cond_dim, 2 * node_dim)
        self.film_edge = nn.Linear(cond_dim, 2 * edge_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        # ---- node self-attention with an edge bias -------------------------
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        bias = self.edge_bias(e).permute(0, 3, 1, 2)  # (B, H, n, n)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale + bias
        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.einsum("bhij,bhjd->bhid", attn, v).reshape(b, n, -1)
        x = self.node_norm(x + self.dropout(self.out(ctx)))

        gamma, beta = self.film_node(cond).unsqueeze(1).chunk(2, dim=-1)
        x = self.node_norm(x + self.node_mlp(x * (1 + gamma) + beta))

        # ---- edge update ---------------------------------------------------
        edge_in = torch.cat(
            [e, x.unsqueeze(2).expand(b, n, n, x.shape[-1]),
             x.unsqueeze(1).expand(b, n, n, x.shape[-1])],
            dim=-1,
        )
        gamma_e, beta_e = self.film_edge(cond).unsqueeze(1).unsqueeze(1).chunk(2, dim=-1)
        delta = self.edge_mlp(edge_in)
        e = self.edge_norm(e + self.dropout(delta) * (1 + gamma_e) + beta_e)
        return x, e


class TransformerStructureDenoiser(nn.Module):
    """``p_theta(A^{t-1} | A^t, y)`` as a transformer over sub-graph tokens."""

    def __init__(
        self,
        num_labels: int = 17,
        num_classes: int = 2,
        node_attn: int = 256,
        edge_attn: int = 64,
        global_attn: int = 64,
        node_mlp: int = 256,
        edge_mlp: int = 128,
        global_mlp: int = 128,
        heads: int = 8,
        layers: int = 2,
        time_dim: int = 64,
        edge_input_dim: int = LABEL_DIFF_BUCKETS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_labels, node_attn)
        self.edge_in = nn.Linear(1 + edge_input_dim, edge_attn)
        self.class_emb = nn.Embedding(num_classes, global_attn)
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = MLP(time_dim, global_mlp, global_attn, 2)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    node_attn, edge_attn, global_attn, heads, node_mlp, edge_mlp, dropout
                )
                for _ in range(layers)
            ]
        )
        self.node_out = MLP(node_attn, node_mlp, num_labels, 2, dropout)
        self.edge_out = MLP(edge_attn, edge_mlp, 1, 2, dropout)

    def condition(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_emb(t)) + self.class_emb(y)

    def forward(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_logits, edge_logits)``."""
        diff = (labels.unsqueeze(2) - labels.unsqueeze(1)).abs().clamp(
            max=LABEL_DIFF_BUCKETS - 1
        )
        diff_onehot = F.one_hot(diff, num_classes=LABEL_DIFF_BUCKETS).to(x_t.dtype)
        e = self.edge_in(torch.cat([x_t.unsqueeze(-1), diff_onehot], dim=-1))

        x = self.label_emb(labels)
        mask = mask.bool()
        for block in self.blocks:
            x, e = block(x, e, cond, mask)
        node_logits = self.node_out(x)
        edge_logits = self.edge_out(e).squeeze(-1)
        return node_logits, edge_logits
