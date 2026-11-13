"""Clean categorical structure denoiser (Sec. 3.3 and Appendix A.3).

Time/class embeddings are concatenated and then concatenated to every noisy
node/edge token before projection. Node-edge attention and global updates follow
DiGress's NodeEdgeBlock/Xtoy/Etoy design, with masked statistics for padded
subgraphs and finite statistics for empty/singleton graphs. There is no added
input-to-output skip. Default hidden sizes follow Table 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, SinusoidalPositionEmbedding

LABEL_DIFF_BUCKETS = 8


def _masked_statistics(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DiGress mean/min/max/sample-std pooling restricted to valid tokens."""
    tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
    mask = mask.reshape(mask.shape[0], -1, 1).bool()
    count = mask.sum(1)
    mean = (tokens * mask).sum(1) / count.clamp(min=1)
    minimum = tokens.masked_fill(~mask, torch.inf).amin(1)
    maximum = tokens.masked_fill(~mask, -torch.inf).amax(1)
    minimum = torch.where(count > 0, minimum, torch.zeros_like(minimum))
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    variance = ((tokens - mean.unsqueeze(1)).square() * mask).sum(1) / (count - 1).clamp(min=1)
    std = torch.where(count > 1, (variance + 1e-8).sqrt(), torch.zeros_like(variance))
    return torch.cat([mean, minimum, maximum, std], dim=-1)


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
        global_mlp_dim: int | None = None,
    ) -> None:
        super().__init__()
        if heads < 1 or node_dim % heads:
            raise ValueError("node_dim must be divisible by a positive number of heads")
        self.heads = heads
        self.head_dim = node_dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(node_dim, 3 * node_dim)
        self.out = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_ff_norm = nn.LayerNorm(node_dim)
        self.node_mlp = MLP(node_dim, node_mlp_dim, node_dim, 2, dropout)
        self.edge_mul = nn.Linear(edge_dim, node_dim)
        self.edge_add = nn.Linear(edge_dim, node_dim)
        self.edge_out = nn.Linear(node_dim, edge_dim)
        self.edge_mlp = MLP(edge_dim, edge_mlp_dim, edge_dim, 2, dropout)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_ff_norm = nn.LayerNorm(edge_dim)
        self.film_node = nn.Linear(cond_dim, 2 * node_dim)
        self.film_edge = nn.Linear(cond_dim, 2 * node_dim)
        self.global_in = nn.Linear(cond_dim, cond_dim)
        self.node_global = nn.Linear(4 * node_dim, cond_dim)
        self.edge_global = nn.Linear(4 * edge_dim, cond_dim)
        self.global_out = MLP(cond_dim, cond_dim, cond_dim, 2)
        self.global_mlp = MLP(cond_dim, global_mlp_dim or cond_dim, cond_dim, 2, dropout)
        self.global_norm = nn.LayerNorm(cond_dim)
        self.global_ff_norm = nn.LayerNorm(cond_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e, cond, mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        mask = mask.bool()
        node_mask = mask.unsqueeze(-1)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        edge_mask = pair_mask & ~torch.eye(n, dtype=torch.bool, device=x.device)
        global_delta = self.global_out(
            self.global_in(cond)
            + self.node_global(_masked_statistics(x, mask))
            + self.edge_global(_masked_statistics(e, edge_mask))
        )
        q, k, v = (value * node_mask for value in self.qkv(x).split(d, dim=-1))
        q = q.reshape(b, n, self.heads, self.head_dim).unsqueeze(2)
        k = k.reshape(b, n, self.heads, self.head_dim).unsqueeze(1)
        v = v.reshape(b, n, self.heads, self.head_dim).unsqueeze(1)
        scores = q * k * self.scale
        mul = (self.edge_mul(e) * pair_mask.unsqueeze(-1)).reshape(b, n, n, self.heads, self.head_dim)
        add = (self.edge_add(e) * pair_mask.unsqueeze(-1)).reshape(b, n, n, self.heads, self.head_dim)
        scores = scores * (1 + mul) + add
        key_mask = mask[:, None, :, None, None]
        weights = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min).softmax(dim=2)
        weights = self.dropout(weights * key_mask)
        context = (weights * v).sum(2).flatten(start_dim=2)
        gamma_x, beta_x = self.film_node(cond).unsqueeze(1).chunk(2, dim=-1)
        delta_x = self.out(context * (1 + gamma_x) + beta_x) * node_mask
        x = self.node_norm(x + self.dropout(delta_x))
        x = self.node_ff_norm(x + self.dropout(self.node_mlp(x))) * node_mask
        gamma_e, beta_e = self.film_edge(cond)[:, None, None, :].chunk(2, dim=-1)
        delta_e = self.edge_out(scores.flatten(start_dim=3) * (1 + gamma_e) + beta_e)
        e = self.edge_norm(e + self.dropout(delta_e))
        e = self.edge_ff_norm(e + self.dropout(self.edge_mlp(e)))
        e = 0.5 * (e + e.transpose(1, 2)) * edge_mask.unsqueeze(-1)
        cond = self.global_norm(cond + self.dropout(global_delta))
        cond = self.global_ff_norm(cond + self.dropout(self.global_mlp(cond)))
        return x, e, cond


class TransformerStructureDenoiser(nn.Module):
    """Predict clean DRNL categories and binary-edge log odds."""

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
        self.num_labels = int(num_labels)
        self.label_emb = nn.Embedding(num_labels, node_attn)
        self.node_in = MLP(node_attn + global_attn, node_mlp, node_attn, 2)
        self.edge_in = MLP(2 + global_attn, edge_mlp, edge_attn, 2)
        self.class_emb = nn.Embedding(num_classes, global_attn)
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = MLP(time_dim, global_mlp, global_attn, 2)
        self.condition_in = MLP(2 * global_attn, global_mlp, global_attn, 2)
        self.blocks = nn.ModuleList([
            _TransformerBlock(node_attn, edge_attn, global_attn, heads, node_mlp,
                              edge_mlp, dropout, global_mlp)
            for _ in range(layers)
        ])
        self.node_out = MLP(node_attn, node_mlp, num_labels, 2, dropout)
        self.edge_out = MLP(edge_attn, edge_mlp, 1, 2, dropout)

    def condition(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.condition_in(torch.cat([self.time_mlp(self.time_emb(t)), self.class_emb(y)], dim=-1))

    def forward(self, x_t, labels, mask, cond) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = labels.shape
        mask = mask.bool()
        labels = labels.masked_fill(~mask, 0)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        edge_mask = pair_mask & ~torch.eye(n, dtype=torch.bool, device=x_t.device)
        node_cond = cond.unsqueeze(1).expand(-1, n, -1)
        edge_cond = cond[:, None, None, :].expand(-1, n, n, -1)
        x = self.node_in(torch.cat([self.label_emb(labels), node_cond], dim=-1)) * mask.unsqueeze(-1)
        categorical_edges = F.one_hot(x_t.long(), 2).to(x.dtype)
        e = self.edge_in(torch.cat([categorical_edges, edge_cond], dim=-1)) * edge_mask.unsqueeze(-1)
        for block in self.blocks:
            x, e, cond = block(x, e, cond, mask)
        node_logits = self.node_out(x) * mask.unsqueeze(-1)
        edge_logits = self.edge_out(e).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2)) * edge_mask
        return node_logits, edge_logits
