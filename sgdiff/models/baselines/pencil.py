"""PENCIL-style plain Transformer baseline for sub-graph link prediction.

Reference: *Plain Transformers are Surprisingly Powerful Link Predictors*
(Truong et al., arXiv:2602.01553, ICML 2026).

The paper's point is that an encoder-only Transformer over a sampled local
sub-graph, with **no** hand-crafted structural heuristics and **no** node-ID
embeddings, generalises several sub-graph heuristics implicitly and is a very
strong link predictor.  This implementation keeps the *inputs* of the SEAL /
SGDiff pipeline (node attributes plus the DRNL label, which is needed by every
model in this benchmark) and replaces only the encoder, so that a comparison
against ``SEALBaseline`` and ``SGDiff`` isolates the encoder choice.

Differences from the paper (kept deliberately small, documented in
``docs/benchmark.md``):

* DRNL labels and endpoint role embeddings are still used, because they are part
  of the shared data pipeline; ``use_labels=False`` switches them off and turns
  this into the "features + topology only" variant.
* Read-out is a symmetric pair read-out ``[h_u, h_v, |h_u - h_v|, h_u * h_v]``
  rather than the paper's concatenation of the two endpoint tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MLP, apply_node_mask


class _MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with an optional additive bias / key mask."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self,
        h: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if bias is not None:
            att = att + bias.unsqueeze(1)
        if key_mask is not None:
            att = att.masked_fill(~key_mask[:, None, None, :], torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = torch.matmul(att, v).transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.out(out)


class _Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, heads: int, dropout: float, ff_mult: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )
        self.dropout = dropout

    def forward(self, h, bias=None, key_mask=None):
        h = h + F.dropout(self.attn(self.norm1(h), bias, key_mask), p=self.dropout, training=self.training)
        h = h + F.dropout(self.ff(self.norm2(h)), p=self.dropout, training=self.training)
        return h


class PENCILBaseline(nn.Module):
    """Transformer sub-graph classifier: ``P(y | A, X)`` with full self-attention."""

    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 2,
        attn_mode: str = "full",
        use_labels: bool = True,
    ) -> None:
        super().__init__()
        if attn_mode not in {"full", "topology"}:
            raise ValueError(f"unknown attn_mode '{attn_mode}'")
        self.attn_mode = attn_mode
        self.use_labels = bool(use_labels)
        self.hidden_dim = int(hidden_dim)

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.label_emb = nn.Embedding(num_labels, hidden_dim) if self.use_labels else None
        self.role_emb = nn.Embedding(3, hidden_dim)  # 0 = other, 1 = source, 2 = destination
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [_Block(hidden_dim, heads, dropout, ff_mult) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = MLP(4 * hidden_dim, hidden_dim, 2, 2, dropout)

    # ------------------------------------------------------------------ helpers
    def _token_embedding(self, batch) -> torch.Tensor:
        h = self.feature_proj(batch.x)
        if self.label_emb is not None:
            h = h + self.label_emb(batch.labels.clamp(max=self.label_emb.num_embeddings - 1))
        role = torch.zeros_like(batch.labels)
        if hasattr(batch, "u_idx"):
            rows = torch.arange(batch.u_idx.shape[0], device=h.device)
            role[rows, batch.u_idx] = 1
            role[rows, batch.v_idx] = 2
        h = h + self.role_emb(role)
        return self.in_norm(h)

    def _attention_bias(self, batch, dtype) -> Optional[torch.Tensor]:
        if self.attn_mode != "topology":
            return None
        n = batch.adj.shape[-1]
        keep = batch.adj > 0
        keep = keep | torch.eye(n, device=keep.device, dtype=torch.bool).unsqueeze(0)
        bias = torch.zeros(keep.shape, device=keep.device, dtype=dtype)
        return bias.masked_fill(~keep, torch.finfo(dtype).min)

    def _pair_embeddings(self, h: torch.Tensor, batch) -> torch.Tensor:
        rows = torch.arange(h.shape[0], device=h.device)
        if hasattr(batch, "u_idx"):
            hu = h[rows, batch.u_idx]
            hv = h[rows, batch.v_idx]
        else:  # graceful fallback for hand-built batches
            pooled = (h * batch.mask.unsqueeze(-1)).sum(1) / batch.mask.sum(1, keepdim=True).clamp(min=1)
            hu = hv = pooled
        return torch.cat([hu, hv, (hu - hv).abs(), hu * hv], dim=-1)

    # --------------------------------------------------------------- inference
    def forward(self, batch) -> torch.Tensor:
        h = self._token_embedding(batch)
        bias = self._attention_bias(batch, h.dtype)
        for block in self.blocks:
            h = block(h, bias, batch.mask)
            h = apply_node_mask(h, batch.mask)
        h = self.out_norm(h)
        return self.head(self._pair_embeddings(h, batch))

    def predict_logit(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        logits = self(batch)
        return logits[:, 1] - logits[:, 0]

    def predict_proba(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        return torch.softmax(self(batch), dim=-1)[:, 1]
