"""Conditional discrete diffusion over the sub-graph adjacency (Sec. 3.3).

The adjacency of an enclosing sub-graph is a binary symmetric matrix, so we use
a uniform-state D3PM (Austin et al., 2021) with ``K = 2`` states per edge and
condition the denoiser on the class ``y`` (link / non-link).  Both the training
objective and the likelihood score are the per-step KL

``KL[ q(A^{t-1} | A^t, A^0) || p_theta(A^{t-1} | A^t, y) ]``

which is exactly the term used by Algorithm 1 in the paper's appendix.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EdgeMessageLayer, MLP, SinusoidalPositionEmbedding
from .schedules import UniformDiscreteSchedule

LABEL_DIFF_BUCKETS = 8
EPS = 1e-8


class StructureDiffusionModel(nn.Module):
    """``p_theta(A^{t-1} | A^t, y)`` implemented as an edge-featured GNN."""

    def __init__(
        self,
        num_labels: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 16,
        num_classes: int = 2,
        num_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        time_dim: int = 32,
        dropout: float = 0.1,
        normalize_score: bool = True,
        num_mc_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.schedule = UniformDiscreteSchedule(num_steps, beta_start, beta_end)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.normalize_score = bool(normalize_score)
        self.num_mc_samples = int(num_mc_samples or num_steps)

        self.label_emb = nn.Embedding(num_labels, hidden_dim)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = MLP(time_dim, hidden_dim, hidden_dim, 2)
        self.edge_in = nn.Linear(1 + LABEL_DIFF_BUCKETS, edge_dim)
        self.edge_cond = MLP(hidden_dim, hidden_dim, edge_dim, 2)
        self.layers = nn.ModuleList(
            [EdgeMessageLayer(hidden_dim, edge_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.out = MLP(edge_dim, hidden_dim, 1, 2, dropout)

    # ------------------------------------------------------------------ utils
    def _table(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return table.to(t.device)[t - 1].view(-1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample ``A^t`` given ``A^0`` (``t`` is 1-based)."""
        a_bar = self._table(self.schedule.alphas_bar, t)
        flip = (torch.rand_like(x0) < (1.0 - a_bar)).to(x0.dtype)
        return x0 * (1.0 - flip) + (1.0 - x0) * flip

    def posterior_state1(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``q(A^{t-1} = 1 | A^t, A^0)`` for every edge."""
        a_prev = self._table(self.schedule.alphas_bar_prev, t)
        beta = self._table(self.schedule.betas, t)
        p_stay = 1.0 - 0.5 * beta
        p_flip = 0.5 * beta

        same = x0 == x_t
        is_one = x0 > 0.5
        num1 = torch.where(
            is_one,
            torch.where(same, p_stay * a_prev, p_flip * a_prev),
            torch.where(same, p_flip * (1 - a_prev), p_stay * (1 - a_prev)),
        )
        num0 = torch.where(
            is_one,
            torch.where(same, p_flip * (1 - a_prev), p_stay * (1 - a_prev)),
            torch.where(same, p_stay * a_prev, p_flip * a_prev),
        )
        return num1 / (num1 + num0 + EPS)

    # -------------------------------------------------------------- denoiser
    def forward(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Per-edge logits of ``p(A^{t-1} = 1 | A^t, y)``, shape ``(B, n, n)``."""
        cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)
        h = self.label_emb(labels) + cond.unsqueeze(1)

        diff = (labels.unsqueeze(2) - labels.unsqueeze(1)).abs().clamp(max=LABEL_DIFF_BUCKETS - 1)
        diff_onehot = F.one_hot(diff, num_classes=LABEL_DIFF_BUCKETS).to(x_t.dtype)
        e = self.edge_in(torch.cat([x_t.unsqueeze(-1), diff_onehot], dim=-1))
        e = e + self.edge_cond(cond).unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            h, e = layer(h, e, mask)
        logits = self.out(e).squeeze(-1)
        return 0.5 * (logits + logits.transpose(1, 2))

    # ------------------------------------------------------------------- loss
    @staticmethod
    def candidate_mask(mask: torch.Tensor) -> torch.Tensor:
        """Upper-triangular mask over valid nodes, shape ``(B, n, n, 1)``."""
        n = mask.shape[1]
        upper = torch.triu(torch.ones(n, n, device=mask.device, dtype=torch.bool), diagonal=1)
        return (mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper.unsqueeze(0)).unsqueeze(-1)

    def compute_kl(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-graph KL summed over edges (mean-normalised when configured)."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (b,), device=x0.device)
        x_t = self.q_sample(x0, t)
        logits = self.forward(x_t, labels, mask, y, t)
        pq = self.posterior_state1(x0, x_t, t).unsqueeze(-1)
        p = torch.sigmoid(logits).unsqueeze(-1).clamp(EPS, 1 - EPS)
        kl = pq * torch.log((pq + EPS) / p) + (1 - pq) * torch.log((1 - pq + EPS) / (1 - p))

        cmask = self.candidate_mask(mask)
        total = (kl * cmask).sum(dim=(1, 2, 3))
        if self.normalize_score:
            total = total / cmask.sum(dim=(1, 2, 3)).clamp(min=1)
        return total

    def loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Class-conditional ELBO loss (mean over the batch)."""
        return self.compute_kl(x0, labels, mask, y).mean()

    @torch.no_grad()
    def log_likelihood(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Monte-Carlo estimate of ``log p(A | y)`` (higher is better)."""
        num_samples = int(num_samples or self.num_mc_samples)
        steps = torch.linspace(1, self.num_steps, num_samples, device=x0.device).round().long()
        scores = []
        for step in steps:
            t = torch.full((x0.shape[0],), int(step), device=x0.device, dtype=torch.long)
            scores.append(self.compute_kl(x0, labels, mask, y, t))
        return -torch.stack(scores, dim=0).mean(dim=0)
