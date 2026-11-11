"""Conditional discrete diffusion over the sub-graph adjacency (Sec. 3.3).

The adjacency of an enclosing sub-graph is a binary symmetric matrix, so we use
a uniform-state D3PM (Austin et al., 2021) with ``K = 2`` states per edge and
condition the denoiser on the class ``y`` (link / non-link).  Both the training
objective and the likelihood score are the per-step KL

``KL[ q(A^{t-1} | A^t, A^0) || p_theta(A^{t-1} | A^t, y) ]``

which is exactly the term used by Algorithm 1 in the paper's appendix.

Appendix A.3 specifies a *transformer* denoiser conditioned on `time + class`,
trained with the VLB plus the "sum of node and feature cross-entropy loss"
(hybrid D3PM objective).  ``arch="edge_gnn"`` keeps a lighter GNN variant.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EdgeMessageLayer, MLP, SinusoidalPositionEmbedding
from .schedules import UniformDiscreteSchedule
from .transformer_denoiser import LABEL_DIFF_BUCKETS, TransformerStructureDenoiser

EPS = 1e-8


class StructureDiffusionModel(nn.Module):
    """``p_theta(A^{t-1} | A^t, y)`` with a transformer or edge-GNN denoiser."""

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
        arch: str = "edge_gnn",
        node_attn: int = 256,
        edge_attn: int = 64,
        global_attn: int = 64,
        node_mlp: int = 256,
        edge_mlp: int = 128,
        global_mlp: int = 128,
        heads: int = 8,
        aux_ce_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.schedule = UniformDiscreteSchedule(num_steps, beta_start, beta_end)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.normalize_score = bool(normalize_score)
        self.num_mc_samples = int(num_mc_samples or num_steps)
        self.arch = arch
        self.num_labels = int(num_labels)
        self.aux_ce_weight = float(aux_ce_weight)

        if arch == "transformer":
            self.denoiser = TransformerStructureDenoiser(
                num_labels=num_labels,
                num_classes=num_classes,
                node_attn=node_attn,
                edge_attn=edge_attn,
                global_attn=global_attn,
                node_mlp=node_mlp,
                edge_mlp=edge_mlp,
                global_mlp=global_mlp,
                heads=heads,
                layers=num_layers,
                time_dim=max(time_dim, global_attn),
                dropout=dropout,
            )
        elif arch == "edge_gnn":
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
            self.node_out = MLP(hidden_dim, hidden_dim, num_labels, 2, dropout)
        else:
            raise ValueError(f"unknown structure arch '{arch}'")

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_logits, edge_logits)`` of ``p(A^{t-1} | A^t, y)``."""
        if self.arch == "transformer":
            cond = self.denoiser.condition(t, y)
            node_logits, edge_logits = self.denoiser(x_t, labels, mask, cond)
            return node_logits, 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)
        h = self.label_emb(labels) + cond.unsqueeze(1)

        diff = (labels.unsqueeze(2) - labels.unsqueeze(1)).abs().clamp(max=LABEL_DIFF_BUCKETS - 1)
        diff_onehot = F.one_hot(diff, num_classes=LABEL_DIFF_BUCKETS).to(x_t.dtype)
        e = self.edge_in(torch.cat([x_t.unsqueeze(-1), diff_onehot], dim=-1))
        e = e + self.edge_cond(cond).unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            h, e = layer(h, e, mask)
        logits = self.out(e).squeeze(-1)
        return self.node_out(h), 0.5 * (logits + logits.transpose(1, 2))

    def edge_logits(self, x_t, labels, mask, y, t) -> torch.Tensor:
        """Per-edge logits only (used by the likelihood score)."""
        return self.forward(x_t, labels, mask, y, t)[1]

    # ------------------------------------------------------------------- loss
    @staticmethod
    def candidate_mask(mask: torch.Tensor) -> torch.Tensor:
        """Upper-triangular mask over valid nodes, shape ``(B, n, n, 1)``."""
        n = mask.shape[1]
        upper = torch.triu(torch.ones(n, n, device=mask.device, dtype=torch.bool), diagonal=1)
        return (mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper.unsqueeze(0)).unsqueeze(-1)

    def _kl(self, x0, labels, mask, y, t, x_t=None, logits=None) -> torch.Tensor:
        if x_t is None:
            x_t = self.q_sample(x0, t)
        if logits is None:
            logits = self.edge_logits(x_t, labels, mask, y, t)
        pq = self.posterior_state1(x0, x_t, t).unsqueeze(-1)
        # ``log p`` / ``log(1 - p)`` come straight from the logits: writing the
        # KL as a difference of logs is algebraically identical to
        # ``log((pq + eps) / p)`` but stays finite when the sigmoid saturates.
        # Dividing by ``p ~ 1e-8`` instead produces gradients of order 1e8, which
        # ``clip_grad_norm_`` turns into inf and then NaN (observed on NS seed 1).
        log_p = -F.softplus(-logits).unsqueeze(-1)
        log_1mp = -F.softplus(logits).unsqueeze(-1)
        kl = pq * (torch.log(pq + EPS) - log_p) + (1 - pq) * (torch.log(1 - pq + EPS) - log_1mp)

        cmask = self.candidate_mask(mask)
        total = (kl * cmask).sum(dim=(1, 2, 3))
        if self.normalize_score:
            total = total / cmask.sum(dim=(1, 2, 3)).clamp(min=1)
        return total

    def compute_kl(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-graph KL summed over edges (mean-normalised when configured)."""
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (x0.shape[0],), device=x0.device)
        return self._kl(x0, labels, mask, y, t)

    def loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """VLB plus the hybrid node/edge cross-entropy (paper Sec. A.3)."""
        b = x0.shape[0]
        t = torch.randint(1, self.num_steps + 1, (b,), device=x0.device)
        x_t = self.q_sample(x0, t)
        node_logits, edge_logits = self.forward(x_t, labels, mask, y, t)
        loss = self._kl(x0, labels, mask, y, t, x_t=x_t, logits=edge_logits).mean()
        if self.aux_ce_weight <= 0:
            return loss

        cmask = self.candidate_mask(mask)
        denom = cmask.sum().clamp(min=1)
        edge_ce = F.binary_cross_entropy_with_logits(
            edge_logits.unsqueeze(-1), x0.unsqueeze(-1), reduction="none"
        )
        edge_ce = (edge_ce * cmask).sum() / denom
        node_ce = F.cross_entropy(
            node_logits.transpose(1, 2), labels.clamp(min=0), ignore_index=0, reduction="sum"
        ) / mask.sum().clamp(min=1)
        return loss + self.aux_ce_weight * (edge_ce + node_ce)

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
