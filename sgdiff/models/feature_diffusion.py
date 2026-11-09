"""Conditional Gaussian diffusion over sub-graph node features (Sec. 3.4).

A standard DDPM is trained on the node attribute matrix ``X`` of the enclosing
sub-graph, conditioned on the clean adjacency ``A`` (through a GNN encoder) and
on the class ``y``.  The likelihood proxy used by the fusion step is the
denoising error ``||eps - eps_theta(X^t, A, y)||^2`` averaged over timesteps,
i.e. exactly the ``FeatureScore`` of Algorithm 1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DenseGATLayer, DenseGCNLayer, DenseSAGELayer, MLP, SinusoidalPositionEmbedding
from .schedules import GaussianSchedule

EPS = 1e-8

_DENSE_LAYERS = {"gcn": DenseGCNLayer, "sage": DenseSAGELayer, "gat": DenseGATLayer}


class FeatureDiffusionModel(nn.Module):
    """``eps_theta(X^t, A, y)``: a GNN that predicts the injected Gaussian noise."""

    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_dim: int = 32,
        num_classes: int = 2,
        arch: str = "gcn",
        dropout: float = 0.1,
        num_mc_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        if arch not in _DENSE_LAYERS:
            raise ValueError(f"unknown arch '{arch}', expected one of {sorted(_DENSE_LAYERS)}")
        self.schedule = GaussianSchedule(num_steps, beta_start, beta_end)
        self.num_steps = int(num_steps)
        self.num_mc_samples = int(num_mc_samples or num_steps)
        self.feature_dim = int(feature_dim)

        layer_cls = _DENSE_LAYERS[arch]
        self.label_emb = nn.Embedding(num_labels, hidden_dim)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = MLP(time_dim, hidden_dim, hidden_dim, 2)

        self.input_proj = nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                layer_cls(hidden_dim, hidden_dim, heads=4, dropout=dropout)
                if arch == "gat"
                else layer_cls(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = MLP(hidden_dim, hidden_dim, feature_dim, 2, dropout)

    # ------------------------------------------------------------------ utils
    def _table(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return table.to(t.device)[t - 1].view(-1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Return ``(x_t, eps)`` with ``x_t = sqrt(a_bar) x0 + sqrt(1-a_bar) eps``."""
        noise = torch.randn_like(x0) if noise is None else noise
        a_bar = self._table(self.schedule.alphas_bar, t)
        x_t = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise
        return x_t, noise

    # -------------------------------------------------------------- denoiser
    def forward(
        self,
        x_t: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)
        h = torch.cat([x_t, self.label_emb(labels), cond.unsqueeze(1).expand(-1, x_t.shape[1], -1)], dim=-1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = h + F.relu(layer(h, adj, mask))
        return self.out(self.norm(h)) * mask.unsqueeze(-1).to(x_t.dtype)

    # ------------------------------------------------------------------- loss
    def compute_mse(
        self,
        x0: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-graph mean squared denoising error (the ``FeatureScore``)."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (b,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = self.forward(x_t, adj, labels, mask, y, t)
        err = ((pred - noise) ** 2).mean(dim=-1)  # (B, n)
        weights = mask.to(err.dtype)
        return (err * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def loss(
        self,
        x0: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Standard DDPM epsilon-prediction loss."""
        return self.compute_mse(x0, adj, labels, mask, y).mean()

    @torch.no_grad()
    def log_likelihood(
        self,
        x0: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Monte-Carlo estimate of ``log p(X | A, y)`` (higher is better).

        Following Eq. (13) of the paper the score is the (negated) average
        denoising error over the sampled diffusion steps.
        """
        num_samples = int(num_samples or self.num_mc_samples)
        steps = torch.linspace(1, self.num_steps, num_samples, device=x0.device).round().long()
        scores = []
        for step in steps:
            t = torch.full((x0.shape[0],), int(step), device=x0.device, dtype=torch.long)
            scores.append(self.compute_mse(x0, adj, labels, mask, y, t))
        return -torch.stack(scores, dim=0).mean(dim=0)
