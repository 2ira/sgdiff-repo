"""Conditional feature DDPM: Eq. (10), Eq. (11) and Algorithm 1 of SGDiff.

The paper denoiser is a GCN over noisy attributes with the connection class
concatenated to each node. DRNL belongs to the structure branch, not this one.
Optional time embeddings and non-GCN encoders are explicit architecture variants.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DenseGATLayer, DenseSAGELayer, SinusoidalPositionEmbedding
from .schedules import GaussianSchedule


class FeatureDiffusionModel(nn.Module):
    """Predict Gaussian noise with an attribute-to-attribute GCN (Eq. 11)."""

    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 17,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_dim: int = 0,
        num_classes: int = 2,
        arch: str = "gcn",
        dropout: float = 0.0,
        num_mc_samples: Optional[int] = None,
        normalize_score: bool = False,
    ) -> None:
        super().__init__()
        if arch not in {"gcn", "sage", "gat"}:
            raise ValueError(f"unknown feature encoder: {arch}")
        if min(feature_dim, hidden_dim, num_layers, num_steps, num_classes) < 1 or time_dim < 0:
            raise ValueError("feature dimensions, layers and steps must be positive")
        self.schedule = GaussianSchedule(num_steps, beta_start, beta_end)
        self.num_steps = int(num_steps)
        self.num_mc_samples = int(num_steps if num_mc_samples is None else num_mc_samples)
        if self.num_mc_samples < 1:
            raise ValueError("num_mc_samples must be positive")
        self.normalize_score = bool(normalize_score)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.arch = arch
        self.dropout = float(dropout)
        self.time_emb = SinusoidalPositionEmbedding(time_dim) if time_dim else None
        dims = [feature_dim + num_classes + time_dim] + [hidden_dim] * (num_layers - 1) + [feature_dim]
        if arch == "gcn":
            self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(num_layers))
        else:
            layer_cls = DenseSAGELayer if arch == "sage" else DenseGATLayer
            self.layers = nn.ModuleList(layer_cls(dims[i], dims[i + 1]) for i in range(num_layers))

    def _table(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1 or torch.any(t < 1) or torch.any(t > self.num_steps):
            raise ValueError("timesteps must be a vector in [1, num_steps]")
        return table.to(t.device)[t - 1].view(-1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Eq. (10): sqrt(alpha_bar) X0 + sqrt(1-alpha_bar) epsilon."""
        noise = torch.randn_like(x0) if noise is None else noise
        if noise.shape != x0.shape:
            raise ValueError("noise and attributes must have the same shape")
        a_bar = self._table(self.schedule.alphas_bar, t).to(x0.dtype)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise, noise

    def forward(self, x_t, adj, labels, mask, y, t) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x_t.dtype)
        condition = F.one_hot(y, self.num_classes).to(x_t.dtype)
        inputs = [x_t * weights, condition.unsqueeze(1).expand(-1, x_t.shape[1], -1)]
        if self.time_emb is not None:
            inputs.append(self.time_emb(t).to(x_t.dtype).unsqueeze(1).expand(-1, x_t.shape[1], -1))
        h = torch.cat(inputs, dim=-1) * weights
        if self.arch == "gcn":
            valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            a = adj * valid.to(adj.dtype)
            a = a + torch.diag_embed(mask.to(adj.dtype))
            inv_sqrt = a.sum(-1).clamp(min=1.0).rsqrt()
            normalized = a * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
            for i, layer in enumerate(self.layers):
                h = torch.matmul(normalized, layer(h))
                if i < len(self.layers) - 1:
                    h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
                h = h * weights
        else:
            for i, layer in enumerate(self.layers):
                h = layer(h, adj, mask)
                if i < len(self.layers) - 1:
                    h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h * weights

    def compute_mse(self, x0, adj, labels, mask, y, t=None, noise=None) -> torch.Tensor:
        """Per-graph MSE, excluding padded nodes and averaging attribute dimensions."""
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (x0.shape[0],), device=x0.device)
        x_t, noise = self.q_sample(x0, t, noise)
        pred = self.forward(x_t, adj, labels, mask, y, t)
        err = ((pred - noise) ** 2).mean(dim=-1)
        weights = mask.to(err.dtype)
        return (err * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def loss(self, x0, adj, labels, mask, y) -> torch.Tensor:
        return self.compute_mse(x0, adj, labels, mask, y).mean()

    def score_mse(self, x0, adj, labels, mask, y, t=None, noise=None):
        """Eq. (10) squared norm; element-mean scores are an explicit variant."""
        mse = self.compute_mse(x0, adj, labels, mask, y, t, noise)
        return mse if self.normalize_score else mse * mask.sum(1) * self.feature_dim

    def class_scores(self, x0, adj, labels, mask, num_classes=None, num_samples=None,
                     random_timesteps=False, t=None, noise=None, conditions=None):
        """Algorithm 1: each class sees exactly the same timestep and noise."""
        b = x0.shape[0]
        classes = self.num_classes if num_classes is None else int(num_classes)
        if conditions is None:
            if not 1 <= classes <= self.num_classes:
                raise ValueError("invalid number of classes")
            conditions = torch.arange(classes, device=x0.device)
        conditions = torch.as_tensor(conditions, device=x0.device, dtype=torch.long)
        if conditions.ndim == 1:
            conditions = conditions[None, :].expand(b, -1)
        if conditions.ndim != 2 or conditions.shape[0] != b or conditions.shape[1] < 1:
            raise ValueError("conditions must have shape (C,) or (B,C)")
        if t is None:
            count = self.num_mc_samples if num_samples is None else int(num_samples)
            if count < 1:
                raise ValueError("num_samples must be positive")
            if random_timesteps:
                steps = torch.randint(1, self.num_steps + 1, (count, b), device=x0.device)
            else:
                values = torch.linspace(1, self.num_steps, count, device=x0.device).round().long()
                steps = values[:, None].expand(-1, b)
        else:
            steps = torch.as_tensor(t, device=x0.device, dtype=torch.long)
            if steps.ndim == 1:
                steps = steps[None, :]
            if steps.ndim != 2 or steps.shape[1] != b:
                raise ValueError("t must have shape (B,) or (S,B)")
        scores = []
        for index, step in enumerate(steps):
            epsilon = torch.randn_like(x0) if noise is None else (noise[index] if noise.ndim == 4 else noise)
            scores.append(torch.stack([
                -self.score_mse(x0, adj, labels, mask, conditions[:, cls], step, epsilon)
                for cls in range(conditions.shape[1])
            ], dim=1))
        return torch.stack(scores).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x0, adj, labels, mask, y, num_samples=None) -> torch.Tensor:
        """Negative average epsilon MSE proxy, not a full normalized log density."""
        scores = self.class_scores(x0, adj, labels, mask, num_samples=num_samples)
        return scores.gather(1, y[:, None]).squeeze(1)
