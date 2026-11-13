"""Categorical diffusion of DRNL node labels and undirected edges (Sec. 3.3).

The denoiser predicts clean S0 distributions, trained with node + edge CE
(Appendix A.3). Equation (3) converts these predictions to reverse transition
probabilities. Algorithm 1 scores both node and edge posterior KL using the same
noisy graph for every class condition. Linear or DiGress-style cosine betas are
implementation choices; empirical marginals are supplied with ``set_marginals``.
Scores are Algorithm 1's negative mean KL, not a complete normalized likelihood.
The unconditional graph-size prior and terminal KL are omitted as class-common
terms; t=1 posterior KL already includes reconstruction because q(S0) is a delta.
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
    """Clean-x0 categorical denoiser with transformer and legacy edge-GNN backends."""

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
        node_marginal: Optional[torch.Tensor] = None,
        edge_marginal: Optional[torch.Tensor] = None,
        noise_schedule: str = "linear",
        loss_mode: str = "ce",
    ) -> None:
        super().__init__()
        if num_labels < 1 or num_classes < 1:
            raise ValueError("num_labels and num_classes must be positive")
        if loss_mode not in {"ce", "hybrid"}:
            raise ValueError("loss_mode must be 'ce' or 'hybrid'")
        if not torch.isfinite(torch.tensor(aux_ce_weight)) or aux_ce_weight < 0:
            raise ValueError("aux_ce_weight must be finite and nonnegative")
        self.noise_schedule = noise_schedule
        self.loss_mode = loss_mode
        self.schedule = UniformDiscreteSchedule(num_steps, beta_start, beta_end, noise_schedule)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.normalize_score = bool(normalize_score)
        self.num_mc_samples = int(num_mc_samples or num_steps)
        self.arch = arch
        self.num_labels = int(num_labels)
        self.aux_ce_weight = float(aux_ce_weight)  # Used only by the explicit hybrid variant.
        self.register_buffer("node_marginal", torch.full((num_labels,), 1.0 / num_labels))
        self.register_buffer("edge_marginal", torch.full((2,), 0.5))
        self.set_marginals(node_marginal, edge_marginal)

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

    @torch.no_grad()
    def set_marginals(self, node_marginal=None, edge_marginal=None) -> None:
        """Set nonnegative empirical counts/probabilities; normalization is automatic."""
        for name, value in (("node_marginal", node_marginal), ("edge_marginal", edge_marginal)):
            if value is None:
                continue
            target = getattr(self, name)
            value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if value.shape != target.shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be a finite nonnegative vector of shape {tuple(target.shape)}")
            if value.sum() <= 0:
                raise ValueError(f"{name} must have positive total mass")
            target.copy_(value / value.sum())

    @staticmethod
    def candidate_mask(mask: torch.Tensor) -> torch.Tensor:
        """Upper-triangular valid edge mask, shape ``(B, N, N, 1)``."""
        mask = mask.bool()
        n = mask.shape[1]
        upper = torch.triu(torch.ones(n, n, device=mask.device, dtype=torch.bool), diagonal=1)
        return (mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper.unsqueeze(0)).unsqueeze(-1)

    @staticmethod
    def _apply_transition(probs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        flat = probs.reshape(probs.shape[0], -1, probs.shape[-1])
        return torch.bmm(flat, matrix.to(probs)).reshape_as(probs)

    @staticmethod
    def _draw(probs: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.rand(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
        else:
            noise = torch.as_tensor(noise, device=probs.device, dtype=probs.dtype)
            if noise.shape != probs.shape[:-1] or not torch.isfinite(noise).all():
                raise ValueError("categorical noise must be finite and match the state tensor shape")
            if torch.any((noise < 0) | (noise >= 1)):
                raise ValueError("categorical noise must lie in [0, 1)")
        return (noise.unsqueeze(-1) >= probs.cumsum(-1)).sum(-1).clamp(max=probs.shape[-1] - 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, mask=None, noise=None) -> torch.Tensor:
        """Sample binary edges, mirror the upper triangle and clear padding/diagonal."""
        clean = F.one_hot(x0.long(), num_classes=2).to(self.edge_marginal.dtype)
        probs = self._apply_transition(clean, self.schedule.cumulative(t, self.edge_marginal))
        sampled = self._draw(probs, noise)
        if mask is None:
            mask = torch.ones(x0.shape[:2], dtype=torch.bool, device=x0.device)
        upper = sampled * self.candidate_mask(mask).squeeze(-1)
        return (upper + upper.transpose(1, 2)).to(x0.dtype)

    def q_sample_nodes(self, labels: torch.Tensor, t: torch.Tensor, mask=None, noise=None) -> torch.Tensor:
        """Diffuse every valid categorical DRNL label, including label zero."""
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        clean = F.one_hot(labels.masked_fill(~mask.bool(), 0), self.num_labels).to(self.node_marginal.dtype)
        probs = self._apply_transition(clean, self.schedule.cumulative(t, self.node_marginal))
        return self._draw(probs, noise).masked_fill(~mask.bool(), 0)

    @staticmethod
    def _split_noise(noise):
        if noise is None:
            return None, None
        if isinstance(noise, dict):
            return noise.get("node"), noise.get("edge")
        if isinstance(noise, (tuple, list)) and len(noise) == 2:
            return noise[0], noise[1]
        raise ValueError("noise must be a (node_uniform, edge_uniform) pair or a node/edge dict")

    def _noisy_structure(self, x0, labels, mask, t, noise=None):
        node_noise, edge_noise = self._split_noise(noise)
        return (self.q_sample(x0, t, mask, edge_noise),
                self.q_sample_nodes(labels, t, mask, node_noise))

    def posterior_distribution(self, clean_probs, noisy_states, t, marginal):
        """Normalized Eq. (3): ``(clean_probs @ Qbar[t-1]) * Qt[:, observation]``."""
        prior = self._apply_transition(clean_probs, self.schedule.cumulative(t - 1, marginal))
        observation = F.one_hot(noisy_states.long(), marginal.numel()).to(clean_probs)
        likelihood = self._apply_transition(observation, self.schedule.transition(t, marginal).transpose(-1, -2))
        unnormalized = prior * likelihood
        denom = unnormalized.sum(-1, keepdim=True)
        posterior = unnormalized / denom.clamp(min=torch.finfo(unnormalized.dtype).tiny)
        return torch.where(denom > 0, posterior, prior)

    def posterior_state1(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Legacy binary-edge posterior API, now using the exact categorical kernel."""
        clean = F.one_hot(x0.long(), 2).to(self.edge_marginal.dtype)
        return self.posterior_distribution(clean, x_t, t, self.edge_marginal)[..., 1]

    @staticmethod
    def _edge_probs(logits):
        return torch.softmax(torch.stack([torch.zeros_like(logits), logits], dim=-1), dim=-1)

    def reverse_probabilities(self, node_logits, edge_logits, labels_t, x_t, t):
        """Map clean node/edge predictions to normalized reverse distributions."""
        node = self.posterior_distribution(node_logits.softmax(-1), labels_t, t, self.node_marginal)
        edge = self.posterior_distribution(self._edge_probs(edge_logits), x_t, t, self.edge_marginal)
        return node, edge

    def forward(self, x_t, labels, mask, y, t) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clean-S0 ``(node_logits[B,N,K], edge_log_odds[B,N,N])``."""
        mask = mask.bool()
        labels = labels.masked_fill(~mask, 0)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        if self.arch == "transformer":
            node_logits, edge_logits = self.denoiser(x_t, labels, mask, self.denoiser.condition(t, y))
        else:
            cond = self.time_mlp(self.time_emb(t)) + self.class_emb(y)
            h = (self.label_emb(labels) + cond.unsqueeze(1)) * mask.unsqueeze(-1)
            diff = (labels.unsqueeze(2) - labels.unsqueeze(1)).abs().clamp(max=LABEL_DIFF_BUCKETS - 1)
            diff_onehot = F.one_hot(diff, num_classes=LABEL_DIFF_BUCKETS).to(x_t.dtype)
            e = self.edge_in(torch.cat([x_t.unsqueeze(-1), diff_onehot], dim=-1))
            e = e + self.edge_cond(cond).unsqueeze(1).unsqueeze(1)
            for layer in self.layers:
                h, e = layer(h, e, mask)
                e = 0.5 * (e + e.transpose(1, 2)) * pair_mask.unsqueeze(-1)
            node_logits, edge_logits = self.node_out(h), self.out(e).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        pair_mask = pair_mask & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
        return node_logits * mask.unsqueeze(-1), edge_logits * pair_mask

    def edge_logits(self, x_t, labels, mask, y, t) -> torch.Tensor:
        """Legacy edge-only clean-logit interface."""
        return self.forward(x_t, labels, mask, y, t)[1]

    def _kl(self, x0, labels, mask, y, t, x_t=None, logits=None, labels_t=None, noise=None):
        if x_t is None or labels_t is None:
            sampled_edges, sampled_nodes = self._noisy_structure(x0, labels, mask, t, noise)
            x_t = sampled_edges if x_t is None else x_t
            labels_t = sampled_nodes if labels_t is None else labels_t
        if isinstance(logits, (tuple, list)):
            node_logits, edge_logits = logits
        else:
            node_logits, edge_logits = self.forward(x_t, labels_t, mask, y, t)
            if logits is not None:
                edge_logits = logits
        node_clean = F.one_hot(labels.masked_fill(~mask.bool(), 0), self.num_labels).to(node_logits)
        edge_clean = F.one_hot(x0.long(), 2).to(edge_logits)
        q_node = self.posterior_distribution(node_clean, labels_t, t, self.node_marginal)
        q_edge = self.posterior_distribution(edge_clean, x_t, t, self.edge_marginal)
        p_node, p_edge = self.reverse_probabilities(node_logits, edge_logits, labels_t, x_t, t)
        node_kl = (q_node * (q_node.clamp(min=EPS).log() - p_node.clamp(min=EPS).log())).sum(-1)
        edge_kl = (q_edge * (q_edge.clamp(min=EPS).log() - p_edge.clamp(min=EPS).log())).sum(-1)
        cmask = self.candidate_mask(mask).squeeze(-1)
        total = (node_kl * mask).sum(1) + (edge_kl * cmask).sum((1, 2))
        if self.normalize_score:
            total = total / (mask.sum(1) + cmask.sum((1, 2))).clamp(min=1)
        return total.clamp(min=0)

    def compute_kl(self, x0, labels, mask, y, t=None, noise=None, x_t=None, labels_t=None):
        """Per-graph node + undirected-edge posterior KL; supports supplied shared noise."""
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (x0.shape[0],), device=x0.device)
        return self._kl(x0, labels, mask, y, t, x_t=x_t, labels_t=labels_t, noise=noise)

    def loss(self, x0, labels, mask, y, t=None, noise=None) -> torch.Tensor:
        """Appendix A.3 node + edge CE; explicit hybrid adds posterior KL.

        ``loss_mode='ce'`` ignores ``aux_ce_weight``. The non-paper hybrid variant
        preserves the old weighting convention: KL + aux_ce_weight * CE.
        """
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (x0.shape[0],), device=x0.device)
        x_t, labels_t = self._noisy_structure(x0, labels, mask, t, noise)
        node_logits, edge_logits = self.forward(x_t, labels_t, mask, y, t)
        targets = labels.masked_fill(~mask.bool(), 0)
        node_ce = F.cross_entropy(node_logits.transpose(1, 2), targets, reduction="none")
        node_ce = (node_ce * mask).sum() / mask.sum().clamp(min=1)
        cmask = self.candidate_mask(mask).squeeze(-1)
        edge_ce = F.binary_cross_entropy_with_logits(edge_logits, x0, reduction="none")
        edge_ce = (edge_ce * cmask).sum() / cmask.sum().clamp(min=1)
        ce = node_ce + edge_ce
        if self.loss_mode == "hybrid":
            kl = self._kl(x0, labels, mask, y, t, x_t=x_t, labels_t=labels_t,
                          logits=(node_logits, edge_logits)).mean()
            return kl + self.aux_ce_weight * ce
        return ce

    def class_scores(self, x0, labels, mask, num_samples=None, t=None, noise=None, conditions=None,
                     num_classes=None, random_timesteps=False):
        """Differentiable ``(B,C)`` negative mean KL with a shared noisy graph per step.

        ``t`` accepts ``(B,)`` or ``(S,B)``; absent timesteps cover the schedule
        unless ``random_timesteps=True``. ``noise`` is a node/edge pair (or dict),
        optionally with leading sample dimension S. ``conditions`` accepts class
        IDs ``(C,)`` or ``(B,C)``; otherwise ``num_classes`` selects classes 0..C-1.
        """
        b = x0.shape[0]
        if t is None:
            samples = self.num_mc_samples if num_samples is None else int(num_samples)
            if samples < 1:
                raise ValueError("num_samples must be positive")
            if random_timesteps:
                timesteps = torch.randint(1, self.num_steps + 1, (samples, b), device=x0.device)
            else:
                steps = torch.linspace(1, self.num_steps, samples, device=x0.device).round().long()
                timesteps = steps[:, None].expand(-1, b)
        else:
            timesteps = torch.as_tensor(t, device=x0.device, dtype=torch.long)
            if timesteps.ndim == 1:
                timesteps = timesteps.unsqueeze(0)
            if timesteps.ndim != 2 or timesteps.shape[1] != b:
                raise ValueError("t must have shape (B,) or (S,B)")
        if conditions is None:
            classes = self.num_classes if num_classes is None else int(num_classes)
            if not 1 <= classes <= self.num_classes:
                raise ValueError("num_classes must lie in [1, model.num_classes]")
            conditions = torch.arange(classes, device=x0.device)
        conditions = torch.as_tensor(conditions, device=x0.device, dtype=torch.long)
        if conditions.ndim == 1:
            conditions = conditions.unsqueeze(0).expand(b, -1)
        if conditions.ndim != 2 or conditions.shape[0] != b or conditions.shape[1] < 1:
            raise ValueError("conditions must have shape (C,) or (B,C)")
        node_noise, edge_noise = self._split_noise(noise)
        scores = []
        for index, step_t in enumerate(timesteps):
            step_node = node_noise[index] if node_noise is not None and node_noise.ndim == 3 else node_noise
            step_edge = edge_noise[index] if edge_noise is not None and edge_noise.ndim == 4 else edge_noise
            x_t, labels_t = self._noisy_structure(x0, labels, mask, step_t, (step_node, step_edge))
            scores.append(torch.stack([
                -self._kl(x0, labels, mask, conditions[:, cls], step_t, x_t=x_t, labels_t=labels_t)
                for cls in range(conditions.shape[1])
            ], dim=1))
        return torch.stack(scores).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x0, labels, mask, y, num_samples=None, t=None, noise=None):
        """Negative mean posterior KL for the supplied per-graph class conditions."""
        return self.class_scores(x0, labels, mask, num_samples, t, noise, y.unsqueeze(1))[:, 0]
