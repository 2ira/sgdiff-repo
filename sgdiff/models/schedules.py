"""Noise schedules for the structure (discrete) and feature (Gaussian) diffusions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianSchedule:
    """Linear ``beta`` schedule of DDPM (Ho et al., 2020)."""

    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1), self.alphas_bar[:-1]])

    def to(self, device: torch.device) -> "GaussianSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        return self


@dataclass
class UniformDiscreteSchedule:
    """Categorical schedule with a uniform or empirical marginal prior.

    ``Q_t = (1 - beta_t) I + beta_t 1 m^T`` and
    ``Q_bar_t = a_t I + (1 - a_t) 1 m^T``. For a uniform binary prior,
    the flip probability is ``(1 - a_t) / 2``. Linear and DiGress-style cosine
    schedules are configurable choices, not explicitly specified by SGDiff.
    Cosine uses the same offset 0.008 but T one-based transitions, preserving
    Qbar(0)=I rather than copying DiGress's additional noisy t=0 transition.
    """

    num_steps: int = 20
    beta_start: float = 0.02
    beta_end: float = 0.5
    noise_schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.num_steps < 1 or not (0 <= self.beta_start <= 1 and 0 <= self.beta_end <= 1):
            raise ValueError("num_steps must be positive and betas must lie in [0, 1]")
        if self.noise_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        elif self.noise_schedule == "cosine":
            grid = torch.linspace(0, 1, self.num_steps + 1, dtype=torch.float64)
            cumulative = torch.cos((grid + 0.008) / 1.008 * torch.pi / 2).square()
            cumulative = cumulative / cumulative[0]
            betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 1).float()
            betas[-1] = 1
        else:
            raise ValueError("noise_schedule must be 'linear' or 'cosine'")
        self.betas = betas
        self.alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1), self.alphas_bar[:-1]])

    @staticmethod
    def _matrix(alpha: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
        k = marginal.numel()
        eye = torch.eye(k, device=alpha.device, dtype=marginal.dtype)
        prior = marginal.to(alpha.device).unsqueeze(0).expand(k, -1)
        return alpha[..., None, None] * eye + (1 - alpha[..., None, None]) * prior

    def transition(self, t: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
        """Row-stochastic one-step matrices; ``t`` is one-based."""
        if torch.any((t < 1) | (t > self.num_steps)):
            raise ValueError("timesteps must lie in [1, num_steps]")
        return self._matrix(1 - self.betas.to(t.device)[t - 1], marginal)

    def cumulative(self, t: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
        """Cumulative matrices including the identity at ``t = 0``."""
        if torch.any((t < 0) | (t > self.num_steps)):
            raise ValueError("cumulative timesteps must lie in [0, num_steps]")
        table = torch.cat([torch.ones(1, device=t.device), self.alphas_bar.to(t.device)])
        return self._matrix(table[t], marginal)

    def to(self, device: torch.device) -> "UniformDiscreteSchedule":
        self.betas = self.betas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        return self
