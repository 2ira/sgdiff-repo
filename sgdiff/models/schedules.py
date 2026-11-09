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
    """Uniform (D3PM) transition ``Q_t = (1 - beta_t) I + beta_t / K * 11^T`` for ``K = 2``.

    The cumulative transition is ``Q_bar_t = a_t I + (1 - a_t) / K * 11^T`` with
    ``a_t = prod_s (1 - beta_s)``, i.e. every binary edge entry is flipped with
    probability ``1 - a_t``.
    """

    num_steps: int = 20
    beta_start: float = 0.02
    beta_end: float = 0.5

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        self.betas = betas
        self.alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1), self.alphas_bar[:-1]])

    def to(self, device: torch.device) -> "UniformDiscreteSchedule":
        self.betas = self.betas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        return self
