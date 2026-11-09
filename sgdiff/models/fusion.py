"""Learnable fusion of the two conditional likelihoods (Eqs. 12-13)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LikelihoodFusion(nn.Module):
    """``log P(A, X | y) = eta_1 log P(X|A,y) + eta_2 log P(A|y) + delta``.

    ``eta_1``, ``eta_2`` and ``delta`` are optimised through the cross-entropy
    between ``softmax_y(log P(A, X | y) + log prior)`` and the true link label.
    Both ``eta`` scalars are initialised to ``+1`` because the likelihood
    proxies are already "higher = better".
    """

    def __init__(
        self,
        num_classes: int = 2,
        init_eta: float = 1.0,
        learnable: bool = True,
        use_feature: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_feature = bool(use_feature)
        eta1 = torch.full((1,), float(init_eta) if self.use_feature else 0.0)
        eta2 = torch.full((1,), float(init_eta))
        delta = torch.zeros(1)
        if learnable and self.use_feature:
            self.eta1 = nn.Parameter(eta1)
        else:
            self.register_buffer("eta1", eta1)
        if learnable:
            self.eta2 = nn.Parameter(eta2)
            self.delta = nn.Parameter(delta)
        else:
            self.register_buffer("eta2", eta2)
            self.register_buffer("delta", delta)
        self.register_buffer("log_prior", -torch.log(torch.full((num_classes,), float(num_classes))))

    def log_joint(self, feature_scores: torch.Tensor, structure_scores: torch.Tensor) -> torch.Tensor:
        """Fuse per-class scores into ``log P(A, X | y)`` of shape ``(B, C)``."""
        return self.eta1 * feature_scores + self.eta2 * structure_scores + self.delta

    def forward(
        self,
        feature_scores: torch.Tensor,
        structure_scores: torch.Tensor,
        use_prior: bool = True,
    ) -> torch.Tensor:
        logits = self.log_joint(feature_scores, structure_scores)
        if use_prior:
            logits = logits + self.log_prior.to(logits.device)
        return logits

    def extra_repr(self) -> str:
        return f"eta1={self.eta1.item():.3f}, eta2={self.eta2.item():.3f}, delta={self.delta.item():.3f}"


def fusion_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy over the fused class logits."""
    return F.cross_entropy(logits, labels)
