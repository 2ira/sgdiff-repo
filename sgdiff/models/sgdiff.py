"""SGDiff: structure diffusion + feature diffusion + learnable fusion (Sec. 3)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .feature_diffusion import FeatureDiffusionModel
from .fusion import LikelihoodFusion
from .structure_diffusion import StructureDiffusionModel


class SGDiff(nn.Module):
    """The full SGDIFF model.

    The module is stateless with respect to data: it consumes a
    :class:`~sgdiff.engine.batch.SubgraphBatch` and returns per-pair scores.
    """

    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 17,
        num_classes: int = 2,
        structure_kwargs: Optional[dict] = None,
        feature_kwargs: Optional[dict] = None,
        fusion_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        structure_kwargs = dict(structure_kwargs or {})
        feature_kwargs = dict(feature_kwargs or {})
        fusion_kwargs = dict(fusion_kwargs or {})

        structure_kwargs.setdefault("num_labels", num_labels)
        structure_kwargs.setdefault("num_classes", num_classes)
        feature_kwargs.setdefault("num_labels", num_labels)
        feature_kwargs.setdefault("num_classes", num_classes)

        self.structure = StructureDiffusionModel(**structure_kwargs)
        self.feature = FeatureDiffusionModel(feature_dim=feature_dim, **feature_kwargs)
        self.fusion = LikelihoodFusion(num_classes=num_classes, **fusion_kwargs)

    # ------------------------------------------------------------------ losses
    def diffusion_losses(self, batch) -> dict[str, torch.Tensor]:
        """Class-conditional ELBO of the structure model and DDPM loss of the feature model."""
        structure = self.structure.loss(batch.adj, batch.labels, batch.mask, batch.y)
        feature = self.feature.loss(batch.x, batch.adj, batch.labels, batch.mask, batch.y)
        return {"structure": structure, "feature": feature}

    # ------------------------------------------------------------------ scores
    @torch.no_grad()
    def class_scores(self, batch, num_samples: Optional[int] = None) -> tuple:
        """Return ``(feature_scores, structure_scores)`` each of shape ``(B, C)``."""
        feat, struct = [], []
        b = batch.batch_size
        for cls in range(self.fusion.num_classes):
            y = torch.full((b,), cls, device=batch.y.device, dtype=torch.long)
            feat.append(
                self.feature.log_likelihood(
                    batch.x, batch.adj, batch.labels, batch.mask, y, num_samples=num_samples
                )
            )
            struct.append(
                self.structure.log_likelihood(
                    batch.adj, batch.labels, batch.mask, y, num_samples=num_samples
                )
            )
        return torch.stack(feat, dim=1), torch.stack(struct, dim=1)

    def forward(self, batch, num_samples: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Fuse both likelihoods into class logits."""
        feat, struct = self.class_scores(batch, num_samples=num_samples)
        logits = self.fusion(feat, struct)
        return {"logits": logits, "feature_scores": feat, "structure_scores": struct}

    @torch.no_grad()
    def predict_proba(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        """Probability of the "link" class for every pair in ``batch``."""
        logits = self.forward(batch, num_samples=num_samples)["logits"]
        return torch.softmax(logits, dim=-1)[:, 1]

    @torch.no_grad()
    def predict_logit(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        """Log-odds of the "link" class (used for ranking metrics)."""
        logits = self.forward(batch, num_samples=num_samples)["logits"]
        return logits[:, 1] - logits[:, 0]

    # ------------------------------------------------------------- training
    def train_logits(self, batch, num_samples: int = 1) -> tuple[torch.Tensor, dict]:
        """Differentiable fused logits (one sampled timestep per class).

        Used by the fusion/fine-tuning stage: the two conditional scores are
        computed with gradient, so ``eta_1``/``eta_2``/``delta`` -- and
        optionally both diffusion networks -- are optimised through the
        cross-entropy of Eq. (12).
        """
        feat, struct = [], []
        b = batch.batch_size
        num_samples = max(int(num_samples), 1)
        for cls in range(self.fusion.num_classes):
            y = torch.full((b,), cls, device=batch.y.device, dtype=torch.long)
            feat_c, struct_c = [], []
            for _ in range(num_samples):
                feat_c.append(-self.feature.compute_mse(batch.x, batch.adj, batch.labels, batch.mask, y))
                struct_c.append(-self.structure.compute_kl(batch.adj, batch.labels, batch.mask, y))
            feat.append(torch.stack(feat_c).mean(0))
            struct.append(torch.stack(struct_c).mean(0))
        feature_scores = torch.stack(feat, dim=1)
        structure_scores = torch.stack(struct, dim=1)
        logits = self.fusion(feature_scores, structure_scores)
        return logits, {"feature_scores": feature_scores, "structure_scores": structure_scores}
