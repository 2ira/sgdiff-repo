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
        feature_enabled: bool = True,
    ) -> None:
        super().__init__()
        structure_kwargs = dict(structure_kwargs or {})
        feature_kwargs = dict(feature_kwargs or {})
        fusion_kwargs = dict(fusion_kwargs or {})

        structure_kwargs.setdefault("num_labels", num_labels)
        structure_kwargs.setdefault("num_classes", num_classes)
        feature_kwargs.setdefault("num_labels", num_labels)
        feature_kwargs.setdefault("num_classes", num_classes)

        self.feature_enabled = bool(feature_enabled)
        self.feature_dim = int(feature_dim)
        self.structure = StructureDiffusionModel(**structure_kwargs)
        if self.feature_enabled:
            self.feature = FeatureDiffusionModel(feature_dim=feature_dim, **feature_kwargs)
        else:  # featureless datasets (USAir / NS / Router) and cross-dataset transfer
            self.feature = None
        fusion_kwargs.setdefault("use_feature", self.feature_enabled)
        self.fusion = LikelihoodFusion(num_classes=num_classes, **fusion_kwargs)

    # ------------------------------------------------------------------ losses
    def diffusion_losses(self, batch) -> dict[str, torch.Tensor]:
        """Clean categorical CE (Appendix A.3) and conditional epsilon MSE."""
        structure = self.structure.loss(batch.adj, batch.labels, batch.mask, batch.y)
        if self.feature is None:
            feature = torch.zeros((), device=batch.adj.device)
        else:
            feature = self.feature.loss(batch.x, batch.adj, batch.labels, batch.mask, batch.y)
        return {"structure": structure, "feature": feature}

    # ------------------------------------------------------------------ scores
    @torch.no_grad()
    def class_scores(self, batch, num_samples: Optional[int] = None) -> tuple:
        """Return ``(feature_scores, structure_scores)`` each of shape ``(B, C)``."""
        return self._branch_scores(batch, num_samples=num_samples)

    def _branch_scores(self, batch, num_samples=None, random_timesteps=False):
        struct = self.structure.class_scores(
            batch.adj, batch.labels, batch.mask,
            num_classes=self.fusion.num_classes, num_samples=num_samples,
            random_timesteps=random_timesteps,
        )
        if self.feature is None:
            feat = torch.zeros_like(struct)
        else:
            feat = self.feature.class_scores(
                batch.x, batch.adj, batch.labels, batch.mask,
                num_classes=self.fusion.num_classes, num_samples=num_samples,
                random_timesteps=random_timesteps,
            )
        return feat, struct

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
    def train_logits(self, batch, num_samples: Optional[int] = None) -> tuple[torch.Tensor, dict]:
        """Eq. (13) fusion objective with the same scoring path as inference.

        Frozen denoisers stay in evaluation mode during fusion calibration.
        Updating denoisers through this objective is an explicit variant, not
        the paper-default two-stage training protocol.
        """
        feature_scores, structure_scores = self._branch_scores(batch, num_samples=num_samples)
        logits = self.fusion(feature_scores, structure_scores)
        return logits, {"feature_scores": feature_scores, "structure_scores": structure_scores}
