"""Two-stage training loop for SGDiff.

Stage 1 (``pretrain_epochs``): class-conditional MLE of the structure and
feature diffusion models.
Stage 2: the fused logits of Eq. (12) are trained with cross-entropy, which
optimises ``eta_1``/``eta_2``/``delta`` together with the two denoisers.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device
from .batch import PairSubgraphDataset, collate_subgraphs
from .evaluator import score_pairs

LOGGER = get_logger(__name__)


class Trainer:
    """Fit / validate / checkpoint a :class:`~sgdiff.models.sgdiff.SGDiff` model."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        device=None,
        run_dir: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(device) if device is not None else resolve_device(config.train.device)
        self.model.to(self.device)

        name = config.train.run_name or f"{config.data.name}-{int(time.time())}"
        self.run_dir = ensure_dir(Path(run_dir or config.train.run_dir) / name)
        self.log_path = self.run_dir / "metrics.jsonl"
        self.best_path = self.run_dir / "best.pt"
        self.best_score = -np.inf
        self.history: list[dict] = []

    # ------------------------------------------------------------- optimizers
    def _optimizers(self):
        fusion_params = list(self.model.fusion.parameters())
        fusion_ids = {id(p) for p in fusion_params}
        diffusion_params = [p for p in self.model.parameters() if id(p) not in fusion_ids]
        main = torch.optim.Adam(
            diffusion_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
        )
        fusion = torch.optim.Adam(fusion_params, lr=self.cfg.train.fusion_lr)
        return main, fusion

    @staticmethod
    def _scheduler(optimizer, epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    @staticmethod
    def _loader(dataset: PairSubgraphDataset, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        train_dataset: PairSubgraphDataset,
        val_dataset: Optional[PairSubgraphDataset] = None,
        test_dataset: Optional[PairSubgraphDataset] = None,
        max_steps_per_epoch: Optional[int] = None,
    ) -> dict:
        cfg = self.cfg
        main_opt, fusion_opt = self._optimizers()
        main_sched = self._scheduler(main_opt, cfg.train.epochs)
        fusion_sched = self._scheduler(fusion_opt, cfg.train.epochs)
        train_loader = self._loader(train_dataset, cfg.train.batch_size, True, cfg.train.num_workers)

        best_state = None
        bad_epochs = 0
        for epoch in range(cfg.train.epochs):
            stage = "pretrain" if epoch < cfg.train.pretrain_epochs else "finetune"
            train_metrics = self._train_epoch(
                train_loader, main_opt, fusion_opt, stage, max_steps_per_epoch
            )
            main_sched.step()
            fusion_sched.step()

            record = {"epoch": epoch, "stage": stage, **train_metrics}
            if val_dataset is not None and (epoch + 1) % cfg.train.eval_every == 0:
                val_metrics = self.evaluate(val_dataset)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                score = float(val_metrics.get("auc", float("nan")))
                if np.isfinite(score) and score > self.best_score:
                    self.best_score = score
                    bad_epochs = 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    if cfg.train.save_checkpoint:
                        self.save_checkpoint(self.best_path, epoch=epoch, metrics=val_metrics)
                else:
                    bad_epochs += 1
            self._log(record)

            if val_dataset is not None and bad_epochs >= cfg.train.patience:
                LOGGER.info("early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        result: dict = {"run_dir": str(self.run_dir), "best_val_auc": self.best_score}
        if test_dataset is not None:
            test_metrics = self.evaluate(test_dataset)
            result["test"] = test_metrics
            self._log(
                {"epoch": "final", "stage": "test", **{f"test_{k}": v for k, v in test_metrics.items()}}
            )
            save_json(result, self.run_dir / "result.json")
        return result

    def _train_epoch(
        self,
        loader: DataLoader,
        main_opt: torch.optim.Optimizer,
        fusion_opt: torch.optim.Optimizer,
        stage: str,
        max_steps: Optional[int] = None,
    ) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "structure": 0.0, "feature": 0.0, "ce": 0.0}
        count = 0
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = batch.to(self.device)
            losses = self.model.diffusion_losses(batch)
            w_s, w_f = self.cfg.structure.weight, self.cfg.feature.weight

            if stage == "pretrain":
                loss = w_s * losses["structure"] + w_f * losses["feature"]
                ce_value = torch.zeros((), device=self.device)
            else:
                logits, _ = self.model.train_logits(
                    batch, num_samples=self.cfg.train.finetune_mc_samples
                )
                ce_value = F.cross_entropy(logits, batch.y)
                loss = ce_value
                if not self.cfg.train.freeze_diffusion:
                    loss = loss + 0.1 * (w_s * losses["structure"] + w_f * losses["feature"])

            main_opt.zero_grad(set_to_none=True)
            fusion_opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            if stage == "pretrain" or not self.cfg.train.freeze_diffusion:
                main_opt.step()
            fusion_opt.step()

            totals["loss"] += float(loss.detach())
            totals["structure"] += float(losses["structure"].detach())
            totals["feature"] += float(losses["feature"].detach())
            totals["ce"] += float(ce_value.detach())
            count += 1

        return {k: v / max(count, 1) for k, v in totals.items()}

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, dataset: PairSubgraphDataset) -> dict:
        self.model.eval()
        scores = score_pairs(
            self.model,
            dataset.extractor,
            dataset.pairs,
            batch_size=self.cfg.train.eval_batch_size,
            device=self.device,
            num_workers=self.cfg.train.num_workers,
            seed=self.cfg.train.seed,
        )
        return summarize_scores(dataset.labels, scores)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path, epoch: int = -1, metrics: Optional[dict] = None) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "model_state": self.model.state_dict(),
            "config": self.cfg.to_dict(),
            "feature_dim": int(self.model.feature_dim),
            "num_labels": int(self.model.structure.num_labels),
            "epoch": int(epoch),
            "metrics": metrics or {},
        }
        torch.save(payload, path)
        LOGGER.info("saved checkpoint to %s", path)
        return path

    def _log(self, record: dict) -> None:
        self.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}
        LOGGER.info("%s", printable)


def load_checkpoint(path, device="cpu"):
    """Rebuild ``(model, config)`` from a checkpoint written by :class:`Trainer`."""
    from ..config import ExperimentConfig
    from .factory import build_model

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    cfg = ExperimentConfig.from_dict(payload["config"])
    model = build_model(
        cfg, feature_dim=int(payload["feature_dim"]), num_labels=int(payload["num_labels"])
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, cfg
