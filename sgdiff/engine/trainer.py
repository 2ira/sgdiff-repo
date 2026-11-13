"""Diffusion pretraining followed by Eq. (13) fusion calibration.

``epochs`` is the diffusion budget; ``finetune_epochs`` is a separate fusion
budget. Each phase has its own optimizer, validation criterion and checkpoint.
Training schedules are reproducible choices: the paper does not specify them.
"""

from __future__ import annotations

import json
import random
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
from ..version import METHOD_VERSION
from .batch import PairSubgraphDataset, collate_subgraphs
from .evaluator import score_pairs

LOGGER = get_logger(__name__)


def _rng_state():
    state = np.random.get_state()
    result = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


class Trainer:
    """Train SGDiff without selecting pre-fusion weights as the final model."""

    def __init__(self, model, config: ExperimentConfig, device=None, run_dir=None, train_ablation=None):
        self.model = model
        self.cfg = config
        self.device = torch.device(device) if device is not None else resolve_device(config.train.device)
        self.model.to(self.device)
        self.train_ablation = train_ablation
        name = config.train.run_name or f"{config.data.name}-{int(time.time())}"
        self.run_dir = ensure_dir(Path(run_dir or config.train.run_dir) / name)
        self.log_path = self.run_dir / "metrics.jsonl"
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"
        self.best_score = -np.inf
        self.history: list[dict] = []

    def _optimizers(self):
        fusion_params = list(self.model.fusion.parameters())
        fusion_ids = {id(p) for p in fusion_params}
        diffusion_params = [p for p in self.model.parameters() if id(p) not in fusion_ids]
        main = torch.optim.Adam(diffusion_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        fusion = torch.optim.Adam(fusion_params, lr=self.cfg.train.fusion_lr) if fusion_params else None
        return main, fusion

    @staticmethod
    def _loader(dataset, batch_size, shuffle, num_workers, seed=0):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
        )

    def _state_copy(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _set_stage(self, stage):
        fusion_ids = {id(p) for p in self.model.fusion.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(id(p) in fusion_ids if stage == "finetune" and self.cfg.train.freeze_diffusion else True)
        if stage == "finetune" and self.cfg.train.freeze_diffusion:
            self.model.eval()
            self.model.fusion.train()
        else:
            self.model.train()

    def fit(self, train_dataset, val_dataset=None, test_dataset=None, max_steps_per_epoch=None):
        from .factory import initialize_structure_marginals

        cfg = self.cfg
        if cfg.train.epochs < 1 or cfg.train.finetune_epochs < 0:
            raise ValueError("diffusion epochs must be positive and fusion epochs non-negative")
        if cfg.train.eval_every < 1:
            raise ValueError("eval_every must be positive")
        if self.log_path.exists() and not cfg.train.resume:
            raise FileExistsError(f"run already exists: {self.run_dir}; use a new name or train.resume=last.pt")
        main_opt, fusion_opt = self._optimizers()
        if fusion_opt is None and cfg.train.finetune_epochs:
            LOGGER.warning("fusion is fixed; skipping fusion optimization")
        budgets = {"pretrain": cfg.train.epochs, "finetune": cfg.train.finetune_epochs if fusion_opt else 0}
        best_state = None
        resume_stage, resume_epoch, resume_best, resume_bad = "pretrain", 0, -np.inf, 0
        if cfg.train.resume:
            payload = _load_payload(cfg.train.resume, self.device)
            if "training_state" not in payload:
                raise ValueError("resume requires last.pt, including optimizer and RNG states")
            previous = ExperimentConfig.from_dict(payload["config"])
            old, new = previous.to_dict(), cfg.to_dict()
            for key in ("device", "resume", "run_dir", "run_name", "num_workers"):
                old["train"].pop(key, None)
                new["train"].pop(key, None)
            if old != new:
                raise ValueError("resume configuration differs; only device/run location/workers may change")
            self.model.load_state_dict(payload["model_state"])
            training = payload["training_state"]
            main_opt.load_state_dict(training["main_optimizer"])
            if fusion_opt and training["fusion_optimizer"] is not None:
                fusion_opt.load_state_dict(training["fusion_optimizer"])
            resume_stage, resume_epoch = training["stage"], int(training["next_epoch"])
            resume_best, resume_bad = training["best_criterion"], int(training["bad_epochs"])
            best_state = training["best_model_state"]
            self.best_score = float(training.get("selected_val_auc", -np.inf))
            self.preprocessing = payload.get("preprocessing", {})
            _restore_rng(training["rng"])
        else:
            initialize_structure_marginals(self.model, train_dataset)
            self.preprocessing = dict(train_dataset.extractor.graph.meta.get("preprocessing", {}))
            positive = float(np.mean(train_dataset.labels))
            prior = torch.tensor([1.0 - positive, positive], device=self.device).clamp(min=1e-12).log()
            self.model.fusion.log_prior.copy_(prior)
        self.preprocessing["max_steps_per_epoch"] = max_steps_per_epoch
        save_json({"method_version": METHOD_VERSION, "config": cfg.to_dict(), "preprocessing": self.preprocessing},
                  self.run_dir / "resolved_config.json")

        completed_fusion = False
        for stage in ("pretrain", "finetune"):
            if resume_stage == "finetune" and stage == "pretrain":
                continue
            start = resume_epoch if stage == resume_stage else 0
            criterion = resume_best if stage == resume_stage else -np.inf
            bad = resume_bad if stage == resume_stage else 0
            if stage != resume_stage:
                best_state = None
            self._set_stage(stage)
            for epoch in range(start, budgets[stage]):
                loader = self._loader(train_dataset, cfg.train.batch_size, True, cfg.train.num_workers,
                                      cfg.train.seed + epoch + (100000 if stage == "finetune" else 0))
                metrics = self._train_epoch(loader, main_opt, fusion_opt, stage, max_steps_per_epoch)
                record = {"epoch": epoch, "stage": stage, **metrics}
                score = None
                if (epoch + 1) % cfg.train.eval_every == 0 or epoch + 1 == budgets[stage]:
                    if val_dataset is not None:
                        val = self._validation_losses(val_dataset, stage)
                        record.update({f"val_{k}": v for k, v in val.items()})
                        score = -val["loss"]
                        if stage == "finetune":
                            val_metrics = self.evaluate(val_dataset)
                            record.update({f"val_{k}": v for k, v in val_metrics.items()})
                    else:
                        score = -metrics["loss"]
                    if score > criterion:
                        criterion, bad, best_state = score, 0, self._state_copy()
                        if stage == "finetune" and val_dataset is not None:
                            self.best_score = float(val_metrics["auc"])
                        if cfg.train.save_checkpoint:
                            self.save_checkpoint(self.run_dir / f"best_{stage}.pt", epoch, record)
                    else:
                        bad += 1
                self._log(record)
                patience = cfg.train.patience if stage == "pretrain" else cfg.train.fusion_patience
                minimum = min(cfg.train.pretrain_epochs, budgets[stage]) if stage == "pretrain" else 1
                stop = val_dataset is not None and patience > 0 and bad >= patience and epoch + 1 >= minimum
                training_state = {
                    "main_optimizer": main_opt.state_dict(),
                    "fusion_optimizer": fusion_opt.state_dict() if fusion_opt else None,
                    "stage": stage, "next_epoch": budgets[stage] if stop else epoch + 1,
                    "best_criterion": float(criterion), "bad_epochs": bad,
                    "best_model_state": best_state, "rng": _rng_state(),
                    "selected_val_auc": float(self.best_score),
                }
                if cfg.train.save_checkpoint:
                    self.save_checkpoint(self.last_path, epoch, record, training_state)
                if stop:
                    LOGGER.info("early stopping %s at epoch %d", stage, epoch)
                    break
            if best_state is not None:
                self.model.load_state_dict(best_state)
            if stage == "finetune" and budgets[stage]:
                completed_fusion = True
            resume_epoch, resume_best, resume_bad = 0, -np.inf, 0

        self._set_stage("pretrain")
        self.model.eval()
        result = {
            "method_version": METHOD_VERSION, "run_dir": str(self.run_dir),
            "best_val_auc": float(self.best_score) if np.isfinite(self.best_score) else None,
            "fusion_calibrated": completed_fusion,
            "protocol": {"structure_training_loss": cfg.structure.loss_mode,
                         "feature_training_loss": "epsilon_mse", "fusion_frozen": cfg.train.freeze_diffusion,
                         "preprocessing": self.preprocessing,
                         "feature_normalization": cfg.subgraph.feature_normalization,
                         "structure_noise_schedule": cfg.structure.noise_schedule,
                         "schedule_status": "chosen reproducible budget; training epochs unspecified in paper"},
        }
        if test_dataset is not None:
            result["test"] = self.evaluate(test_dataset)
            self._log({"epoch": "final", "stage": "test", **{f"test_{k}": v for k, v in result["test"].items()}})
        if cfg.train.save_checkpoint:
            self.save_checkpoint(self.best_path, metrics=result.get("test"))
        save_json(result, self.run_dir / "result.json")
        return result

    def _train_epoch(self, loader, main_opt, fusion_opt, stage, max_steps=None):
        self._set_stage(stage)
        totals = {"loss": 0.0, "structure": 0.0, "feature": 0.0, "ce": 0.0}
        count = 0
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = batch.to(self.device)
            main_opt.zero_grad(set_to_none=True)
            if fusion_opt:
                fusion_opt.zero_grad(set_to_none=True)
            if stage == "pretrain":
                condition = self.train_ablation.apply_to_condition(batch) if self.train_ablation is not None else batch
                losses = self.model.diffusion_losses(condition)
                ce = torch.zeros((), device=self.device)
                loss = self.cfg.structure.weight * losses["structure"] + self.cfg.feature.weight * losses["feature"]
            else:
                logits, _ = self.model.train_logits(batch, num_samples=self.cfg.train.finetune_mc_samples)
                ce = F.cross_entropy(logits, batch.y)
                loss = ce
                losses = {"structure": torch.zeros_like(ce), "feature": torch.zeros_like(ce)}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at step {step}; no silent skipped updates")
            loss.backward()
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"non-finite {stage} gradient at step {step}")
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters, self.cfg.train.grad_clip, error_if_nonfinite=True)
            if stage == "pretrain" or not self.cfg.train.freeze_diffusion:
                main_opt.step()
            if stage == "finetune" and fusion_opt:
                fusion_opt.step()
            size = batch.batch_size
            for key, value in {"loss": loss, "structure": losses["structure"], "feature": losses["feature"], "ce": ce}.items():
                totals[key] += float(value.detach()) * size
            count += size
        if not count:
            raise ValueError("training epoch has no batches")
        return {k: v / count for k, v in totals.items()}

    @torch.no_grad()
    def _validation_losses(self, dataset, stage):
        state = _rng_state()
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            torch.manual_seed(self.cfg.train.eval_seed)
            self.model.eval()
            loader = self._loader(dataset, self.cfg.train.eval_batch_size, False, self.cfg.train.num_workers,
                                  self.cfg.train.eval_seed)
            total, count = 0.0, 0
            for batch in loader:
                batch = batch.to(self.device)
                if stage == "pretrain":
                    losses = self.model.diffusion_losses(batch)
                    loss = self.cfg.structure.weight * losses["structure"] + self.cfg.feature.weight * losses["feature"]
                else:
                    logits, _ = self.model.train_logits(batch, num_samples=self.cfg.train.finetune_mc_samples)
                    loss = F.cross_entropy(logits, batch.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation loss")
                total += float(loss) * batch.batch_size
                count += batch.batch_size
            if not count:
                raise ValueError("validation dataset is empty")
            return {"loss": total / count}
        finally:
            _restore_rng(state)
            for module, training in modes:
                module.training = training

    @torch.no_grad()
    def evaluate(self, dataset):
        scores = score_pairs(self.model, dataset.extractor, dataset.pairs,
                             batch_size=self.cfg.train.eval_batch_size, device=self.device,
                             num_workers=self.cfg.train.num_workers, seed=self.cfg.train.eval_seed)
        return summarize_scores(dataset.labels, scores)

    def save_checkpoint(self, path, epoch=-1, metrics=None, training_state=None):
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "method_version": METHOD_VERSION, "model_state": self.model.state_dict(),
            "config": self.cfg.to_dict(), "feature_dim": int(self.model.feature_dim),
            "preprocessing": getattr(self, "preprocessing", {}),
            "num_labels": int(self.model.structure.num_labels), "epoch": int(epoch), "metrics": metrics or {},
        }
        if training_state is not None:
            payload["training_state"] = training_state
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _log(self, record):
        record = {"method_version": METHOD_VERSION, **record}
        self.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        LOGGER.info("%s", {k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()})


def _load_payload(path, device):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("method_version") != METHOD_VERSION:
        raise ValueError("legacy SGDiff checkpoint is not compatible with corrected method; retrain into a new run")
    return payload


def load_checkpoint(path, device="cpu"):
    """Load only corrected-method, tensor/primitive checkpoints (no pickle fallback)."""
    from .factory import build_model

    payload = _load_payload(path, device)
    cfg = ExperimentConfig.from_dict(payload["config"])
    model = build_model(cfg, int(payload["feature_dim"]), int(payload["num_labels"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, cfg
