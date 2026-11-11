"""Scoring- and training-time ablations of the SGDiff likelihood.

Motivation (GCCM, arXiv:2605.05689, 2026): a conditional diffusion model used as
a *predictor* can pass the accuracy bar while the conditioning carries no
information - consistency/denoising training collapses the score to a
deterministic function of the input, and the reported "generative" advantage is
produced by the fusion head rather than by the likelihood.  This module provides
the experiments that decide whether that happens in SGDiff.

Two families:

``scoring-time`` (no retraining)
    Recompute the same checkpoint with a perturbed condition, perturbed DRNL
    labels, a single diffusion step instead of the Monte-Carlo average, or a
    single likelihood stream.  ``same_condition`` is the sanity check: both
    classes are scored with ``y = 0``, so any AUC above chance cannot come from
    the condition; ``single_t_*`` is the GCCM test proper - if the score barely
    changes between t = 1, t = T/2 and t = T, the "diffusion depth" is
    decorative.
``training-time`` (one retrain)
    Train stage 1 with a **random** condition, then evaluate normally.  If AUC
    stays high, the denoiser learns a class-discriminative sub-graph model
    without ever seeing the condition, and the paper's generative reading is
    unsupported.

Every variant is scored on the *same* evaluation pairs as
``scripts/run_benchmark.py``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..models.sgdiff import SGDiff
from ..utils.io import save_json
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device, set_seed
from .batch import PairSubgraphDataset, SubgraphBatch, collate_subgraphs
from .evaluator import score_pairs
from .factory import build_model, load_split, prepare_datasets
from .trainer import Trainer

CONDITIONS = ("true", "same", "flip", "random")
LABEL_MODES = ("true", "shuffle")
STREAMS = ("both", "structure", "feature")
METRICS = ("auc", "ap", "hit@100")


@dataclass(frozen=True)
class Ablation:
    """One perturbation of the likelihood computation.

    Parameters
    ----------
    name:
        Label used in the result table.
    condition:
        ``true`` (paper), ``same`` (both classes use ``y = 0``), ``flip``
        (class ``c`` uses ``y = 1 - c``) or ``random`` (one random label per
        pair, shared by both classes).
    labels:
        ``true`` or ``shuffle`` (permute the DRNL labels inside each sub-graph).
    timestep_frac:
        ``None`` keeps the paper's Monte-Carlo average over diffusion steps;
        a float in ``(0, 1]`` evaluates a **single** step at that fraction of
        each model's own step count (the two streams have different counts).
    stream:
        Keep both likelihoods, or drop one of them from the fusion.
    seed:
        Seed for the random condition / label permutation.
    """

    name: str
    condition: str = "true"
    labels: str = "true"
    timestep_frac: Optional[float] = None
    stream: str = "both"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.labels not in LABEL_MODES:
            raise ValueError(f"labels must be one of {LABEL_MODES}")
        if self.stream not in STREAMS:
            raise ValueError(f"stream must be one of {STREAMS}")
        if self.timestep_frac is not None and not 0.0 < self.timestep_frac <= 1.0:
            raise ValueError("timestep_frac must be in (0, 1]")

    # ------------------------------------------------------------------ inputs
    def y_for(self, batch: SubgraphBatch, cls: int, num_classes: int = 2) -> torch.Tensor:
        """Link condition passed to the denoisers for class ``cls``."""
        b = batch.batch_size
        device = batch.y.device
        if self.condition == "true":
            return torch.full((b,), int(cls), dtype=torch.long, device=device)
        if self.condition == "same":
            return torch.zeros(b, dtype=torch.long, device=device)
        if self.condition == "flip":
            return torch.full((b,), int(num_classes - 1 - cls), dtype=torch.long, device=device)
        generator = torch.Generator().manual_seed(int(self.seed))
        return torch.randint(0, num_classes, (b,), generator=generator).to(device)

    def labels_for(self, batch: SubgraphBatch) -> torch.Tensor:
        """DRNL labels (optionally permuted inside every sub-graph)."""
        if self.labels == "true":
            return batch.labels
        labels = batch.labels.detach().cpu().clone()
        mask = batch.mask.detach().cpu()
        generator = torch.Generator().manual_seed(int(self.seed))
        for i in range(labels.shape[0]):
            index = mask[i].nonzero(as_tuple=False).flatten()
            if index.numel() < 2:
                continue
            perm = index[torch.randperm(index.numel(), generator=generator)]
            labels[i, index] = labels[i, perm]
        return labels.to(batch.labels.device)

    def apply_to_condition(self, batch: SubgraphBatch) -> SubgraphBatch:
        """Training-time variation: replace ``y`` by a random label.

        Only the diffusion losses see the perturbed batch; the cross-entropy
        target of the fusion stage stays the true label.
        """
        if self.condition != "random":
            raise ValueError("training-time ablation supports only condition='random'")
        y = self.y_for(batch, cls=0, num_classes=2)
        return replace(batch, y=y)

    def describe(self) -> str:
        parts = [f"condition={self.condition}", f"labels={self.labels}", f"stream={self.stream}"]
        parts.append(
            "t=MC-avg" if self.timestep_frac is None else f"t={self.timestep_frac:g}·T"
        )
        return ", ".join(parts)


def fusion_parameters(model: SGDiff) -> dict:
    """Read the learned fusion weights ``{eta_1, eta_2, delta}`` (Eq. 13).

    ``eta_1`` multiplies the feature likelihood, ``eta_2`` the structure one.
    A near-zero ``eta_1`` is the direct evidence behind the "the feature stream
    contributes nothing" reading, instead of inferring it from the AUC of
    ``feature_only``.
    """
    fusion = model.fusion
    return {
        "eta1_feature": float(fusion.eta1.detach().cpu()),
        "eta2_structure": float(fusion.eta2.detach().cpu()),
        "delta": float(fusion.delta.detach().cpu()),
        "feature_stream": bool(model.feature is not None),
    }


@torch.no_grad()
def stream_contributions(
    model: SGDiff,
    dataset: PairSubgraphDataset,
    batch_size: int = 64,
    device=None,
    num_samples: Optional[int] = None,
) -> dict:
    """How much of the logit spread each likelihood stream actually contributes.

    The fusion logit difference is ``eta_1 * d_feat + eta_2 * d_struct`` where
    ``d_stream = score(class 1) - score(class 0)``.  Reporting ``eta`` alone is
    misleading: a large weight on a nearly constant stream contributes nothing.
    This function returns the weights *and* the standard deviation of each term,
    plus the share of the total variance, so "the feature stream is dead" becomes
    a measurement instead of an inference from AUC.
    """
    device = device or next(model.parameters()).device
    scorer = AblatedScorer(model, Ablation("contributions")).to(device)
    scorer.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
    )
    feat_diff, struct_diff = [], []
    for batch in loader:
        feat, struct = scorer.class_scores(batch.to(device))
        feat_diff.append((feat[:, 1] - feat[:, 0]).float().cpu())
        struct_diff.append((struct[:, 1] - struct[:, 0]).float().cpu())
    d_feat = torch.cat(feat_diff).numpy().astype(np.float64)
    d_struct = torch.cat(struct_diff).numpy().astype(np.float64)

    eta1 = float(model.fusion.eta1.detach().cpu())
    eta2 = float(model.fusion.eta2.detach().cpu())
    term_feat = eta1 * d_feat
    term_struct = eta2 * d_struct
    var_feat = float(np.var(term_feat))
    var_struct = float(np.var(term_struct))
    total = max(var_feat + var_struct, 1e-12)
    out = fusion_parameters(model)
    out.update(
        {
            "std_feature_term": float(np.std(term_feat)),
            "std_structure_term": float(np.std(term_struct)),
            "share_feature": var_feat / total,
            "share_structure": var_struct / total,
            "corr_terms": float(np.corrcoef(term_feat, term_struct)[0, 1])
            if np.std(term_feat) > 1e-12 and np.std(term_struct) > 1e-12
            else float("nan"),
        }
    )
    return out


def standard_ablations(seed: int = 0) -> list[Ablation]:
    """The evaluation-only suite run on every checkpoint."""
    return [
        Ablation("reference", seed=seed),
        Ablation("same_condition", condition="same", seed=seed),
        Ablation("condition_flip", condition="flip", seed=seed),
        Ablation("label_shuffle", labels="shuffle", seed=seed),
        Ablation("single_t_low", timestep_frac=0.05, seed=seed),
        Ablation("single_t_mid", timestep_frac=0.5, seed=seed),
        Ablation("single_t_full", timestep_frac=1.0, seed=seed),
        Ablation("structure_only", stream="structure", seed=seed),
        Ablation("feature_only", stream="feature", seed=seed),
    ]


class AblatedScorer(nn.Module):
    """Wrap a trained :class:`SGDiff` and score pairs under an :class:`Ablation`.

    Mirrors ``SGDiff.class_scores`` (which hard-codes ``y = cls`` and averages
    over the Monte-Carlo steps) but lets every input of the likelihood be
    perturbed.
    """

    def __init__(self, model: SGDiff, ablation: Ablation) -> None:
        super().__init__()
        self.model = model
        self.ablation = ablation
        self.num_classes = model.fusion.num_classes
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    # ------------------------------------------------------------- per stream
    def _structure_score(self, batch, labels, y):
        if self.ablation.stream == "feature":
            return torch.zeros(batch.batch_size, device=batch.y.device)
        structure = self.model.structure
        if self.ablation.timestep_frac is None:
            return structure.log_likelihood(batch.adj, labels, batch.mask, y)
        step = max(1, round(self.ablation.timestep_frac * structure.num_steps))
        t = torch.full((batch.batch_size,), step, dtype=torch.long, device=batch.adj.device)
        return -structure.compute_kl(batch.adj, labels, batch.mask, y, t)

    def _feature_score(self, batch, labels, y):
        if self.model.feature is None or self.ablation.stream == "structure":
            return torch.zeros(batch.batch_size, device=batch.y.device)
        feature = self.model.feature
        if self.ablation.timestep_frac is None:
            return feature.log_likelihood(batch.x, batch.adj, labels, batch.mask, y)
        step = max(1, round(self.ablation.timestep_frac * feature.num_steps))
        t = torch.full((batch.batch_size,), step, dtype=torch.long, device=batch.adj.device)
        return -feature.compute_mse(batch.x, batch.adj, labels, batch.mask, y, t)

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def class_scores(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        labels = self.ablation.labels_for(batch)
        feat, struct = [], []
        for cls in range(self.num_classes):
            y = self.ablation.y_for(batch, cls, self.num_classes)
            feat.append(self._feature_score(batch, labels, y))
            struct.append(self._structure_score(batch, labels, y))
        return torch.stack(feat, dim=1), torch.stack(struct, dim=1)

    @torch.no_grad()
    def forward(self, batch, num_samples: Optional[int] = None) -> dict:
        feat, struct = self.class_scores(batch)
        logits = self.model.fusion(feat, struct)
        return {"logits": logits, "feature_scores": feat, "structure_scores": struct}

    @torch.no_grad()
    def predict_logit(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        logits = self.forward(batch)["logits"]
        return logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def predict_proba(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        return torch.softmax(self.forward(batch)["logits"], dim=-1)[:, 1]


@torch.no_grad()
def evaluate_ablation(
    model: SGDiff,
    dataset: PairSubgraphDataset,
    ablation: Ablation,
    batch_size: int = 64,
    device=None,
    num_samples: Optional[int] = None,
) -> dict:
    """AUC / AP / Hit@100 of ``model`` on ``dataset`` under ``ablation``."""
    scorer = AblatedScorer(model, ablation).to(next(model.parameters()).device)
    scorer.eval()
    scores = score_pairs(
        scorer,
        dataset.extractor,
        dataset.pairs,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
        seed=ablation.seed,
    )
    metrics = summarize_scores(dataset.labels, scores)
    metrics["score_std"] = float(np.std(scores))
    return metrics


def format_markdown(result: dict) -> str:
    """Render the ablation table."""
    header = ["variant", "training", "scoring", "AUC", "AP", "Hit@100", "score std"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in result["rows"]:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    row["training"],
                    row["scoring"],
                    f"{metrics['auc']:.4f}",
                    f"{metrics['ap']:.4f}",
                    f"{metrics['hit@100']:.4f}",
                    f"{metrics['score_std']:.4f}",
                ]
            )
            + " |"
        )
    fusion_rows = result.get("fusion") or []
    if fusion_rows:
        lines += [
            "",
            "Fusion weights and what each stream contributes to the logit spread "
            "(`d = score(class 1) - score(class 0)` on the test pairs):",
            "",
            "| model | eta_1 (feature) | eta_2 (structure) | delta | std(eta_1*d_feat) | "
            "std(eta_2*d_struct) | var share feat | corr |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in fusion_rows:
            lines.append(
                f"| {row['model']} | {row['eta1_feature']:.4f} | {row['eta2_structure']:.4f} | "
                f"{row['delta']:.4f} | {row['std_feature_term']:.4f} | "
                f"{row['std_structure_term']:.4f} | {row['share_feature']:.3f} | "
                f"{row['corr_terms']:.3f} |"
            )
    return "\n".join(lines)


def run_ablation(
    config: ExperimentConfig,
    seed: int = 0,
    sgdiff_epochs: Optional[int] = None,
    device: str = "auto",
    output_dir: str = "results",
    run_root: str = "runs/ablation",
    ablations: Optional[Sequence[Ablation]] = None,
    train_without_condition: bool = True,
    unfrozen_variant: bool = True,
    tag: str = "",
    write: bool = True,
    verbose: bool = True,
) -> dict:
    """Train the reference model, run every scoring ablation, optionally retrain.

    A second model is trained with ``condition='random'`` during stage 1 so that
    the conditioning is destroyed while everything else (data, architecture,
    schedule) is held fixed.  With ``unfrozen_variant`` a third model repeats
    that run while letting the fusion stage back-propagate into the denoisers -
    the control that shows how the fusion loss can re-inject the condition.
    """
    device = resolve_device(device)
    cfg = copy.deepcopy(config)
    cfg.data.seed = int(seed)
    cfg.train.seed = int(seed)
    if sgdiff_epochs is not None:
        cfg.train.epochs = int(sgdiff_epochs)
        cfg.train.pretrain_epochs = min(cfg.train.pretrain_epochs, cfg.train.epochs)
    set_seed(int(seed))

    graph, split = load_split(cfg)
    datasets = prepare_datasets(graph, split, cfg)
    test = datasets["test"]
    feature_dim = datasets["train"].feature_dim

    prefix = f"_{tag}" if tag else ""

    def _fit(
        run_name: str, ablation: Optional[Ablation], freeze_diffusion: Optional[bool] = None
    ) -> tuple[SGDiff, dict]:
        model = build_model(cfg, feature_dim)
        run_cfg = copy.deepcopy(cfg)
        run_cfg.train.run_name = f"{run_name}{prefix}"
        if freeze_diffusion is not None:
            run_cfg.train.freeze_diffusion = bool(freeze_diffusion)
        trainer = Trainer(model, run_cfg, run_dir=f"{run_root}/{cfg.data.name}", train_ablation=ablation)
        summary = trainer.fit(datasets["train"], datasets["val"], test)
        return model, summary["test"]

    started = time.perf_counter()
    reference_model, reference_metrics = _fit(f"reference-s{seed}", None)
    reference_seconds = time.perf_counter() - started
    contributions = stream_contributions(
        reference_model,
        test,
        batch_size=cfg.train.eval_batch_size,
        device=device,
        num_samples=cfg.train.finetune_mc_samples,
    )
    fusion_rows = [{"model": "reference", **contributions}]
    if verbose:
        print(
            f"fusion: eta1(feature)={contributions['eta1_feature']:.4f} "
            f"eta2(structure)={contributions['eta2_structure']:.4f} "
            f"delta={contributions['delta']:.4f}"
        )
        print(
            f"logit spread: std(eta1*d_feat)={contributions['std_feature_term']:.4f} "
            f"std(eta2*d_struct)={contributions['std_structure_term']:.4f} "
            f"variance share feature={contributions['share_feature']:.3f}"
        )

    ablations = list(ablations or standard_ablations(seed))
    rows = []
    for ablation in ablations:
        began = time.perf_counter()
        metrics = evaluate_ablation(
            reference_model,
            test,
            ablation,
            batch_size=cfg.train.eval_batch_size,
            device=device,
            num_samples=cfg.train.finetune_mc_samples,
        )
        rows.append(
            {
                "variant": ablation.name,
                "training": "condition=true",
                "scoring": ablation.describe(),
                "metrics": metrics,
                "seconds": float(time.perf_counter() - began),
            }
        )
        if verbose:
            print(
                f"{ablation.name:22s} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
                f"hit@100={metrics['hit@100']:.4f} std={metrics['score_std']:.4f}"
            )

    result = {
        "dataset": cfg.data.name,
        "seed": int(seed),
        "protocol": {
            "split": "80/5/15 positives with equal-size random negatives",
            "num_test_pairs": int(len(test)),
            "epochs": int(cfg.train.epochs),
            "pretrain_epochs": int(cfg.train.pretrain_epochs),
            "num_mc_samples": int(cfg.train.finetune_mc_samples),
            "device": str(device),
        },
        "reference_training": {
            "metrics": reference_metrics,
            "seconds": float(reference_seconds),
        },
        "rows": rows,
    }

    if train_without_condition:
        # The *clean* variant freezes the denoisers during stage 2: otherwise the
        # fusion cross-entropy back-propagates into them and re-teaches the
        # condition that stage 1 never saw, which defeats the ablation.
        variants = [("trained_without_condition", True)]
        if unfrozen_variant:
            variants.append(("trained_without_condition_unfrozen", False))
        for variant, freeze in variants:
            started = time.perf_counter()
            without, _ = _fit(
                f"{variant}-s{seed}",
                Ablation("random_condition", condition="random", seed=seed),
                freeze_diffusion=freeze,
            )
            metrics = evaluate_ablation(
                without,
                test,
                Ablation("reference", seed=seed),
                batch_size=cfg.train.eval_batch_size,
                device=device,
                num_samples=cfg.train.finetune_mc_samples,
            )
            result["rows"].append(
                {
                    "variant": variant,
                    "training": "condition=random" + (", denoisers frozen in stage 2" if freeze else ""),
                    "scoring": "condition=true (paper scoring)",
                    "metrics": metrics,
                    "seconds": float(time.perf_counter() - started),
                }
            )
            fusion = stream_contributions(
                without,
                test,
                batch_size=cfg.train.eval_batch_size,
                device=device,
                num_samples=cfg.train.finetune_mc_samples,
            )
            fusion_rows.append({"model": variant, **fusion})
            if verbose:
                print(
                    f"{variant:34s} auc={metrics['auc']:.4f} "
                    f"ap={metrics['ap']:.4f} hit@100={metrics['hit@100']:.4f} "
                    f"eta1={fusion['eta1_feature']:.3f} eta2={fusion['eta2_structure']:.3f}"
                )

    result["fusion"] = fusion_rows
    result["markdown"] = format_markdown(result)
    if write:
        suffix = f"_{tag}" if tag else ""
        stem = f"{output_dir}/ablation_{cfg.data.name}{suffix}"
        save_json(result, f"{stem}.json")
        with open(f"{stem}.md", "w", encoding="utf-8") as handle:
            handle.write(_header(result) + result["markdown"] + "\n")
    return result


def _header(result: dict) -> str:
    return "\n".join(
        [
            f"# Likelihood ablations: {result['dataset']}",
            "",
            "Generated by `scripts/run_ablation.py`. All rows use the same checkpoint",
            "and the same evaluation pairs, except `trained_without_condition`, which is",
            "a second run with a randomized stage-1 condition. See `docs/ablation.md`.",
            "",
            f"* seed: {result['seed']}",
            f"* test pairs: {result['protocol']['num_test_pairs']}",
            f"* reference: auc={result['reference_training']['metrics']['auc']:.4f}, "
            f"ap={result['reference_training']['metrics']['ap']:.4f}, "
            f"hit@100={result['reference_training']['metrics']['hit@100']:.4f}",
            "",
            "",
        ]
    )
