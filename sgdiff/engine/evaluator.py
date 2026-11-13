"""Scoring and metric computation for link prediction."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.subgraph import SubgraphExtractor
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device
from .batch import PairSubgraphDataset, collate_subgraphs


def _loader(dataset: PairSubgraphDataset, batch_size: int, shuffle: bool = False, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
    )


@contextmanager
def _evaluation_context(model: torch.nn.Module, seed: Optional[int] = 0):
    """Isolate scoring RNGs and restore every module's training mode on exit."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_state = torch.mps.get_rng_state() if mps_available else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if seed is not None:
            torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        for module, training in modes:
            module.training = training


@torch.no_grad()
def score_pairs(
    model: torch.nn.Module,
    extractor: SubgraphExtractor,
    pairs: np.ndarray,
    batch_size: int = 64,
    device=None,
    num_samples: Optional[int] = None,
    output: str = "logit",
    num_workers: int = 0,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Return a ranking score for every ``(u, v)`` pair.

    ``output`` is ``"logit"`` (log-odds of the link class) or ``"proba"``.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    device = resolve_device(device) if isinstance(device, str) else device
    if device is None:
        device = next(model.parameters()).device
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    dataset = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
    loader = _loader(dataset, batch_size, num_workers=num_workers)
    if output not in {"logit", "proba"}:
        raise ValueError("output must be 'logit' or 'proba'")
    scores: list[np.ndarray] = []
    with _evaluation_context(model, seed):
        for sub_batch in loader:
            sub_batch = sub_batch.to(device)
            if output == "proba":
                value = model.predict_proba(sub_batch, num_samples=num_samples)
            else:
                value = model.predict_logit(sub_batch, num_samples=num_samples)
            scores.append(value.detach().float().cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def score_pairs_logits(
    model: torch.nn.Module,
    extractor: SubgraphExtractor,
    pairs: np.ndarray,
    batch_size: int = 64,
    device=None,
    num_workers: int = 0,
    seed: Optional[int] = 0,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Return raw ``(N, 2)`` fused logits without advancing training RNGs."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    device = resolve_device(device) if isinstance(device, str) else device
    if device is None:
        device = next(model.parameters()).device
    dataset = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
    loader = _loader(dataset, batch_size, num_workers=num_workers)
    out: list[np.ndarray] = []
    with _evaluation_context(model, seed):
        for sub_batch in loader:
            sub_batch = sub_batch.to(device)
            result = model(sub_batch) if num_samples is None else model(sub_batch, num_samples=num_samples)
            out.append(result["logits"].detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)


def evaluate_link_prediction(
    model: torch.nn.Module,
    extractor: SubgraphExtractor,
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    batch_size: int = 64,
    device=None,
    num_samples: Optional[int] = None,
    num_workers: int = 0,
    hit_k: int = 100,
    seed: Optional[int] = 0,
) -> dict:
    """AUC / AP / pooled precision@k (legacy ``hit@k``) for the candidate set."""
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    neg_pairs = np.asarray(neg_pairs, dtype=np.int64).reshape(-1, 2)
    pos_scores = score_pairs(
        model, extractor, pos_pairs, batch_size, device, num_samples,
        num_workers=num_workers, seed=seed,
    )
    neg_scores = score_pairs(
        model, extractor, neg_pairs, batch_size, device, num_samples,
        num_workers=num_workers, seed=seed,
    )
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    metrics = summarize_scores(labels, scores, k=hit_k)
    metrics.update({"num_pos": int(len(pos_scores)), "num_neg": int(len(neg_scores))})
    return metrics
