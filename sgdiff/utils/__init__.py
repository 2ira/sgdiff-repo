"""Utility helpers (metrics, seeding, logging, IO)."""

from .io import ensure_dir, load_json, save_json
from .logging import get_logger
from .metrics import average_precision, hit_at_k, roc_auc, summarize_scores
from .seed import set_seed, resolve_device

__all__ = [
    "average_precision",
    "ensure_dir",
    "get_logger",
    "hit_at_k",
    "load_json",
    "resolve_device",
    "roc_auc",
    "save_json",
    "set_seed",
    "summarize_scores",
]
