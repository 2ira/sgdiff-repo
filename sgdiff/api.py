"""High-level Python API.

Everything the CLI does is available as a plain function so that the
reproduction can be scripted from notebooks / other projects::

    from sgdiff import api

    model, result = api.train(config_path="configs/sgdiff_cora.yaml")
    scores = api.predict(model, graph, [(0, 1), (2, 3)], config=result["config"])
    metrics = api.evaluate(model, graph, pos_pairs, neg_pairs, config=result["config"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import ExperimentConfig, load_config
from .data.datasets import DATASETS, available_datasets, describe_dataset, load_graph
from .data.splits import LinkSplit, make_link_split
from .data.subgraph import SubgraphExtractor
from .engine.evaluator import evaluate_link_prediction, score_pairs
from .engine.factory import build_extractor, build_model, load_split, prepare_datasets
from .engine.trainer import Trainer, load_checkpoint
from .utils.logging import get_logger
from .utils.seed import set_seed

LOGGER = get_logger(__name__)

__all__ = [
    "DATASETS",
    "available_datasets",
    "build_extractor",
    "build_model",
    "describe_dataset",
    "evaluate",
    "load_checkpoint",
    "load_config",
    "load_graph",
    "load_split",
    "predict",
    "prepare",
    "train",
]


def prepare(config: ExperimentConfig, download: bool = True) -> dict:
    """Load a dataset, build the 80/5/15 split and the sub-graph datasets."""
    graph, split = load_split(config)
    datasets = prepare_datasets(graph, split, config)
    return {"graph": graph, "split": split, "datasets": datasets}


def train(
    config: Optional[ExperimentConfig] = None,
    config_path: Optional[str | Path] = None,
    overrides: Optional[Sequence[str]] = None,
    run_dir: Optional[str | Path] = None,
) -> tuple:
    """Train SGDiff and return ``(model, summary, config)``."""
    config = config or load_config(config_path, list(overrides or []))
    set_seed(config.train.seed)
    prepared = prepare(config)
    datasets = prepared["datasets"]
    model = build_model(config, datasets["train"].feature_dim)
    trainer = Trainer(model, config, run_dir=run_dir)
    summary = trainer.fit(datasets["train"], datasets["val"], datasets["test"])
    summary["config"] = config
    return model, summary, config


def predict(
    model,
    graph,
    pairs: Sequence,
    config: ExperimentConfig,
    eval_graph=None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Ranking scores (higher = more likely a link) for ``pairs``."""
    extractor = build_extractor(eval_graph or graph, config)
    return score_pairs(
        model,
        extractor,
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        batch_size=batch_size or config.train.eval_batch_size,
        device=device,
        num_samples=num_samples,
    )


def evaluate(
    model,
    graph,
    pos_pairs,
    neg_pairs,
    config: ExperimentConfig,
    eval_graph=None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    """AUC / AP / Hit@100 for a positive and a negative pair set."""
    extractor = build_extractor(eval_graph or graph, config)
    return evaluate_link_prediction(
        model,
        extractor,
        np.asarray(pos_pairs, dtype=np.int64),
        np.asarray(neg_pairs, dtype=np.int64),
        batch_size=batch_size or config.train.eval_batch_size,
        device=device,
    )


def make_extractor(graph, config: ExperimentConfig) -> SubgraphExtractor:
    """Convenience wrapper around :func:`build_extractor`."""
    return build_extractor(graph, config)


def make_split(graph, config: ExperimentConfig) -> LinkSplit:
    """Convenience wrapper around :func:`make_link_split`."""
    return make_link_split(
        graph,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        neg_ratio=config.data.neg_ratio,
        seed=config.data.seed,
    )
