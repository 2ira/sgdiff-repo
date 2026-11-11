"""Data-identical benchmarking harness for link prediction models.

The reproduction compares SGDiff against baselines that must be trained and
evaluated on **exactly** the same data.  This module makes that guarantee
explicit and checkable:

* one :func:`~sgdiff.data.splits.make_link_split` per seed, shared by every
  model (``split_fingerprint`` hashes the six pair arrays);
* one :func:`~sgdiff.engine.factory.prepare_datasets` per seed, so the
  enclosing sub-graphs, DRNL labels and node features are bit-identical across
  models (``subgraph_fingerprint`` hashes extracted sub-graphs);
* one metric implementation (:mod:`sgdiff.utils.metrics`) and one evaluation
  pair set, so AUC / AP / Hit@100 are directly comparable.

Adding a model means adding a :class:`ModelSpec` to
:func:`build_registry`; the training entry points are the existing
``train_node_pair_baseline`` / ``train_subgraph_classifier`` / ``Trainer``.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.graph import GraphData
from ..data.splits import LinkSplit
from ..models.baselines import NodePairLinkPredictor, PENCILBaseline, SEALBaseline, VGAE
from ..models.sgdiff import SGDiff
from ..utils.io import save_json
from ..utils.seed import resolve_device, set_seed
from .baseline_trainer import (
    node_feature_matrix,
    train_node_pair_baseline,
    train_subgraph_classifier,
)
from .factory import build_model, load_split, prepare_datasets, training_pairs
from .trainer import Trainer

METRICS = ("auc", "ap", "hit@100")
KINDS = ("node_pair", "subgraph", "sgdiff")

# Budgets shared by scripts/run_benchmark.py and scripts/run_ablation.py; they
# mirror the profiles of scripts/run_experiments.py so that every table in
# docs/ is directly comparable. ``paper`` keeps the config defaults.
BUDGET_PROFILES: dict[str, list[str]] = {
    "paper": [],
    "cpu": [
        "data.max_train_pairs=2000",
        "data.max_eval_pairs=500",
        "train.epochs=8",
        "train.pretrain_epochs=3",
        "structure.num_mc_samples=4",
        "feature.num_mc_samples=4",
        "subgraph.max_nodes_total=40",
    ],
    # Table-4 data and hyper-parameters with a longer, still CPU-feasible
    # schedule (mirrors scripts/run_experiments.py).
    "full": [
        "data.max_train_pairs=null",
        "data.max_eval_pairs=500",
        "train.epochs=20",
        "train.pretrain_epochs=4",
        "train.eval_every=4",
        "structure.num_mc_samples=8",
        "feature.num_mc_samples=8",
        "subgraph.max_nodes_total=60",
    ],
    "smoke": [
        "data.name=synthetic",
        "data.max_train_pairs=128",
        "data.max_eval_pairs=40",
        "subgraph.max_nodes_per_hop=8",
        "structure.num_steps=4",
        "feature.num_steps=4",
        "structure.hidden_dim=16",
        "feature.hidden_dim=16",
        "structure.num_layers=1",
        "feature.num_layers=1",
        "structure.num_mc_samples=2",
        "feature.num_mc_samples=2",
        "train.epochs=2",
        "train.pretrain_epochs=1",
        "train.batch_size=16",
        "train.eval_batch_size=16",
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    """How to build and train one comparable model.

    ``kind`` selects the training loop:

    ``node_pair``
        Full-graph encoder + dot-product decoder, trained with
        :func:`~sgdiff.engine.baseline_trainer.train_node_pair_baseline`.
    ``subgraph``
        Sub-graph classifier over the shared enclosing sub-graphs, trained with
        :func:`~sgdiff.engine.baseline_trainer.train_subgraph_classifier`.
    ``sgdiff``
        The diffusion model of the paper, trained with
        :class:`~sgdiff.engine.trainer.Trainer`.
    """

    name: str
    kind: str
    builder: Callable[[int, ExperimentConfig], torch.nn.Module]
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind '{self.kind}', expected one of {KINDS}")


def build_registry(cfg: ExperimentConfig, feature_dim: Optional[int] = None) -> dict[str, ModelSpec]:
    """Return the model zoo used by the benchmark, keyed by name."""
    num_labels = cfg.subgraph.max_label + 1
    registry = {
        "gcn": ModelSpec(
            "gcn",
            "node_pair",
            lambda fd, c: NodePairLinkPredictor(fd, 64, 64, 2, arch="gcn"),
            note="full graph, 2-layer GCN + dot product",
        ),
        "gat": ModelSpec(
            "gat",
            "node_pair",
            lambda fd, c: NodePairLinkPredictor(fd, 64, 64, 2, arch="gat"),
            note="full graph, 2-layer GAT + dot product",
        ),
        "sage": ModelSpec(
            "sage",
            "node_pair",
            lambda fd, c: NodePairLinkPredictor(fd, 64, 64, 2, arch="sage"),
            note="full graph, 2-layer GraphSAGE + dot product",
        ),
        "vgae": ModelSpec("vgae", "node_pair", lambda fd, c: VGAE(fd), note="variational graph auto-encoder"),
        "seal": ModelSpec(
            "seal",
            "subgraph",
            lambda fd, c: SEALBaseline(fd, num_labels=num_labels, hidden_dim=32, num_layers=4),
            note="4-layer GCN on the shared enclosing sub-graphs",
        ),
        "pencil": ModelSpec(
            "pencil",
            "subgraph",
            lambda fd, c: PENCILBaseline(
                fd, num_labels=num_labels, hidden_dim=64, num_layers=2, heads=4, dropout=0.1
            ),
            note="plain Transformer on the shared enclosing sub-graphs (arXiv:2602.01553)",
        ),
        "sgdiff": ModelSpec(
            "sgdiff",
            "sgdiff",
            lambda fd, c: build_model(c, fd),
            note="the paper's sub-graph diffusion model",
        ),
    }
    if feature_dim is not None:  # early validation of the shared feature width
        for spec in registry.values():
            spec.builder(feature_dim, cfg)
    return registry


# --------------------------------------------------------------------------- #
# Data-integrity fingerprints
# --------------------------------------------------------------------------- #
def split_fingerprint(split: LinkSplit) -> str:
    """Stable hash of the six positive/negative pair arrays of a split."""
    digest = hashlib.sha1()
    for name in ("train_pos", "train_neg", "val_pos", "val_neg", "test_pos", "test_neg"):
        pairs = np.asarray(getattr(split, name), dtype=np.int64).reshape(-1, 2)
        canon = np.sort(pairs, axis=1)
        digest.update(name.encode("ascii"))
        digest.update(str(canon.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(canon).tobytes())
    return digest.hexdigest()[:16]


def pairs_fingerprint(pairs: np.ndarray, labels: np.ndarray) -> str:
    """Stable hash of a training pair set (order-free, label-aware)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    order = np.lexsort((np.sort(pairs, axis=1)[:, 1], np.sort(pairs, axis=1)[:, 0], labels))
    canon = np.concatenate([np.sort(pairs[order], axis=1), labels[order, None]], axis=1)
    return hashlib.sha1(np.ascontiguousarray(canon).tobytes()).hexdigest()[:16]


def subgraph_fingerprint(datasets: dict, limit: int = 32) -> str:
    """Stable hash of the first ``limit`` test sub-graphs (nodes / labels / adjacency)."""
    dataset = datasets.get("test")
    if dataset is None or len(dataset) == 0:
        return "empty"
    digest = hashlib.sha1()
    for index in range(min(limit, len(dataset))):
        sub, label = dataset[index]
        digest.update(np.ascontiguousarray(sub.nodes, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(sub.labels, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(sub.adj, dtype=np.float32).tobytes())
        digest.update(str(int(label)).encode("ascii"))
        if sub.x is not None:
            digest.update(np.ascontiguousarray(sub.x, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Training dispatch
# --------------------------------------------------------------------------- #
def _metrics_from_result(kind: str, result: dict) -> dict:
    if kind == "sgdiff":
        return {k: float(v) for k, v in result.get("test", {}).items() if k in METRICS}
    if kind == "subgraph":
        return {
            k[len("test_") :]: float(v)
            for k, v in result.items()
            if k.startswith("test_") and k[len("test_") :] in METRICS
        }
    return {k: float(result[k]) for k in METRICS if k in result}


def capped_train_split(split: LinkSplit, cfg: ExperimentConfig) -> LinkSplit:
    """Restrict the training pairs to the ones the sub-graph models actually use.

    Without this, ``data.max_train_pairs`` (and the 1% ``train_ratio`` study)
    would shrink the sub-graph training set while the full-graph baselines still
    saw every training edge - a silent budget asymmetry.
    """
    pairs, labels = training_pairs(split, cfg)
    if len(pairs) == len(split.train_pos) + len(split.train_neg):
        return split
    capped = copy.copy(split)
    capped.train_pos = pairs[labels == 1]
    capped.train_neg = pairs[labels == 0]
    return capped


def _fit(
    spec: ModelSpec,
    cfg: ExperimentConfig,
    graph: GraphData,
    split: LinkSplit,
    datasets: dict,
    seed: int,
    sgdiff_epochs: Optional[int],
    subgraph_epochs: int,
    node_pair_epochs: int,
    device: str,
    run_root: str,
) -> tuple[dict, int]:
    """Train one model and return ``(metrics, num_params)``."""
    if spec.kind == "node_pair":
        features = node_feature_matrix(graph)
        model = spec.builder(features.shape[1], cfg)
        num_params = sum(p.numel() for p in model.parameters())
        result = train_node_pair_baseline(
            model,
            graph,
            capped_train_split(split, cfg),
            device=device,
            epochs=node_pair_epochs,
            is_vgae=(spec.name == "vgae"),
            seed=seed,
        )
    elif spec.kind == "subgraph":
        model = spec.builder(datasets["train"].feature_dim, cfg)
        num_params = sum(p.numel() for p in model.parameters())
        result = train_subgraph_classifier(
            model, datasets, device=device, epochs=subgraph_epochs, seed=seed
        )
    else:
        model = spec.builder(datasets["train"].feature_dim, cfg)
        num_params = sum(p.numel() for p in model.parameters())
        cfg_run = copy.deepcopy(cfg)
        cfg_run.train.run_name = f"{spec.name}-s{seed}"
        trainer = Trainer(model, cfg_run, run_dir=run_root)
        result = trainer.fit(datasets["train"], datasets["val"], datasets["test"])
    return _metrics_from_result(spec.kind, result), int(num_params)


def _aggregate(runs: Sequence[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    names = sorted({run["model"] for run in runs})
    for name in names:
        subset = [run for run in runs if run["model"] == name]
        per_metric: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            values = [run["metrics"][metric] for run in subset if metric in run["metrics"]]
            values = [v for v in values if np.isfinite(v)]
            if values:
                per_metric[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        per_metric["seconds"] = {
            "mean": float(np.mean([run["seconds"] for run in subset])),
            "std": 0.0,
            "n": len(subset),
        }
        summary[name] = per_metric
    return summary


def format_markdown(result: dict) -> str:
    """Render the aggregated results as a markdown table."""
    summary = result["summary"]
    header = ["model", "AUC", "AP", "Hit@100", "#params", "fit (s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    ranked = sorted(summary.items(), key=lambda kv: -kv[1].get("auc", {}).get("mean", -1.0))
    for name, metrics in ranked:
        cells = [name]
        for metric in METRICS:
            entry = metrics.get(metric)
            if entry is None:
                cells.append("-")
            elif entry["n"] > 1:
                cells.append(f"{entry['mean']:.4f} ± {entry['std']:.4f}")
            else:
                cells.append(f"{entry['mean']:.4f}")
        params = next((run["num_params"] for run in result["runs"] if run["model"] == name), 0)
        cells.append(f"{params:,}")
        cells.append(f"{metrics['seconds']['mean']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_benchmark(
    config: ExperimentConfig,
    models: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0,),
    sgdiff_epochs: Optional[int] = None,
    subgraph_epochs: int = 40,
    node_pair_epochs: int = 300,
    device: str = "auto",
    output_dir: str = "results",
    run_root: str = "runs/benchmark",
    write: bool = True,
    quiet: bool = False,
    verbose: bool = True,
) -> dict:
    """Train every model on the same split / sub-graphs and aggregate the metrics.

    For each seed the split and the sub-graph datasets are rebuilt from that
    seed, then *all* models consume those exact objects; the resulting
    fingerprints are stored next to the metrics so a reader can verify that the
    comparison is data-identical.
    """
    if quiet:
        logging.getLogger("sgdiff").setLevel(logging.WARNING)
    device = resolve_device(device)

    # Cheap pre-flight: the registry validates every builder against the shared
    # feature width of the first seed before any training starts.
    registry = build_registry(config)
    models = list(models) if models else ["sgdiff", "seal", "pencil", "gcn", "gat", "sage", "vgae"]
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise KeyError(f"unknown models {unknown}; available: {sorted(registry)}")

    runs: list[dict] = []
    integrity: dict[int, dict] = {}
    total = len(seeds) * len(models)
    done = 0

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.data.seed = int(seed)
        cfg.train.seed = int(seed)
        set_seed(int(seed))

        graph, split = load_split(cfg)
        datasets = prepare_datasets(graph, split, cfg)
        split_fp = split_fingerprint(split)
        sub_fp = subgraph_fingerprint(datasets)
        train_pairs, train_labels = training_pairs(split, cfg)
        train_fp = pairs_fingerprint(train_pairs, train_labels)
        integrity[int(seed)] = {
            "split_fingerprint": split_fp,
            "subgraph_fingerprint": sub_fp,
            "train_pairs_fingerprint": train_fp,
            "num_train_pairs": int(len(train_pairs)),
            "num_val_pairs": int(len(datasets["val"])),
            "num_test_pairs": int(len(datasets["test"])),
        }

        for name in models:
            spec = registry[name]
            started = time.perf_counter()
            metrics, num_params = _fit(
                spec,
                cfg,
                graph,
                split,
                datasets,
                int(seed),
                sgdiff_epochs,
                subgraph_epochs,
                node_pair_epochs,
                device,
                run_root=f"{run_root}/{cfg.data.name}",
            )
            elapsed = time.perf_counter() - started
            runs.append(
                {
                    "model": name,
                    "seed": int(seed),
                    "metrics": metrics,
                    "seconds": float(elapsed),
                    "num_params": num_params,
                    "split_fingerprint": split_fp,
                    "subgraph_fingerprint": sub_fp,
                    "train_pairs_fingerprint": train_fp,
                }
            )
            done += 1
            if verbose:
                printable = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[{done}/{total}] {cfg.data.name} seed={seed} {name}: {printable} ({elapsed:.1f}s)")

    _check_identity(runs)
    result = {
        "dataset": config.data.name,
        "models": models,
        "seeds": [int(s) for s in seeds],
        "protocol": {
            "split": "80/5/15 (positives) with equal-size random negatives",
            "neg_ratio": config.data.neg_ratio,
            "subgraph": {
                "hops": config.subgraph.hops,
                "max_nodes_per_hop": config.subgraph.max_nodes_per_hop,
                "max_nodes_total": config.subgraph.max_nodes_total,
                "remove_target_edge": config.subgraph.remove_target_edge,
            },
            "metrics": list(METRICS),
            "epochs": {
                "sgdiff": sgdiff_epochs if sgdiff_epochs is not None else config.train.epochs,
                "subgraph": subgraph_epochs,
                "node_pair": node_pair_epochs,
            },
            "device": str(device),
        },
        "data_integrity": {str(k): v for k, v in integrity.items()},
        "runs": runs,
        "summary": _aggregate(runs),
    }
    result["markdown"] = format_markdown(result)

    if write:
        stem = f"{output_dir}/benchmark_{config.data.name}"
        save_json(result, f"{stem}.json")
        with open(f"{stem}.md", "w", encoding="utf-8") as fh:
            fh.write(_header(result) + result["markdown"] + "\n")
    return result


def _check_identity(runs: Sequence[dict]) -> None:
    """Fail loudly if two models in the same seed saw different data."""
    by_seed: dict[int, set[tuple[str, str, str]]] = {}
    for run in runs:
        by_seed.setdefault(run["seed"], set()).add(
            (
                run["split_fingerprint"],
                run["subgraph_fingerprint"],
                run.get("train_pairs_fingerprint", ""),
            )
        )
    for seed, fingerprints in by_seed.items():
        if len(fingerprints) > 1:
            raise RuntimeError(f"seed {seed}: models were trained on different data: {fingerprints}")


def _header(result: dict) -> str:
    lines = [
        f"# Benchmark: {result['dataset']}",
        "",
        "Generated by `scripts/run_benchmark.py`. Every row was trained on the same",
        "80/5/15 split and the same enclosing sub-graphs (see `data_integrity` in the",
        "matching JSON).",
        "",
        f"* seeds: {result['seeds']}",
        f"* metrics: {', '.join(result['protocol']['metrics'])} (equal-size random negatives)",
        "* `#params` = trainable parameters of the scorer",
        "",
        "",
    ]
    return "\n".join(lines)
