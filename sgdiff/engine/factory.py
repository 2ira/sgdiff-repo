"""Build models / datasets / extractors from an :class:`ExperimentConfig`."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.datasets import load_graph
from ..data.graph import GraphData
from ..data.splits import LinkSplit, make_link_split, subsample_pairs
from ..data.subgraph import SubgraphExtractor
from ..models.sgdiff import SGDiff
from .batch import PairSubgraphDataset


def build_model(cfg: ExperimentConfig, feature_dim: int, num_labels: Optional[int] = None) -> SGDiff:
    """Instantiate SGDiff from the experiment configuration."""
    num_labels = int(num_labels if num_labels is not None else cfg.subgraph.max_label + 1)
    structure_kwargs = dict(
        arch=cfg.structure.arch,
        node_attn=cfg.structure.node_attn,
        edge_attn=cfg.structure.edge_attn,
        global_attn=cfg.structure.global_attn,
        node_mlp=cfg.structure.node_mlp,
        edge_mlp=cfg.structure.edge_mlp,
        global_mlp=cfg.structure.global_mlp,
        heads=cfg.structure.heads,
        aux_ce_weight=cfg.structure.aux_ce_weight,
        noise_schedule=cfg.structure.noise_schedule,
        loss_mode=cfg.structure.loss_mode,
        hidden_dim=cfg.structure.hidden_dim,
        num_layers=cfg.structure.num_layers,
        edge_dim=cfg.structure.edge_dim,
        num_steps=cfg.structure.num_steps,
        beta_start=cfg.structure.beta_start,
        beta_end=cfg.structure.beta_end,
        time_dim=cfg.structure.time_dim,
        dropout=cfg.structure.dropout,
        normalize_score=cfg.structure.normalize_score,
        num_mc_samples=cfg.structure.num_mc_samples,
    )
    feature_kwargs = dict(
        hidden_dim=cfg.feature.hidden_dim,
        num_layers=cfg.feature.num_layers,
        num_steps=cfg.feature.num_steps,
        beta_start=cfg.feature.beta_start,
        beta_end=cfg.feature.beta_end,
        time_dim=cfg.feature.time_dim,
        dropout=cfg.feature.dropout,
        arch=cfg.feature.arch,
        normalize_score=cfg.feature.normalize_score,
        num_mc_samples=cfg.feature.num_mc_samples,
    )
    fusion_kwargs = dict(learnable=cfg.model.fusion_learnable, init_eta=cfg.model.init_eta)
    return SGDiff(
        feature_dim=feature_dim,
        num_labels=num_labels,
        structure_kwargs=structure_kwargs,
        feature_kwargs=feature_kwargs,
        fusion_kwargs=fusion_kwargs,
        feature_enabled=cfg.feature.enabled,
    )


def build_extractor(
    graph: GraphData,
    cfg: ExperimentConfig,
    seed: Optional[int] = None,
    target_feature_dim: Optional[int] = None,
) -> SubgraphExtractor:
    """Create a sub-graph extractor with the configured hyper-parameters.

    ``target_feature_dim`` enables cross-dataset transfer: node attributes are
    projected onto the dimension the checkpoint expects.
    """
    return SubgraphExtractor(
        graph,
        hops=cfg.subgraph.hops,
        max_nodes_per_hop=cfg.subgraph.max_nodes_per_hop,
        max_nodes_total=cfg.subgraph.max_nodes_total,
        max_label=cfg.subgraph.max_label,
        feature_mode=cfg.subgraph.feature_mode,
        feature_normalization=cfg.subgraph.feature_normalization,
        remove_target=cfg.subgraph.remove_target_edge,
        seed=cfg.data.seed if seed is None else seed,
        target_feature_dim=target_feature_dim,
    )


def prepare_datasets(
    graph: GraphData,
    split: LinkSplit,
    cfg: ExperimentConfig,
    train_graph: Optional[GraphData] = None,
    eval_graph: Optional[GraphData] = None,
) -> dict:
    """Build train/val/test datasets.

    Training and validation use training links only. Testing explicitly uses
    the transductive train + validation graph, never test links. Custom graphs
    retain their perturbations but have the corresponding held-out links removed.
    """
    held_out = split.val_pos.tolist() + split.test_pos.tolist()
    train_graph = (graph if train_graph is None else train_graph).drop_edges(held_out)
    eval_graph = (graph if eval_graph is None else eval_graph).drop_edges(split.test_pos.tolist())

    train_extractor = build_extractor(train_graph, cfg)
    val_extractor = build_extractor(train_graph, cfg, seed=cfg.data.seed + 1)
    eval_extractor = build_extractor(eval_graph, cfg, seed=cfg.data.seed + 1)
    split.meta["preprocessing"] = {
        "train_context": "train", "val_context": "train",
        "test_context": "train+val (transductive)",
        "drnl": "final_subgraph_opposite_endpoint_removed; endpoints=1; unreachable=0",
        "feature_mode": train_extractor.feature_mode,
        "feature_normalization": train_extractor.feature_normalization,
        "feature_dim": train_extractor.feature_dim,
        "max_label": train_extractor.max_label,
        "hops": train_extractor.hops,
        "max_nodes_per_hop": train_extractor.max_nodes_per_hop,
        "max_nodes_total": train_extractor.max_nodes_total,
        "remove_target": train_extractor.remove_target,
        "hit_metric": "precision@k over the pooled candidates (legacy hit@k key)",
    }

    train_pairs, train_labels = training_pairs(split, cfg)

    datasets = {"train": PairSubgraphDataset(train_pairs, train_labels, train_extractor)}
    for name in ("val", "test"):
        pairs, labels = split.pairs(name)
        if cfg.data.max_eval_pairs is not None:
            pairs, labels = _cap_eval(pairs, labels, int(cfg.data.max_eval_pairs))
        extractor = val_extractor if name == "val" else eval_extractor
        datasets[name] = PairSubgraphDataset(pairs, labels, extractor)
    train_graph.meta["preprocessing"] = dict(split.meta["preprocessing"])
    eval_graph.meta["preprocessing"] = dict(split.meta["preprocessing"])
    return datasets


def initialize_training_marginals(model, train_dataset: PairSubgraphDataset) -> dict:
    """Initialize categorical marginals from only the actual training subgraphs.

    DRNL zero is a valid category. Edge counts include absent edges, exclude
    diagonal entries, and count each unpadded undirected pair exactly once.
    Models without ``structure.set_marginals`` are left unchanged.
    """
    structure = getattr(model, "structure", None)
    setter = getattr(structure, "set_marginals", None)
    if not callable(setter):
        return {}
    num_labels = int(structure.num_labels)
    node_counts = np.zeros(num_labels, dtype=np.int64)
    edge_counts = np.zeros(2, dtype=np.int64)
    for src, dst in train_dataset.pairs:
        sub = train_dataset.extractor.build(int(src), int(dst))
        labels = sub.labels
        if np.any((labels < 0) | (labels >= num_labels)):
            raise ValueError("training DRNL label is outside the model's categorical vocabulary")
        node_counts += np.bincount(labels, minlength=num_labels)
        edges = sub.adj[np.triu_indices(sub.num_nodes, k=1)]
        if not np.all(np.isin(edges, [0, 1])):
            raise ValueError("training adjacency must be binary")
        edge_counts += np.bincount(edges.astype(np.int64), minlength=2)
    if node_counts.sum() == 0 or edge_counts.sum() == 0:
        raise ValueError("cannot initialize marginals from an empty training dataset")
    node_marginal = node_counts / node_counts.sum()
    edge_marginal = edge_counts / edge_counts.sum()
    setter(torch.as_tensor(node_marginal, dtype=torch.float32),
           torch.as_tensor(edge_marginal, dtype=torch.float32))
    metadata = {
        "source": "training_pairs_subgraphs", "num_pairs": len(train_dataset),
        "node_counts": node_counts.tolist(), "edge_counts": edge_counts.tolist(),
        "node_marginal": node_marginal.tolist(), "edge_marginal": edge_marginal.tolist(),
        "edge_counting": "valid strict upper triangle; absent=0; present=1",
    }
    train_dataset.extractor.graph.meta.setdefault("preprocessing", {})["marginals"] = metadata
    return metadata


initialize_structure_marginals = initialize_training_marginals


def training_pairs(split: LinkSplit, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """The exact ``(pairs, labels)`` used to train the sub-graph models.

    Applies ``data.train_ratio`` (the 1% study) and ``data.max_train_pairs``
    (the compute cap) in the same order as :func:`prepare_datasets`.  The
    benchmark harness reuses it so that full-graph baselines are trained on the
    very same pairs instead of the full split.
    """
    if cfg.data.train_ratio < 1.0:
        pos = subsample_pairs(split.train_pos, cfg.data.train_ratio, seed=cfg.data.seed)
        neg = subsample_pairs(split.train_neg, cfg.data.train_ratio, seed=cfg.data.seed)
    else:
        pos, neg = split.train_pos, split.train_neg
    pairs = _stack(pos, neg)
    labels = _labels(len(pos), len(neg))
    if cfg.data.max_train_pairs is not None:
        pairs, labels = _cap_eval(pairs, labels, int(cfg.data.max_train_pairs), cfg.data.seed)
    return pairs, labels


def load_split(cfg: ExperimentConfig):
    """Load the dataset and create the 80/5/15 link split from the configuration."""
    graph = load_graph(cfg.data.name, root=cfg.data.root, download=cfg.data.download, use_cache=True)
    split = make_link_split(
        graph,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        neg_ratio=cfg.data.neg_ratio,
        seed=cfg.data.seed,
    )
    return graph, split


def _stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.int64).reshape(-1, 2)
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return np.vstack([a, b])


def _labels(n_pos: int, n_neg: int) -> np.ndarray:
    return np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])


def _cap_eval(pairs: np.ndarray, labels: np.ndarray, cap: int, seed: int = 0):
    """Randomly keep ``cap/2`` positives and ``cap/2`` negatives.

    The subsample is always class-balanced: slicing the raw ``(pos, neg)``
    concatenation instead would keep only positives whenever ``cap`` is smaller
    than the number of positive pairs.
    """
    rng = np.random.default_rng(seed)
    half = max(cap // 2, 1)
    pos_idx = rng.permutation(np.flatnonzero(labels == 1))[:half]
    neg_idx = rng.permutation(np.flatnonzero(labels == 0))[:half]
    idx = np.concatenate([pos_idx, neg_idx])
    return pairs[idx], labels[idx]
