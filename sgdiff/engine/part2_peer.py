"""Adapters and training helpers for the aligned CN/GCN/SGDiff study."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..models.baselines.node_pair import AblatedGCN, GCN_ABLATIONS, NodePairLinkPredictor
from .baseline_trainer import _edge_index, node_feature_matrix, train_node_pair_baseline
from .factory import build_extractor
from .model_scores import sgdiff_streams
from .reproduction import archive, digest as file_digest


GCN_VARIANTS = ("gcn", *GCN_ABLATIONS)
SGDIFF_VARIANTS = ("sgdiff", "sgdiff_structure", "sgdiff_feature")


def gcn_checkpoint(root: Path, dataset: str, seed: int, variant: str) -> Path:
    return Path(root) / dataset / "gcn" / f"seed_{seed}" / f"{variant}.pt"


def build_gcn(feature_dim: int, variant: str):
    if variant == "gcn":
        return NodePairLinkPredictor(feature_dim, 64, 64, 2, arch="gcn", dropout=0.5)
    if variant in GCN_ABLATIONS:
        return AblatedGCN(feature_dim, variant, 64, 64, 2, 0.5)
    raise ValueError(f"unknown GCN variant: {variant}")


def train_gcn(graph, split, dataset: str, seed: int, variant: str, checkpoint: Path,
              device: str, epochs: int = 100) -> dict:
    """Train one clean-graph GCN variant with peer-style early stopping."""
    if checkpoint.exists():
        return load_gcn(graph, dataset, seed, variant, checkpoint, device)[1]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = build_gcn(graph.num_features, variant)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    metrics = train_node_pair_baseline(
        model,
        graph,
        split,
        device=device,
        epochs=epochs,
        lr=0.01,
        weight_decay=0.0,
        patience=30,
        eval_every=5,
        seed=seed,
    )
    record = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "epochs": epochs,
        "eval_every": 5,
        "patience": 30,
        "optimizer": "Adam(lr=0.01, weight_decay=0)",
        "training_context": "train positives only",
        "validation_context": "train positives only",
        "test_context": "train+validation positives",
        "feature_preprocessing": "column standardization over all nodes",
        "seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "metrics": metrics,
    }
    payload = {"model_state": model.state_dict(), "record": record}
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    return record


def load_gcn(graph, dataset: str, seed: int, variant: str, checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    record = payload["record"]
    identity = (record["dataset"], int(record["seed"]), record["variant"])
    if identity != (dataset, seed, variant):
        raise ValueError(f"wrong GCN checkpoint identity at {checkpoint}: {identity}")
    model = build_gcn(graph.num_features, variant)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, record


class GCNPeerPredictor:
    """Expose a frozen full-graph GCN to Junxiong's native evaluator."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = torch.device(device)

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def context(self, graph):
        x = torch.as_tensor(node_feature_matrix(graph), dtype=torch.float32, device=self.device)
        edges = _edge_index(graph, self.device)
        return self.model.encode(x, edges, graph.num_nodes)

    @torch.no_grad()
    def score(self, context, pairs):
        return self.model.score_pairs(context, pairs)


def _array_digest(*arrays) -> str:
    out = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        out.update(str(value.shape).encode())
        out.update(str(value.dtype).encode())
        out.update(value.tobytes())
    return out.hexdigest()


class SGDiffSharedScores:
    """Compute fusion and both single-stream readouts once per attacked graph."""

    def __init__(self, model, cfg, pairs, positive_count: int, device: str, seed: int,
                 checkpoint: Path, root: Path):
        self.model = model
        self.cfg = cfg
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.positive_count = int(positive_count)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.pair_digest = _array_digest(self.pairs)
        self.checkpoint_digest = file_digest(checkpoint)
        self.root = Path(root) / self.checkpoint_digest[:16]
        self.root.mkdir(parents=True, exist_ok=True)

    def scores(self, graph):
        graph_digest = _array_digest(graph.edges, graph.x)
        path = self.root / f"{graph_digest}.npz"
        if path.exists():
            with np.load(path) as saved:
                if str(saved["pair_digest"]) != self.pair_digest:
                    raise ValueError(f"SGDiff shared-score query mismatch: {path}")
                return {name: saved[name] for name in SGDIFF_VARIANTS}
        extractor = build_extractor(graph, self.cfg, seed=self.cfg.data.seed + 1)
        cohorts = (self.pairs[: self.positive_count], self.pairs[self.positive_count :])
        values = sgdiff_streams(
            self.model,
            extractor,
            cohorts,
            self.cfg.train.eval_batch_size,
            self.device,
            self.seed,
        )
        archive(
            path,
            pair_digest=np.asarray(self.pair_digest),
            graph_digest=np.asarray(graph_digest),
            **values,
        )
        return values


class SGDiffPeerPredictor:
    """One SGDiff component view backed by shared three-readout score files."""

    def __init__(self, shared: SGDiffSharedScores, component: str, num_nodes: int):
        if component not in SGDIFF_VARIANTS:
            raise ValueError(component)
        self.shared = shared
        self.component = component
        self.device = shared.device
        self.num_nodes = int(num_nodes)
        keys = shared.pairs[:, 0] * int(num_nodes) + shared.pairs[:, 1]
        self.order = np.argsort(keys)
        self.sorted_keys = keys[self.order]

    def eval(self):
        self.shared.model.eval()
        return self

    def context(self, graph):
        return self.shared.scores(graph)[self.component][self.order]

    def score(self, context, pairs):
        keys = pairs[:, 0].detach().cpu().numpy() * self.num_nodes + pairs[:, 1].detach().cpu().numpy()
        indices = np.searchsorted(self.sorted_keys, keys)
        np.testing.assert_array_equal(self.sorted_keys[indices], keys)
        return torch.as_tensor(context[indices], device=self.device)
