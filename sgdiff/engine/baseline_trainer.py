"""Training loops for the baseline link predictors."""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.graph import GraphData
from ..data.splits import LinkSplit
from ..utils.logging import get_logger
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device
from .batch import PairSubgraphDataset, collate_subgraphs

LOGGER = get_logger(__name__)


def node_feature_matrix(graph: GraphData, degree_bins: int = 16) -> np.ndarray:
    """Standardised attributes, or a degree encoding for featureless graphs."""
    if graph.has_features:
        x = np.asarray(graph.x, dtype=np.float32)
        mean, std = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (x - mean) / std
    deg = graph.degrees().astype(np.float32)
    bins = np.clip(np.floor(np.log2(np.maximum(deg, 1.0))).astype(np.int64), 0, degree_bins - 1)
    one_hot = np.zeros((graph.num_nodes, degree_bins), dtype=np.float32)
    one_hot[np.arange(graph.num_nodes), bins] = 1.0
    scaled = ((deg - deg.mean()) / max(deg.std(), 1e-6))[:, None]
    return np.concatenate([one_hot, scaled], axis=1)


def _edge_index(graph: GraphData, device) -> torch.Tensor:
    e = graph.edges
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=device)


def train_node_pair_baseline(
    model,
    graph: GraphData,
    split: LinkSplit,
    device: str = "auto",
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    is_vgae: bool = False,
    kl_weight: float = 0.001,  # the raw ELBO KL term collapses the latent code
    patience: int = 30,
    seed: int = 0,
) -> dict:
    """Train a full-graph encoder with a link-prediction decoder."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    model.to(device)

    train_graph = graph.drop_edges(split.val_pos.tolist() + split.test_pos.tolist())
    eval_graph = graph.drop_edges(split.test_pos.tolist())
    x_train_np = node_feature_matrix(train_graph)
    x_eval_np = node_feature_matrix(eval_graph)
    x_train = torch.as_tensor(x_train_np, device=device)
    x_eval = torch.as_tensor(x_eval_np, device=device)
    ei_train = _edge_index(train_graph, device)
    ei_eval = _edge_index(eval_graph, device)

    train_pairs = np.vstack([split.train_pos, split.train_neg])
    train_labels = np.concatenate(
        [np.ones(len(split.train_pos)), np.zeros(len(split.train_neg))]
    ).astype(np.float32)
    train_pairs_t = torch.as_tensor(train_pairs, device=device)
    train_labels_t = torch.as_tensor(train_labels, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_auc, bad = None, -np.inf, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if is_vgae:
            z, kl = model(x_train, ei_train, train_graph.num_nodes)
            logits = model.decode(z, train_pairs_t)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels_t) + kl_weight * kl
        else:
            z = model.encode(x_train, ei_train, train_graph.num_nodes)
            logits = model.score_pairs(z, train_pairs_t)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 and len(split.val_pos):
            metrics = evaluate_node_pair_baseline(
                model, x_eval, ei_eval, eval_graph.num_nodes, split.val_pos, split.val_neg, device
            )
            if metrics["auc"] > best_auc:
                best_auc, bad = metrics["auc"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience // 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_node_pair_baseline(
        model, x_eval, ei_eval, eval_graph.num_nodes, split.test_pos, split.test_neg, device
    )
    test["best_val_auc"] = float(best_auc)
    test["epochs_run"] = epoch + 1
    return test


def train_rgcn(
    model,
    graph: GraphData,
    split: LinkSplit,
    device: str = "auto",
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    patience: int = 30,
    seed: int = 0,
) -> dict:
    """Train a Bayesian robust GCN (Zhu et al., KDD 2019).

    Loss = BCE on the dot-product logits + ``var_weight * mean(sigma)``.  The
    penalty is the only difference from ``train_node_pair_baseline``; the
    reparameterised sample happens inside ``model.forward`` (train mode only), so
    the evaluation path stays deterministic.
    """
    device = resolve_device(device)
    torch.manual_seed(seed)
    model.to(device)

    train_graph = graph.drop_edges(split.val_pos.tolist() + split.test_pos.tolist())
    eval_graph = graph.drop_edges(split.test_pos.tolist())
    x_train = torch.as_tensor(node_feature_matrix(train_graph), device=device)
    x_eval = torch.as_tensor(node_feature_matrix(eval_graph), device=device)
    ei_train = _edge_index(train_graph, device)
    ei_eval = _edge_index(eval_graph, device)

    train_pairs = np.vstack([split.train_pos, split.train_neg])
    train_labels = np.concatenate(
        [np.ones(len(split.train_pos)), np.zeros(len(split.train_neg))]
    ).astype(np.float32)
    train_pairs_t = torch.as_tensor(train_pairs, device=device)
    train_labels_t = torch.as_tensor(train_labels, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    var_weight = float(getattr(model, "var_weight", 0.0))
    best_state, best_auc, bad = None, -np.inf, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z, sigma = model(x_train, ei_train, train_graph.num_nodes)
        logits = model.score_pairs(z, train_pairs_t)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels_t)
        if var_weight > 0:
            loss = loss + var_weight * sigma.mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 and len(split.val_pos):
            metrics = evaluate_node_pair_baseline(
                model, x_eval, ei_eval, eval_graph.num_nodes, split.val_pos, split.val_neg, device
            )
            if metrics["auc"] > best_auc:
                best_auc, bad = metrics["auc"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience // 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_node_pair_baseline(
        model, x_eval, ei_eval, eval_graph.num_nodes, split.test_pos, split.test_neg, device
    )
    test["best_val_auc"] = float(best_auc)
    test["epochs_run"] = epoch + 1
    return test


@torch.no_grad()
def evaluate_node_pair_baseline(model, x, edge_index, num_nodes, pos_pairs, neg_pairs, device) -> dict:
    model.eval()
    is_vgae = hasattr(model, "decode")
    if is_vgae:
        z, _ = model(x, edge_index, num_nodes)
    else:
        z = model.encode(x, edge_index, num_nodes)
    scorer = model.decode if is_vgae else model.score_pairs
    pos = torch.as_tensor(np.asarray(pos_pairs), device=z.device, dtype=torch.long)
    neg = torch.as_tensor(np.asarray(neg_pairs), device=z.device, dtype=torch.long)
    pos_scores = scorer(z, pos)
    neg_scores = scorer(z, neg)
    scores = torch.cat([pos_scores, neg_scores]).float().cpu().numpy()
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return summarize_scores(labels, scores)


def train_subgraph_classifier(
    model,
    datasets: dict,
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 10,
    seed: int = 0,
) -> dict:
    """Train a sub-graph classifier (SEAL) on the same datasets as SGDiff."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    model.to(device)
    train_ds: PairSubgraphDataset = datasets["train"]
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_subgraphs, feature_dim=train_ds.feature_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_auc, bad = None, -np.inf, 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            loss = F.cross_entropy(model(batch), batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = _eval_subgraph(model, datasets.get("val"), device)
        if np.isfinite(val["auc"]) and val["auc"] > best_auc:
            best_auc, bad = val["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result = {"best_val_auc": float(best_auc), "epochs_run": epoch + 1}
    test = _eval_subgraph(model, datasets.get("test"), device)
    result.update({f"test_{k}": v for k, v in test.items()})
    return result


@torch.no_grad()
def _eval_subgraph(model, dataset: Optional[PairSubgraphDataset], device) -> dict:
    if dataset is None:
        return {"auc": float("nan")}
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_subgraphs, feature_dim=dataset.feature_dim),
    )
    model.eval()
    scores = []
    for batch in loader:
        scores.append(model.predict_logit(batch.to(device)).float().cpu().numpy())
    return summarize_scores(dataset.labels, np.concatenate(scores))
