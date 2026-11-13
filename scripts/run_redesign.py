#!/usr/bin/env python
"""Component-level redesign: change one building block, retrain, re-probe.

``scripts/run_deep_probe.py`` shows *where* the damage enters (layer, operator,
edge kind) on frozen weights.  Attribution on frozen weights can only say what
the trained network does; it cannot say whether the component is *causally*
responsible for the fragility.  This script closes that loop: each row below
retrains a GCN from scratch with exactly **one** component changed, and the same
probe suite is applied to the retrained model.

Configurations (all else identical: hidden 64, 300 epochs, lr 1e-2, dropout 0.5,
early stop on the validation AUC):

======================  ==========================================================
``base``                2 layers, symmetric ``1/sqrt(d_i d_j)``, self-loops, raw X
``depth-L1`` / ``-L3``  1 or 2 hops of message passing (hypothesis: the structural
                        damage enters at the second hop, so L1 should be hit less)
``noloop``              trained with ``add_self_loops = False`` (hypothesis: the
                        self-loop is the robustness carrier, so deletion must hurt
                        more when the model never had an identity path)
``row``                 trained with row-mean normalisation (hypothesis: the
                        normalisation is *not* the carrier, so this should match
                        ``base`` within seed noise even at train time)
``+cnfeat``             raw X concatenated with 2-hop structural descriptors
                        (degree, mean / max common neighbours over incident edges,
                        clustering coefficient, 2-hop reach, mean neighbour degree).
                        Hypothesis: giving the model the local-evidence channel
                        explicitly shrinks the deletion gap.
``+smooth``             input attributes rebuilt from the structure (1 round,
                        alpha = 0.5).  Hypothesis: the attribute channel becomes
                        partly redundant, so ``feature_shuffle`` hurts less.
``+cnfeat+smooth``      both data-side changes at once (the only combination that
                        is a candidate deployment fix).
``+edgedropout``        robust training: 30 % of the training edges dropped in
                        every epoch.  Hypothesis: the standard train-time defence
                        shrinks the deletion gap.
======================  ==========================================================

All augmented inputs are rebuilt **from the graph being scored** (a deployed
model cannot see the clean graph either), and the test positives are withheld
from every inference graph, exactly as in ``run_probe.py`` / ``run_deep_probe.py``.

    python scripts/run_redesign.py --dataset cora --seeds 0 1 2
    python scripts/run_redesign.py --dataset citeseer --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.engine.baseline_trainer import _edge_index, node_feature_matrix  # noqa: E402
from sgdiff.models.layers import spmm  # noqa: E402
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.metrics import summarize_scores  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

PROBES = ("del@50%", "del_hub@50%", "ins_hub@50%", "feature_shuffle")
PROBE_LABEL = {"del@50%": "del", "del_hub@50%": "del_hub", "ins_hub@50%": "ins_hub", "feature_shuffle": "feat_shuffle"}


# --------------------------------------------------------------------------- #
# The model: GCN with every component switchable
# --------------------------------------------------------------------------- #
class FlexGCN(nn.Module):
    """``score(u, v) = <z_u, z_v>`` with the propagation block configurable."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        norm: str = "sym",
        self_loops: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.self_loops = self_loops
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def _adjacency(self, edge_index, num_nodes: int):
        row, col = edge_index[0], edge_index[1]
        if self.self_loops:
            loop = torch.arange(num_nodes, device=row.device)
            row = torch.cat([row, loop])
            col = torch.cat([col, loop])
        ones = torch.ones(row.numel(), device=row.device, dtype=torch.float32)
        deg = torch.zeros(num_nodes, device=row.device).index_add_(0, row, ones).clamp(min=1.0)
        if self.norm == "sym":
            w = deg.pow(-0.5)[row] * deg.pow(-0.5)[col]
        elif self.norm == "row":
            w = deg.pow(-1.0)[row]
        else:
            w = ones
        return row, col, w

    def encode(self, x, edge_index, num_nodes):
        row, col, weights = self._adjacency(edge_index, num_nodes)
        for i, lin in enumerate(self.lins):
            x = spmm(row, col, weights, x, num_nodes)
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

    def score_pairs(self, z, pairs):
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Input builders (data-side components)
# --------------------------------------------------------------------------- #
_STRUCT_CACHE: dict[tuple, np.ndarray] = {}


def _struct_block(graph) -> np.ndarray:
    """Six standardised 2-hop descriptors per node, computed from ``graph`` only."""
    key = (graph.name, int(graph.num_nodes), int(graph.num_edges))
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    adj = graph.adjacency().tocsr().astype(np.float64)
    n = graph.num_nodes
    edges = np.asarray(graph.edges, dtype=np.int64)
    deg = np.asarray(adj.sum(1)).ravel()
    a2 = (adj @ adj).tocsr()
    cn = np.asarray(a2[edges[:, 0], edges[:, 1]]).ravel() if len(edges) else np.zeros(0)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst_cn = np.concatenate([cn, cn])
    sum_cn = np.bincount(src, weights=dst_cn, minlength=n) if len(src) else np.zeros(n)
    max_cn = np.zeros(n)
    if len(src):
        np.maximum.at(max_cn, src, dst_cn)
    cnt = np.maximum(np.bincount(src, minlength=n) if len(src) else np.zeros(n), 1.0)
    tri = np.asarray(((adj @ adj).multiply(adj)).sum(1)).ravel() / 2.0
    cc = np.where(deg > 1, 2.0 * tri / np.maximum(deg * (deg - 1), 1.0), 0.0)
    reach2 = np.asarray((a2 > 0).sum(1)).ravel() - deg
    nbr_deg_mean = np.asarray(adj @ deg).ravel() / cnt
    block = np.stack(
        [
            np.log1p(deg),
            sum_cn / cnt,
            max_cn,
            cc,
            np.log1p(np.maximum(reach2, 0)),
            nbr_deg_mean,
        ],
        axis=1,
    ).astype(np.float32)
    mean, std = block.mean(0, keepdims=True), block.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    scaled = ((block - mean) / std).astype(np.float32)
    _STRUCT_CACHE[key] = scaled
    return scaled


def smoothed_features(graph, alpha: float = 0.5, rounds: int = 1) -> np.ndarray:
    x = node_feature_matrix(graph)
    adj = graph.adjacency().tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    prop = sp.diags(1.0 / deg) @ adj
    for _ in range(rounds):
        x = (1.0 - alpha) * x + alpha * (prop @ x)
    return x.astype(np.float32)


def _concat(*blocks: np.ndarray) -> np.ndarray:
    return np.concatenate(blocks, axis=1).astype(np.float32)


FEATURE_BUILDERS = {
    "raw": node_feature_matrix,
    "cnfeat": lambda g: _concat(node_feature_matrix(g), _struct_block(g)),
    "smooth": smoothed_features,
    "cn_smooth": lambda g: _concat(smoothed_features(g), _struct_block(g)),
}


# --------------------------------------------------------------------------- #
# Training (mirrors sgdiff.engine.baseline_trainer.train_node_pair_baseline)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    name: str
    num_layers: int = 2
    norm: str = "sym"
    self_loops: bool = True
    features: str = "raw"
    edge_dropout: float = 0.0


CONFIGS = [
    Config("base", features="raw"),
    Config("depth-L1", num_layers=1),
    Config("depth-L3", num_layers=3),
    Config("noloop", self_loops=False),
    Config("row", norm="row"),
    Config("+cnfeat", features="cnfeat"),
    Config("+smooth", features="smooth"),
    Config("+cnfeat+smooth", features="cn_smooth"),
    Config("+edgedropout", edge_dropout=0.3),
]


def _eval_auc(model, x, edge_index, num_nodes, pairs, labels) -> float:
    with torch.no_grad():
        z = model.encode(x, edge_index, num_nodes)
        scores = model.score_pairs(z, torch.as_tensor(pairs, device=z.device, dtype=torch.long)).float().cpu().numpy()
    return float(summarize_scores(labels, scores)["auc"])


def _eval_split_auc(model, x, edge_index, num_nodes, pos_pairs, neg_pairs) -> float:
    """Validation AUC on separate positive / negative arrays."""
    pos, neg = np.asarray(pos_pairs, dtype=np.int64), np.asarray(neg_pairs, dtype=np.int64)
    pairs = np.vstack([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
    return _eval_auc(model, x, edge_index, num_nodes, pairs, labels)


def train_config(cfg: Config, graph, split, seed: int, device: str, epochs: int = 300, lr: float = 1e-2):
    set_seed(seed)
    torch.manual_seed(seed)
    train_graph = graph.drop_edges(split.val_pos.tolist() + split.test_pos.tolist())
    eval_graph = graph.drop_edges(split.test_pos.tolist())

    builder = FEATURE_BUILDERS[cfg.features]
    x_train = torch.as_tensor(builder(train_graph), device=device)
    x_eval_np = builder(eval_graph)
    x_eval = torch.as_tensor(x_eval_np, device=device)
    ei_train = _edge_index(train_graph, device)
    ei_eval = _edge_index(eval_graph, device)
    train_edges = np.asarray(train_graph.edges, dtype=np.int64)

    train_pairs = np.vstack([split.train_pos, split.train_neg])
    train_labels = np.concatenate([np.ones(len(split.train_pos)), np.zeros(len(split.train_neg))]).astype(np.float32)
    pairs_t = torch.as_tensor(train_pairs, device=device)
    labels_t = torch.as_tensor(train_labels, device=device)

    model = FlexGCN(
        x_train.shape[1],
        num_layers=cfg.num_layers,
        norm=cfg.norm,
        self_loops=cfg.self_loops,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 17)
    best_state, best_auc, bad = None, -np.inf, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if cfg.edge_dropout > 0:
            keep = rng.random(len(train_edges)) >= cfg.edge_dropout
            dropped = train_graph.drop_edges(train_edges[~keep].tolist())
            ei_used = _edge_index(dropped, device)
        else:
            ei_used = ei_train
        z = model.encode(x_train, ei_used, train_graph.num_nodes)
        loss = F.binary_cross_entropy_with_logits(model.score_pairs(z, pairs_t), labels_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            auc = _eval_split_auc(model, x_eval, ei_eval, eval_graph.num_nodes, split.val_pos, split.val_neg)
            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, builder, eval_graph


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(cfg, seeds, configs, device="cpu", max_pairs=500, epochs=300) -> dict:
    scores: dict[tuple[str, str, int], float] = {}
    extras: dict = {"clean_auc": {}, "train_seconds": {}}
    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        set_seed(int(seed))
        graph, split = rp.load_split(cfg_seed)
        pos, neg = rp._cap_pairs(split, max_pairs)
        pairs = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()

        eval_graphs = {"clean": graph.drop_edges(test_edges)}
        for probe in PROBES:
            try:
                eval_graphs[probe] = PERTURBATIONS[probe](graph, int(seed)).drop_edges(test_edges)
            except ValueError:
                continue

        for conf in configs:
            started = time.perf_counter()
            model, builder, _ = train_config(conf, graph, split, int(seed), device, epochs=epochs)
            elapsed = time.perf_counter() - started
            for probe, eval_graph in eval_graphs.items():
                x = torch.as_tensor(builder(eval_graph), device=device)
                auc = _eval_auc(model, x, _edge_index(eval_graph, device), eval_graph.num_nodes, pairs, labels)
                scores[(conf.name, probe, int(seed))] = auc
            extras["clean_auc"].setdefault(conf.name, []).append(
                scores[(conf.name, "clean", int(seed))]
            )
            extras["train_seconds"].setdefault(conf.name, []).append(elapsed)
            print(
                f"[{cfg.data.name} seed={seed}] {conf.name}: clean={scores[(conf.name, 'clean', int(seed))]:.3f} "
                f"fit={elapsed:.1f}s",
                flush=True,
            )
    return _summarise(cfg, seeds, configs, scores, extras)


def _agg(values: list[float]) -> dict:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def _summarise(cfg, seeds, configs, scores, extras) -> dict:
    table: dict[str, dict[str, dict]] = {}
    for conf in configs:
        table[conf.name] = {}
        for probe in ("clean",) + tuple(p for p in PROBES if any(k[1] == p for k in scores)):
            vals = [scores[(conf.name, probe, int(s))] for s in seeds if (conf.name, probe, int(s)) in scores]
            table[conf.name][probe] = _agg(vals)
    gaps: dict[str, dict[str, dict]] = {}
    for conf in configs:
        gaps[conf.name] = {}
        for probe in PROBES:
            paired = [
                scores[(conf.name, "clean", int(s))] - scores[(conf.name, probe, int(s))]
                for s in seeds
                if (conf.name, "clean", int(s)) in scores and (conf.name, probe, int(s)) in scores
            ]
            if paired:
                gaps[conf.name][probe] = _agg(paired)
    return {
        "dataset": cfg.data.name,
        "seeds": [int(s) for s in seeds],
        "configs": [c.__dict__ for c in configs],
        "scores": table,
        "damage": gaps,
        "train_seconds": {k: _agg(v) for k, v in extras["train_seconds"].items()},
        "protocol": {
            "setting": "evasion; the model is retrained clean per configuration, probes only change the inference graph",
            "features": "augmented inputs are rebuilt from the graph being scored; test positives withheld",
            "pair_budget": "same capped test pairs for every row",
        },
    }


def _f(entry, digits=3):
    if not entry or not entry.get("n") or not np.isfinite(entry.get("mean", float("nan"))):
        return "-"
    return f"{entry['mean']:.{digits}f}"


def to_markdown(result: dict, base: str = "base") -> str:
    probes = [p for p in PROBES if all(p in per for per in result["scores"].values())]
    names = list(result["scores"])
    lines = [f"# Component redesign: {result['dataset']}", ""]
    lines += [
        "* seeds: " + str(result["seeds"]),
        f"* every row retrains from scratch with one component changed; `{base}` is the reference",
        "* augmented inputs are rebuilt from the graph being scored (test positives withheld)",
        "",
        "## 1. AUC after retraining",
        "",
        "| config | clean | " + " | ".join(PROBE_LABEL[p] for p in probes) + " |",
        "|---|" + "---:|" * (len(probes) + 1),
    ]
    for name in names:
        per = result["scores"][name]
        row = [name, _f(per.get("clean"))] + [_f(per.get(p)) for p in probes]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 2. Damage = clean − probe (smaller is more robust)",
        "",
        "| config | " + " | ".join(PROBE_LABEL[p] for p in probes) + " | Δ vs base (mean) |",
        "|---|" + "---:|" * len(probes) + "---:|",
    ]
    base_damage = result["damage"].get(base, {})
    for name in names:
        per = result["damage"][name]
        deltas = [
            per[p]["mean"] - base_damage[p]["mean"]
            for p in probes
            if p in per and p in base_damage and np.isfinite(per[p].get("mean", np.nan))
        ]
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        row = [name] + [f"{per[p]['mean']:.3f}" if p in per and per[p].get("n") else "-" for p in probes]
        row.append(f"{mean_delta:+.3f}" if np.isfinite(mean_delta) else "-")
        lines.append("| " + " | ".join(row) + " |")
    clean_delta = result["scores"][base]["clean"]["mean"]
    lines += ["", f"Reference clean AUC (`{base}`): {clean_delta:.3f}.", ""]
    lines.append("## 3. Training cost")
    lines.append("")
    lines.append("| config | seconds / fit |")
    lines.append("|---|---:|")
    for name in names:
        lines.append(f"| {name} | {result['train_seconds'][name]['mean']:.1f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--configs", nargs="*", default=[c.name for c in CONFIGS])
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"] + rp.PROFILES[args.profile] + args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)
    device = resolve_device(args.device)
    configs = [c for c in CONFIGS if c.name in set(args.configs)]

    result = run(cfg, seeds=args.seeds, configs=configs, device=device, max_pairs=args.max_pairs, epochs=args.epochs)
    result["markdown"] = to_markdown(result)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.output}/redesign_{cfg.data.name}{suffix}"
    save_json(result, f"{stem}.json")
    Path(f"{stem}.md").write_text(result["markdown"] + "\n", encoding="utf-8")
    print()
    print(result["markdown"])
    print(f"\nwrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
