#!/usr/bin/env python
"""Train and evaluate the baseline link predictors for one dataset.

Usage
-----
    python scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.splits import make_link_split  # noqa: E402
from sgdiff.data.datasets import load_graph  # noqa: E402
from sgdiff.engine.baseline_trainer import (  # noqa: E402
    node_feature_matrix,
    train_node_pair_baseline,
    train_subgraph_classifier,
)
from sgdiff.engine.factory import prepare_datasets  # noqa: E402
from sgdiff.models.baselines import (  # noqa: E402
    NodePairLinkPredictor,
    PENCILBaseline,
    SEALBaseline,
    VGAE,
)
from sgdiff.utils.io import save_json  # noqa: E402
from sgdiff.utils.seed import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--baselines", nargs="*", default=["gcn", "gat", "sage", "vgae", "seal"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(None, [f"data.name={args.dataset}", f"data.root={args.root}"])
    graph = load_graph(args.dataset, root=args.root)
    split = make_link_split(graph, seed=args.seed)
    results: dict = {}

    for name in args.baselines:
        if name in {"gcn", "gat", "sage", "vgae"}:
            feats = node_feature_matrix(graph)
            if name == "vgae":
                model = VGAE(feats.shape[1])
            else:
                model = NodePairLinkPredictor(feats.shape[1], hidden_dim=64, out_dim=64, num_layers=2, arch=name)
            results[name] = train_node_pair_baseline(
                model, graph, split, device=args.device, epochs=args.epochs,
                is_vgae=(name == "vgae"), seed=args.seed,
            )
        elif name in {"seal", "pencil"}:
            datasets = prepare_datasets(graph, split, cfg)
            if name == "seal":
                model = SEALBaseline(
                    datasets["train"].feature_dim, num_labels=cfg.subgraph.max_label + 1
                )
            else:
                model = PENCILBaseline(
                    datasets["train"].feature_dim, num_labels=cfg.subgraph.max_label + 1
                )
            results[name] = train_subgraph_classifier(
                model, datasets, device=args.device, epochs=max(args.epochs // 10, 10), seed=args.seed
            )
        else:
            print(f"unknown baseline '{name}', skipping", file=sys.stderr)
        print(name, results.get(name))

    if args.output:
        save_json(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
