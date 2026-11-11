#!/usr/bin/env python
"""Benchmark SGDiff against the baselines on identical data.

Every model in a run is trained and evaluated on the *same* 80/5/15 link split
and the *same* enclosing sub-graphs; the split / sub-graph fingerprints are
stored next to the metrics so the comparison is auditable.

Examples
--------
    # CPU-friendly comparison on Cora (one seed)
    python scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0

    # table-4 budget (GPU recommended)
    python scripts/run_benchmark.py --dataset cora --profile paper --seeds 0 1 2

    # only the sub-graph models, extra overrides
    python scripts/run_benchmark.py --dataset citeseer --models seal pencil sgdiff \\
        --profile cpu --overrides subgraph.hops=2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.config import load_config  # noqa: E402
from sgdiff.engine.benchmark import (  # noqa: E402
    BUDGET_PROFILES as PROFILES,
    build_registry,
    run_benchmark,
)

DEFAULT_MODELS = ["sgdiff", "seal", "pencil", "gcn", "gat", "sage", "vgae"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora", help="dataset name (default: cora)")
    parser.add_argument("--config", default=None, help="config path; defaults to configs/sgdiff_<dataset>.yaml")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="cpu")
    parser.add_argument("--sgdiff-epochs", type=int, default=None, help="override train.epochs for SGDiff")
    parser.add_argument("--subgraph-epochs", type=int, default=40)
    parser.add_argument("--node-pair-epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--overrides", nargs="*", default=[], help="extra section.key=value overrides")
    parser.add_argument("--list-models", action="store_true", help="print the model zoo and exit")
    args = parser.parse_args()

    if args.list_models:
        cfg = load_config(None, [])
        for name, spec in build_registry(cfg).items():
            print(f"{name:8s} {spec.kind:10s} {spec.note}")
        return 0

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"]
    overrides += PROFILES[args.profile]
    overrides += args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)

    result = run_benchmark(
        cfg,
        models=args.models,
        seeds=args.seeds,
        sgdiff_epochs=args.sgdiff_epochs,
        subgraph_epochs=args.subgraph_epochs,
        node_pair_epochs=args.node_pair_epochs,
        device=args.device,
        output_dir=args.output,
    )
    print()
    print(result["markdown"])
    print()
    print(f"wrote {args.output}/benchmark_{cfg.data.name}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
