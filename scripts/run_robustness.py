#!/usr/bin/env python
"""RQ3 for *every* model: robustness under RF / DICE / EA.

Each model of the zoo (SGDiff, SEAL, PENCIL, GCN, GAT, SAGE, VGAE) is trained
once on the clean graph and then re-scored on inference graphs perturbed at the
given budgets. The clean column is produced by the same scoring code, so
`drop = clean - attacked` is a like-for-like comparison.

Examples
--------
    # Cora, all models, three attacks, 10/25/50 % budget (CPU profile)
    python scripts/run_robustness.py --dataset cora --profile cpu --seeds 0

    # paper's RQ3 models only, EA at 10 % (EA is dense: Cora-sized graphs only)
    python scripts/run_robustness.py --dataset cora --models sgdiff --attacks ea --ratios 0.1

    # sub-graph models only, longer node-pair schedule
    python scripts/run_robustness.py --dataset citeseer --models sgdiff seal pencil \\
        --node-pair-epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.config import load_config  # noqa: E402
from sgdiff.engine.benchmark import BUDGET_PROFILES as PROFILES  # noqa: E402
from sgdiff.engine.benchmark import build_registry  # noqa: E402
from sgdiff.engine.robustness import ATTACKS, run_robustness  # noqa: E402

DEFAULT_MODELS = ["sgdiff", "seal", "pencil", "gcn", "gat", "sage", "vgae"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora", help="dataset name (default: cora)")
    parser.add_argument("--config", default=None, help="config path; defaults to configs/sgdiff_<dataset>.yaml")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--attacks", nargs="*", default=["rf", "dice", "ea"],
                        choices=[a for a in ATTACKS if a != "clean"])
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.1, 0.25, 0.5],
                        help="perturbation budget as a fraction of |E|")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="cpu")
    parser.add_argument("--sgdiff-epochs", type=int, default=None, help="override train.epochs for SGDiff")
    parser.add_argument("--subgraph-epochs", type=int, default=40)
    parser.add_argument("--node-pair-epochs", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="per-class cap on the scored test pairs (0 = full test split)")
    parser.add_argument("--ea-max-nodes", type=int, default=4000,
                        help="EA is dense; larger graphs are skipped with a note")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--overrides", nargs="*", default=[], help="extra section.key=value overrides")
    parser.add_argument("--list-models", action="store_true", help="print the model zoo and exit")
    args = parser.parse_args()

    if args.list_models:
        for name, spec in build_registry(load_config(None, [])).items():
            print(f"{name:8s} {spec.kind:10s} {spec.note}")
        return 0

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"]
    overrides += PROFILES[args.profile]
    overrides += args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)

    result = run_robustness(
        cfg,
        models=args.models,
        attacks=args.attacks,
        ratios=args.ratios,
        seeds=args.seeds,
        sgdiff_epochs=args.sgdiff_epochs,
        subgraph_epochs=args.subgraph_epochs,
        node_pair_epochs=args.node_pair_epochs,
        device=args.device,
        max_pairs=args.max_pairs or None,
        output_dir=args.output,
        ea_max_nodes=args.ea_max_nodes,
    )
    print()
    print(result["markdown"])
    if result["skipped"]:
        print()
        for skip in result["skipped"]:
            print(f"skipped {skip['variant']}: {skip['reason']}")
    print()
    print(f"wrote {args.output}/robustness_{cfg.data.name}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
