#!/usr/bin/env python
"""Does SGDIFF's likelihood actually depend on the link condition?

Runs the ablations of :mod:`sgdiff.engine.ablation` on one dataset and writes
``results/ablation_<dataset>.{json,md}``:

* scoring-time variants on a single checkpoint (``same_condition``,
  ``condition_flip``, ``label_shuffle``, single-timestep scoring, single-stream
  scoring);
* one retrained model whose stage-1 condition is randomized
  (``trained_without_condition``).

Examples
--------
    python scripts/run_ablation.py --dataset cora --profile cpu --seed 0
    python scripts/run_ablation.py --dataset synthetic --profile smoke --no-retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgdiff.config import load_config  # noqa: E402
from sgdiff.engine.ablation import run_ablation  # noqa: E402
from sgdiff.engine.benchmark import BUDGET_PROFILES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None, help="defaults to configs/sgdiff_<dataset>.yaml")
    parser.add_argument("--profile", choices=sorted(BUDGET_PROFILES), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs for SGDiff")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--tag", default="", help="suffix for the output stem, e.g. 'full'")
    parser.add_argument("--no-retrain", action="store_true", help="skip the randomized-condition retrain")
    parser.add_argument(
        "--no-unfrozen-variant",
        action="store_true",
        help="skip the control run that lets stage 2 re-learn the condition",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"]
    overrides += BUDGET_PROFILES[args.profile]
    overrides += args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)

    result = run_ablation(
        cfg,
        seed=args.seed,
        sgdiff_epochs=args.epochs,
        device=args.device,
        output_dir=args.output,
        train_without_condition=not args.no_retrain,
        unfrozen_variant=not args.no_unfrozen_variant,
        tag=args.tag,
    )
    print()
    print(result["markdown"])
    print()
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"wrote {args.output}/ablation_{cfg.data.name}{suffix}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
