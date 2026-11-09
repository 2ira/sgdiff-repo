"""``sgdiff`` command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from ..version import __version__
from . import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgdiff",
        description="SGDiff: sub-graph based diffusion model for link prediction (reproduction).",
    )
    parser.add_argument("--version", action="version", version=f"sgdiff {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- prepare
    p = sub.add_parser("prepare", help="download and preprocess the benchmark graphs")
    p.add_argument("--datasets", nargs="*", default=None, help="dataset names (default: all)")
    p.add_argument("--root", default="data", help="data directory")
    p.add_argument("--no-download", action="store_true", help="use local files only")
    p.set_defaults(func=commands.cmd_prepare)

    # ------------------------------------------------------------------ train
    t = sub.add_parser("train", help="train SGDiff on one dataset")
    t.add_argument("--config", default="configs/sgdiff_cora.yaml")
    t.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.name=usair")
    t.add_argument("--run-dir", default=None)
    t.add_argument("--max-steps", type=int, default=None, help="cap batches per epoch (debug)")
    t.set_defaults(func=commands.cmd_train)

    # --------------------------------------------------------------- evaluate
    e = sub.add_parser("evaluate", help="evaluate a checkpoint (optionally under attack)")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset", default=None)
    e.add_argument("--split", default="test", choices=["train", "val", "test"])
    e.add_argument("--max-pairs", type=int, default=None)
    e.add_argument("--attack", default="none", choices=["none", "random", "dice"])
    e.add_argument("--attack-mode", default="flip", choices=["flip", "insert", "delete"])
    e.add_argument("--ratio", type=float, default=0.1)
    e.add_argument("--device", default=None)
    e.add_argument("--output", default=None)
    e.set_defaults(func=commands.cmd_evaluate)

    # --------------------------------------------------------------- transfer
    r = sub.add_parser("transfer", help="zero-shot cross-dataset transfer")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--targets", nargs="*", default=None)
    r.add_argument("--root", default="data")
    r.add_argument("--max-pairs", type=int, default=None)
    r.add_argument("--device", default=None)
    r.add_argument("--output", default=None)
    r.set_defaults(func=commands.cmd_transfer)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
