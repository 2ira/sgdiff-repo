"""Run from the repository root. One seed is one independent paired experiment."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.team_study import run_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["cora", "citeseer", "pubmed"], default=["cora", "citeseer"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--profile", choices=["local", "team", "reference", "paper-rq3"], default="paper-rq3",
                        help="paper-rq3 matches the SGDiff RQ3 anchor; reference matches Line C; local is diagnostic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/team-evaluation")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        for seed in args.seeds:
            run_seed(dataset, seed, args.profile, args.data_root, args.output, args.device)
