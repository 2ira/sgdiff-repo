"""Train, resume, or score the selected DATA8002 methods using one graph bank."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.reproduction import experiment, OWNED, PRIMARY

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "evaluate", "run"])
    p.add_argument("--datasets", nargs="+", choices=["cora", "citeseer"], default=["cora", "citeseer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--models", nargs="+", choices=PRIMARY, default=list(OWNED))
    p.add_argument("--weights", type=Path, default=Path("runs/pair"))
    p.add_argument("--output", type=Path, default=Path("results/paired-reproduction"))
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--max-padded-pairs", type=int, default=8192)
    a = p.parse_args()
    if a.max_padded_pairs < 0:
        p.error("--max-padded-pairs must be non-negative (0 uses native dense execution)")
    torch.set_num_threads(a.threads)
    for d in a.datasets:
        for s in a.seeds:
            experiment(d, s, a.models, a.data_root, a.weights, a.output, a.device,
                       a.action, a.max_padded_pairs or None)
