"""Generate Junxiong-compatible EA caches from the released PubMed splits on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def generate(peer, splits, output):
    source = peer / "experiments/new_link_models/protocol.py"
    protocol_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if protocol_sha != "7567b820fa0ea9eca5152bfe25bde882031247a64113f3a8cce1990c3f6cb98d":
        raise ValueError("EA protocol differs from peer revision 95083da")
    sys.path.insert(0, str(peer.resolve()))
    from experiments.new_link_models.protocol import digest, perturb, save_npz
    from sgdiff.data.graph import GraphData

    output.mkdir(exist_ok=True, parents=True)
    manifest = dict(
        peer_commit="95083da808e500c02da458e205f4b861ceb80871",
        protocol_sha256=protocol_sha,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        numpy=np.__version__,
        device=torch.cuda.get_device_name(),
        graphs=[],
    )
    for seed in range(3):
        split_path = splits / f"pubmed_seed{seed}.npz"
        with np.load(split_path) as f:
            split = SimpleNamespace(**dict(f))
        base = GraphData("pubmed", 19717, np.vstack((split.train_pos, split.val_pos)))
        clean = base.edge_set()
        for ratio in (0.1, 0.25, 0.5):
            path = output / f"pubmed_s{seed}_ea_{ratio:g}.npz"
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            edited, stats = perturb(base, split, "ea", ratio, seed, device="cuda")
            changed = edited.edge_set()
            added = np.asarray(sorted(changed - clean), dtype=np.int64).reshape(-1, 2)
            removed = np.asarray(sorted(clean - changed), dtype=np.int64).reshape(-1, 2)
            assert (len(added), len(removed)) == (stats["inserted"], stats["deleted"])
            save_npz(
                path,
                edges=edited.edges,
                added=added,
                removed=removed,
                base_sha256=digest(base.edges),
                statistics=json.dumps(stats),
                split_file_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            )
            manifest["graphs"].append(
                dict(
                    file=path.name,
                    **stats,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    seconds=round(time.monotonic() - started, 2),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                )
            )
            print(
                path.name, stats["effective_edits"], manifest["graphs"][-1]["seconds"], flush=True
            )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("peer", "splits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    generate(args.peer, args.splits, args.output)
