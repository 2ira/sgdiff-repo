"""Create Jiatong's node-rank evaluator manifests for the saved Part 2 scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    repo, output = a.repo.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = repo / "results/part2-peer-cn-gcn-sgdiff-all/per_seed.csv"
    with table.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (table.parent / "provenance.json").open() as f:
        nodes = {r["dataset"]: int(r["num_nodes"]) for r in json.load(f)}
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], int(r["seed"])), []).append(r)
    for (dataset, seed), group in sorted(groups.items()):
        if a.pilot and (dataset, seed) != ("cora", 0):
            continue
        split_path = repo / "runs/part2-peer/evaluation/splits" / f"{dataset}_seed{seed}.npz"
        with np.load(split_path) as f:
            pos_hash, neg_hash = digest(f["test_pos"]), digest(f["test_neg"])
        records = []
        for r in group:
            if a.pilot and (r["model"], r["condition"]) != ("cn", "clean"):
                continue
            path = repo / "runs/part2-peer/evaluation/runs" / f"{dataset}_{r['model']}_s{seed}" / f"predictions_{r['condition']}.npz"
            records.append(dict(model=r["family"], variant=r["model"], condition=r["condition"],
                                scores_path=str(path), pos_key="positive", neg_key="negative",
                                positive_sha256=pos_hash, negative_sha256=neg_hash,
                                query_order_evidence="scripts/run_part2_peer_models.py scores split.test_pos then split.test_neg; per-condition original AUC verified against peer evaluator",
                                expected_auc=float(r["auc"]), graph_sha256=r["graph_sha256"]))
        manifest = dict(dataset=dataset, seed=seed, num_nodes=nodes[dataset], split_path=str(split_path),
                        protocol_commit="95083da808e500c02da458e205f4b861ceb80871", records=records)
        path = output / f"{dataset}_seed{seed}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(path, len(records), flush=True)


if __name__ == "__main__":
    main()
