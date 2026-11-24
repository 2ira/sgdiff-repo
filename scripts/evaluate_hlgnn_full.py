"""Evaluate 27 trained HL-GNN checkpoints across 14 cached peer-protocol conditions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_peer_protocol import bind_peer
from experiments.hlgnn.model import HLGNNLinkModel, VARIANTS, configuration
from scripts.train_hlgnn_full import HLGNNPredictor
from sgdiff.data.datasets import load_graph
from sgdiff.engine.reproduction import digest as file_hash


def main(a) -> None:
    protocol, native = bind_peer(a.peer)
    choices = json.loads((a.training / "selection.json").read_text())
    if set(choices) != {"cora", "citeseer", "pubmed"}:
        raise ValueError("Incomplete configuration selection")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows, receipts = [], []
    metric_max_abs_error = 0.0

    for dataset in a.datasets:
        graph = load_graph(dataset, root=str(a.data_root), download=False)
        profile = choices[dataset]["profile"]
        cfg = configuration(dataset, profile)
        for seed in a.seeds:
            split, fingerprint = protocol.split_data(graph, seed, a.output)
            pairs = np.vstack([split.test_pos, split.test_neg])
            base = native.graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
            for attack, ratio in native.variants():
                tag = attack if not ratio else f"{attack}_{ratio:g}"
                if attack == "feature_shuffle":
                    continue
                file = a.output / "attacks" / f"{dataset}_s{seed}_{tag}.npz"
                if not file.exists():
                    raise FileNotFoundError(f"Missing cached attack graph: {file}")
                with np.load(file) as cached:
                    if str(cached["base_sha256"]) != protocol.digest(base.edges):
                        raise ValueError(f"Incorrect graph input {file}")
                    stats = json.loads(str(cached["statistics"]))
                    if stats["graph_sha256"] != protocol.digest(cached["edges"]):
                        raise ValueError(f"Incorrect cached edges {file}")

            for variant in a.variants:
                directory = a.training / "models" / f"{dataset}_{profile}_{variant}_s{seed}"
                saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                if saved["receipt"]["split"] != fingerprint:
                    raise ValueError(f"Checkpoint split mismatch at {directory}")
                if saved["config"] != cfg or saved["variant"] != variant:
                    raise ValueError(f"Checkpoint configuration mismatch at {directory}")
                if saved["receipt"]["features"] != protocol.digest(graph.x):
                    raise ValueError(f"Checkpoint feature mismatch at {directory}")

                model = HLGNNLinkModel(graph.num_features, saved["config"], variant).to(a.device)
                model.load_state_dict(saved["model"])
                predictor = HLGNNPredictor(model, a.device)

                run = a.output / "runs" / f"{dataset}_{variant}_s{seed}"
                run.mkdir(exist_ok=True, parents=True)
                receipt = dict(
                    dataset=dataset,
                    variant=variant,
                    seed=seed,
                    profile=profile,
                    split=fingerprint,
                    checkpoint_sha256=file_hash(directory / "best.pt"),
                    evaluator_sha256=file_hash(Path(native.__file__)),
                    features=protocol.digest(graph.x),
                    query_sha256=protocol.digest(pairs),
                )
                if (run / "receipt.json").exists() and json.loads((run / "receipt.json").read_text()) != receipt:
                    raise ValueError(f"Evaluation inputs changed at {run}")
                (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                np.savez_compressed(run / "queries.npz", pairs=pairs, positive_count=len(split.test_pos))
                receipts.append(receipt)

                args = SimpleNamespace(
                    dataset=dataset,
                    seed=seed,
                    model="hlgnn",
                    variant=variant,
                    output=a.output,
                    eval_batch_size=512,
                )
                records = native.evaluate(predictor, graph, split, args, run)
                assert len(records) == 14
                for record in records:
                    attack, ratio = record["attack"], record["ratio"]
                    tag = attack if not ratio else f"{attack}_{ratio:g}"
                    with np.load(run / f"predictions_{tag}.npz") as scores:
                        pos, neg = scores["positive"], scores["negative"]
                        assert len(pos) == len(split.test_pos) and len(neg) == len(split.test_neg)
                        assert np.isfinite(pos).all() and np.isfinite(neg).all()
                        vals = protocol.metrics(pos, neg)
                    for k in ("auc", "ap", "hits100", "precision100"):
                        diff = abs(vals[k] - record["metrics"][k])
                        metric_max_abs_error = max(metric_max_abs_error, diff)
                        assert math.isclose(vals[k], record["metrics"][k], rel_tol=0.0, abs_tol=1e-12)
                    rows.append(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            variant=variant,
                            profile=profile,
                            condition=tag,
                            graph_sha256=record["statistics"]["graph_sha256"],
                            **{k: vals[k] for k in ("auc", "ap", "hits100", "precision100")},
                        )
                    )

    expected = len(a.datasets) * len(a.seeds) * len(a.variants) * 14
    assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"
    with (a.output / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "provenance.json").write_text(json.dumps(receipts, indent=2) + "\n")
    audit = dict(
        native_evaluator_cells=len(rows),
        selected_independent_fits=len(a.datasets) * len(a.seeds) * len(a.variants),
        metric_recomputation_max_abs_error=metric_max_abs_error,
        metric_recomputation_absolute_tolerance=1e-12,
        selected_using="historically selected readme_exact profile, fixed before retraining",
        epochs_selected_by_validation=True,
        test_used_for_selection=False,
    )
    (a.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("peer", "data-root", "training", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    main(p.parse_args())
