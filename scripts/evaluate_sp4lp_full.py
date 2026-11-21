"""Score selected full-data checkpoints on Junxiong's exact cached attack graphs."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_peer_protocol import bind_peer
from train_sp4lp_full import Predictor
from experiments.sp4lp_full.model import FullSP4LP, input_graph, configuration
from sgdiff.data.datasets import load_graph
from sgdiff.engine.reproduction import digest as file_hash


def main(a):
    protocol, native = bind_peer(a.peer)
    choices = json.loads((a.training / "selection.json").read_text())
    if set(choices) != {"cora", "citeseer", "pubmed"}:
        raise ValueError("Incomplete configuration selection")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, receipts = [], []
    for dataset in ("cora", "citeseer", "pubmed"):
        graph = load_graph(dataset, root=str(a.data_root), download=False)
        profile = choices[dataset]["profile"]
        graph = input_graph(graph, configuration(dataset, profile))
        for seed in (0, 1, 2):
            split, fingerprint = protocol.split_data(graph, seed, a.output)
            pairs = np.vstack([split.test_pos, split.test_neg])
            base = native.graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
            for attack, ratio in native.variants():
                tag = attack if not ratio else f"{attack}_{ratio:g}"
                if attack == "feature_shuffle":
                    continue
                file = a.output / "attacks" / f"{dataset}_s{seed}_{tag}.npz"
                with np.load(file) as cached:
                    if str(cached["base_sha256"]) != protocol.digest(base.edges):
                        raise ValueError(f"Incorrect graph input {file}")
                    stats = json.loads(str(cached["statistics"]))
                    if stats["graph_sha256"] != protocol.digest(cached["edges"]):
                        raise ValueError(f"Incorrect cached edges {file}")
            for variant in ("full", "endpoints", "featureless"):
                directory = a.training / "models" / f"{dataset}_{profile}_{variant}_s{seed}"
                saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                if saved["receipt"]["split"] != fingerprint:
                    raise ValueError("Checkpoint split mismatch")
                if saved["config"] != configuration(dataset, profile) or saved["variant"] != variant:
                    raise ValueError("Checkpoint configuration mismatch")
                if saved["receipt"]["features"] != protocol.digest(graph.x):
                    raise ValueError("Checkpoint feature preprocessing mismatch")
                model = FullSP4LP(graph.num_features, saved["config"], variant).to(a.device)
                model.load_state_dict(saved["model"])
                path_dir = a.output / "paths" / f"{dataset}_s{seed}"
                path_dir.mkdir(exist_ok=True, parents=True)
                predictor = Predictor(model, pairs, path_dir, protocol, a.device)
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
                if (run / "receipt.json").exists() and json.loads(
                    (run / "receipt.json").read_text()
                ) != receipt:
                    raise ValueError("Evaluation inputs changed")
                (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                np.savez_compressed(
                    run / "queries.npz", pairs=pairs, positive_count=len(split.test_pos)
                )
                receipts.append(receipt)
                args = SimpleNamespace(
                    dataset=dataset,
                    seed=seed,
                    model="sp4lp_full_training",
                    variant=variant,
                    output=a.output,
                    eval_batch_size=512,
                )
                records = native.evaluate(predictor, graph, split, args, run)
                if variant == "featureless":
                    # CUDA GCN scatter sums can vary slightly even for identical inputs.
                    repeated = []
                    for _ in range(2):
                        native.seed_all(seed + 20000)
                        scores, _ = native.predict(predictor, base, pairs, 512)
                        repeated.append(scores)
                    rng = np.random.default_rng(seed + 7183)
                    shuffled = native.GraphData(base.name, base.num_nodes, base.edges,
                                                x=base.x[rng.permutation(base.num_nodes)], y=base.y)
                    native.seed_all(seed + 20000)
                    scores, _ = native.predict(predictor, shuffled, pairs, 512)
                    np.testing.assert_allclose(repeated[0], scores, rtol=1e-5, atol=1e-5)
                    control = dict(
                        repeated_clean_max_abs_error=float(np.max(np.abs(repeated[0] - repeated[1]))),
                        shuffled_features_max_abs_error=float(np.max(np.abs(repeated[0] - scores))),
                        rtol=1e-5, atol=1e-5,
                    )
                    (run / "numerical_control.json").write_text(json.dumps(control, indent=2) + "\n")
                for record in records:
                    attack, ratio = record["attack"], record["ratio"]
                    tag = attack if not ratio else f"{attack}_{ratio:g}"
                    with np.load(run / f"predictions_{tag}.npz") as scores:
                        vals = protocol.metrics(scores["positive"], scores["negative"])
                        assert vals == record["metrics"]
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
    assert len(rows) == 378
    with (a.output / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "provenance.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print("Verified 378 native evaluation cells.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("peer", "data-root", "training", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
