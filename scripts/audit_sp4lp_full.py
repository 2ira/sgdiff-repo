"""Verify remote archives and matched evaluation, then summarize full-supervision SP4LP."""
import argparse
import io
import json
import math
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from evaluate_peer_protocol import bind_peer


def audit(peer, root):
    protocol, native = bind_peer(peer)
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    df = pd.read_csv(root / "per_seed.csv", float_precision="round_trip")
    keys = ["dataset", "seed", "variant", "condition"]
    assert len(df) == 378 and not df.duplicated(keys).any()
    choices = json.loads((root / "selection.json").read_text())
    previous = pd.concat(
        [
            pd.read_csv("results/peer-protocol/per_seed.csv"),
            pd.read_csv("results/peer-protocol/pubmed/per_seed.csv"),
        ]
    )
    previous = previous[previous.model == "sp4lp"].set_index(["dataset", "seed", "condition"])
    old_provenance = {}
    for source in (
        "results/peer-protocol/provenance.json",
        "results/peer-protocol/pubmed/provenance.json",
    ):
        for r in json.loads(Path(source).read_text()):
            if r["model"] == "sp4lp":
                old_provenance[r["dataset"], r["seed"]] = r
    records = []
    featureless_shuffle_max_abs_error = 0.0
    metric_max_abs_error = 0.0
    repeated_clean_max_abs_error = 0.0
    with ZipFile(root / "evaluation-inputs.zip") as inputs:
        with ZipFile("results/peer-protocol/predictions-and-graphs.zip") as original_graphs:
            for dataset in ("cora", "citeseer"):
                for seed in (0, 1, 2):
                    for ratio in (0.1, 0.25, 0.5):
                        name = f"attacks/{dataset}_s{seed}_ea_{ratio:g}.npz"
                        assert inputs.read("evaluation/" + name) == original_graphs.read(name)
        for dataset in ("cora", "citeseer", "pubmed"):
            selection = choices[dataset]
            best = max(selection["candidates"], key=lambda r: r["best_validation"]["auc"])
            assert best["run"] == f"{dataset}_{selection['profile']}_full_s0"
            for seed in (0, 1, 2):
                with np.load(
                    io.BytesIO(inputs.read(f"evaluation/splits/{dataset}_seed{seed}.npz"))
                ) as split:
                    positive, negative = split["test_pos"], split["test_neg"]
                    supervised = len(split["train_pos"])
                    original = old_provenance[dataset, seed]
                    assert protocol.digest(positive) == original["positive"]
                    assert protocol.digest(negative) == original["negative"]
                with ZipFile(root / f"{dataset}_seed{seed}.zip") as z:
                    assert z.testzip() is None
                    for variant in ("full", "endpoints", "featureless"):
                        label = f"{dataset}_{selection['profile']}_{variant}_s{seed}"
                        prefix = f"training/models/{label}"
                        history = json.loads(z.read(prefix + "/history.json"))
                        result = json.loads(z.read(prefix + "/complete.json"))
                        receipt = json.loads(z.read(prefix + "/receipt.json"))
                        raw = z.read(prefix + "/best.pt")
                        assert hashlib.sha256(raw).hexdigest() == result["checkpoint_sha256"]
                        saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                        assert saved["receipt"] == receipt
                        chosen = max(
                            (r for r in history if "val_auc" in r), key=lambda r: r["val_auc"]
                        )
                        assert chosen["epoch"] == saved["epoch"] == result["best_epoch"]
                        assert chosen["val_auc"] == result["best_validation"]["auc"]
                        assert len(history) == result["epochs_run"]
                        assert (
                            result["updates"] == math.ceil(supervised / 512) * result["epochs_run"]
                        )
                        assert all(np.isfinite(r["train_loss"]) for r in history)
                        records.append(
                            dict(
                                dataset=dataset,
                                seed=seed,
                                variant=variant,
                                profile=selection["profile"],
                                train_positives=supervised,
                                train_negatives=supervised,
                                best_epoch=result["best_epoch"],
                                epochs_run=result["epochs_run"],
                                updates=result["updates"],
                                validation_auc=chosen["val_auc"],
                            )
                        )
                        run = f"evaluation/runs/{dataset}_{variant}_s{seed}"
                        provenance = json.loads(z.read(run + "/receipt.json"))
                        assert provenance["checkpoint_sha256"] == result["checkpoint_sha256"]
                        assert (
                            provenance["evaluator_sha256"]
                            == hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
                        )
                        with np.load(io.BytesIO(z.read(run + "/queries.npz"))) as q:
                            np.testing.assert_array_equal(
                                q["pairs"], np.vstack((positive, negative))
                            )
                        subset = df[
                            (df.dataset == dataset) & (df.seed == seed) & (df.variant == variant)
                        ]
                        assert len(subset) == 14
                        if variant == "featureless":
                            control = json.loads(z.read(run + "/numerical_control.json"))
                            repeated_clean_max_abs_error = max(repeated_clean_max_abs_error, control["repeated_clean_max_abs_error"])
                            with np.load(io.BytesIO(z.read(run + "/predictions_clean.npz"))) as clean_scores, np.load(
                                io.BytesIO(z.read(run + "/predictions_feature_shuffle.npz"))
                            ) as shuffled_scores:
                                for key in ("positive", "negative"):
                                    featureless_shuffle_max_abs_error = max(
                                        featureless_shuffle_max_abs_error,
                                        float(np.max(np.abs(clean_scores[key] - shuffled_scores[key]))),
                                    )
                                    np.testing.assert_allclose(clean_scores[key], shuffled_scores[key], rtol=1e-5, atol=1e-5)
                        for row in subset.itertuples():
                            tag = "clean" if row.condition == "feature_shuffle" else row.condition
                            with np.load(
                                io.BytesIO(
                                    inputs.read(f"evaluation/attacks/{dataset}_s{seed}_{tag}.npz")
                                )
                            ) as graph:
                                assert protocol.digest(graph["edges"]) == row.graph_sha256
                            assert (
                                previous.loc[(dataset, seed, row.condition), "graph_sha256"]
                                == row.graph_sha256
                            )
                            with np.load(
                                io.BytesIO(z.read(run + f"/predictions_{row.condition}.npz"))
                            ) as pred:
                                assert len(pred["positive"]) == len(positive) and len(
                                    pred["negative"]
                                ) == len(negative)
                                metrics = protocol.metrics(pred["positive"], pred["negative"])
                            for k in ("auc", "ap", "hits100", "precision100"):
                                metric_max_abs_error = max(metric_max_abs_error, abs(metrics[k] - getattr(row, k)))
                                assert math.isclose(metrics[k], getattr(row, k), rel_tol=0, abs_tol=1e-12), (
                                    dataset, seed, variant, row.condition, k, metrics[k], getattr(row, k)
                                )
    pd.DataFrame(records).to_csv(root / "training_summary.csv", index=False)
    metrics = ["auc", "ap", "hits100", "precision100"]
    full = df[df.variant == "full"].drop(columns="variant")
    clean = df[df.condition == "clean"].drop(columns="condition")
    paired = df.merge(
        full, on=["dataset", "seed", "condition"], suffixes=("", "_full"), validate="many_to_one"
    )
    paired = paired.merge(
        clean, on=["dataset", "seed", "variant"], suffixes=("", "_clean"), validate="many_to_one"
    )
    fields = []
    for k in metrics:
        paired[k + "_drop"] = paired[k + "_clean"] - paired[k]
        paired[k + "_full_minus_variant"] = paired[k + "_full"] - paired[k]
        fields.extend([k, k + "_drop", k + "_full_minus_variant"])
    summary = paired.groupby(["dataset", "variant", "condition"])[fields].agg(["mean", "std"])
    summary.columns = ["_".join(k) for k in summary.columns]
    summary.to_csv(root / "summary.csv")
    current = full.set_index(["dataset", "seed", "condition"])
    comparison = current[metrics].join(
        previous[metrics], rsuffix="_previous", validate="one_to_one"
    )
    for k in metrics:
        comparison[k + "_change"] = comparison[k] - comparison[k + "_previous"]
    comparison.to_csv(root / "paired_previous.csv")
    result = dict(
        native_evaluator_cells=378,
        selected_independent_fits=27,
        matched_old_graphs_and_queries=True,
        original_cora_citeseer_ea_files_byte_identical=18,
        featureless_shuffle_max_abs_error=featureless_shuffle_max_abs_error,
        repeated_clean_max_abs_error=repeated_clean_max_abs_error,
        metric_recomputation_max_abs_error=metric_max_abs_error,
        metric_recomputation_absolute_tolerance=1e-12,
        checksums_valid=True,
        selected_using="seed 0 full validation AUROC, six configurations per dataset",
        epochs_selected_by_validation=True,
        test_used_for_selection=False,
    )
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    audit(a.peer, a.root)
