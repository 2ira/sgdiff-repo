"""Train and evaluate CN, GCN and SGDiff on the pinned Part-2 peer protocol."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_peer_protocol import FrozenPredictor, REVISION, bind_peer
from sgdiff.engine.course_baselines import build_course_model
from sgdiff.engine.factory import load_split
from sgdiff.engine.part2_peer import (
    GCNPeerPredictor,
    GCN_VARIANTS,
    SGDiffPeerPredictor,
    SGDiffSharedScores,
    SGDIFF_VARIANTS,
    gcn_checkpoint,
    load_gcn,
    train_gcn,
)
from sgdiff.engine.reproduction import (
    checkpoint_path,
    digest as file_digest,
    restore,
    training,
)
from sgdiff.engine.team_study import settings
from summarize_peer_protocol import summarize


FAMILIES = {
    "cn": ("cn",),
    "gcn": GCN_VARIANTS,
    "sgdiff": SGDIFF_VARIANTS,
}


def copy_verified(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_digest(source) != file_digest(target):
            raise ValueError(f"conflicting peer asset: {target}")
        return
    shutil.copy2(source, target)


def stage_attacks(dataset: str, seed: int, assets: Path, peer_assets: Path, output: Path):
    """Stage published graphs; never regenerate the expensive EA attacks."""
    target = output / "attacks"
    if dataset in ("cora", "citeseer"):
        for source in (assets / "attacks").glob(f"{dataset}_s{seed}_*.npz"):
            if "_ea_" not in source.name:
                copy_verified(source, target / source.name)
        graphs = peer_assets / "graphs"
        for ratio in ("0.1", "0.25", "0.5"):
            name = f"{dataset}_s{seed}_ea_{ratio}.npz"
            copy_verified(graphs / name, target / name)
    else:
        for ratio in ("0.1", "0.25", "0.5"):
            name = f"pubmed_s{seed}_ea_{ratio}.npz"
            copy_verified(assets / "pubmed" / "attacks" / name, target / name)


def verify_split(dataset: str, seed: int, split, assets: Path):
    root = assets / ("pubmed/splits" if dataset == "pubmed" else "splits")
    with np.load(root / f"{dataset}_seed{seed}.npz") as saved:
        for name in saved.files:
            np.testing.assert_array_equal(getattr(split, name), saved[name])


def published_graph_hashes(root: Path):
    rows = []
    for source in (root / "results/peer-protocol/per_seed.csv",
                   root / "results/peer-protocol/pubmed/per_seed.csv"):
        rows.append(pd.read_csv(source))
    expected = {}
    for frame in rows:
        for row in frame.to_dict("records"):
            key = (row["dataset"], int(row["seed"]), row["condition"])
            value = row.get("peer_graph_sha256") or row["graph_sha256"]
            if isinstance(value, float) and np.isnan(value):
                value = row["graph_sha256"]
            if key in expected and expected[key] != value:
                raise ValueError(f"published graph hashes disagree for {key}")
            expected[key] = value
    return expected


def sgdiff_config(dataset: str, seed: int, data_root: str, device: str,
                  batch_size: int, eval_batch_size: int, num_workers: int,
                  finetune_epochs: int):
    cfg = settings(dataset, seed, "paper-rq3", data_root, device)
    cfg.train.batch_size = batch_size
    cfg.train.eval_batch_size = eval_batch_size
    cfg.train.num_workers = num_workers
    cfg.train.finetune_epochs = finetune_epochs
    return cfg


def source_receipt(family, variant, dataset, seed, checkpoint, graph, split_sha, cfg=None):
    stable_config = None
    if cfg is not None:
        stable_config = cfg.to_dict()
        for name in ("device", "run_name", "resume", "run_dir", "num_workers"):
            stable_config["train"].pop(name, None)
    return {
        "family": family,
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "peer_commit": REVISION,
        "checkpoint": "parameter-free" if checkpoint is None else file_digest(checkpoint),
        "split_sha256": split_sha,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "config": stable_config,
    }


def comparable_receipt(value):
    value = json.loads(json.dumps(value))
    config = value.get("config")
    if config is not None:
        for name in ("device", "run_name", "resume", "run_dir", "num_workers"):
            config["train"].pop(name, None)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "evaluate", "run"))
    p.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--datasets", nargs="+", choices=("cora", "citeseer", "pubmed"),
                   default=["cora", "citeseer", "pubmed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-root", default="data")
    p.add_argument("--peer", type=Path, required=True,
                   help="Detached checkout of the pinned peer evaluator commit")
    p.add_argument("--assets", type=Path, default=Path("runs/peer-assets"))
    p.add_argument("--peer-ea-assets", type=Path, required=True,
                   help="Directory containing Cora/CiteSeer peer EA graphs")
    p.add_argument("--weights", type=Path, default=Path("runs/part2-peer/weights"))
    p.add_argument("--output", type=Path, default=Path("runs/part2-peer/evaluation"))
    p.add_argument("--tables", type=Path, default=Path("results/part2-peer-cn-gcn-sgdiff"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gcn-epochs", type=int, default=100)
    p.add_argument("--sgdiff-batch-size", type=int, default=32)
    p.add_argument("--sgdiff-eval-batch-size", type=int, default=64)
    p.add_argument("--sgdiff-finetune-epochs", type=int, default=5,
                   help="Frozen-diffusion fusion-calibration budget")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; may change when resuming a checkpoint")
    p.add_argument("--max-padded-pairs", type=int, default=8192)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if any(value < 1 for value in (a.gcn_epochs, a.sgdiff_batch_size,
                                    a.sgdiff_eval_batch_size, a.sgdiff_finetune_epochs,
                                    a.threads)) or a.num_workers < 0:
        p.error("training and thread budgets must be positive")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.set_num_threads(a.threads)
    protocol, native = bind_peer(a.peer)
    expected_hash = published_graph_hashes(Path(__file__).resolve().parents[1])
    rows, provenance = [], []

    for dataset in a.datasets:
        for seed in a.seeds:
            cfg = sgdiff_config(dataset, seed, a.data_root, a.device,
                                a.sgdiff_batch_size, a.sgdiff_eval_batch_size, a.num_workers,
                                a.sgdiff_finetune_epochs)
            graph, split = load_split(cfg)
            verify_split(dataset, seed, split, a.assets)
            split_sha = protocol.split_data(graph, seed, a.output)[1]
            stage_attacks(dataset, seed, a.assets, a.peer_ea_assets, a.output)

            if a.action in ("train", "run"):
                if "gcn" in a.models:
                    for variant in GCN_VARIANTS:
                        train_gcn(graph, split, dataset, seed, variant,
                                  gcn_checkpoint(a.weights, dataset, seed, variant),
                                  a.device, a.gcn_epochs)
                if "sgdiff" in a.models:
                    training(cfg, dataset, seed, ["sgdiff"], a.weights, a.device,
                             a.max_padded_pairs)
            if a.action == "train":
                continue

            pairs = np.vstack((split.test_pos, split.test_neg))
            shared_sgdiff = None
            for family in a.models:
                for variant in FAMILIES[family]:
                    checkpoint = None
                    model_cfg = None
                    if family == "cn":
                        predictor = FrozenPredictor(build_course_model("cn", graph.num_features))
                    elif family == "gcn":
                        checkpoint = gcn_checkpoint(a.weights, dataset, seed, variant)
                        model, _ = load_gcn(graph, dataset, seed, variant, checkpoint, a.device)
                        predictor = GCNPeerPredictor(model, a.device)
                    else:
                        checkpoint = checkpoint_path(a.weights, dataset, seed, "sgdiff")
                        if shared_sgdiff is None:
                            sgdiff = restore("sgdiff", cfg, graph, checkpoint, a.device,
                                             a.max_padded_pairs)
                            shared_sgdiff = SGDiffSharedScores(
                                sgdiff, cfg, pairs, len(split.test_pos), a.device, seed,
                                checkpoint, a.output / "shared-sgdiff" / dataset / f"seed_{seed}",
                            )
                        predictor = SGDiffPeerPredictor(shared_sgdiff, variant, graph.num_nodes)
                        model_cfg = cfg

                    run = a.output / "runs" / f"{dataset}_{variant}_s{seed}"
                    run.mkdir(parents=True, exist_ok=True)
                    receipt = source_receipt(family, variant, dataset, seed, checkpoint,
                                             graph, split_sha, model_cfg)
                    receipt_path = run / "receipt.json"
                    if receipt_path.exists():
                        previous = json.loads(receipt_path.read_text())
                        if comparable_receipt(previous) != comparable_receipt(receipt):
                            raise ValueError(f"changed model/configuration at {run}")
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                    provenance.append(receipt)
                    args = SimpleNamespace(dataset=dataset, seed=seed, model=family,
                                           variant=variant, output=a.output, eval_batch_size=512)
                    records = native.evaluate(predictor, graph, split, args, run)
                    for record in records:
                        attack, ratio = record["attack"], record["ratio"]
                        condition = attack if attack in ("clean", "feature_shuffle") else f"{attack}_{ratio:g}"
                        with np.load(run / f"predictions_{condition}.npz") as saved:
                            positive, negative = saved["positive"], saved["negative"]
                        replay = protocol.metrics(positive, negative)
                        for metric, value in replay.items():
                            if value != record["metrics"][metric]:
                                raise ValueError(f"metric replay mismatch: {run}/{condition}/{metric}")
                        actual = record["statistics"]["graph_sha256"]
                        expected = expected_hash[(dataset, seed, condition)]
                        rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "family": family,
                            "model": variant,
                            "condition": condition,
                            "positives": len(positive),
                            "negatives": len(negative),
                            "graph_hash_matches": int(actual == expected),
                            "graph_sha256": actual,
                            "peer_graph_sha256": expected,
                            **{name: replay[name] for name in ("auc", "ap", "hits100", "precision100")},
                        })
                    del predictor
                    if a.device == "cuda":
                        torch.cuda.empty_cache()

    if a.action == "train":
        return
    a.tables.mkdir(parents=True, exist_ok=True)
    table = a.tables / "per_seed.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (a.tables / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if set(a.seeds) == {0, 1, 2}:
        summarize(a.tables)
    expected_cells = len(a.datasets) * len(a.seeds) * sum(len(FAMILIES[x]) for x in a.models) * 14
    if len(rows) != expected_cells or not all(row["graph_hash_matches"] for row in rows):
        raise ValueError(f"incomplete or graph-misaligned result grid: {len(rows)}/{expected_cells}")
    print(f"Completed {len(rows)} aligned cells in {a.tables}")


if __name__ == "__main__":
    main()
