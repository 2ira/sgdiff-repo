"""Full supervision and minibatch target-masked SP4LP, selected using validation only."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_peer_protocol import bind_peer
from evaluate_pubmed_extension import fixed_query_paths
from experiments.sp4lp_full.model import FullSP4LP, configuration, input_graph
from sgdiff.data.graph import GraphData
from sgdiff.data.datasets import load_graph
from sgdiff.engine.baseline_trainer import _edge_index
from sgdiff.engine.reproduction import digest as file_hash


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def path_cache(graph, pairs, path, protocol):
    receipt = dict(graph=protocol.digest(graph.edges), pairs=protocol.digest(pairs))
    with protocol.cache_lock(path):
        if path.exists():
            payload = torch.load(path, weights_only=False, map_location="cpu")
            if payload["receipt"] != receipt:
                raise ValueError(f"Path inputs changed at {path}")
            return payload["store"]
        store = fixed_query_paths(graph, pairs)
        atomic_save(dict(receipt=receipt, store=store), path)
        return store


class Predictor:
    def __init__(self, model, pairs, cache, protocol, device):
        self.model = model
        self.pairs = pairs
        self.cache = cache
        self.protocol = protocol
        self.device = torch.device(device)

    def eval(self):
        self.model.eval()

    def context(self, graph):
        if not self.model.endpoints_only:
            key = self.protocol.digest(graph.edges)
            self.model.store = path_cache(
                graph, self.pairs, self.cache / f"{key}.pt", self.protocol
            )
        x = torch.tensor(graph.x, device=self.device)
        return self.model.encode(x, _edge_index(graph, self.device), graph.num_nodes)

    def score(self, ctx, pairs):
        return self.model.score_pairs(ctx, pairs)


def train(a, protocol, native):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph = load_graph(a.dataset, root=str(a.data_root), download=False)
    split, fingerprint = protocol.split_data(graph, a.seed, a.output)
    cfg = configuration(a.dataset, a.profile)
    graph = input_graph(graph, cfg)
    cache = a.output / "paths" / f"{a.dataset}_s{a.seed}"
    cache.mkdir(parents=True, exist_ok=True)
    run = a.output / "models" / f"{a.dataset}_{a.profile}_{a.variant}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        dataset=a.dataset,
        seed=a.seed,
        profile=a.profile,
        variant=a.variant,
        config=cfg,
        split=fingerprint,
        features=protocol.digest(graph.x),
        max_epochs=a.epochs,
        eval_every=5,
        patience=15,
        batch_positive=512,
        feature_preprocessing="standardized" if cfg.get("standardize", False) else "raw",
        training_context="fixed shuffled minibatch positives removed from both propagation and paths",
        validation_context="train_pos",
        optimizer="Adam",
        clip_norm=1.0,
    )
    contract = run / "receipt.json"
    if contract.exists() and json.loads(contract.read_text()) != receipt:
        raise ValueError(f"Training receipt changed at {run}")
    contract.write_text(json.dumps(receipt, indent=2) + "\n")
    if (run / "complete.json").exists():
        return run
    model = FullSP4LP(graph.num_features, cfg, a.variant).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batches = []
    order = np.random.default_rng(a.seed + 903).permutation(len(split.train_pos))
    # Fixed groups make exact batch-masked paths reusable across epochs.
    for start in range(0, len(order), 512):
        idx = order[start : start + 512]
        keep = np.ones(len(order), dtype=bool)
        keep[idx] = False
        visible = GraphData(graph.name, graph.num_nodes, split.train_pos[keep], x=graph.x)
        pairs = np.vstack([split.train_pos[idx], split.train_neg[idx]])
        assert not set(map(tuple, pairs)) & visible.edge_set()
        store = (
            None
            if model.endpoints_only
            else path_cache(visible, pairs, cache / f"train_{start}.pt", protocol)
        )
        batches.append(
            (
                _edge_index(visible, a.device),
                torch.tensor(pairs, device=a.device),
                torch.tensor(
                    np.r_[np.ones(len(idx)), np.zeros(len(idx))],
                    dtype=torch.float32,
                    device=a.device,
                ),
                store,
            )
        )
    valid_pairs = np.vstack([split.val_pos, split.val_neg])
    visible = GraphData(graph.name, graph.num_nodes, split.train_pos, x=graph.x)
    predictor = Predictor(model, valid_pairs, cache, protocol, a.device)
    x = torch.tensor(graph.x, device=a.device)
    best, bad, updates, last = -float("inf"), 0, 0, 0
    history = []
    if (run / "last.pt").exists():
        saved = torch.load(run / "last.pt", map_location=a.device, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        best, bad, updates, last = saved["best"], saved["bad"], saved["updates"], saved["epoch"]
        history = saved["history"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(saved["gpu_rng"].cpu())
    print(
        json.dumps(
            dict(
                event="training_started",
                run=str(run),
                batches=len(batches),
                positive_examples=len(order),
                parameters=sum(p.numel() for p in model.parameters()),
            )
        ),
        flush=True,
    )
    for epoch in range(last + 1, a.epochs + 1):
        started = time.monotonic()
        model.train()
        total = 0.0
        for i in np.random.default_rng(a.seed + epoch).permutation(len(batches)):
            edges, pairs, labels, store = batches[i]
            opt.zero_grad(set_to_none=True)
            model.store = store
            z = model.encode(x, edges, graph.num_nodes)
            logits = model.score_pairs(z, pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updates += 1
            total += float(loss.detach()) * len(labels)
        row = dict(
            epoch=epoch,
            updates=updates,
            train_loss=total / (2 * len(order)),
            train_seconds=time.monotonic() - started,
        )
        if epoch % 5 == 0 or epoch == a.epochs:
            scores, _ = native.predict(predictor, visible, valid_pairs, 512)
            vals = protocol.metrics(scores[: len(split.val_pos)], scores[len(split.val_pos) :])
            row.update({"val_" + k: v for k, v in vals.items()})
            if vals["auc"] > best:
                best, bad = vals["auc"], 0
                atomic_save(
                    dict(
                        model=model.state_dict(),
                        config=cfg,
                        variant=a.variant,
                        feature_dim=graph.num_features,
                        receipt=receipt,
                        epoch=epoch,
                        updates=updates,
                        validation=vals,
                    ),
                    run / "best.pt",
                )
            else:
                bad += 1
            print(json.dumps(dict(run=run.name, **row)), flush=True)
        history.append(row)
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch % 5 == 0 or epoch == a.epochs:
            atomic_save(
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    best=best,
                    bad=bad,
                    updates=updates,
                    epoch=epoch,
                    history=history,
                    cpu_rng=torch.get_rng_state(),
                    gpu_rng=torch.cuda.get_rng_state()
                    if torch.cuda.is_available()
                    else torch.get_rng_state(),
                ),
                run / "last.pt",
            )
        if bad >= 15:
            break
    selected = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    result = dict(
        run=run.name,
        best_epoch=selected["epoch"],
        best_validation=selected["validation"],
        epochs_run=epoch,
        updates=updates,
        checkpoint_sha256=file_hash(run / "best.pt"),
    )
    (run / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=["cora", "citeseer", "pubmed"], required=True)
    p.add_argument(
        "--profile",
        choices=[
            "full64",
            "paper128",
            "release256",
            "full64_std",
            "paper128_std",
            "release256_std",
        ],
        required=True,
    )
    p.add_argument("--variant", choices=["full", "endpoints", "featureless"], default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    protocol, native = bind_peer(args.peer)
    train(args, protocol, native)
