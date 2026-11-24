"""Train HL-GNN full and ablation variants on Lin Junxiong's pinned peer protocol splits."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_peer_protocol import bind_peer
from experiments.hlgnn.model import HLGNNLinkModel, PROFILES, VARIANTS, compute_init_temp, configuration
from sgdiff.data.datasets import load_graph
from sgdiff.data.graph import GraphData
from sgdiff.engine.baseline_trainer import _edge_index
from sgdiff.engine.reproduction import digest as file_hash


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class HLGNNPredictor:
    def __init__(self, model: HLGNNLinkModel, device: str | torch.device):
        self.model = model
        self.device = torch.device(device)

    def eval(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def context(self, graph: GraphData) -> torch.Tensor:
        x = torch.tensor(graph.x, dtype=torch.float32, device=self.device)
        ei = _edge_index(graph, self.device)
        return self.model.encode(x, ei, graph.num_nodes)

    @torch.no_grad()
    def score(self, ctx: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return self.model.score_pairs(ctx, pairs.to(self.device))


def train(a) -> Path:
    protocol, native = bind_peer(a.peer)
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    graph = load_graph(a.dataset, root=str(a.data_root), download=False)
    split, fingerprint = protocol.split_data(graph, a.seed, a.output)
    cfg = configuration(a.dataset, a.profile)

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
        eval_every=a.eval_every,
        patience=a.patience,
        batch_positive=512,
        feature_preprocessing="raw Planetoid attributes, float32",
        training_context="per-epoch shuffled minibatch positives removed from propagation graph",
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

    device = torch.device(a.device)
    model = HLGNNLinkModel(graph.num_features, cfg, a.variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_pos = split.train_pos
    train_neg = split.train_neg
    num_pos = len(train_pos)
    pos_keys = set(map(tuple, train_pos))
    neg_keys = set(map(tuple, train_neg))
    assert len(pos_keys) == num_pos, "Duplicate edges in split.train_pos"
    assert not (pos_keys & neg_keys), "Overlap between split.train_pos and split.train_neg"

    pos_u = torch.tensor(train_pos[:, 0], dtype=torch.long, device=device)
    pos_v = torch.tensor(train_pos[:, 1], dtype=torch.long, device=device)
    neg_u = torch.tensor(train_neg[:, 0], dtype=torch.long, device=device)
    neg_v = torch.tensor(train_neg[:, 1], dtype=torch.long, device=device)
    x = torch.tensor(graph.x, dtype=torch.float32, device=device)

    valid_pairs = np.vstack([split.val_pos, split.val_neg])
    visible_val = GraphData(graph.name, graph.num_nodes, train_pos, x=graph.x)
    predictor = HLGNNPredictor(model, device)

    init_temp = compute_init_temp(model.encoder.K, cfg["init"], cfg["alpha"]).tolist()
    best, bad, updates = -float("inf"), 0, 0
    history = []
    batch_pos = 512

    print(
        json.dumps(
            dict(
                event="training_started",
                run=str(run),
                positive_examples=num_pos,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        ),
        flush=True,
    )

    for epoch in range(1, a.epochs + 1):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(a.seed + epoch).permutation(num_pos)
        total = 0.0
        n_examples = 0

        for start in range(0, num_pos, batch_pos):
            idx_np = order[start : start + batch_pos]
            if epoch == 1:
                keep_np = np.ones(num_pos, dtype=bool)
                keep_np[idx_np] = False
                visible_set = set(map(tuple, train_pos[keep_np]))
                batch_set = set(map(tuple, np.vstack([train_pos[idx_np], train_neg[idx_np]])))
                assert not (visible_set & batch_set), "Target leakage into visible training graph"

            idx_t = torch.tensor(idx_np, dtype=torch.long, device=device)
            mask = torch.ones(num_pos, dtype=torch.bool, device=device)
            mask[idx_t] = False

            vis_u = pos_u[mask]
            vis_v = pos_v[mask]
            vis_ei = torch.stack([torch.cat([vis_u, vis_v]), torch.cat([vis_v, vis_u])], dim=0)

            pairs_t = torch.stack(
                [torch.cat([pos_u[idx_t], neg_u[idx_t]]), torch.cat([pos_v[idx_t], neg_v[idx_t]])],
                dim=1,
            )
            labels = torch.cat(
                [
                    torch.ones(len(idx_np), dtype=torch.float32, device=device),
                    torch.zeros(len(idx_np), dtype=torch.float32, device=device),
                ]
            )

            opt.zero_grad(set_to_none=True)
            z = model.encode(x, vis_ei, graph.num_nodes)
            logits = model.score_pairs(z, pairs_t)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updates += 1
            total += float(loss.detach()) * len(labels)
            n_examples += len(labels)

        row = dict(
            epoch=epoch,
            updates=updates,
            train_loss=total / n_examples,
            train_seconds=time.monotonic() - started,
        )
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            scores, _ = native.predict(predictor, visible_val, valid_pairs, 512)
            vals = protocol.metrics(scores[: len(split.val_pos)], scores[len(split.val_pos) :])
            row.update({"val_" + k: v for k, v in vals.items()})
            if vals["auc"] > best:
                best, bad = vals["auc"], 0
                atomic_save(
                    dict(
                        model=model.state_dict(),
                        config=cfg,
                        variant=a.variant,
                        profile=a.profile,
                        feature_dim=graph.num_features,
                        receipt=receipt,
                        epoch=epoch,
                        updates=updates,
                        validation=vals,
                        init_temp=init_temp,
                        learned_temp=model.encoder.temp.detach().cpu().tolist(),
                    ),
                    run / "best.pt",
                )
            else:
                bad += 1
            print(json.dumps(dict(run=run.name, **row)), flush=True)
        history.append(row)
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if bad >= a.patience:
            break

    selected = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    result = dict(
        run=run.name,
        dataset=a.dataset,
        profile=a.profile,
        variant=a.variant,
        seed=a.seed,
        best_epoch=selected["epoch"],
        best_validation=selected["validation"],
        epochs_run=epoch,
        updates=updates,
        checkpoint_sha256=file_hash(run / "best.pt"),
        init_temp=init_temp,
        learned_temp=selected["learned_temp"],
    )
    (run / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("cora", "citeseer", "pubmed"), required=True)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    # Per-fit ownership lets independent GPU workers share the same recovery queue.
    # This changes orchestration only; the training routine above is unchanged.
    import fcntl
    args = parse_args()
    lock_dir = args.output / "models" / f"{args.dataset}_{args.profile}_{args.variant}_s{args.seed}"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "worker.lock").open("a") as lock:
        deadline = time.monotonic() + 7200
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timed out waiting for fit owner: {lock_dir}")
                time.sleep(1)
        train(args)
