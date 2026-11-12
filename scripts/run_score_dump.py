#!/usr/bin/env python
"""Dump per-pair scores: the raw material for operating-point and calibration analysis.

``run_probe.py`` keeps only ``auc`` / ``ap`` / ``hit@100`` per (model, probe) and
throws the score vector away.  Every question of the form "did the attack move
the *whole* score distribution or only the head?" needs those vectors, so this
script trains each model once per seed and writes, per probe graph:

* the model's score for every scored test pair (``<model>__<variant>``),
* per-pair structure of *that* graph: ``cn`` (common neighbours), ``null_cn``
  (degree-preserving configuration-model expectation ``d_u d_v / 2m``), the two
  endpoint degrees, and the same-label flag.

Features are recomputed on every inference graph (a deployed model cannot see the
clean graph either); the test positives are withheld from all of them.

    python scripts/run_score_dump.py --dataset cora --seeds 0 1
    python scripts/run_score_dump.py --dataset citeseer --seeds 0 1 --models gcn seal

Output: ``results/scores/<dataset>_s<seed>.npz`` (consumed by
``run_operating_point.py`` and ``run_evidence_calibration.py``).
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_probe as rp  # noqa: E402
from sgdiff.config import load_config  # noqa: E402
from sgdiff.data.perturb import PERTURBATIONS  # noqa: E402
from sgdiff.engine.baseline_trainer import node_feature_matrix  # noqa: E402
from sgdiff.engine.benchmark import build_registry, fit_model  # noqa: E402
from sgdiff.engine.factory import load_split, prepare_datasets  # noqa: E402
from sgdiff.engine.robustness import _cap_pairs  # noqa: E402
from sgdiff.utils.seed import resolve_device, set_seed  # noqa: E402

VARIANTS = ("clean", "del@50%", "del_hub@50%", "ins_hub@50%", "feature_shuffle")


def pair_features(graph, pairs: np.ndarray) -> dict[str, np.ndarray]:
    """Structural description of each pair *on this graph*."""
    n = graph.num_nodes
    edges = np.asarray(graph.edges, dtype=np.int64)
    m = max(len(edges), 1)
    deg = np.zeros(n, dtype=np.float64)
    if len(edges):
        np.add.at(deg, edges[:, 0], 1.0)
        np.add.at(deg, edges[:, 1], 1.0)
    adj = graph.adjacency().tocsr().astype(np.float64)
    a2 = (adj @ adj).tocsr()
    u, v = pairs[:, 0], pairs[:, 1]
    cn = np.asarray(a2[u, v]).ravel()
    du, dv = deg[u], deg[v]
    out = {
        "cn": cn.astype(np.float32),
        "null_cn": (du * dv / (2.0 * m)).astype(np.float32),
        "deg_u": du.astype(np.float32),
        "deg_v": dv.astype(np.float32),
    }
    labels = getattr(graph, "y", None)
    if labels is not None:
        out["same_label"] = (labels[u] == labels[v]).astype(np.int8)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--profile", choices=sorted(rp.PROFILES), default="cpu")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="results/scores")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    config_path = args.config or f"configs/sgdiff_{args.dataset}.yaml"
    overrides = [f"data.name={args.dataset}", f"data.root={args.root}"] + rp.PROFILES[args.profile] + args.overrides
    cfg = load_config(config_path if Path(config_path).exists() else None, overrides)
    device = resolve_device(args.device)
    registry = build_registry(cfg)
    models = args.models or list(registry)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.data.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        cfg_seed.train.device = str(device)
        set_seed(int(seed))

        graph, split = load_split(cfg_seed)
        datasets = prepare_datasets(graph, split, cfg_seed)
        pos, neg = _cap_pairs(split, args.max_pairs)
        pairs = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
        test_edges = split.test_pos.tolist()
        clean_eval = graph.drop_edges(test_edges)

        eval_graphs: dict[str, object] = {}
        for variant in args.variants:
            if variant == "clean":
                eval_graphs[variant] = clean_eval
                continue
            try:
                eval_graphs[variant] = PERTURBATIONS[variant](graph, int(seed)).drop_edges(test_edges)
            except ValueError as exc:  # e.g. feature_shuffle on a featureless graph
                print(f"skip {variant}: {exc}", flush=True)

        payload: dict[str, np.ndarray] = {"pairs": pairs, "labels": labels}
        for variant, eval_graph in eval_graphs.items():
            for key, values in pair_features(eval_graph, pairs).items():
                payload[f"feat__{variant}__{key}"] = values

        for name in models:
            spec = registry[name]
            started = time.perf_counter()
            set_seed(int(seed))
            model, _train, _params = fit_model(
                spec,
                cfg_seed,
                graph,
                split,
                datasets,
                int(seed),
                device=device,
                run_root=f"runs/scores/{cfg_seed.data.name}",
            )
            fit_seconds = time.perf_counter() - started
            for variant, eval_graph in eval_graphs.items():
                set_seed(int(seed))
                features = None
                if variant == "feature_shuffle" and spec.kind in rp.NODE_PAIR_KINDS:
                    features = node_feature_matrix(eval_graph)
                elif spec.kind in rp.NODE_PAIR_KINDS:
                    features = node_feature_matrix(clean_eval)
                scores = rp.score_model(
                    spec.kind, model, cfg_seed, eval_graph, pairs, device, int(seed), features=features
                )
                payload[f"score__{name}__{variant}"] = np.asarray(scores, dtype=np.float32)
            print(
                f"[{cfg_seed.data.name} s{seed}] {name}: fit={fit_seconds:.1f}s "
                f"variants={len(eval_graphs)}",
                flush=True,
            )

        path = out_dir / f"{cfg_seed.data.name}_s{int(seed)}.npz"
        np.savez_compressed(path, **payload)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
