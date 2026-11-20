"""Paired, protected-query evasion study. See docs/team_study.zh-CN.md."""
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import torch

from ..config import load_config
from ..data.observation import Condition, degrade, graph_fingerprint, masked_node_ids
from ..models.baselines.fsd import propagate_features
from ..utils.io import save_json
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device, set_seed
from .baseline_trainer import _edge_index, node_feature_matrix, train_node_pair_baseline
from .benchmark import BUDGET_PROFILES, build_registry, capped_train_split, fit_model, pairs_fingerprint, split_fingerprint
from .course_baselines import score_course_model
from .evaluator import _evaluation_context, _loader, score_pairs
from .factory import build_extractor, load_split, prepare_datasets
from .batch import PairSubgraphDataset
from .robustness import _cap_pairs

PRIMARY = ("cn", "feature_cosine", "katz", "gae", "fsd", "pencil", "sp4lp", "sgdiff")
COMPONENTS = {
    "katz": ("katz_h2", "katz_h3"),
    "sp4lp": ("sp4lp_endpoints", "sp4lp_featureless"),
    "sgdiff": ("sgdiff_structure", "sgdiff_feature"),
}
# GAE and FSD use the same encoder. They differ only when an attribute row is missing.
FIT_NAMES = ("sgdiff", "pencil", "gae", "sp4lp", *COMPONENTS["sp4lp"],
             "cn", "feature_cosine", "katz", *COMPONENTS["katz"])


def study_plan():
    plan = [Condition("clean")]
    for pct in (25, 50):
        r = pct / 100
        for kind in ("random_delete", "degree_delete", "random_flip", "dice"):
            plan.append(Condition(f"{kind}_{pct}", kind, r))
        plan.append(Condition(f"feature_mask_{pct}", feature_ratio=r))
        for kind in ("random_delete", "degree_delete"):
            plan.append(Condition(f"joint_{kind}_{pct}", kind, r, r))
    return plan


def settings(dataset, seed, profile, data_root, device):
    cfg = load_config(Path("configs") / f"sgdiff_{dataset}.yaml")
    cfg.data.root = str(data_root)
    cfg.data.seed = cfg.train.seed = seed
    cfg.train.device = device
    if profile == "paper-rq3":
        return cfg
    if profile == "reference":
        return cfg.apply_overrides(BUDGET_PROFILES["cpu"])
    if profile not in {"team", "local"}:
        raise ValueError(f"unknown study profile: {profile}")
    local = profile == "local"
    cfg.data.max_train_pairs = 256 if local else 2000
    cfg.data.max_eval_pairs = 128 if local else 1000
    cfg.subgraph.max_nodes_total = 16 if local else 40
    cfg.train.epochs = cfg.train.finetune_epochs = 4 if local else 8
    cfg.train.pretrain_epochs = cfg.train.epochs
    mc = 2 if local else 4
    cfg.structure.num_mc_samples = cfg.feature.num_mc_samples = cfg.train.finetune_mc_samples = mc
    cfg.train.batch_size = cfg.train.eval_batch_size = 16 if local else 32
    return cfg


def study_datasets(graph, split, cfg, profile):
    datasets = prepare_datasets(graph, split, cfg)
    if profile in {"reference", "paper-rq3"}:
        # Prior SP4LP and Line C tables cap validation by total, test per class.
        pos, neg = _cap_pairs(split, 500)
        datasets["test"] = PairSubgraphDataset(
            np.vstack((pos, neg)), np.repeat([1, 0], [len(pos), len(neg)]),
            datasets["test"].extractor,
        )
    return datasets


def fit_all(cfg, graph, split, datasets, profile, device, run_root):
    registry = build_registry(cfg)
    models = {}
    node_epochs = {"local": 30, "team": 200, "reference": 300, "paper-rq3": 300}[profile]
    for name in FIT_NAMES:
        seed = cfg.data.seed
        set_seed(seed)
        print(f"fit {name}", flush=True)
        if name == "gae":
            model = registry["fsd_none"].builder(graph.num_features, cfg)
            selected = {k: (d.pairs[d.labels == 1], d.pairs[d.labels == 0])
                        for k, d in datasets.items() if k != "train"}
            train_node_pair_baseline(model, graph, capped_train_split(split, cfg), device=device,
                                    epochs=node_epochs, seed=seed, evaluation_pairs=selected)
        else:
            model, _, _ = fit_model(registry[name], cfg, graph, split, datasets, seed,
                                    node_pair_epochs=node_epochs, subgraph_epochs=10 if profile == "local" else 40,
                                    device=device, run_root=str(run_root))
        models[name] = model.eval()
    return models


@torch.no_grad()
def sgdiff_streams(sg, extractor, cohorts, batch_size, device, seed):
    """Score both ablations from the same MC draws as their full prediction."""
    parts = {key: [] for key in ("sgdiff", *COMPONENTS["sgdiff"])}
    for pairs in cohorts:
        ds = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
        # Native RQ3 resets the scoring RNG once per cohort (score_pairs).
        with _evaluation_context(sg, seed):
            for batch in _loader(ds, batch_size):
                out = sg(batch.to(device))
                feat, struct = out["feature_scores"], out["structure_scores"]
                logits = (out["logits"], sg.fusion(torch.zeros_like(feat), struct),
                          sg.fusion(feat, torch.zeros_like(struct)))
                for key, value in zip(parts, logits):
                    parts[key].append((value[:, 1] - value[:, 0]).cpu().numpy())
    return {key: np.concatenate(values) for key, values in parts.items()}


@torch.no_grad()
def predict_all(models, cfg, graph, pairs, features, observed, device, cohort_boundary=None):
    scores = {}
    ei = _edge_index(graph, device)
    x = torch.as_tensor(features, device=device)
    ids = torch.as_tensor(pairs, device=device)
    gae = models["gae"]
    scores["gae"] = gae.score_pairs(gae.encoder(x, ei, len(x)), ids).cpu().numpy()
    if observed.all():
        scores["fsd"] = scores["gae"].copy()
    else:
        filled = propagate_features(features, ei.cpu().numpy(), observed, method="fsd", gamma=1.2)
        z = gae.encoder(torch.as_tensor(filled, device=device), ei, len(x))
        scores["fsd"] = gae.score_pairs(z, ids).cpu().numpy()

    extractor = build_extractor(graph, cfg, seed=cfg.data.seed + 1)
    bs = cfg.train.eval_batch_size
    scores["pencil"] = score_pairs(models["pencil"], extractor, pairs, bs, device, seed=cfg.data.seed)
    cohorts = (pairs,) if cohort_boundary is None else (pairs[:cohort_boundary], pairs[cohort_boundary:])
    scores.update(sgdiff_streams(models["sgdiff"], extractor, cohorts, bs, device, cfg.data.seed))
    for name in FIT_NAMES:
        if name not in scores:
            scores[name] = score_course_model(models[name], graph, pairs, features, device)
    return scores


def run_seed(dataset, seed, profile, data_root, output, device):
    out = Path(output) / profile / dataset / f"seed_{seed}"
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a fresh --output for a rerun")
    device = str(resolve_device(device))
    cfg = settings(dataset, seed, profile, data_root, device)
    graph, split = load_split(cfg)
    ds = study_datasets(graph, split, cfg, profile)
    test = ds["test"]
    clean = test.extractor.graph
    forbidden = np.vstack((split.test_pos, split.test_neg, split.val_neg))
    test_set = {tuple(pair) for pair in forbidden}
    base_x = node_feature_matrix(clean)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent = Path("runs/team_study") / profile / dataset
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(mkdtemp(prefix=f"s{seed}-", dir=checkpoint_parent))
    print(f"{dataset} seed {seed} ({profile})", flush=True)
    models = fit_all(cfg, graph, split, ds, profile, device, run_root)
    manifest = dict(dataset=dataset, seed=seed, profile=profile, config=cfg.to_dict(),
                    split=split_fingerprint(split), train_graph=graph_fingerprint(ds["train"].extractor.graph),
                    queries={k: pairs_fingerprint(d.pairs, d.labels) for k, d in ds.items()},
                    n_queries={k: len(d) for k, d in ds.items()}, conditions={})
    arrays = {f"{k}_{field}": getattr(d, field) for k, d in ds.items() for field in ("pairs", "labels")}
    rows = []
    cohort_boundary = int(test.labels.sum()) if profile == "paper-rq3" else None
    for condition in study_plan():
        g, info = degrade(clean, condition, seed, forbidden)
        if test_set.intersection(g.edge_set()) or info["deleted"] + info["inserted"] != info["budget"]:
            raise RuntimeError(f"invalid query protection or edit budget for {condition.name}")
        observed = np.ones(g.num_nodes, dtype=bool)
        observed[masked_node_ids(g.num_nodes, condition.feature_ratio, seed)] = False
        features = base_x.copy()
        features[~observed] = 0
        predictions = predict_all(models, cfg, g, test.pairs, features, observed, device, cohort_boundary)
        deg = g.degrees()
        info["isolated_fraction"] = float((deg == 0).mean())
        info["query_with_isolate_fraction"] = float((deg[test.pairs].min(1) == 0).mean())
        manifest["conditions"][condition.name] = info
        for model, scores in predictions.items():
            metrics = summarize_scores(test.labels, scores)
            rows.append(dict(dataset=dataset, seed=seed, model=model, condition=condition.name,
                             auc=metrics["auc"], ap=metrics["ap"]))
            arrays[f"{condition.name}__{model}"] = scores
    with TemporaryDirectory(dir=out.parent) as temp:
        # Publish the three data files and training trace as one complete directory.
        stage = Path(temp) / "complete"
        stage.mkdir()
        np.savez_compressed(stage / "queries_and_scores.npz", **arrays)
        save_json(manifest, stage / "manifest.json")
        save_json(rows, stage / "metrics.json")
        copyfile(run_root / f"sgdiff-s{seed}" / "metrics.jsonl", stage / "sgdiff_training.jsonl")
        stage.rename(out)
    print(f"{len(rows)} paired cells saved in {out}", flush=True)
