"""Individual training recipes, one shared evaluation cohort and persisted graph bank."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.attack import apply_attack
from ..data.graph import GraphData
from filelock import FileLock
from ..data.observation import Condition, degrade, graph_fingerprint, masked_node_ids
from ..utils.metrics import summarize_scores
from ..utils.seed import set_seed
from .baseline_trainer import node_feature_matrix, train_node_pair_baseline
from .benchmark import build_registry, capped_train_split, fit_model, pairs_fingerprint, split_fingerprint
from .course_baselines import build_course_model
from .factory import build_model, load_split
from .model_scores import frozen_scores
from ..models.baselines.node_pair import AblatedGAE, GAE_ABLATIONS
from .team_study import COMPONENTS as HISTORICAL_COMPONENTS, PRIMARY, settings, study_datasets, study_plan
from .trainer import Trainer, load_checkpoint

OWNED = ("gae", "sp4lp", "katz")
COMPONENTS = {**HISTORICAL_COMPONENTS, "gae": GAE_ABLATIONS}
HEURISTICS = {"cn", "feature_cosine", "katz", *COMPONENTS["katz"]}
PROTOCOL = "paired-citation-v1"


def expanded(names):
    for name in dict.fromkeys(names):
        yield name
        if name in ("katz", "sp4lp", "gae"):
            yield from COMPONENTS[name]


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def archive(path, **arrays):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def json_file(path, value):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def conditions():
    return study_plan() + [Condition(f"rq3_{attack}_{pct}", attack, pct / 100)
                           for attack in ("rf", "dice", "ea") for pct in (25, 50)]


def bank(graph, split, ds, seed, directory):
    """Generate once; historical attacks replicate edit-then-hide, missingness hides first."""
    directory.mkdir(parents=True, exist_ok=True)
    clean = ds["test"].extractor.graph
    pairs = ds["test"].pairs
    excluded = np.vstack((split.test_pos, split.test_neg, split.val_neg))
    identity = dict(protocol=PROTOCOL, dataset=graph.name, seed=seed, split=split_fingerprint(split),
                    graph=graph_fingerprint(graph), clean=graph_fingerprint(clean),
                    test=pairs_fingerprint(pairs, ds["test"].labels),
                    classes_sha256=hashlib.sha256(np.ascontiguousarray(graph.y).tobytes()).hexdigest())
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored["identity"] != identity:
            raise ValueError(f"incompatible graph bank: {directory}")
        return stored
    rows = {}
    clean_edges = clean.edge_set()
    neg = set(map(tuple, split.test_neg))
    pos = set(map(tuple, split.test_pos))
    for c in conditions():
        if c.name.startswith("rq3_"):
            attacked = apply_attack(graph, c.mechanism, ratio=c.edge_ratio, seed=seed, device="cpu")
            observed = attacked.drop_edges(split.test_pos.tolist())
            meta = dict(attack=attacked.meta["attack"], budget_base="original",
                        budget_base_edges=graph.num_edges)
        else:
            observed, meta = degrade(clean, c, seed, excluded)
            meta.update(budget_base="visible", budget_base_edges=clean.num_edges)
        edges = observed.edge_set()
        assert not (pos & edges)
        if not c.name.startswith("rq3_") and (neg & edges):
            raise ValueError("protected candidate was inserted")
        missing = masked_node_ids(graph.num_nodes, c.feature_ratio, seed)
        target = directory / (c.name + ".npz")
        archive(target, edges=observed.edges, missing=missing)
        rows[c.name] = dict(**meta, requested_edge_ratio=c.edge_ratio, requested_feature_ratio=c.feature_ratio,
                            file_sha256=digest(target), visible_deleted=len(clean_edges - edges),
                            visible_inserted=len(edges - clean_edges), test_neg_in_context=len(neg & edges))
    archive(directory / "test.npz", pairs=pairs, labels=ds["test"].labels)
    result = dict(identity=identity, test_sha256=digest(directory / "test.npz"), conditions=rows)
    json_file(manifest_path, result)
    return result


def training(cfg, dataset, seed, names, directory, device, padded_limit):
    graph, split = load_split(cfg)
    profile = "paper-rq3" if names == ["sgdiff"] else "reference"
    datasets = study_datasets(graph, split, cfg, profile)
    registry = build_registry(cfg)
    for name in dict.fromkeys("gae" if n == "fsd" else n for n in expanded(names)):
        if name in HEURISTICS:
            continue
        ckpt = checkpoint_path(directory, dataset, seed, name)
        if ckpt.exists():
            continue
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        if name == "sgdiff":
            cfg.train.run_name = ckpt.parent.name
            previous = ckpt.with_name("last.pt")
            if previous.exists():
                cfg.train.resume = str(previous)
            model = build_model(cfg, datasets["train"].feature_dim)
            model.structure.denoiser.max_padded_pairs = padded_limit
            Trainer(model, cfg, run_dir=ckpt.parent.parent).fit(datasets["train"], datasets["val"])
            continue
        if name == "gae" or name in GAE_ABLATIONS:
            model = (registry["fsd_none"].builder(graph.num_features, cfg) if name == "gae"
                     else AblatedGAE(graph.num_features, name))
            selected = {k: (d.pairs[d.labels == 1], d.pairs[d.labels == 0])
                        for k, d in datasets.items() if k != "train"}
            metrics = train_node_pair_baseline(model, graph, capped_train_split(split, cfg), device=device,
                                              epochs=300, seed=seed, evaluation_pairs=selected)
        else:
            model, metrics, _ = fit_model(registry[name], cfg, graph, split, datasets, seed, device=device,
                                          node_pair_epochs=300, subgraph_epochs=40)
        temp = ckpt.with_suffix(".tmp")
        torch.save(dict(model_state=model.state_dict(), config=cfg.to_dict(), name=name, metrics=metrics), temp)
        temp.replace(ckpt)


def checkpoint_path(root, dataset, seed, name):
    if name == "sgdiff":
        return root / dataset / "sgdiff" / f"sgdiff-s{seed}" / "best.pt"
    family = "sp4lp" if name.startswith("sp4lp") else "baselines"
    filename = "gae" if name == "fsd" else name
    return root / dataset / family / f"seed_{seed}" / f"{filename}.pt"


def restore(name, cfg, graph, ckpt, device, padded_limit):
    if name in HEURISTICS:
        return build_course_model(name, graph.num_features)
    if name == "sgdiff":
        model, actual = load_checkpoint(ckpt, device)
        model.structure.denoiser.max_padded_pairs = padded_limit
    else:
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        actual = ExperimentConfig.from_dict(payload["config"])
        if payload["name"] != ("gae" if name == "fsd" else name):
            raise ValueError(f"wrong checkpoint variant: {ckpt}")
        if name in GAE_ABLATIONS:
            model = AblatedGAE(graph.num_features, name)
        else:
            spec = build_registry(cfg)["fsd_none" if name in ("gae", "fsd") else name]
            model = spec.builder(graph.num_features, cfg)
        model.load_state_dict(payload["model_state"])
        model.to(device)
    for section in ("data", "structure", "feature", "subgraph", "train", "model"):
        expected = vars(getattr(cfg, section)).copy()
        saved = vars(getattr(actual, section)).copy()
        ignored = {"root"} if section == "data" else {"device", "run_name", "resume", "run_dir"} if section == "train" else set()
        for field in ignored:
            expected.pop(field, None)
            saved.pop(field, None)
        if expected != saved:
            raise ValueError(f"checkpoint {section} differs from declared recipe: {ckpt}")
    return model.eval()


def experiment(dataset, seed, names, data_root, weights, output, device, action, padded_limit,
               condition_names=None):
    ref = settings(dataset, seed, "reference", data_root, device)
    paper = settings(dataset, seed, "paper-rq3", data_root, device)
    if action in ("train", "run"):
        other = [name for name in names if name != "sgdiff"]
        if other:
            training(ref, dataset, seed, other, weights, device, padded_limit)
        if "sgdiff" in names:
            training(paper, dataset, seed, ["sgdiff"], weights, device, padded_limit)
        if action == "train":
            return
    graph, split = load_split(ref)
    datasets = study_datasets(graph, split, ref, "reference")
    clean = datasets["test"].extractor.graph
    output = output / dataset / f"seed_{seed}"
    graphs = output / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    with FileLock(str(graphs / ".bank.lock")):
        manifest = bank(graph, split, datasets, seed, graphs)
    pairs, labels = datasets["test"].pairs, datasets["test"].labels
    if digest(graphs / "test.npz") != manifest["test_sha256"]:
        raise ValueError("test archive checksum mismatch")
    with np.load(graphs / "test.npz") as test:
        np.testing.assert_array_equal(pairs, test["pairs"])
        np.testing.assert_array_equal(labels, test["labels"])
    clean_x = node_feature_matrix(clean)
    for name in expanded(names):
        cfg = paper if name == "sgdiff" else ref
        ckpt = checkpoint_path(weights, dataset, seed, name)
        model = restore(name, cfg, graph, ckpt, device, padded_limit)
        signature = "parameter-free" if name in HEURISTICS else digest(ckpt)
        dest = output / name
        dest.mkdir(exist_ok=True)
        recipe = dict(model=name, checkpoint=signature, device=device,
                      layout_limit=padded_limit if name == "sgdiff" else None,
                      protocol=PROTOCOL, split=manifest["identity"]["split"],
                      test=manifest["test_sha256"], training_recipe="paper-rq3" if name=="sgdiff" else "reference")
        receipt = dest / "recipe.json"
        if receipt.exists() and json.loads(receipt.read_text()) != recipe:
            raise ValueError(f"changed model or input; select a fresh output: {dest}")
        json_file(receipt, recipe)
        for cond, info in manifest["conditions"].items():
            if condition_names is not None and cond not in condition_names:
                continue
            src, target = graphs / f"{cond}.npz", dest / f"{cond}.npz"
            if digest(src) != info["file_sha256"]:
                raise ValueError(f"corrupt observation: {src}")
            if target.exists():
                with np.load(target) as cache:
                    if str(cache["graph_sha"]) != info["file_sha256"]:
                        raise ValueError(f"observation mismatch: {target}")
                continue
            with np.load(src) as entry:
                edges, missing = entry["edges"], entry["missing"]
            x, features = clean.x, clean_x
            observed = np.ones(graph.num_nodes, dtype=bool)
            if len(missing):
                observed[missing] = False
                x, features = x.copy(), features.copy()
                x[missing], features[missing] = 0, 0
            g = GraphData(graph.name, graph.num_nodes, edges, x=x, y=graph.y)
            scores = frozen_scores(name, model, cfg, g, pairs, features, observed, device, seed,
                                   cohort_boundary=int(labels.sum()))
            for values in scores.values():
                if values.shape != labels.shape or not np.isfinite(values).all():
                    raise ValueError(f"invalid scores for {name}/{cond}")
            archive(target, graph_sha=info["file_sha256"], **scores)
            print(dataset, seed, name, cond, round(summarize_scores(labels, scores[name])["auc"], 6), flush=True)
