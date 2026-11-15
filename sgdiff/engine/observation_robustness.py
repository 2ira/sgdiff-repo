"""Reproducible, resumable local investigation of observation robustness.

One train-only context, disjoint negatives, frozen clean preprocessing and
identical saved evaluation pairs for all models. No fitting on degraded data.
"""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..data.graph import GraphData
from ..data.observation import Condition, conditions, degrade, graph_fingerprint
from ..models.baselines.node_pair import NodePairLinkPredictor, VGAE
from ..models.baselines.seal import SEALBaseline
from ..models.baselines.sp4lp import SP4LPBaseline, path_store
from ..utils.metrics import summarize_scores
from ..utils.seed import set_seed
from .baseline_trainer import _edge_index, train_subgraph_classifier
from .batch import PairSubgraphDataset
from .batch import collate_subgraphs
from .factory import load_split, build_extractor, training_pairs, _cap_eval
from .evaluator import _loader
from .benchmark import split_fingerprint, pairs_fingerprint

PROFILES = {
    "smoke": dict(train_pairs=64, val_pairs=24, test_pairs=32,
                  sg_epochs=2, pretrain=1, epochs=5, sub_epochs=2, mc=2, nodes=16),
    "local": dict(train_pairs=4000, val_pairs=512, test_pairs=1024,
                  sg_epochs=12, pretrain=3, epochs=200, sub_epochs=30, mc=4, nodes=40),
}
ALL_MODELS = ("cn", "aa", "pa", "feature_dot", "feature_cosine", "fusion_lr",
              "gcn", "vgae", "seal", "sp4lp", "sp4lp_endpoints",
              "sp4lp_featureless")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def make_experiment(dataset, seed, profile="local", sg_device="cpu"):
    budget = PROFILES[profile]
    config_path = Path("configs") / f"sgdiff_{dataset}.yaml"
    cfg = load_config(config_path if config_path.exists() else None)
    cfg.data.name, cfg.data.seed, cfg.train.seed = dataset, seed, seed
    cfg.data.max_train_pairs = budget["train_pairs"]
    cfg.data.max_eval_pairs = budget["test_pairs"]
    cfg.subgraph.max_nodes_total = budget["nodes"]
    cfg.structure.num_mc_samples = cfg.feature.num_mc_samples = budget["mc"]
    cfg.train.epochs, cfg.train.pretrain_epochs = budget["sg_epochs"], budget["pretrain"]
    cfg.train.device = sg_device
    cfg.train.eval_every = 2
    cfg.train.run_name = "sgdiff"
    cfg.train.eval_batch_size = 64
    graph, split = load_split(cfg)
    context = graph.drop_edges(np.vstack([split.val_pos, split.test_pos]))
    pairs = {}
    pairs["train"] = training_pairs(split, cfg)
    for name in ("val", "test"):
        p, y = split.pairs(name)
        pairs[name] = _cap_eval(p, y, budget[f"{name}_pairs"], seed + (name == "test"))
    # These moments are fit ONCE on the clean attributes, without edge labels.
    mean = graph.x.mean(0, keepdims=True)
    std = graph.x.std(0, keepdims=True)
    std[std < 1e-6] = 1
    return cfg, graph, split, context, pairs, mean, std


def extractor_for(graph, cfg, mean, std):
    extractor = build_extractor(graph, cfg, seed=cfg.data.seed + 1)
    # The original extractor refits moments on every graph. Freeze them here.
    extractor._features = ((graph.x - mean) / std).astype(np.float32)
    return extractor


from .course_baselines import heuristic_scores, fusion_inputs


def fit_fusion(context, pairs, out):
    train_scores = heuristic_scores(context, pairs["train"][0], remove_target=True)
    scaler = StandardScaler().fit(fusion_inputs(train_scores))
    x = scaler.transform(fusion_inputs(train_scores))
    model = LogisticRegression(C=1., max_iter=1000, random_state=0).fit(x, pairs["train"][1])
    save_json(out / "fusion_lr.json", dict(
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        coef=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        fit="clean training pairs only; C=1 fixed before test evaluation"))
    return scaler, model


@torch.no_grad()
def node_scores(model, graph, pairs, mean, std, batch_size=512, store=None):
    model.eval()
    x = torch.from_numpy(((graph.x - mean) / std).astype(np.float32))
    ei = _edge_index(graph, "cpu")
    if isinstance(model, SP4LPBaseline) and not model.endpoints_only:
        model.store = store if store is not None else path_store(graph, pairs)
    if isinstance(model, VGAE):
        z, _ = model(x, ei, graph.num_nodes)
        score = model.decode
    else:
        z = model.encode(x, ei, graph.num_nodes)
        score = model.score_pairs
    result = []
    for start in range(0, len(pairs), batch_size):
        p = torch.as_tensor(pairs[start:start + batch_size], dtype=torch.long)
        result.append(score(z, p).detach().numpy())
    return np.concatenate(result)


def fit_node(name, context, pairs, mean, std, seed, budget, out):
    set_seed(seed)
    fd = context.num_features
    if name.startswith("sp4lp"):
        model = SP4LPBaseline(fd, endpoints_only=name == "sp4lp_endpoints",
                              featureless=name == "sp4lp_featureless")
    elif name == "vgae":
        model = VGAE(fd)
    elif name == "gcn":
        model = NodePairLinkPredictor(fd, hidden_dim=64, out_dim=64)
    else:
        raise ValueError(name)
    checkpoint = out / f"{name}.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state"])
        return model, payload["training"]
    all_pairs = np.vstack([pairs["train"][0], pairs["val"][0]])
    store = None
    if isinstance(model, SP4LPBaseline) and not model.endpoints_only:
        before = time.perf_counter()
        store = path_store(context, all_pairs)
        model.store = store
        print(f"  {name} clean paths {time.perf_counter()-before:.1f}s", flush=True)
    x = torch.from_numpy(((context.x - mean) / std).astype(np.float32))
    ei = _edge_index(context, "cpu")
    train_p, train_y = pairs["train"]
    lr = .001 if name.startswith("sp4lp") else .01
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1., None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, budget["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        if isinstance(model, VGAE):
            z, kl = model(x, ei, context.num_nodes)
            scorer = model.decode
        else:
            z, kl = model.encode(x, ei, context.num_nodes), None
            scorer = model.score_pairs
        order = np.random.permutation(len(train_p))
        losses = []
        for start in range(0, len(order), 512):
            idx = order[start:start + 512]
            logits = scorer(z, torch.as_tensor(train_p[idx], dtype=torch.long))
            loss = F.binary_cross_entropy_with_logits(logits, torch.as_tensor(train_y[idx], dtype=torch.float32))
            weight = len(idx) / len(train_p)
            weighted = weight * (loss + .001 * kl if kl is not None else loss)
            weighted.backward(retain_graph=start + 512 < len(order))
            losses.append(float(loss.detach()) * weight)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if epoch % 5 == 0 or epoch == budget["epochs"]:
            scores = node_scores(model, context, pairs["val"][0], mean, std, store=store)
            val = summarize_scores(pairs["val"][1], scores)
            history.append(dict(epoch=epoch, loss=sum(losses), val_auc=val["auc"]))
            if val["auc"] > best:
                best, bad = val["auc"], 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if epoch % 25 == 0:
                print(f"  {name} epoch={epoch} val_auc={val['auc']:.4f}", flush=True)
            if bad >= 6:
                break
            if isinstance(model, SP4LPBaseline):
                model.store = store
    model.load_state_dict(best_state)
    training = dict(best_val_auc=best, epochs=epoch, seconds=time.perf_counter()-started,
                    parameters=sum(p.numel() for p in model.parameters()), history=history)
    torch.save(dict(state=model.state_dict(), training=training,
                    settings=getattr(model, "settings", {})), checkpoint)
    save_json(out / f"{name}_training.json", training)
    return model, training


def fit_subgraph(name, context, pairs, mean, std, cfg, budget, out):
    if name != "seal":
        raise ValueError("Observation runner does not train SGDiff; use the upstream v2 harness")
    set_seed(cfg.data.seed)
    extractor = extractor_for(context,cfg,mean,std)
    datasets = {k:PairSubgraphDataset(p,y,extractor) for k,(p,y) in pairs.items()}
    model = SEALBaseline(context.num_features,num_labels=cfg.subgraph.max_label+1)
    checkpoint=out/f"{name}.pt"
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(payload["state"])
        return model,payload["training"]
    started=time.perf_counter()
    training=train_subgraph_classifier(model,{"train":datasets["train"],"val":datasets["val"]},
        device="cpu",epochs=budget["sub_epochs"],seed=cfg.data.seed)
    training={k:v for k,v in training.items() if not k.startswith("test_")}
    training.update(seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()))
    torch.save(dict(state=model.state_dict(),training=training,config=cfg.to_dict()),checkpoint)
    save_json(out/f"{name}_training.json",training)
    return model,training


@torch.no_grad()
def score_seal(model, extractor, pairs, batch_size=64, device="cpu", seed=0):
    model.eval()
    data=PairSubgraphDataset(pairs,np.zeros(len(pairs),dtype=int),extractor)
    chunks=[model.predict_logit(batch.to(device)).cpu().numpy()
            for batch in _loader(data,batch_size)]
    return np.concatenate(chunks)


def diagnostics(graph, pairs, clean_degrees, clean_paths, store):
    deg = graph.degrees()
    u, v = pairs.T
    scores = heuristic_scores(graph, pairs)
    rows = store.rows_for(pairs)
    reach = store.reach[rows]
    lengths = np.diff(store.offs)[rows] - 1
    clean_rows = clean_paths.rows_for(pairs)
    changed = []
    for row, old_row in zip(rows, clean_rows):
        a = store.data[store.offs[row]:store.offs[row+1]]
        b = clean_paths.data[clean_paths.offs[old_row]:clean_paths.offs[old_row+1]]
        changed.append(not np.array_equal(a, b) or store.reach[row] != clean_paths.reach[old_row])
    return dict(isolated_node_fraction=float(np.mean(deg == 0)),
                isolated_endpoint_fraction=float(np.mean((deg[u] == 0) | (deg[v] == 0))),
                mean_endpoint_degree=float(np.mean(np.r_[deg[u], deg[v]])),
                mean_common_neighbors=float(scores["cn"].mean()),
                zero_cn_fraction=float(np.mean(scores["cn"] == 0)),
                unreachable_query_fraction=float(np.mean(~reach)),
                mean_reachable_path_edges=float(lengths[reach].mean()) if reach.any() else None,
                changed_path_fraction=float(np.mean(changed))), dict(
                    degree_min=np.minimum(clean_degrees[u], clean_degrees[v]),
                    observed_degree_min=np.minimum(deg[u], deg[v]),
                    cn=scores["cn"], reachable=reach, path_edges=lengths,
                    changed_path=np.asarray(changed))


def run_one(dataset, seed, profile, model_names, output_root, condition_names=None, sg_device="cpu"):
    unknown=set(model_names)-set(ALL_MODELS)
    if unknown: raise ValueError(f"Unsupported observation models: {unknown}")
    out = Path(output_root) / profile / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg, graph, split, context, pairs, mean, std = make_experiment(dataset, seed, profile, sg_device)
    budget = PROFILES[profile]
    query_p, query_y = pairs["test"]
    manifest = dict(dataset=dataset, seed=seed, profile=profile, budget=budget,
                    protocol="clean train-only context for training, validation and test; frozen clean feature moments",
                    split_fingerprint=split_fingerprint(split),
                    context_fingerprint=graph_fingerprint(context),
                    query_fingerprint=pairs_fingerprint(query_p, query_y),
                    train_fingerprint=pairs_fingerprint(*pairs["train"]),
                    pair_counts={k: len(v[0]) for k,v in pairs.items()},
                    graph=dict(nodes=graph.num_nodes, edges=graph.num_edges, features=graph.num_features),
                    config=cfg.to_dict(), torch=torch.__version__, platform=platform.platform(),
                    threads=torch.get_num_threads(),
                    source_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip())
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        # A review/commit of the finished work must not invalidate identical
        # protocol/data. Keep the original provenance; algorithm changes still
        # require a new output directory, as documented.
        comparable = lambda value: {k:v for k,v in value.items() if k != "source_commit"}
        if comparable(previous) != comparable(manifest):
            raise RuntimeError(f"protocol changed in {out}; use a new output root")
        manifest = previous
    else:
        save_json(manifest_path, manifest)
    np.savez_compressed(out / "split_and_queries.npz", **{
        f"{k}_{s}": getattr(split, f"{k}_{s}") for k in ("train","val","test") for s in ("pos","neg")},
        **{f"actual_{k}_pairs": v[0] for k,v in pairs.items()},
        **{f"actual_{k}_labels": v[1] for k,v in pairs.items()}, feature_mean=mean, feature_std=std)
    print(f"START {dataset} seed={seed} pairs={manifest['pair_counts']}", flush=True)
    scaler, fusion = fit_fusion(context, pairs, out)
    fitted = {}
    for name in model_names:
        if name in ("cn","aa","pa","feature_dot","feature_cosine","fusion_lr"):
            continue
        print(f"FIT {dataset} seed={seed} {name}", flush=True)
        if name in ("seal",):
            fitted[name], _ = fit_subgraph(name, context, pairs, mean, std, cfg, budget, out)
        else:
            fitted[name], _ = fit_node(name, context, pairs, mean, std, seed, budget, out)
    print(f"PATHS clean evaluation {dataset} seed={seed}", flush=True)
    clean_paths = path_store(context, query_p)
    all_rows = []
    forbidden = np.vstack([split.val_pos, split.test_pos, split.val_neg, split.test_neg])
    selected = conditions()
    if condition_names:
        selected = [c for c in selected if c.name in condition_names]
    if profile == "smoke" and not condition_names:
        selected = [c for c in selected if c.name in ("clean","random_delete_40","degree_delete_40","feature_mask_40","joint_degree_40","dice_25")]
    clean_scores = {}
    for condition in selected:
        artifact = out / "scores" / f"{condition.name}.npz"
        meta_path = out / "scores" / f"{condition.name}.json"
        if artifact.exists() and meta_path.exists():
            data = dict(np.load(artifact))
            if all(f"score_{m}" in data for m in model_names):
                info = json.loads(meta_path.read_text())
                all_rows.extend(info["rows"])
                if condition.name == "clean":
                    clean_scores = {m: data[f"score_{m}"] for m in model_names}
                print(f"RESUME {dataset} seed={seed} {condition.name}", flush=True)
                continue
        started = time.perf_counter()
        observed, info = degrade(context, condition, seed + 701, forbidden)
        assert not (observed.edge_set() & {tuple(sorted(p)) for p in query_p.tolist()})
        store = clean_paths if condition.mechanism == "none" else path_store(observed, query_p)
        diagnostic, per_pair = diagnostics(observed, query_p, context.degrees(), clean_paths, store)
        hs = heuristic_scores(observed, query_p)
        hs["fusion_lr"] = fusion.decision_function(scaler.transform(fusion_inputs(hs)))
        rows, arrays = [], dict(pairs=query_p, labels=query_y, **per_pair)
        extractor = None
        for name in model_names:
            if name in hs:
                scores = hs[name]
            elif name in ("seal",):
                if extractor is None:
                    extractor = extractor_for(observed, cfg, mean, std)
                scores = score_seal(fitted[name], extractor, query_p, batch_size=64, device="cpu", seed=seed)
            else:
                scores = node_scores(fitted[name], observed, query_p, mean, std, store=store)
            if not np.isfinite(scores).all():
                raise RuntimeError(f"nonfinite {dataset} {name} {condition.name}")
            if condition.name == "clean":
                clean_scores[name] = scores
            if condition.feature_ratio == 0 and name in ("feature_dot", "feature_cosine"):
                if name in clean_scores:
                    np.testing.assert_array_equal(scores, clean_scores[name])
            metrics = summarize_scores(query_y, scores)
            # The inherited hit@100 is pooled precision, not OGB Hits@K.
            metrics["precision_at_100"] = metrics.pop("hit@100")
            row = dict(dataset=dataset, seed=seed, model=name, condition=condition.name,
                       query_fingerprint=manifest["query_fingerprint"],
                       graph_fingerprint=info["graph_fingerprint"], **metrics)
            rows.append(row)
            arrays[f"score_{name}"] = scores
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, **arrays)
        save_json(meta_path, dict(perturbation=info, diagnostics=diagnostic, rows=rows,
                                 seconds=time.perf_counter()-started))
        all_rows.extend(rows)
        print(f"DONE {dataset} seed={seed} {condition.name} {time.perf_counter()-started:.1f}s " +
              " ".join(f"{r['model']}={r['auc']:.3f}" for r in rows if r['model'] in ('sp4lp','sgdiff','cn')), flush=True)
    # A selected-condition resume must preserve prior and exploratory rows.
    all_rows = []
    for metadata in sorted((out / "scores").glob("*.json")):
        all_rows.extend(json.loads(metadata.read_text())["rows"])
    save_json(out / "metrics.json", all_rows)
    return all_rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--profile", choices=PROFILES, default="local")
    parser.add_argument("--output", default="results/observation_robustness_v2")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sg-device", choices=("cpu","mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for dataset in args.datasets:
        for seed in args.seeds:
            run_one(dataset, seed, args.profile, args.models, args.output, args.conditions, args.sg_device)


if __name__ == "__main__":
    main()
