#!/usr/bin/env python
"""Exploratory control: remove 40% of edges without creating new isolates.

Motivated by Cora seed-0 results, not a preregistered confirmatory experiment.
All checkpoints must already exist; this script never trains a model.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sgdiff.data.graph import GraphData
from sgdiff.data.observation import graph_fingerprint
from sgdiff.engine.observation_robustness import (
    ALL_MODELS,PROFILES,make_experiment,fit_node,fit_subgraph,heuristic_scores,
    fusion_inputs,extractor_for,node_scores,score_seal,diagnostics,save_json,
)
from sgdiff.models.baselines.sp4lp import path_store
from sgdiff.engine.benchmark import pairs_fingerprint
from sgdiff.utils.metrics import summarize_scores


def without_new_isolates(graph,ratio,seed):
    budget=round(ratio*graph.num_edges)
    degrees=graph.degrees().copy()
    keep=np.ones(graph.num_edges,dtype=bool)
    count=0
    for i in np.random.default_rng(seed).permutation(graph.num_edges):
        u,v=graph.edges[i]
        if degrees[u]<=1 or degrees[v]<=1:
            continue
        keep[i]=False
        degrees[u]-=1;degrees[v]-=1;count+=1
        if count==budget:break
    if count!=budget:
        raise RuntimeError(f"Only {count}/{budget} edges can be removed without new isolates")
    out=GraphData(graph.name,graph.num_nodes,graph.edges[keep],graph.x,graph.y,dict(graph.meta))
    assert np.array_equal(out.degrees()==0,graph.degrees()==0)
    return out


def run(dataset,seed,root):
    out=Path(root)/dataset/f"seed{seed}"
    manifest=json.loads((out/"manifest.json").read_text())
    cfg,graph,split,context,pairs,mean,std=make_experiment(dataset,seed,"local",manifest["config"]["train"]["device"])
    query,y=pairs["test"]
    assert pairs_fingerprint(query,y)==manifest["query_fingerprint"]
    assert graph_fingerprint(context)==manifest["context_fingerprint"]
    clean=dict(np.load(out/"scores/clean.npz"))
    np.testing.assert_array_equal(query,clean["pairs"])
    condition="nonisolating_delete_40"
    file=out/"scores"/f"{condition}.npz"
    if file.exists():
        print("RESUME",dataset,seed,condition,flush=True);return
    started=time.perf_counter()
    observed=without_new_isolates(context,.4,seed+701)
    store=path_store(observed,query)
    clean_store=path_store(context,query)
    diagnostic,arrays=diagnostics(observed,query,context.degrees(),clean_store,store)
    arrays.update(pairs=query,labels=y)
    hs=heuristic_scores(observed,query)
    f=json.loads((out/"fusion_lr.json").read_text())
    x=(fusion_inputs(hs)-np.asarray(f["mean"]))/np.asarray(f["scale"])
    hs["fusion_lr"]=x@np.asarray(f["coef"])[0]+f["intercept"][0]
    extractor=extractor_for(observed,cfg,mean,std)
    rows=[]
    for name in ALL_MODELS:
        if name in hs:
            scores=hs[name]
        else:
            checkpoint=out/f"{name}.pt"
            assert checkpoint.exists(),f"Refusing to train missing checkpoint {checkpoint}"
            if name == "seal":
                model,_=fit_subgraph(name,context,pairs,mean,std,cfg,PROFILES["local"],out)
                scores=score_seal(model,extractor,query,device="cpu",seed=seed)
            else:
                model,_=fit_node(name,context,pairs,mean,std,seed,PROFILES["local"],out)
                scores=node_scores(model,observed,query,mean,std,store=store)
            del model
        assert np.isfinite(scores).all()
        arrays[f"score_{name}"]=scores
        metrics=summarize_scores(y,scores)
        metrics["precision_at_100"]=metrics.pop("hit@100")
        rows.append(dict(dataset=dataset,seed=seed,model=name,condition=condition,
                         query_fingerprint=manifest["query_fingerprint"],
                         graph_fingerprint=graph_fingerprint(observed),**metrics))
    budget=round(.4*context.num_edges)
    info=dict(condition=condition,mechanism="nonisolating_delete",requested_ratio=.4,
              budget=budget,deleted=budget,inserted=0,masked_nodes=0,masked_nonzero_entries=0,
              clean_edges=context.num_edges,observed_edges=observed.num_edges,
              graph_fingerprint=graph_fingerprint(observed),exploratory=True,
              motivation="Added after observing Cora seed-0 isolation differences")
    np.savez_compressed(file,**arrays)
    save_json(file.with_suffix(".json"),dict(perturbation=info,diagnostics=diagnostic,rows=rows,
               seconds=time.perf_counter()-started))
    all_rows=[]
    for meta in sorted((out/"scores").glob("*.json")):
        all_rows.extend(json.loads(meta.read_text())["rows"])
    save_json(out/"metrics.json",all_rows)
    print("DONE",dataset,seed,condition,f"{time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="results/observation_robustness_v2/local")
    p.add_argument("--datasets",nargs="+",default=["cora","pubmed"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    for dataset in a.datasets:
        for seed in a.seeds:run(dataset,seed,a.root)
