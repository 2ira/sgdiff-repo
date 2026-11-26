"""Frozen training-graph node centrality -> incident-edge AUROC curves (no training)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

VERSION = 'node-rank-auc-v1'


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge_array(a, n, name):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in 'iu':
        raise ValueError(f'{name}: require integer [M,2] edges')
    if a.size and (a.min() < 0 or a.max() >= n or np.any(a[:, 0] == a[:, 1])):
        raise ValueError(f'{name}: invalid node or self loop')
    canonical = np.sort(a, axis=1)
    if len(np.unique(canonical, axis=0)) != len(a):
        raise ValueError(f'{name}: duplicated undirected edges')
    return a


def ranks(n, train, dataset, seed):
    graph = nx.Graph()
    graph.add_nodes_from(range(n)); graph.add_edges_from(train.tolist())
    degree = np.array([graph.degree(i) for i in range(n)], dtype=float)
    pr = nx.pagerank(graph, alpha=.85, tol=1e-12, max_iter=1000, weight=None)
    pagerank = np.array([pr[i] for i in range(n)])
    tie = [hashlib.sha256(f'node-rank-v1|{dataset}|{seed}|{i}'.encode()).hexdigest() for i in range(n)]
    result = {}
    for method, values in [('degree', degree), ('pagerank', pagerank)]:
        order = np.array(sorted(range(n), key=lambda i: (-values[i], tie[i])))
        position = np.empty(n, dtype=np.int64); position[order] = np.arange(1, n+1)
        result[method] = (values, order, position)
    return result, tie


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError('Non-finite score')
    if not len(pos) or not len(neg):
        return None
    # Mann-Whitney U; average ranks give tied positive-negative pairs weight 0.5.
    r = rankdata(np.r_[pos, neg], method='average')
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def masks(position, pairs, percents=range(101)):
    n = len(position)
    entrance = position[pairs].min(axis=1)  # OR rule; each undirected edge counted once.
    return {k: entrance <= math.ceil(n*k/100) for k in percents}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    m = json.loads(manifest_path.read_text())
    n, dataset, seed = int(m['num_nodes']), m['dataset'], int(m['seed'])
    if n < 1:
        raise ValueError('num_nodes must be positive')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    resolve = lambda p: (manifest_path.parent / p).resolve()
    split_path = resolve(m['split_path'])
    with np.load(split_path, allow_pickle=False) as f:
        train, tp, tn = [edge_array(f[k], n, k).copy() for k in ('train_pos', 'test_pos', 'test_neg')]
    def keys(edges): return set(map(tuple, np.sort(edges, axis=1).tolist()))
    if keys(tp) & keys(tn) or keys(train) & (keys(tp) | keys(tn)):
        raise ValueError('Test classes overlap or a test candidate is in training graph')
    ranking, tie = ranks(n, train, dataset, seed)
    hashes = dict(train_pos=digest(train), test_pos=digest(tp), test_neg=digest(tn))
    base = dict(protocol=VERSION, dataset=dataset, seed=seed)
    all_rows, audits, seen = [], [], set()
    for record in m['records']:
        ident = tuple(record[k] for k in ('model','variant','condition'))
        if ident in seen: raise ValueError(f'Duplicate record: {ident}')
        seen.add(ident)
        path = resolve(record['scores_path'])
        with np.load(path, allow_pickle=False) as f:
            pos = np.asarray(f[record.get('pos_key','positive')]).copy()
            neg = np.asarray(f[record.get('neg_key','negative')]).copy()
            if 'test_pos' in f and 'test_neg' in f:
                if not np.array_equal(f['test_pos'],tp) or not np.array_equal(f['test_neg'],tn):
                    raise ValueError('Score archive test pair order mismatch')
                evidence = 'explicit test pair arrays checked'
            else:
                if record.get('positive_sha256') != hashes['test_pos'] or record.get('negative_sha256') != hashes['test_neg']:
                    raise ValueError('Ordered query fingerprints required for scores without pair IDs')
                if not record.get('query_order_evidence'): raise ValueError('Missing order provenance')
                evidence = record['query_order_evidence']
        if pos.ndim != 1 or neg.ndim != 1 or len(pos) != len(tp) or len(neg) != len(tn):
            raise ValueError('Score shape/length mismatch')
        global_auc = auc(pos,neg)
        if global_auc is None: raise ValueError('Global test set must include both labels')
        independent = roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg])
        if abs(global_auc-independent) > 1e-12: raise AssertionError('AUC disagrees with sklearn')
        if 'expected_auc' in record and abs(global_auc-float(record['expected_auc'])) > record.get('auc_tolerance',1e-10):
            raise ValueError(f'Published global AUC mismatch: {ident}: {global_auc} vs {record["expected_auc"]}')
        for method, (values, order, position) in ranking.items():
            pm, nm = masks(position,tp), masks(position,tn)
            for k in range(101):
                count = math.ceil(n*k/100)
                p, q = pos[pm[k]], neg[nm[k]]
                value = auc(p,q)
                if k==100 and value != global_auc: raise AssertionError('100% must equal full AUC')
                cutoff = values[order[count-1]] if count else None
                all_rows.append(dict(**base,model=ident[0],variant=ident[1],condition=ident[2],rank=method,
                    k=k,n_selected_nodes=count,actual_node_pct=100*count/n,n_pos=len(p),n_neg=len(q),
                    pos_coverage=len(p)/len(pos),neg_coverage=len(q)/len(neg),
                    positive_fraction=len(p)/(len(p)+len(q)) if len(p)+len(q) else None,
                    auc=value,low_support=int(min(len(p),len(q))<20),
                    undefined_reason='no_positive' if not len(p) else 'no_negative' if not len(q) else '',
                    cutoff_score=cutoff,boundary_tie_size=int(np.sum(values==cutoff)) if count else 0,
                    test_pos_sha256=hashes['test_pos'],test_neg_sha256=hashes['test_neg']))
        audits.append(dict(model=ident[0],variant=ident[1],condition=ident[2],scores_file=str(path),
            scores_sha256=file_hash(path),positive_scores_sha256=digest(pos),negative_scores_sha256=digest(neg),
            query_order_evidence=evidence,graph_sha256=record.get('graph_sha256'),
            global_auc=global_auc,sklearn_error=abs(global_auc-independent),
            published_auc_error=abs(global_auc-float(record['expected_auc'])) if 'expected_auc' in record else None))
    clean={(r['model'],r['variant'],r['rank'],r['k']):r for r in all_rows if r['condition']=='clean'}
    deltas=[]
    for r in all_rows:
        if r['condition']=='clean': continue
        key=(r['model'],r['variant'],r['rank'],r['k'])
        if key not in clean: raise ValueError(f'Missing paired clean condition: {key}')
        c=clean[key]
        if (r['n_pos'],r['n_neg'])!=(c['n_pos'],c['n_neg']): raise AssertionError('Subset changed')
        deltas.append({**r,'clean_auc':c['auc'],'delta_auc':r['auc']-c['auc'] if r['auc'] is not None and c['auc'] is not None else None})
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'per_seed_curves.csv',all_rows); write_csv(output/'paired_deltas.csv',deltas)
    rank_rows=[]
    for method,(values,order,position) in ranking.items():
        rank_rows.extend(dict(**base,rank=method,node_id=i,value=float(values[i]),position=int(position[i]),tie_key=tie[i]) for i in range(n))
    write_csv(output/'node_ranks.csv',rank_rows)
    audit=dict(**base,num_nodes=n,protocol_commit=m.get('protocol_commit'),manifest_sha256=file_hash(manifest_path),
        split_sha256=file_hash(split_path),array_hashes=hashes,implementation_sha256=file_hash(__file__),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','networkx']},
        rank_source='train_pos only',pagerank=dict(alpha=.85,tol=1e-12,max_iter=1000),records=audits,
        n_curve_rows=len(all_rows),n_delta_rows=len(deltas),k100_check='exact match to global scores for every record')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'{dataset} seed {seed}: {len(audits)} records, {len(all_rows)} curve rows; all 100% checks passed',flush=True)
    return all_rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.output)
