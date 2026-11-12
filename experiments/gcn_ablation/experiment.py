"""GCN link prediction: paired ablations and source-only model selection for transfer."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
NAMES = ['Cora', 'Citeseer', 'Pubmed', 'Router', 'USAir', 'NS']
MODES = ['structure', 'feature', 'combined']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def negative_edges(n, positives, count, rng, excluded=()):
    blocked = set((positives[:, 0] * n + positives[:, 1]).tolist()) | set(excluded)
    if count > n * (n - 1) // 2 - len(blocked):
        raise ValueError('Not enough distinct non-edges')
    result, seen = [], set()
    while len(result) < count:
        pairs = rng.integers(0, n, size=(max(256, 2 * (count - len(result))), 2))
        pairs.sort(axis=1)
        for u, v in pairs:
            key = int(u * n + v)
            if u != v and key not in blocked and key not in seen:
                seen.add(key)
                result.append((u, v))
                if len(result) == count:
                    break
    return np.asarray(result, dtype=np.int64)


def make_split(n, edges, seed):
    rng = np.random.default_rng(seed)
    ordered = edges[rng.permutation(len(edges))]
    nv, nt = int(len(edges) * .05), int(len(edges) * .15)
    train, valid, test = ordered[nv + nt:], ordered[:nv], ordered[nv:nv + nt]
    neg = negative_edges(n, edges, nv + nt, rng)
    train_neg = negative_edges(n, edges, len(train), rng,
                               neg[:, 0] * n + neg[:, 1])
    return dict(train=train, valid=valid, test=test, valid_neg=neg[:nv],
                test_neg=neg[nv:], train_neg=train_neg)


def sparse_tensor(matrix):
    c = matrix.tocoo()
    idx = torch.from_numpy(np.vstack([c.row, c.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(c.data.astype(np.float32)),
                                   c.shape).coalesce()


def row_normalize(x):
    sums = np.asarray(x.sum(axis=1)).ravel()
    inv = np.zeros_like(sums, dtype=np.float32)
    np.divide(1., sums, out=inv, where=sums != 0)
    return (sp.diags(inv) @ x).tocsr()


@dataclass
class Graph:
    name: str
    n: int
    edges: np.ndarray
    split: dict
    adjacency: torch.Tensor
    raw: object
    structural: torch.Tensor
    projected: object


def load_graph(name, seed, out, degree_dim=32, projection_dim=64):
    base = ROOT / 'data/processed'
    data = np.load(base / f'{name}.npz')
    edges, n = data['edges'], int(data['n'])
    split_path = out / 'splits' / f'{name}_seed{seed}.npz'
    split_path.parent.mkdir(parents=True, exist_ok=True)
    s = make_split(n, edges, seed)
    if split_path.exists():
        old = np.load(split_path)
        if set(old.files) != set(s) or any(not np.array_equal(old[k], s[k]) for k in s):
            raise RuntimeError(f'Saved split differs: {split_path}')
    else:
        np.savez_compressed(split_path, **s)
    train = s['train']
    a = sp.csr_matrix((np.ones(2 * len(train), dtype=np.float32),
                       (np.r_[train[:, 0], train[:, 1]], np.r_[train[:, 1], train[:, 0]])),
                      shape=(n, n))
    degree = np.asarray(a.sum(axis=1)).ravel()
    structural = torch.eye(degree_dim)[np.minimum(degree.astype(int), degree_dim - 1)]
    a = a + sp.eye(n, dtype=np.float32)
    inv = np.power(np.asarray(a.sum(axis=1)).ravel(), -.5)
    norm = sparse_tensor(sp.diags(inv) @ a @ sp.diags(inv))
    path = base / f'{name}_features.npz'
    raw, projected = None, None
    if path.exists():
        raw = row_normalize(sp.load_npz(path))
        # Dataset-specific fixed projections align dimensions, not semantic feature meanings.
        code = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        rng = np.random.default_rng(np.random.SeedSequence([seed, code, 7183]))
        p = rng.standard_normal((raw.shape[1], projection_dim), dtype=np.float32)
        projected = torch.from_numpy(np.asarray(raw @ (p / np.sqrt(projection_dim))))
        projected = F.normalize(projected, p=2, dim=1)
        raw = sparse_tensor(raw)
    return Graph(name, n, edges, s, norm, raw, structural, projected)


def model_input(g, mode, protocol):
    if protocol == 'within':
        return None if mode == 'structure' else g.raw
    # Same input width and parameter count for all three transfer ablations.
    d = g.structural if mode != 'feature' else torch.zeros_like(g.structural)
    p = (g.projected if mode != 'structure' else torch.zeros((g.n, 64)))
    if p is None:
        raise ValueError('Original attributes unavailable')
    return torch.cat([d, p], dim=1)


class LinkModel(nn.Module):
    def __init__(self, input_dim, hidden, embedding, dropout, propagate):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Linear(hidden, embedding)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.dropout, self.propagate = dropout, propagate

    def forward(self, x, a):
        # x is A_norm X for GCN, X for MLP; X=I uses A_norm as cached x.
        h = (torch.sparse.mm(x, self.weight) if x.is_sparse else x @ self.weight)
        h = F.dropout(F.relu(h + self.bias), self.dropout, self.training)
        z = h @ self.output.weight.T
        if self.propagate:
            z = torch.sparse.mm(a, z)
        return z + self.output.bias

    @staticmethod
    def score(z, edges):
        return (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)


def cached_input(g, mode, protocol):
    x = model_input(g, mode, protocol)
    if x is None:
        return g.adjacency
    if mode == 'feature':
        return x
    # A X is constant; precomputation is exact because there is no input dropout.
    return torch.sparse.mm(g.adjacency, x).coalesce() if x.is_sparse else torch.sparse.mm(g.adjacency, x)


def metrics(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    score = np.r_[positive, negative]
    # OGB-style global negative-pool Hits@100 with conservative strict tie handling.
    # Fewer than 100 negatives makes this metric trivially 1, documented in the report.
    hits = 1. if len(negative) < 100 else float(np.mean(positive > np.partition(negative, -100)[-100]))
    return dict(auc=float(roc_auc_score(y, score)),
                ap=float(average_precision_score(y, score)), hits100=hits)


@torch.no_grad()
def evaluate(model, g, x, part):
    model.eval()
    z = model(x, g.adjacency)
    pos = model.score(z, torch.from_numpy(g.split[part])).cpu().numpy()
    neg = model.score(z, torch.from_numpy(g.split[part + '_neg'])).cpu().numpy()
    return metrics(pos, neg), pos, neg


def train_model(g, mode, protocol, seed, args):
    seed_all(seed)
    x = cached_input(g, mode, protocol)
    model = LinkModel(x.shape[1], args.hidden, args.embedding, args.dropout, mode != 'feature')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = torch.from_numpy(g.split['train'])
    neg = torch.from_numpy(g.split['train_neg'])
    candidates = torch.cat([pos, neg])
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    best, best_state, best_epoch, best_metrics, stale = -np.inf, None, 0, None, 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(x, g.adjacency)
        loss = F.binary_cross_entropy_with_logits(model.score(z, candidates), labels)
        loss.backward()
        opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid, _, _ = evaluate(model, g, x, 'valid')
            history.append(dict(epoch=epoch, loss=float(loss), **valid))
            if valid['auc'] > best + 1e-8:
                best, best_state, best_epoch = valid['auc'], copy.deepcopy(model.state_dict()), epoch
                best_metrics, stale = valid, 0
            else:
                stale += args.eval_every
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch, stopped_epoch=epoch, valid=best_metrics,
                       seconds=time.monotonic() - started, history=history,
                       parameter_count=sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--datasets', nargs='+', choices=NAMES, default=NAMES)
    p.add_argument('--protocols', nargs='+', choices=['within', 'transfer'], default=['within', 'transfer'])
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--embedding', type=int, default=32)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    if min(args.epochs, args.eval_every, args.patience, args.hidden, args.embedding, args.threads) < 1:
        p.error('Epochs, dimensions, patience and threads must be positive')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch=torch.__version__, numpy=np.__version__, device='cpu',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = args.output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('Configuration changed; choose a new --output directory')
    path.write_text(json.dumps(config, indent=2))
    for seed in args.seeds:
        graphs = {name: load_graph(name, seed, args.output) for name in args.datasets}
        for protocol in args.protocols:
            for name, g in graphs.items():
                for mode in args.modes:
                    if mode != 'structure' and g.raw is None:
                        continue
                    run = args.output / 'runs' / f'{protocol}_{name}_{mode}_seed{seed}'
                    if (run / 'complete.json').exists():
                        continue
                    run.mkdir(parents=True, exist_ok=True)
                    model, training = train_model(g, mode, protocol, seed, args)
                    torch.save(model.state_dict(), run / 'model.pt')
                    (run / 'training.json').write_text(json.dumps(training, indent=2))
                    results = []
                    targets = [g] if protocol == 'within' else list(graphs.values())
                    for target in targets:
                        if mode != 'structure' and target.raw is None:
                            continue
                        x = cached_input(target, mode, protocol)
                        values, positive, negative = evaluate(model, target, x, 'test')
                        np.savez_compressed(run / f'predictions_{target.name}.npz',
                                            positive=positive, negative=negative)
                        results.append(dict(protocol=protocol, source=name, target=target.name,
                                            mode=mode, seed=seed, **values,
                                            best_epoch=training['best_epoch'],
                                            parameter_count=training['parameter_count']))
                    (run / 'complete.json').write_text(json.dumps(results, indent=2))
                    print(json.dumps(dict(protocol=protocol, source=name, mode=mode, seed=seed,
                                          seconds=round(training['seconds'], 2),
                                          epochs=training['stopped_epoch'],
                                          results=[dict(target=r['target'], auc=round(r['auc'], 4))
                                                   for r in results])), flush=True)


if __name__ == '__main__':
    main()
