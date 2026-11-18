"""Pinned data, disjoint splits, protected targets, and auditable perturbations."""
import hashlib
import json
import pickle
import os
import fcntl
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from sgdiff.data.graph import GraphData
from sgdiff.data.splits import make_link_split

PROTOCOL_VERSION = 'new-link-models-evasion-v2-common-batch512'
PLANETOID = 'af6abf468a772cc055bec88c52e4d6f51b7c37e3'


@contextmanager
def cache_lock(path):
    """Serialize shared cache creation across independent suite workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as handle:
        # Some network filesystems return EAGAIN even for a blocking flock.
        # Explicit nonblocking retries work on both local and network storage.
        deadline = time.monotonic() + 3600
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Timed out waiting for cache lock: {path}')
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_npz(path, **arrays):
    """Publish a complete archive atomically, including on interrupted runs."""
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as output:
            np.savez_compressed(output, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_data(root, name):
    root = Path(root)
    tag = {'cora': 'Cora', 'citeseer': 'Citeseer'}[name]
    with np.load(root / 'processed' / f'{tag}.npz') as f:
        edges, n = f['edges'], int(f['n'])
    x = sp.load_npz(root / 'processed' / f'{tag}_features.npz').toarray().astype(np.float32)
    # Class labels are available ONLY to the DICE attacker and diagnostics.
    with (root / 'raw' / f'ind.{name}.ally').open('rb') as f:
        ally = pickle.load(f, encoding='latin1')
    with (root / 'raw' / f'ind.{name}.ty').open('rb') as f:
        ty = pickle.load(f, encoding='latin1')
    order = np.loadtxt(root / 'raw' / f'ind.{name}.test.index', dtype=np.int64)
    y = np.full(n, -1, dtype=np.int64)
    known = np.asarray(ally).sum(1) > 0
    y[np.arange(len(ally))[known]] = np.asarray(ally)[known].argmax(1)
    known = np.asarray(ty).sum(1) > 0
    y[order[known]] = np.asarray(ty)[known].argmax(1)
    graph = GraphData(name, n, edges, x=x, y=y)
    return graph


def split_data(graph, seed, output):
    split = make_link_split(graph, seed=seed)
    arrays = {k: getattr(split, k) for k in ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']}
    path = Path(output) / 'splits' / f'{graph.name}_seed{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache_lock(path):
        if path.exists():
            with np.load(path) as old:
                assert set(old.files) == set(arrays)
                assert all(np.array_equal(old[k], v) for k, v in arrays.items()), path
        else:
            save_npz(path, **arrays)
    fingerprint = hashlib.sha256(json.dumps({k: digest(v) for k, v in arrays.items()}, sort_keys=True).encode()).hexdigest()
    return split, fingerprint


def metrics(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    scores = np.r_[pos, neg]
    if not np.all(np.isfinite(scores)):
        raise ValueError('Non-finite prediction')
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    # Strict comparisons make ties fail; a tiny pool is explicitly marked.
    hits = 1. if len(neg) < 100 else float(np.mean(pos > np.partition(neg, -100)[-100]))
    # Average over the complete tie group at the cutoff, not input order.
    k = min(100, len(scores))
    threshold = np.partition(scores, -k)[-k]
    above, ties = scores > threshold, scores == threshold
    precision = (labels[above].sum() + (k - above.sum()) * labels[ties].mean()) / k
    return dict(auc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)),
                hits100=hits, precision100=float(precision), hits100_trivial=len(neg) < 100)


def edge_keys(edges, n):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])


def protected_keys(split, n):
    # Labels are never exposed: the complete candidate set is protected.
    pairs = np.vstack([split.test_pos, split.test_neg, split.val_pos, split.val_neg])
    return set(edge_keys(pairs, n).tolist())


def _ea_candidates(graph, protected, budget, seed, device):
    """Visible-graph linear-surrogate EA, explicitly NOT DeepRobust parity."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = graph.num_nodes
    adj = torch.from_numpy(graph.adjacency().toarray()).to(device)
    weight = torch.nn.Parameter(torch.randn(n, 32, device=device) * .1)
    opt = torch.optim.Adam([weight], lr=.01)
    edges = torch.tensor(graph.edges, device=device)
    existing = graph.edge_set()

    def samples():
        pos = edges[rng.integers(len(edges), size=1024)]
        neg = []
        while len(neg) < 1024:
            u, v = map(int, rng.integers(n, size=2))
            u, v = min(u, v), max(u, v)
            if u != v and (u, v) not in existing and u * n + v not in protected:
                neg.append((u, v))
        pairs = torch.cat([pos, torch.tensor(neg, device=device)])
        labels = torch.cat([torch.ones(1024, device=device), torch.zeros(1024, device=device)])
        return pairs, labels

    for _ in range(60):
        pairs, labels = samples()
        z = (adj / adj.sum(1, keepdim=True).clamp(min=1)) @ weight
        loss = torch.nn.functional.binary_cross_entropy_with_logits((z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    a = adj.detach().requires_grad_(True)
    pairs, labels = samples()
    z = (a / a.sum(1, keepdim=True).clamp(min=1)) @ weight.detach()
    loss = torch.nn.functional.binary_cross_entropy_with_logits((z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1), labels)
    gradient, = torch.autograd.grad(loss, a)
    gains = ((gradient + gradient.T) * (1 - 2 * adj)).detach()
    gains[torch.tril(torch.ones_like(adj, dtype=torch.bool))] = -torch.inf
    if protected:
        blocked = torch.tensor(sorted(protected), device=device)
        gains[blocked // n, blocked % n] = -torch.inf
    selected = gains.flatten().topk(budget).indices.cpu().numpy()
    del adj, a, gains, gradient, weight
    return [(int(k // n), int(k % n)) for k in selected]


def perturb(graph, split, attack, ratio, seed, device='cpu'):
    """Edit only the clean inference graph; return exact realized-edge statistics."""
    n = graph.num_nodes
    original = graph.edge_set()
    protected = protected_keys(split, n)
    # Validation positives are part of the allowed transductive test context.
    protected -= set(edge_keys(split.val_pos, n).tolist())
    budget = int(round(ratio * len(original)))
    rng = np.random.default_rng(seed)
    removed, added = set(), set()
    if attack == 'clean' or budget == 0:
        budget = 0
    elif attack == 'ea':
        for pair in _ea_candidates(graph, protected, budget, seed, device):
            (removed if pair in original else added).add(pair)
    elif attack in ('rf', 'dice', 'delete', 'delete_hub', 'insert'):
        ndel = budget if attack.startswith('delete') else 0 if attack == 'insert' else budget // 2
        candidates = [e for e in sorted(original) if e[0] * n + e[1] not in protected]
        if attack == 'dice':
            candidates = [e for e in candidates if graph.y[e[0]] >= 0 and graph.y[e[0]] == graph.y[e[1]]]
        ndel = min(ndel, len(candidates))
        probability = None
        if attack == 'delete_hub':
            deg = graph.degrees()
            probability = np.asarray([deg[u] + deg[v] for u, v in candidates], dtype=np.float64)
            probability /= probability.sum()
        if ndel:
            indices = rng.choice(len(candidates), ndel, replace=False, p=probability)
            removed = {candidates[i] for i in indices}
        nins = 0 if attack.startswith('delete') else budget - ndel
        tries = 0
        while len(added) < nins:
            tries += 1
            if tries > max(10000, nins * 200):
                raise RuntimeError('Cannot realize the requested protected attack budget')
            u, v = map(int, rng.integers(n, size=2))
            u, v = min(u, v), max(u, v)
            if u == v or (u, v) in original or (u, v) in added or u * n + v in protected:
                continue
            if attack == 'dice' and (graph.y[u] < 0 or graph.y[v] < 0 or graph.y[u] == graph.y[v]):
                continue
            added.add((u, v))
    else:
        raise ValueError(attack)
    edited = (original - removed) | added
    out = GraphData(graph.name, n, np.asarray(sorted(edited), dtype=np.int64).reshape(-1, 2), x=graph.x, y=graph.y)
    assert len(original ^ edited) == len(removed) + len(added) == budget
    assert not set(edge_keys(out.edges, n).tolist()) & set(edge_keys(np.vstack([split.test_pos, split.test_neg]), n).tolist())
    stats = dict(attack=attack, ratio=ratio, requested_budget=budget, deleted=len(removed), inserted=len(added),
                 effective_edits=len(original ^ edited), base_edges=len(original), graph_sha256=digest(out.edges),
                 attacker_labels='node classes, attacker only' if attack == 'dice' else 'none',
                 implementation='visible-graph linear surrogate, 60 steps' if attack == 'ea' else attack)
    return out, stats
