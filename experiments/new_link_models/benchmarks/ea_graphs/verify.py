"""Verify the released EA graph snapshots without running attacks or training."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def keys(edges):
    return {tuple(map(int, edge)) for edge in edges}


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'File hash differs: {name}')
    bases, splits = {}, {}
    for item in manifest['contexts']:
        key = item['dataset'], item['seed']
        with np.load(root / item['clean_file'], allow_pickle=False) as f:
            base = f['edges']
            assert digest(base) == str(f['base_sha256']) == item['base_sha256']
        with np.load(root / item['split_file'], allow_pickle=False) as f:
            split = {name: f[name] for name in f.files}
        expected = keys(np.vstack([split['train_pos'], split['val_pos']]))
        assert keys(base) == expected and len(base) == len(expected)
        sets = [keys(split[name]) for name in split]
        assert sum(map(len, sets)) == len(set.union(*sets))
        bases[key], splits[key] = keys(base), split
    for item in manifest['graphs']:
        key = item['dataset'], item['seed']
        split, base = splits[key], bases[key]
        with np.load(root / item['file'], allow_pickle=False) as f:
            edges = f['edges']
            stats = json.loads(str(f['statistics']))
            assert str(f['base_sha256']) == item['base_sha256']
        actual = keys(edges)
        assert edges.dtype == np.int64 and edges.ndim == 2 and edges.shape[1] == 2
        assert np.all(edges[:, 0] < edges[:, 1]) and len(actual) == len(edges)
        assert edges.min() >= 0 and edges.max() < manifest['num_nodes'][key[0]]
        assert digest(edges) == stats['graph_sha256'] == item['graph_sha256']
        assert stats == item['statistics']
        assert not actual & keys(np.vstack([split['test_pos'], split['test_neg'], split['val_neg']]))
        assert len(base - actual) == stats['deleted']
        assert len(actual - base) == stats['inserted']
        assert len(base ^ actual) == stats['effective_edits'] == stats['requested_budget']
        assert len(base) == stats['base_edges']
        assert stats['requested_budget'] == round(item['ratio'] * len(base))
        assert len(item['used_by']) == 9
    assert len(manifest['graphs']) == 18 and len(manifest['contexts']) == 6
    return {'graphs': 18, 'clean_contexts': 6, 'splits': 6, 'matched_evaluations':
            sum(len(r['used_by']) for r in manifest['graphs']), 'passed': True}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
