"""Download benchmark-author data; preserve raw hashes and canonical undirected graphs."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import requests
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
PLANETOID = 'af6abf468a772cc055bec88c52e4d6f51b7c37e3'
SEAL = 'ca1f019a15fb0c21796042165b4e6bee73981dd3'
NAMES = ['Cora', 'Citeseer', 'Pubmed', 'Router', 'USAir', 'NS']


def main():
    raw = ROOT / 'data/raw'
    out = ROOT / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'sources': [], 'datasets': []}

    def fetch(repo, commit, remote, local):
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{remote}'
        path = raw / local
        # Download the pinned revision, including files previously probed from HEAD.
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
        manifest['sources'].append(dict(url=url, file=str(path.relative_to(ROOT)),
                                       sha256=hashlib.sha256(response.content).hexdigest()))
        return path

    for name in NAMES:
        x = None
        if name in NAMES[:3]:
            tag = name.lower()
            paths = {key: fetch('kimiyoung/planetoid', PLANETOID, f'data/ind.{tag}.{key}',
                                f'ind.{tag}.{key}')
                     for key in ['allx', 'tx', 'graph', 'test.index']}
            # These are the original Planetoid benchmark's Python pickle files.
            def load(key):
                with paths[key].open('rb') as f:
                    return pickle.load(f, encoding='latin1')
            allx, tx, graph = load('allx'), load('tx'), load('graph')
            reorder = np.loadtxt(paths['test.index'], dtype=np.int64)
            ordered = np.sort(reorder)
            if tag == 'citeseer':
                full = np.arange(ordered.min(), ordered.max() + 1)
                extended = sp.lil_matrix((len(full), tx.shape[1]), dtype=np.float32)
                extended[ordered - full.min()] = tx
                tx = extended
            x = sp.vstack([allx, tx]).tolil()
            x[reorder] = x[ordered]
            n = max(x.shape[0], max(graph) + 1,
                    max(v for neighbors in graph.values() for v in neighbors) + 1)
            if x.shape[0] < n:
                x.resize((n, x.shape[1]))
            rows = [u for u, vs in graph.items() for _ in vs]
            cols = [v for vs in graph.values() for v in vs]
            a = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        else:
            path = fetch('muhanzhang/SEAL', SEAL, f'Python/data/{name}.mat', f'{name}.mat')
            data = sio.loadmat(path)
            a = sp.csr_matrix(data['net'])
            n = a.shape[0]
        raw_nonzeros = int(a.nnz)
        a = a.maximum(a.T).tocsr()
        a.setdiag(0)
        a.eliminate_zeros()
        a.data[:] = 1
        edges = np.column_stack(sp.triu(a, k=1).nonzero()).astype(np.int64)
        np.savez_compressed(out / f'{name}.npz', edges=edges, n=np.array(n))
        if x is not None:
            x = x.tocsr().astype(np.float32)
            sp.save_npz(out / f'{name}_features.npz', x)
        record = dict(name=name, nodes=n, raw_nonzeros=raw_nonzeros,
                      undirected_edges=len(edges), symmetric_nonzeros=a.nnz,
                      feature_dim=0 if x is None else x.shape[1])
        manifest['datasets'].append(record)
        print(record, flush=True)
    (ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
