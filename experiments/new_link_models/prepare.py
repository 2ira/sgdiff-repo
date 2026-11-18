"""Prepare pinned Planetoid inputs, including attacker-only node classes."""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import requests
import numpy as np
import scipy.sparse as sp

from sgdiff.data.graph import GraphData

from .protocol import PLANETOID


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--existing-data', type=Path, help='reuse the earlier pinned GCN data download')
    args = p.parse_args()
    raw = args.root / 'raw'
    processed = args.root / 'processed'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    if args.existing_data:
        for name in ['Cora', 'Citeseer']:
            for suffix in ['.npz', '_features.npz']:
                shutil.copyfile(args.existing_data / 'processed' / (name + suffix), processed / (name + suffix))
            for key in ['allx', 'tx', 'graph', 'test.index']:
                file = f'ind.{name.lower()}.{key}'
                shutil.copyfile(args.existing_data / 'raw' / file, raw / file)
    for name in ['cora', 'citeseer']:
        for key in ['allx', 'tx', 'graph', 'test.index', 'ally', 'ty']:
            file = f'ind.{name}.{key}'
            target = raw / file
            if not target.exists():
                r = requests.get(f'https://raw.githubusercontent.com/kimiyoung/planetoid/{PLANETOID}/data/{file}', timeout=60)
                r.raise_for_status()
                target.write_bytes(r.content)
        if not args.existing_data:
            def load(key):
                with (raw / f'ind.{name}.{key}').open('rb') as handle:
                    return pickle.load(handle, encoding='latin1')
            allx, tx, neighbors = load('allx'), load('tx'), load('graph')
            order = np.loadtxt(raw / f'ind.{name}.test.index', dtype=np.int64)
            sorted_order = np.sort(order)
            if name == 'citeseer':
                full = np.arange(sorted_order.min(), sorted_order.max() + 1)
                extended = sp.lil_matrix((len(full), tx.shape[1]), dtype=np.float32)
                extended[sorted_order - full.min()] = tx
                tx = extended
            x = sp.vstack([allx, tx]).tolil()
            x[order] = x[sorted_order]
            n = max(x.shape[0], max(neighbors) + 1,
                    max(v for values in neighbors.values() for v in values) + 1)
            x.resize((n, x.shape[1]))
            graph = GraphData(name, n, [(u, v) for u, values in neighbors.items() for v in values])
            tag = {'cora': 'Cora', 'citeseer': 'Citeseer'}[name]
            np.savez_compressed(processed / f'{tag}.npz', edges=graph.edges, n=np.array(n))
            sp.save_npz(processed / f'{tag}_features.npz', x.tocsr().astype(np.float32))
    manifest = dict(planetoid_commit=PLANETOID,
                    files={str(f.relative_to(args.root)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for folder in [raw, processed] for f in sorted(folder.iterdir()) if f.is_file()})
    (args.root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
