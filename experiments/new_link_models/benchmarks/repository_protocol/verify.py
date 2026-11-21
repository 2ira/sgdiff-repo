"""Verify released main-protocol graphs, splits and metric summaries (NumPy only)."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def keys(array):
    return set(map(tuple, array.tolist()))


def rows(root, name):
    with (root / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def verify(root):
    root = Path(root)
    for line in (root / 'checksums.sha256').read_text().splitlines():
        sha, name = line.split('  ', 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    manifest = json.loads((root / 'input_manifest.json').read_text())
    full_graphs, splits = {}, {}
    for ds in ['cora', 'citeseer', 'pubmed']:
        for seed in range(3):
            with np.load(root / 'splits' / f'{ds}_seed{seed}.npz', allow_pickle=False) as f:
                arrays = {k: f[k] for k in f.files}
            positive = np.vstack([arrays[k] for k in ['train_pos', 'val_pos', 'test_pos']])
            full = np.array(sorted(keys(positive)), dtype=np.int64)
            assert digest(full) == manifest['datasets'][ds]['graph_sha256']
            sets = [keys(arrays[k]) for k in ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']]
            assert sum(map(len, sets)) == len(set.union(*sets))
            assert np.array_equal(arrays['scored_pos'], arrays['test_pos'][:500])
            assert np.array_equal(arrays['scored_neg'], arrays['test_neg'][:500])
            full_graphs[ds, seed], splits[ds, seed] = keys(full), arrays
    seen = set()
    for row in manifest['attacks']:
        ds, seed = row['dataset'], row['seed']
        split, before = splits[ds, seed], full_graphs[ds, seed]
        with np.load(root / row['file'], allow_pickle=False) as f:
            raw, visible = f['raw_edges'], f['edges']
            actual = json.loads(str(f['statistics']))
        assert actual == {k: v for k, v in row.items() if k != 'file'}
        for array in [raw, visible]:
            assert array.dtype == np.int64 and array.shape[1] == 2
            assert len(keys(array)) == len(array) and np.all(array[:, 0] < array[:, 1])
            assert array.min() >= 0 and array.max() < manifest['datasets'][ds]['nodes']
        assert digest(raw) == row['raw_graph_sha256']
        assert digest(visible) == row['graph_sha256']
        after, inference = keys(raw), keys(visible)
        assert inference == after - keys(split['test_pos'])
        assert row['requested_budget'] == round(row['ratio'] * len(before))
        assert row['raw_deleted'] == len(before - after)
        assert row['raw_inserted'] == len(after - before)
        assert row['raw_effective_edits'] == len(before ^ after)
        clean = before - keys(split['test_pos'])
        assert row['visible_effective_edits'] == len(clean ^ inference)
        assert row['inserted_scored_negatives'] == len(inference & keys(split['scored_neg']))
        key = ds, seed, row['attack'], row['ratio']
        assert key not in seen
        seen.add(key)
    assert len(seen) == 81
    assert all(not (ds == 'pubmed' and attack == 'ea') for ds, _, attack, _ in seen)
    data, summary = rows(root, 'per_seed.csv'), rows(root, 'summary.csv')
    assert len(data) == 729 and len(summary) == 243
    groups, seeded = {}, {}
    for row in data:
        key = tuple(row[k] for k in ['dataset', 'model', 'component', 'attack', 'ratio'])
        groups.setdefault(key, []).append(row)
        skey = key + (row['seed'],)
        assert skey not in seeded
        seeded[skey] = row
    for row in summary:
        key = tuple(row[k] for k in ['dataset', 'model', 'component', 'attack', 'ratio'])
        group = groups[key]
        assert len(group) == int(row['n_seeds']) == 3
        for name in ['auc', 'ap', 'hit@100', 'strict_hits100', 'precision100_tie_average', 'drop_auc', 'drop_ap']:
            values = [float(r[name]) for r in group]
            for suffix, value in [('mean', statistics.mean(values)), ('std', statistics.stdev(values))]:
                assert math.isclose(value, float(row[name + '_' + suffix]), abs_tol=1e-12)
    for row in data:
        for name in ['auc', 'ap']:
            clean = seeded[(row['dataset'], row['model'], row['component'], 'clean', '0.0', row['seed'])]
            assert math.isclose(float(row['drop_' + name]), float(clean[name]) - float(row[name]), abs_tol=1e-12)
    paired = rows(root, 'paired_components_summary.csv')
    assert len(paired) == 162
    for row in paired:
        key = tuple(row[k] for k in ['dataset', 'model', 'component', 'attack', 'ratio'])
        for name in ['auc', 'ap']:
            deltas, drops = [], []
            for variant in groups[key]:
                full = seeded[(row['dataset'], row['model'], 'full', row['attack'], row['ratio'], variant['seed'])]
                deltas.append(float(full[name]) - float(variant[name]))
                drops.append(float(variant['drop_' + name]) - float(full['drop_' + name]))
            for prefix, values in [('full_minus_variant_', deltas), ('variant_minus_full_drop_', drops)]:
                for suffix, value in [('mean', statistics.mean(values)), ('std', statistics.stdev(values))]:
                    assert math.isclose(value, float(row[prefix + name + '_' + suffix]), abs_tol=1e-12)
    replay = json.loads((root / 'replay_audit.json').read_text())
    assert replay['passed'] and replay['checkpoints'] == 81 and replay['evaluations'] == 162
    return dict(passed=True, graphs=81, splits=9, evaluations=729, summary_rows=243,
                replayed_checkpoints=81, pubmed_ea='skipped by pinned main limit')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
