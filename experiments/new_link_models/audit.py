"""Independent data, masking, budget and checkpoint-selection audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import digest, edge_keys, load_data, split_data
from .run import dump, graph_with_edges, source_hashes
from .sources import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    for name in ['ncn', 'lpformer', 'refined_gae']:
        verify(args.source_root, name)
    graphs, splits = {}, {}
    runs = list((args.output / 'runs').glob('*/complete.json'))
    for file in runs:
        run = json.loads(file.read_text())
        cfg = run['config']
        ds, seed = cfg['dataset'], cfg['seed']
        if ds not in graphs:
            graphs[ds] = load_data(args.data_root, ds)
        graph = graphs[ds]
        split, sha = split_data(graph, seed, args.output)
        splits[ds, seed] = split
        assert sha == cfg['split_sha256']
        assert cfg['code_sha256'] == source_hashes()
        assert cfg['graph_sha256'] == digest(graph.edges)
        assert cfg['features_sha256'] == digest(graph.x)
        assert cfg['labels_sha256'] == digest(graph.y)
        sets = [set(edge_keys(getattr(split, k), graph.num_nodes)) for k in
                ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']]
        assert sum(map(len, sets)) == len(set.union(*sets))
        selected = max(run['training']['history'], key=lambda row: row['auc'])
        assert selected['epoch'] == run['training']['best_epoch']
        assert selected['auc'] == run['training']['best_val_auc']
        assert cfg['settings']['batch_size'] == 512
        assert run['training']['min_visible_training_edges'] == len(split.train_pos) - 512 > 0
        for evaluation in run['evaluations']:
            a, r = evaluation['attack'], evaluation['ratio']
            name = 'clean' if a == 'clean' else a if a == 'feature_shuffle' else f'{a}_{r:g}'
            with np.load(file.parent / f'predictions_{name}.npz') as values:
                assert values['positive'].shape == (len(split.test_pos),)
                assert values['negative'].shape == (len(split.test_neg),)
                assert all(np.isfinite(values[key]).all() for key in ['positive', 'negative'])
    attacks = []
    for (ds, seed), split in splits.items():
        graph = graphs[ds]
        base = graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
        original = base.edge_set()
        blocked = set(edge_keys(np.vstack([split.test_pos, split.test_neg, split.val_neg]), graph.num_nodes))
        for file in sorted((args.output / 'attacks').glob(f'{ds}_s{seed}_*.npz')):
            with np.load(file) as values:
                assert str(values['base_sha256']) == digest(base.edges)
                edges = values['edges']
                stats = json.loads(str(values['statistics']))
            actual = graph_with_edges(base, edges).edge_set()
            assert not blocked & set(edge_keys(edges, graph.num_nodes))
            assert stats['graph_sha256'] == digest(edges)
            assert stats['deleted'] == len(original - actual)
            assert stats['inserted'] == len(actual - original)
            assert stats['effective_edits'] == len(original ^ actual) == stats['requested_budget']
            assert stats['requested_budget'] == round(stats['ratio'] * len(original))
            if stats['attack'] == 'dice':
                assert all(graph.y[u] >= 0 and graph.y[u] == graph.y[v] for u, v in original - actual)
                assert all(graph.y[u] >= 0 and graph.y[v] >= 0 and graph.y[u] != graph.y[v] for u, v in actual - original)
            attacks.append(dict(file=file.name, **stats))
    result = dict(passed=True, completed_runs=len(runs), datasets_and_seeds=len(splits),
                  attack_graphs=len(attacks), author_sources_verified=True,
                  source_and_data_fingerprints_verified=True, all_test_candidates_protected=True,
                  negatives_disjoint=True, budgets_and_dice_rules_verified=True,
                  best_checkpoint_uses_clean_validation_auc=True, prediction_counts_verified=True,
                  attacks=attacks)
    dump(args.output / 'protocol_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'attacks'}, indent=2))


if __name__ == '__main__':
    main()
