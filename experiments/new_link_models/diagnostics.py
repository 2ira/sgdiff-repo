"""Label-stratified sensitivity diagnostics from saved predictions, no retraining."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .protocol import load_data, metrics, split_data
from .run import graph_with_edges


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    args = p.parse_args()
    strata, correlations = [], []
    for complete in sorted((args.output / 'runs').glob('*/complete.json')):
        run = json.loads(complete.read_text())
        cfg = run['config']
        if cfg['component'] != 'full':
            continue
        graph = load_data(args.data_root, cfg['dataset'])
        split, sha = split_data(graph, cfg['seed'], args.output)
        assert sha == cfg['split_sha256']
        base = graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
        pairs = np.vstack([split.test_pos, split.test_neg])
        labels = np.r_[np.ones(len(split.test_pos), bool), np.zeros(len(split.test_neg), bool)]
        adj, degree = base.adjacency(), base.degrees()
        cn = np.asarray(adj[pairs[:, 0]].multiply(adj[pairs[:, 1]]).sum(1)).ravel()
        min_degree = np.minimum(degree[pairs[:, 0]], degree[pairs[:, 1]])
        norm = np.linalg.norm(base.x, axis=1)
        cosine = (base.x[pairs[:, 0]] * base.x[pairs[:, 1]]).sum(1) / np.maximum(norm[pairs[:, 0]] * norm[pairs[:, 1]], 1e-12)
        with np.load(complete.parent / 'predictions_clean.npz') as f:
            clean = np.r_[f['positive'], f['negative']]
        identity = {k: cfg[k] for k in ['dataset', 'model', 'seed']}
        for label, selected in [('positive', labels), ('negative', ~labels)]:
            for name, values in [('common_neighbors', cn), ('min_degree', min_degree), ('feature_cosine', cosine)]:
                rho = None if np.ptp(values[selected]) == 0 or np.ptp(clean[selected]) == 0 else float(spearmanr(values[selected], clean[selected]).statistic)
                correlations.append(identity | dict(candidate_label=label, attribute=name, n=int(selected.sum()), spearman_rho=rho))
        masks = {'CN=0': cn == 0, 'CN=1': cn == 1, 'CN>=2': cn >= 2,
                 'min-degree<=2': min_degree <= 2, 'min-degree>=3': min_degree >= 3,
                 'cosine=0': cosine == 0, 'cosine>0': cosine > 0}
        for evaluation in run['evaluations']:
            attack, ratio = evaluation['attack'], evaluation['ratio']
            name = 'clean' if attack == 'clean' else attack if attack == 'feature_shuffle' else f'{attack}_{ratio:g}'
            with np.load(complete.parent / f'predictions_{name}.npz') as f:
                scores = np.r_[f['positive'], f['negative']]
            for stratum, selected in masks.items():
                pos, neg = selected & labels, selected & ~labels
                # Keep sample counts visible; don't invent AUC for a one-class stratum.
                if not pos.any() or not neg.any():
                    continue
                clean_values = metrics(clean[pos], clean[neg])
                values = metrics(scores[pos], scores[neg])
                strata.append(identity | dict(stratum=stratum, attack=attack, ratio=ratio,
                                              n_positive=int(pos.sum()), n_negative=int(neg.sum()),
                                              auc=values['auc'], ap=values['ap'],
                                              drop_auc=clean_values['auc'] - values['auc']))
    write_csv(args.output / 'strata_per_seed.csv', strata)
    write_csv(args.output / 'score_correlations.csv', correlations)
    groups = defaultdict(list)
    for row in strata:
        groups[tuple(row[k] for k in ['dataset', 'model', 'stratum', 'attack', 'ratio'])].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        entry = dict(zip(['dataset', 'model', 'stratum', 'attack', 'ratio'], key), n_seeds=len(rows))
        for metric in ['auc', 'ap', 'drop_auc', 'n_positive', 'n_negative']:
            values = [r[metric] for r in rows]
            entry[metric + '_mean'] = float(np.mean(values))
            entry[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        summary.append(entry)
    write_csv(args.output / 'strata_summary.csv', summary)
    (args.output / 'diagnostics.md').write_text(
        '# Mechanism diagnostics\n\n'
        'Strata use common-neighbor count, minimum endpoint degree and feature '
        'cosine from the clean train+validation graph. Each attack is scored on '
        'the same candidate membership. Per-stratum AUC/AP require both classes; '
        'sample counts are reported. AP depends on stratum prevalence.\n\n'
        '`score_correlations.csv` reports clean score Spearman correlations '
        'separately for positive and negative candidates, reducing label-mixture '
        'confounding. A missing rho indicates a constant attribute/score. '
        'These are descriptive associations, not causal feature attributions. '
        'Read them alongside retrained component deltas and frozen probes.\n')
    print(json.dumps(dict(stratum_records=len(strata), correlations=len(correlations))))


if __name__ == '__main__':
    main()
