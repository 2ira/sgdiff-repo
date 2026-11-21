"""Independently reload each supplemental checkpoint and replay two conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sgdiff.utils.metrics import summarize_scores
from .adapters import ADAPTERS
from .protocol import digest
from .repository_protocol import load_inputs
from .repository_pubmed import PUBMED_ADAPTERS
from .run import dump, graph_with_edges, predict, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    files = sorted((args.output / 'runs').glob('*/complete.json'))
    if len(files) != 81:
        raise RuntimeError(f'Expected 81 checkpoints, found {len(files)}')
    records, max_error, max_metric_error = [], 0., 0.
    for file in files:
        completed = json.loads(file.read_text())
        cfg = completed['config']
        ds, model, seed = cfg['dataset'], cfg['model'], cfg['seed']
        graph, split, pos, neg = load_inputs(args.output, ds, seed)
        adapter = PUBMED_ADAPTERS[model] if ds == 'pubmed' else ADAPTERS[model]
        network = adapter(args.source_root, ds, graph.x.shape[1], cfg['component'], args.device,
                          args.output / 'ppr_cache')
        network.load_state_dict(torch.load(file.parent / 'model.pt', map_location=args.device, weights_only=True))
        for attack in ['clean', 'dice_0.5' if ds == 'pubmed' else 'ea_0.5']:
            with np.load(args.output / 'attacks' / f'{ds}_s{seed}_{attack}.npz', allow_pickle=False) as f:
                attacked = graph_with_edges(graph, f['edges'])
                sha = digest(f['edges'])
            seed_all(seed + 20000)
            scores, _ = predict(network, attacked, np.vstack([pos, neg]), 512)
            with np.load(file.parent / f'predictions_{attack}.npz', allow_pickle=False) as f:
                saved = np.r_[f['positive'], f['negative']]
            if not np.allclose(saved, scores, atol=2e-5, rtol=2e-5):
                raise RuntimeError(f'Checkpoint scores changed: {file.parent.name}:{attack}')
            labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            actual, expected = summarize_scores(labels, scores), summarize_scores(labels, saved)
            score_error = float(np.max(np.abs(scores - saved)))
            metric_error = max(abs(actual[k] - expected[k]) for k in ['auc', 'ap'])
            if metric_error > 2e-5:
                raise RuntimeError(f'Checkpoint ranking changed: {file.parent.name}:{attack}')
            records.append(dict(run=file.parent.name, attack=attack, graph_sha256=sha,
                                max_absolute_score_error=score_error, max_auc_ap_error=metric_error))
            max_error, max_metric_error = max(max_error, score_error), max(max_metric_error, metric_error)
        del network
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print(json.dumps(dict(replayed=file.parent.name, completed=len(records) // 2)), flush=True)
    dump(args.output / 'replay_audit.json', dict(passed=True, checkpoints=81, evaluations=162,
         score_atol=2e-5, score_rtol=2e-5, max_absolute_score_error=max_error,
         max_auc_ap_error=max_metric_error, records=records))


if __name__ == '__main__':
    main()
