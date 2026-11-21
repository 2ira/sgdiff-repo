"""Independently reload supplemental checkpoints; optionally watch training progress."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sgdiff.utils.metrics import summarize_scores
from .adapters import ADAPTERS
from .protocol import digest
from .repository_protocol import load_inputs
from .repository_pubmed import PUBMED_ADAPTERS
from .run import dump, graph_with_edges, predict, seed_all


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_checkpoint(file, args):
    completed = json.loads(file.read_text())
    cfg = completed['config']
    ds, model, seed = cfg['dataset'], cfg['model'], cfg['seed']
    conditions = ['clean', 'dice_0.5' if ds == 'pubmed' else 'ea_0.5']
    provenance = dict(checkpoint_sha256=sha(file.parent / 'model.pt'),
                      completion_sha256=sha(file), implementation_sha256=sha(Path(__file__)),
                      predictions_sha256={name: sha(file.parent / f'predictions_{name}.npz') for name in conditions},
                      graphs_sha256={name: sha(args.output / 'attacks' / f'{ds}_s{seed}_{name}.npz') for name in conditions},
                      device=args.device)
    cached = args.output / 'replay_records' / f'{file.parent.name}.json'
    if cached.exists():
        previous = json.loads(cached.read_text())
        if previous['provenance'] == provenance:
            return previous['records']
    graph, split, pos, neg = load_inputs(args.output, ds, seed)
    adapter = PUBMED_ADAPTERS[model] if ds == 'pubmed' else ADAPTERS[model]
    network = adapter(args.source_root, ds, graph.x.shape[1], cfg['component'], args.device,
                      args.output / 'ppr_cache')
    network.load_state_dict(torch.load(file.parent / 'model.pt', map_location=args.device, weights_only=True))
    records = []
    for attack in conditions:
        with np.load(args.output / 'attacks' / f'{ds}_s{seed}_{attack}.npz', allow_pickle=False) as f:
            attacked = graph_with_edges(graph, f['edges'])
            graph_sha = digest(f['edges'])
        seed_all(seed + 20000)
        scores, context = predict(network, attacked, np.vstack([pos, neg]), 512)
        with np.load(file.parent / f'predictions_{attack}.npz', allow_pickle=False) as f:
            saved = np.r_[f['positive'], f['negative']]
        if not np.allclose(saved, scores, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f'Checkpoint scores changed: {file.parent.name}:{attack}')
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        actual, expected = summarize_scores(labels, scores), summarize_scores(labels, saved)
        score_error = float(np.max(np.abs(scores - saved)))
        metric_error = max(abs(actual[k] - expected[k]) for k in ['auc', 'ap'])
        hit_error = abs(actual['hit@100'] - expected['hit@100'])
        if metric_error > 2e-5:
            raise RuntimeError(f'Checkpoint ranking changed: {file.parent.name}:{attack}')
        records.append(dict(run=file.parent.name, attack=attack, graph_sha256=graph_sha,
                            max_absolute_score_error=score_error, max_auc_ap_error=metric_error,
                            legacy_precision100_error=hit_error, replay_metrics=actual, saved_metrics=expected))
    del network, attacked, context
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    dump(cached, dict(provenance=provenance, records=records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    (args.output / 'replay_records').mkdir(exist_ok=True)
    records, done = [], set()
    while True:
        files = sorted((args.output / 'runs').glob('*/complete.json'))
        if not args.watch and len(files) != 81:
            raise RuntimeError(f'Expected 81 checkpoints, found {len(files)}')
        for file in files:
            if file.parent.name in done:
                continue
            records.extend(replay_checkpoint(file, args))
            done.add(file.parent.name)
            dump(args.output / 'replay_progress.json', dict(checkpoints=len(done), evaluations=len(records)))
            print(json.dumps(dict(replayed=file.parent.name, completed=len(done))), flush=True)
        if len(done) == 81:
            break
        if len(done) > 81:
            raise RuntimeError('Unexpected extra checkpoints')
        time.sleep(10)
    dump(args.output / 'replay_audit.json', dict(passed=True, checkpoints=81, evaluations=162,
         score_atol=2e-5, score_rtol=2e-5,
         max_absolute_score_error=max(r['max_absolute_score_error'] for r in records),
         max_auc_ap_error=max(r['max_auc_ap_error'] for r in records),
         max_legacy_precision100_error=max(r['legacy_precision100_error'] for r in records),
         records=records))


if __name__ == '__main__':
    main()
