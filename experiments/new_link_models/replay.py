"""Independently reload checkpoints and recompute selected stored predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import ADAPTERS
from .protocol import digest, load_data, metrics, split_data
from .run import dump, graph_with_edges, predict, seed_all, source_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--attacks', nargs='+', default=['clean', 'dice_0.5', 'feature_shuffle'])
    p.add_argument('--full-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    checked = []
    for complete in sorted((args.output / 'runs').glob('*/complete.json')):
        cfg = json.loads(complete.read_text())['config']
        if args.full_only and cfg['component'] != 'full':
            continue
        if cfg['code_sha256'] != source_hashes():
            raise RuntimeError('Runtime code differs from the completed run')
        graph = load_data(args.data_root, cfg['dataset'])
        split, sha = split_data(graph, cfg['seed'], args.output)
        assert sha == cfg['split_sha256']
        assert digest(graph.x) == cfg['features_sha256']
        pairs = np.vstack([split.test_pos, split.test_neg])
        base = graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
        seed_all(cfg['seed'])
        model = ADAPTERS[cfg['model']](args.source_root, cfg['dataset'], graph.x.shape[1], cfg['component'], args.device, args.output / 'ppr_cache')
        model.load_state_dict(torch.load(complete.parent / 'model.pt', map_location=args.device, weights_only=True))
        for name in args.attacks:
            if name == 'feature_shuffle':
                rng = np.random.default_rng(cfg['seed'] + 7183)
                attacked = graph_with_edges(base, base.edges)
                attacked.x = base.x[rng.permutation(base.num_nodes)]
            else:
                with np.load(args.output / 'attacks' / f"{cfg['dataset']}_s{cfg['seed']}_{name}.npz") as f:
                    attacked = graph_with_edges(base, f['edges'])
            seed_all(cfg['seed'] + 20000)
            scores, _ = predict(model, attacked, pairs, cfg['eval_batch_size'])
            with np.load(complete.parent / f'predictions_{name}.npz') as f:
                expected = np.r_[f['positive'], f['negative']]
            np.testing.assert_allclose(scores, expected, atol=2e-5, rtol=2e-5)
            values = metrics(scores[:len(split.test_pos)], scores[len(split.test_pos):])
            record = dict(run=complete.parent.name, attack=name, max_abs_error=float(np.max(np.abs(scores - expected))), metrics=values)
            checked.append(record)
            print(json.dumps(record), flush=True)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    dump(args.output / 'replay_audit.json', dict(checkpoints=len({r['run'] for r in checked}), evaluations=len(checked),
                                               atol=2e-5, rtol=2e-5, records=checked, passed=True))


if __name__ == '__main__':
    main()
