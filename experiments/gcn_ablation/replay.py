"""Reload an existing checkpoint and verify every saved target prediction, without training."""
import argparse
import json

import numpy as np
import torch

from experiment import ROOT, NAMES, MODES, LinkModel, cached_input, evaluate, load_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=type(ROOT), default=ROOT / 'results')
    p.add_argument('--protocol', choices=['within', 'transfer'], required=True)
    p.add_argument('--source', choices=NAMES, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads((args.output / 'config.json').read_text())
    run = args.output / 'runs' / f'{args.protocol}_{args.source}_{args.mode}_seed{args.seed}'
    records = json.loads((run / 'complete.json').read_text())
    source = load_graph(args.source, args.seed, args.output)
    x = cached_input(source, args.mode, args.protocol)
    model = LinkModel(x.shape[1], cfg['hidden'], cfg['embedding'], cfg['dropout'], args.mode != 'feature')
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
    for record in records:
        target = source if record['target'] == source.name else load_graph(record['target'], args.seed, args.output)
        x = cached_input(target, args.mode, args.protocol)
        scores, pos, neg = evaluate(model, target, x, 'test')
        saved = np.load(run / f'predictions_{target.name}.npz')
        np.testing.assert_allclose(pos, saved['positive'], rtol=0, atol=1e-7)
        np.testing.assert_allclose(neg, saved['negative'], rtol=0, atol=1e-7)
        print(json.dumps(dict(target=target.name, checkpoint_replay='passed', **scores)))


if __name__ == '__main__':
    main()
