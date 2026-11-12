"""Reuse sgdiff's GCN parameters and pair decoder with the archived protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from sgdiff.models.baselines.node_pair import NodePairLinkPredictor
from . import experiment as reference

ReferenceLinkModel = reference.LinkModel
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SharedLinkModel(nn.Module):
    def __init__(self, input_dim, hidden, embedding, dropout, propagate):
        super().__init__()
        # Preserve the reference initialization and subsequent dropout RNG stream.
        initial = ReferenceLinkModel(input_dim, hidden, embedding, dropout, propagate)
        with torch.random.fork_rng(devices=[]):
            self.predictor = NodePairLinkPredictor(
                input_dim, hidden, embedding, num_layers=2, arch='gcn', dropout=dropout)
        with torch.no_grad():
            first, second = self.predictor.encoder.lins
            first.weight.copy_(initial.weight.T)
            first.bias.copy_(initial.bias)
            second.weight.copy_(initial.output.weight)
            second.bias.copy_(initial.output.bias)
        self.propagate = propagate

    def forward(self, x, a):
        return self.predictor.encoder.forward_preaggregated(x, a if self.propagate else None)

    def score(self, z, edges):
        return self.predictor.score_pairs(z, edges)


def backend_record():
    paths = ['experiments/gcn_ablation/experiment.py',
             'experiments/gcn_ablation/shared_backend.py',
             'sgdiff/models/baselines/node_pair.py', 'sgdiff/models/layers.py']
    return dict(backend='sgdiff-gcn-preaggregated-v1',
                sha256={p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
                        for p in paths})


def configure(argv, replay=False):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=reference.ROOT / 'shared_results')
    args, _ = parser.parse_known_args(argv)
    argv = list(argv)
    if not any(a == '--output' or a.startswith('--output=') for a in argv):
        argv += ['--output', str(args.output)]
    if '-h' in argv or '--help' in argv:
        return argv
    marker = args.output / 'backend.json'
    record = backend_record()
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise RuntimeError('Backend changed; choose a new --output directory')
    elif replay or any(args.output.glob('**/model.pt')) or (args.output / 'config.json').exists():
        raise RuntimeError('Missing shared backend provenance; choose a new --output directory')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(record, indent=2) + '\n')
    return argv
