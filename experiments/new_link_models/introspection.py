"""Inspect learned depth/branch weights and real LPFormer attention distributions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import LPFormer
from .protocol import load_data, split_data
from .run import graph_with_edges, seed_all


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    parameters, attention = [], []
    for file in sorted((args.output / 'runs').glob('*/complete.json')):
        cfg = json.loads(file.read_text())['config']
        identity = {k: cfg[k] for k in ['dataset', 'model', 'component', 'seed']}
        state = torch.load(file.parent / 'model.pt', map_location='cpu', weights_only=True)
        row = identity | dict(cn_beta=None, ncn_jk_scale=None, depth1=None, depth2=None, depth3=None, depth4=None)
        if cfg['model'] == 'ncnc':
            row['cn_beta'] = float(state['predictor.beta'].item())
            row['ncn_jk_scale'] = float(state['encoder.jkparams'].item())
        elif cfg['model'] == 'refined_gae':
            weights = state['encoder.alphas'].softmax(0).tolist()
            row.update({f'depth{i+1}': value for i, value in enumerate(weights)})
        parameters.append(row)
        if cfg['model'] != 'lpformer':
            continue
        graph = load_data(args.data_root, cfg['dataset'])
        split, sha = split_data(graph, cfg['seed'], args.output)
        assert sha == cfg['split_sha256']
        base = graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
        n = min(64, len(split.test_pos), len(split.test_neg))
        pairs = np.vstack([split.test_pos[:n], split.test_neg[:n]])
        seed_all(cfg['seed'])
        model = LPFormer(args.source_root, cfg['dataset'], graph.x.shape[1], cfg['component'], args.device, args.output / 'ppr_cache')
        model.load_state_dict(state)
        model.eval()
        ctx = model.context(base)
        p_tensor = torch.tensor(pairs, device=args.device)
        observed_pe = []
        original_pe = model.core.get_pos_encodings

        def record_pe(*a, **kw):
            output = original_pe(*a, **kw)
            observed_pe.append(float(output.abs().max()) if output.numel() else 0.)
            return output

        model.core.get_pos_encodings = record_pe
        with torch.no_grad():
            model.score(ctx, p_tensor)  # Populate author context through the adapter.
            _, weights = model.core(p_tensor.T, return_weights=True)
            indices, alpha = weights[0].long(), weights[1]
            count = torch.bincount(indices, minlength=len(pairs))
            mass = torch.zeros(len(pairs), device=args.device).scatter_add_(0, indices, alpha)
            entropy = torch.zeros_like(mass).scatter_add_(0, indices, -alpha * alpha.clamp_min(1e-30).log())
            torch.testing.assert_close(mass[count > 0], torch.ones_like(mass[count > 0]), atol=2e-5, rtol=2e-5)
            normalized = entropy / count.float().clamp_min(2).log()
            if cfg['component'] == 'uniform_attention':
                torch.testing.assert_close(normalized[count > 1], torch.ones_like(normalized[count > 1]), atol=2e-5, rtol=2e-5)
            if cfg['component'] == 'no_ppr_encoding':
                assert max(observed_pe) == 0.
        for i, (u, v) in enumerate(pairs):
            attention.append(identity | dict(candidate_label='positive' if i < n else 'negative',
                                             node_u=int(u), node_v=int(v), selected_nodes=int(count[i]),
                                             entropy=float(entropy[i]), normalized_entropy=float(normalized[i]),
                                             max_abs_pe=max(observed_pe)))
        print(json.dumps(identity | dict(attention_pairs=len(pairs), mean_selected_nodes=float(count.float().mean()))), flush=True)
        del model, ctx
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    write_csv(args.output / 'learned_parameters.csv', parameters)
    write_csv(args.output / 'attention_per_pair.csv', attention)
    (args.output / 'introspection.md').write_text(
        '# Learned model components\n\n'
        '`learned_parameters.csv` contains registered NCNC beta/JK scalars and '
        'Refined-GAE softmax weights for propagation depths 1–4. CN0 registers '
        'a beta parameter that its forward path does not use. Scalar magnitudes '
        'are not causal importance because learned representations can rescale.\n\n'
        '`attention_per_pair.csv` evaluates the first 64 positive and 64 negative '
        'test candidates from each seeded split on the clean inference graph. '
        'Entropy is normalized by log(candidate count); counts 0/1 have entropy 0. '
        'The uniform-attention variant is independently checked for normalized '
        'entropy 1 when count > 1, and the no-PE variant for zero encoded PE. '
        'Attention describes aggregation behavior, not causal feature attribution.\n')
    print(json.dumps(dict(parameter_runs=len(parameters), attention_pairs=len(attention), passed=True)))


if __name__ == '__main__':
    main()
