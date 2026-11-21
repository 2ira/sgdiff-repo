"""Supplement the author models with the pinned main branch's input/attack protocol.

Keeps the protected-target study immutable. Attacks are generated once by main's
actual dispatcher on the full graph, then test positives are removed, exactly as
in main's robustness harness. Models are retrained on main's standardized node
attributes, sharing the full split and the first 500 test pairs per class.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sgdiff.data.attack import apply_attack, supports_attack
from sgdiff.data.datasets import load_graph
from sgdiff.data.graph import GraphData
from sgdiff.data.splits import make_link_split
from sgdiff.engine.baseline_trainer import node_feature_matrix
from sgdiff.engine.robustness import _cap_pairs
from sgdiff.utils.metrics import summarize_scores

from .adapters import ADAPTERS, VARIANTS, settings, sparse_context
from .protocol import cache_lock, digest, metrics, save_npz
from .run import dump, graph_with_edges, predict, seed_all, source_hashes
from .repository_pubmed import PUBMED_ADAPTERS, pubmed_settings


MAIN_COMMIT = 'd4b150c121625001f52ead23abf15aead1a4b665'
PROTOCOL = 'repository-main-inputs-attacks-v1-standardized-retrained'
MAIN_FILES = ['sgdiff/data/attack.py', 'sgdiff/data/splits.py', 'sgdiff/data/datasets.py',
              'sgdiff/data/graph.py', 'sgdiff/engine/robustness.py',
              'sgdiff/engine/baseline_trainer.py', 'sgdiff/utils/metrics.py']


def plan(num_nodes):
    return [('clean', 0.)] + [(a, r) for a in ['rf', 'dice', 'ea'] if supports_attack(a, num_nodes)
                             for r in [.1, .25, .5]]


def tag(attack, ratio):
    return 'clean' if attack == 'clean' else f'{attack}_{ratio:g}'


def edgeset(edges):
    return set(map(tuple, edges.tolist()))


def pinned_sources():
    repo = Path(__file__).resolve().parents[2]
    hashes = {}
    for name in MAIN_FILES:
        actual = (repo / name).read_bytes()
        pinned = subprocess.check_output(['git', 'show', f'{MAIN_COMMIT}:{name}'], cwd=repo)
        if actual != pinned:
            raise RuntimeError(f'Pinned main source changed: {name}')
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def load_inputs(root, dataset, seed):
    with np.load(root / 'inputs' / f'{dataset}.npz', allow_pickle=False) as f:
        graph = GraphData(dataset, int(f['n']), f['edges'], x=f['features'], y=f['y'])
    with np.load(root / 'splits' / f'{dataset}_seed{seed}.npz', allow_pickle=False) as f:
        from sgdiff.data.splits import LinkSplit
        split = LinkSplit(**{key: f[key] for key in
                         ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']}, seed=seed)
        scored_pos, scored_neg = f['scored_pos'], f['scored_neg']
    return graph, split, scored_pos, scored_neg


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    main_hashes = pinned_sources()
    for folder in ['inputs', 'splits', 'attacks']:
        (root / folder).mkdir(exist_ok=True)
    datasets, attacks, skipped = {}, [], []
    for dataset in args.datasets:
        raw = load_graph(dataset, root=args.data_root, download=False, use_cache=True)
        features = node_feature_matrix(raw)
        save_npz(root / 'inputs' / f'{dataset}.npz', n=np.array(raw.num_nodes), edges=raw.edges,
                 raw_features=raw.x, features=features, y=raw.y)
        datasets[dataset] = dict(nodes=raw.num_nodes, edges=raw.num_edges, feature_dim=features.shape[1],
                                 graph_sha256=digest(raw.edges), raw_features_sha256=digest(raw.x),
                                 features_sha256=digest(features), labels_sha256=digest(raw.y))
        for seed in args.seeds:
            split = make_link_split(raw, seed=seed)
            pos, neg = _cap_pairs(split, 500)
            arrays = {k: getattr(split, k) for k in
                      ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']}
            save_npz(root / 'splits' / f'{dataset}_seed{seed}.npz', **arrays, scored_pos=pos, scored_neg=neg)
            base = raw.drop_edges(split.test_pos.tolist())
            for attack, ratio in plan(raw.num_nodes):
                path = root / 'attacks' / f'{dataset}_s{seed}_{tag(attack, ratio)}.npz'
                with cache_lock(path):
                    if path.exists():
                        with np.load(path, allow_pickle=False) as f:
                            record = json.loads(str(f['statistics']))
                            if record['source_graph_sha256'] != digest(raw.edges):
                                raise RuntimeError('Cached attack source graph differs')
                    else:
                        # Do not repair main's budgets, protect candidates, or
                        # substitute a second EA implementation here.
                        attacked = raw if attack == 'clean' else apply_attack(
                            raw, attack, ratio=ratio, seed=seed, device=args.device)
                        visible = attacked.drop_edges(split.test_pos.tolist())
                        before, after, actual = edgeset(raw.edges), edgeset(attacked.edges), edgeset(visible.edges)
                        clean = edgeset(base.edges)
                        record = dict(dataset=dataset, seed=seed, attack=attack, ratio=ratio,
                                      source_graph_sha256=digest(raw.edges), graph_sha256=digest(visible.edges),
                                      raw_graph_sha256=digest(attacked.edges), source_edges=raw.num_edges,
                                      requested_budget=round(ratio * raw.num_edges),
                                      generator_metadata=attacked.meta.get('attack', {}),
                                      raw_deleted=len(before - after), raw_inserted=len(after - before),
                                      raw_effective_edits=len(before ^ after), clean_visible_edges=base.num_edges,
                                      visible_deleted=len(clean - actual), visible_inserted=len(actual - clean),
                                      visible_effective_edits=len(clean ^ actual),
                                      inserted_scored_negatives=len(actual & edgeset(neg)),
                                      protected_test_positives=not bool(actual & edgeset(split.test_pos)))
                        save_npz(path, edges=visible.edges, raw_edges=attacked.edges,
                                 statistics=json.dumps(record))
                attacks.append(dict(file=str(path.relative_to(root)), **record))
                print(json.dumps(dict(stage='cache', dataset=dataset, seed=seed,
                                      attack=tag(attack, ratio))), flush=True)
            if not supports_attack('ea', raw.num_nodes):
                skipped.append(dict(dataset=dataset, seed=seed, attack='ea', reason='main EA_MAX_NODES=4000'))
    dump(root / 'input_manifest.json', dict(protocol=PROTOCOL, main_commit=MAIN_COMMIT,
         main_source_sha256=main_hashes, datasets=datasets, attacks=attacks, skipped=skipped,
         graph_provenance='newly generated by pinned main implementation; not historical partner snapshots',
         scored_pairs='main _cap_pairs: first 500 positives and first 500 negatives',
         feature_preprocessing='main node_feature_matrix: per-feature standardization on fixed attributes',
         caveat='Author architectures and optimization recipes remain model-specific; not an exact paper-table reproduction'))


def fit_main(model, graph, split, cfg, args, run_dir):
    """Author minibatch training; main node-pair validation context/cadence.

    Main's train_node_pair_baseline validates on train+val edges. This is
    reproduced explicitly, even though the protected-target study validates on
    train edges alone. Author optimizer settings and target masking are retained.
    """
    train_graph = graph_with_edges(graph, split.train_pos)
    valid_graph = graph.drop_edges(split.test_pos.tolist())
    train_context, valid_context = model.context(train_graph), model.context(valid_graph)
    optimizer = torch.optim.AdamW if args.model == 'refined_gae' and args.dataset == 'cora' else torch.optim.Adam
    opt = optimizer(model.parameter_groups(cfg), weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, cfg['decay'])
    valid_pairs = np.vstack([split.val_pos, split.val_neg])
    history, best, best_epoch, best_state = [], -float('inf'), 0, None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order, losses = np.random.permutation(len(split.train_pos)), []
        for begin in range(0, len(order), cfg['batch_size']):
            selected = order[begin:begin + cfg['batch_size']]
            keep = np.ones(len(split.train_pos), dtype=bool)
            keep[selected] = False
            visible = graph_with_edges(graph, split.train_pos[keep])
            context = sparse_context(visible, model.device) if args.model == 'lpformer' else model.context(visible)
            pairs = torch.tensor(np.vstack([split.train_pos[selected], split.train_neg[selected]]), device=model.device)
            labels = torch.cat([torch.ones(len(selected), device=model.device), torch.zeros(len(selected), device=model.device)])
            opt.zero_grad(set_to_none=True)
            scores = model.score(train_context, pairs, propagation=context) if args.model == 'lpformer' else model.score(context, pairs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            seed_all(args.seed + 10000)
            scores, _ = predict(model, valid_graph, valid_pairs, args.eval_batch_size, context=valid_context)
            measured = summarize_scores(np.r_[np.ones(len(split.val_pos)), np.zeros(len(split.val_neg))], scores)
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **measured))
            if measured['auc'] > best:
                best, best_epoch = measured['auc'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(dict(stage='train', dataset=args.dataset, model=args.model, component=args.variant,
                                  seed=args.seed, epoch=epoch, val_auc=measured['auc'])), flush=True)
            seed_all(args.seed * 100003 + epoch + 1)
            if epoch - best_epoch >= 30:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / 'model.pt')
    result = dict(history=history, best_epoch=best_epoch, stopped_epoch=epoch, best_val_auc=best,
                  validation_context='train+val, matching main train_node_pair_baseline',
                  validation_every_epochs=10, patience_epochs=30,
                  seconds=time.monotonic() - started, positive_minibatch_size=cfg['batch_size'])
    dump(run_dir / 'training.json', result)
    return result


def run(args):
    root = args.output
    graph, split, pos, neg = load_inputs(root, args.dataset, args.seed)
    if args.variant not in VARIANTS[args.model]:
        raise ValueError('Invalid component variant')
    seed_all(args.seed)
    torch.set_num_threads(4)
    args.eval_every, args.eval_batch_size, args.patience = 5, 512, 30
    cfg = pubmed_settings(args.model) if args.dataset == 'pubmed' else settings(args.model, args.dataset)
    run_dir = root / 'runs' / f'{args.dataset}_{args.model}_{args.variant}_s{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'input_manifest.json').read_text())
    config = dict(protocol=PROTOCOL, main_commit=MAIN_COMMIT, dataset=args.dataset, model=args.model,
                  component=args.variant, seed=args.seed, epochs=args.epochs, settings=cfg,
                  main_source_sha256=manifest['main_source_sha256'], author_runtime_sha256=source_hashes(),
                  supplement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pubmed_adapter_sha256=hashlib.sha256(Path(__file__).with_name('repository_pubmed.py').read_bytes()).hexdigest(),
                  features_sha256=digest(graph.x), split_sha256={name: digest(getattr(split, name)) for name in
                  ['train_pos', 'val_pos', 'test_pos', 'train_neg', 'val_neg', 'test_neg']},
                  scored_pos_sha256=digest(pos), scored_neg_sha256=digest(neg),
                  feature_preprocessing=manifest['feature_preprocessing'], device=args.device,
                  training='Fresh author-model retrain; shared positive minibatch512 target masking; main node-pair train+val validation context, every10 epochs; author max100/optimizer',
                  graph_provenance=manifest['graph_provenance'])
    if (run_dir / 'config.json').exists() and json.loads((run_dir / 'config.json').read_text()) != config:
        raise RuntimeError('Supplement configuration changed; use a new output directory')
    dump(run_dir / 'config.json', config)
    if (run_dir / 'complete.json').exists():
        print('Already complete:', run_dir, flush=True)
        return
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.reset_peak_memory_stats()
    adapter = PUBMED_ADAPTERS[args.model] if args.dataset == 'pubmed' else ADAPTERS[args.model]
    model = adapter(args.source_root, args.dataset, graph.x.shape[1], args.variant,
                               args.device, root / 'ppr_cache')
    if (run_dir / 'training.json').exists():
        model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=args.device, weights_only=True))
        training = json.loads((run_dir / 'training.json').read_text())
    else:
        training = fit_main(model, graph, split, cfg, args, run_dir)
    evaluations = []
    pairs = np.vstack([pos, neg])
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    for attack, ratio in plan(graph.num_nodes):
        with np.load(root / 'attacks' / f'{args.dataset}_s{args.seed}_{tag(attack, ratio)}.npz',
                     allow_pickle=False) as f:
            attacked = graph_with_edges(graph, f['edges'])
            statistics = json.loads(str(f['statistics']))
        seed_all(args.seed + 20000)
        scores, _ = predict(model, attacked, pairs, args.eval_batch_size)
        measured = summarize_scores(labels, scores)
        additional = metrics(scores[:len(pos)], scores[len(pos):])
        measured.update(strict_hits100=additional['hits100'], precision100_tie_average=additional['precision100'])
        np.savez_compressed(run_dir / f'predictions_{tag(attack, ratio)}.npz',
                            positive=scores[:len(pos)], negative=scores[len(pos):])
        row = dict(dataset=args.dataset, model=args.model, component=args.variant, seed=args.seed,
                   attack=attack, ratio=ratio, metrics=measured, statistics=statistics)
        evaluations.append(row)
        print(json.dumps(dict(stage='evaluate', dataset=args.dataset, model=args.model, component=args.variant,
                              seed=args.seed, attack=tag(attack, ratio), auc=measured['auc'])), flush=True)
    for row in evaluations:
        row['drop'] = {name: evaluations[0]['metrics'][name] - value for name, value in row['metrics'].items()}
    dump(run_dir / 'complete.json', dict(config=config, training=training, evaluations=evaluations,
         peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0))


def suite(args):
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    pending = []
    for index in range(1 if args.full_only else 3):
        for seed in args.seeds:
            for dataset in args.datasets:
                for model in VARIANTS:
                    variant = VARIANTS[model][index]
                    name = f'{dataset}_{model}_{variant}_s{seed}'
                    cmd = [sys.executable, '-m', 'experiments.new_link_models.repository_protocol', 'run',
                           '--output', str(args.output), '--source-root', str(args.source_root),
                           '--model', model, '--variant', variant, '--dataset', dataset,
                           '--seed', str(seed), '--device', args.device, '--epochs', str(args.epochs)]
                    pending.append((name, cmd))
    active = []
    while pending or active:
        while pending and len(active) < args.jobs:
            name, cmd = pending.pop(0)
            handle = (logs / (name + '.log')).open('a')
            process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            active.append((name, process, handle))
            print(json.dumps(dict(stage='started', run=name, pid=process.pid)), flush=True)
        for name, process, handle in list(active):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            active.remove((name, process, handle))
            if code:
                for _, other, stream in active:
                    other.terminate()
                    other.wait()
                    stream.close()
                raise RuntimeError(f'{name} failed with {code}; inspect its log')
            print(json.dumps(dict(stage='completed', run=name)), flush=True)
        if active:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['prepare', 'run', 'suite'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--model', choices=ADAPTERS)
    parser.add_argument('--variant', default='full')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--full-only', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    {'prepare': prepare, 'run': run, 'suite': suite}[args.stage](args)


if __name__ == '__main__':
    main()
