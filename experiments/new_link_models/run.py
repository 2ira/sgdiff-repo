"""Train one author-model configuration and evaluate protected structural evasion."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .adapters import ADAPTERS, VARIANTS, settings, sparse_context
from .protocol import PROTOCOL_VERSION, digest, load_data, metrics, perturb, split_data, cache_lock, save_npz
from sgdiff.data.graph import GraphData


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes():
    root = Path(__file__).parent
    runtime = ['__init__.py', 'adapters.py', 'ppr_backend.py', 'protocol.py', 'run.py', 'sources.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in runtime} | {
        'sources.lock.json': hashlib.sha256((root / 'sources.lock.json').read_bytes()).hexdigest()}


def graph_with_edges(graph, edges):
    return GraphData(graph.name, graph.num_nodes, edges, x=graph.x, y=graph.y)


def predict(model, graph, pairs, batch_size, context=None):
    model.eval()
    ctx = model.context(graph) if context is None else context
    output = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            p = torch.tensor(pairs[start:start + batch_size], device=model.device)
            output.append(model.score(ctx, p).detach().cpu().numpy())
    return np.concatenate(output), ctx


def fit(model, graph, split, cfg, args, run):
    if cfg['batch_size'] >= len(split.train_pos):
        raise ValueError('Target masking would remove the complete training graph; reduce the shared batch size')
    train_graph = graph_with_edges(graph, split.train_pos)
    context = model.context(train_graph)
    opt = torch.optim.AdamW(model.parameter_groups(cfg), weight_decay=cfg['weight_decay']) if args.model == 'refined_gae' and args.dataset == 'cora' else torch.optim.Adam(model.parameter_groups(cfg), weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, cfg['decay'])
    history, best, best_epoch = [], -float('inf'), 0
    best_state = None
    start = time.monotonic()
    valid_pairs = np.vstack([split.val_pos, split.val_neg])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(split.train_pos))
        losses = []
        for begin in range(0, len(order), cfg['batch_size']):
            indices = order[begin:begin + cfg['batch_size']]
            keep = np.ones(len(split.train_pos), dtype=bool)
            keep[indices] = False
            visible = graph_with_edges(graph, split.train_pos[keep])
            # The author LPFormer uses cached train-graph PPR but masks targets
            # from propagation and local pairwise neighborhoods each minibatch.
            ctx = sparse_context(visible, model.device) if args.model == 'lpformer' else model.context(visible)
            pairs = np.vstack([split.train_pos[indices], split.train_neg[indices]])
            p = torch.tensor(pairs, device=model.device)
            labels = torch.cat([torch.ones(len(indices), device=model.device), torch.zeros(len(indices), device=model.device)])
            opt.zero_grad(set_to_none=True)
            scores = model.score(context, p, propagation=ctx) if args.model == 'lpformer' else model.score(ctx, p)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            seed_all(args.seed + 10000)
            scores, _ = predict(model, train_graph, valid_pairs, args.eval_batch_size, context=context)
            values = metrics(scores[:len(split.val_pos)], scores[len(split.val_pos):])
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **values))
            if values['auc'] > best:
                best, best_epoch = values['auc'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(dict(stage='train', dataset=args.dataset, model=args.model, component=args.variant,
                                  seed=args.seed, epoch=epoch, loss=round(float(np.mean(losses)), 5), val_auc=round(values['auc'], 5))), flush=True)
            # Independent training RNG at each epoch: validation sampling cannot
            # reset the next epoch's shuffle or dropout stream to a fixed state.
            seed_all(args.seed * 100003 + epoch + 1)
            if epoch - best_epoch >= args.patience:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, run / 'model.pt')
    training = dict(history=history, best_epoch=best_epoch, stopped_epoch=epoch,
                    best_val_auc=best, seconds=time.monotonic() - start,
                    positive_minibatch_size=cfg['batch_size'],
                    min_visible_training_edges=len(split.train_pos) - cfg['batch_size'],
                    max_visible_training_edges=len(split.train_pos) - (len(split.train_pos) % cfg['batch_size'] or cfg['batch_size']),
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    dump(run / 'training.json', training)
    return training


def variants():
    return [('clean', 0.)] + [(a, r) for a in ['rf', 'dice', 'ea'] for r in [.1, .25, .5]] + [
        ('delete', .5), ('delete_hub', .5), ('insert', .5), ('feature_shuffle', 0.)]


def evaluate(model, graph, split, args, run):
    base = graph_with_edges(graph, np.vstack([split.train_pos, split.val_pos]))
    pairs = np.vstack([split.test_pos, split.test_neg])
    records = []
    for attack, ratio in variants():
        name = 'clean' if attack == 'clean' else attack if attack == 'feature_shuffle' else f'{attack}_{ratio:g}'
        score_path = run / f'predictions_{name}.npz'
        record_path = run / f'evaluation_{name}.json'
        if score_path.exists() and record_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        cache = args.output / 'attacks' / f'{args.dataset}_s{args.seed}_{name}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if attack == 'feature_shuffle':
            rng = np.random.default_rng(args.seed + 7183)
            attacked = GraphData(base.name, base.num_nodes, base.edges,
                                 x=base.x[rng.permutation(base.num_nodes)], y=base.y)
            stats = dict(attack=attack, effective_edits=0, graph_sha256=digest(base.edges))
        else:
            with cache_lock(cache):
                if cache.exists():
                    with np.load(cache) as f:
                        if str(f['base_sha256']) != digest(base.edges):
                            raise RuntimeError('Attack cache base graph changed')
                        attacked = graph_with_edges(base, f['edges'])
                        stats = json.loads(str(f['statistics']))
                else:
                    attacked, stats = perturb(base, split, attack, ratio, args.seed, str(model.device))
                    save_npz(cache, edges=attacked.edges, base_sha256=digest(base.edges),
                             statistics=json.dumps(stats))
        seed_all(args.seed + 20000)
        scores, _ = predict(model, attacked, pairs, args.eval_batch_size)
        positive, negative = scores[:len(split.test_pos)], scores[len(split.test_pos):]
        values = metrics(positive, negative)
        record = dict(dataset=args.dataset, model=args.model, component=args.variant, seed=args.seed,
                      attack=attack, ratio=ratio, metrics=values, statistics=stats)
        np.savez_compressed(score_path, positive=positive, negative=negative)
        dump(record_path, record)
        records.append(record)
        print(json.dumps(dict(stage='evaluate', dataset=args.dataset, model=args.model, component=args.variant,
                              seed=args.seed, attack=name, auc=round(values['auc'], 5))), flush=True)
    clean = records[0]['metrics']
    for record in records:
        record['drop'] = {k: clean[k] - record['metrics'][k] for k in ['auc', 'ap', 'hits100', 'precision100']}
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=ADAPTERS, required=True)
    p.add_argument('--variant', default='full')
    p.add_argument('--dataset', choices=['cora', 'citeseer'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-batch-size', type=int, default=512)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--train-only', action='store_true')
    args = p.parse_args()
    if args.variant not in VARIANTS[args.model]:
        p.error(f'Choose a variant from {VARIANTS[args.model]}')
    if min(args.epochs, args.eval_every, args.eval_batch_size, args.patience, args.threads) < 1:
        p.error('Budgets must be positive')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly choose --device cpu if intended')
    args.output.mkdir(parents=True, exist_ok=True)
    graph = load_data(args.data_root, args.dataset)
    split, fingerprint = split_data(graph, args.seed, args.output)
    run = args.output / 'runs' / f'{args.dataset}_{args.model}_{args.variant}_s{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    cfg = settings(args.model, args.dataset)
    config = dict(protocol=PROTOCOL_VERSION, model=args.model, component=args.variant, dataset=args.dataset,
                  seed=args.seed, epochs=args.epochs, eval_every=args.eval_every, patience=args.patience,
                  eval_batch_size=args.eval_batch_size, torch=torch.__version__, numpy=np.__version__,
                  device=args.device, threads=args.threads, settings=cfg, split_sha256=fingerprint,
                  graph_sha256=digest(graph.edges), features_sha256=digest(graph.x), labels_sha256=digest(graph.y), code_sha256=source_hashes(),
                  author_sources=json.loads(Path(__file__).with_name('sources.lock.json').read_text()),
                  feature_preprocessing='raw Planetoid attributes, float32',
                  ppr_backend='array Andersen push, reference-equivalent LIFO order, alpha=.15 eps=1e-7',
                  training_negatives='fixed, balanced, disjoint from validation and test',
                  model_selection='clean validation AUC, train edges only',
                  test_context='train+validation; all test candidates protected during attacks',
                  training_context='minibatch positive edges removed from propagation and local neighborhoods; LPFormer PPR cached on complete train graph',
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None)
    path = run / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('Configuration/code changed; choose a new output directory')
    dump(path, config)
    if (run / 'complete.json').exists():
        print('Already complete:', run)
        return
    model = ADAPTERS[args.model](args.source_root, args.dataset, graph.x.shape[1], args.variant,
                               args.device, args.output / 'ppr_cache')
    if (run / 'training.json').exists():
        model.load_state_dict(torch.load(run / 'model.pt', map_location=args.device, weights_only=True))
        training = json.loads((run / 'training.json').read_text())
    else:
        training = fit(model, graph, split, cfg, args, run)
    if args.train_only:
        return
    records = evaluate(model, graph, split, args, run)
    dump(run / 'complete.json', dict(config=config, training=training, evaluations=records,
                                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0))


if __name__ == '__main__':
    main()
