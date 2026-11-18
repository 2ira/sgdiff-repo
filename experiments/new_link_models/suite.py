"""Resumable process-isolated suite; one author-model configuration per worker."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .adapters import VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['cora', 'citeseer'])
    p.add_argument('--models', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--full-only', action='store_true')
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    plan = []
    # Finish all complete models first, then the paired component retrains.
    for component_index in range(1 if args.full_only else 3):
        for seed in args.seeds:
            for dataset in args.datasets:
                for model in args.models:
                    variant = VARIANTS[model][component_index]
                    name = f'{dataset}_{model}_{variant}_s{seed}'
                    if (args.output / 'runs' / name / 'complete.json').exists():
                        continue
                    cmd = [sys.executable, '-m', 'experiments.new_link_models.run', '--model', model,
                           '--variant', variant, '--dataset', dataset, '--seed', str(seed),
                           '--data-root', args.data_root, '--source-root', args.source_root,
                           '--output', str(args.output), '--epochs', str(args.epochs), '--device', args.device]
                    plan.append((name, cmd))
    (args.output / 'suite_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    active, done = [], 0
    while plan or active:
        while plan and len(active) < args.jobs:
            name, cmd = plan.pop(0)
            log = (logs / f'{name}.log').open('a')
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            active.append((name, proc, log))
            print(json.dumps(dict(event='started', name=name, pid=proc.pid, remaining=len(plan))), flush=True)
        for name, proc, log in list(active):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            active.remove((name, proc, log))
            if code:
                for _, other, handle in active:
                    other.terminate()
                    other.wait()
                    handle.close()
                raise RuntimeError(f'{name} failed with {code}; inspect {logs / (name + ".log")}')
            done += 1
            print(json.dumps(dict(event='completed', name=name, completed=done)), flush=True)
        if active:
            time.sleep(2)


if __name__ == '__main__':
    main()
