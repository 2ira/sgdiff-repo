"""Merge completed seed shards after verifying identical numerical configurations."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiment import ROOT, NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('shards', nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--allow-active', action='store_true', help='Leave existing incomplete run directories to their original worker')
    args = p.parse_args()
    base = json.loads((args.output / 'config.json').read_text())
    config_dir = args.output / 'shard_configs'
    config_dir.mkdir(exist_ok=True)
    for shard in args.shards:
        config = json.loads((shard / 'config.json').read_text())
        assert set(config['seeds']) <= set(base['seeds'])
        for key in base.keys() | config.keys():
            if key not in ['seeds', 'output']:
                assert base[key] == config[key], f'Configuration differs: {key}'
        expected = {f'{protocol}_{source}_{mode}_seed{seed}'
                    for seed in config['seeds'] for protocol in config['protocols']
                    for source in config['datasets'] for mode in config['modes']
                    if mode == 'structure' or source in NAMES[:3]}
        complete = {f.parent.name for f in (shard / 'runs').glob('*/complete.json')}
        if complete != expected:
            raise RuntimeError(f'{shard}: {len(complete)}/{len(expected)} runs completed')
        for path in (shard / 'splits').glob('*.npz'):
            target = args.output / 'splits' / path.name
            if target.exists():
                a, b = np.load(target), np.load(path)
                assert set(a.files) == set(b.files)
                assert all(np.array_equal(a[k], b[k]) for k in a.files)
            else:
                shutil.copy2(path, target)
        added = 0
        active = 0
        for name in sorted(complete):
            source, target = shard / 'runs' / name, args.output / 'runs' / name
            if (target / 'complete.json').exists():
                assert json.loads((target / 'complete.json').read_text()) == json.loads((source / 'complete.json').read_text())
                continue
            if target.exists():
                if args.allow_active:
                    active += 1
                    continue
                raise RuntimeError(f'Destination is being trained or is incomplete: {target}')
            temporary = target.with_name('.merge_' + target.name)
            shutil.copytree(source, temporary)
            temporary.rename(target)
            added += 1
        shutil.copy2(shard / 'config.json', config_dir / f'{shard.name}.json')
        print(f'{shard}: verified {len(complete)} runs; added {added}; left {active} active runs to their original worker')


if __name__ == '__main__':
    main()
