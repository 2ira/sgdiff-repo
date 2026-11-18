"""Fetch and verify external author implementations without redistributing them."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LOCK = Path(__file__).with_name('sources.lock.json')


def verify(root, name):
    spec = json.loads(LOCK.read_text())[name]
    directory = Path(root) / name
    for relative, expected in spec['python_sha256'].items():
        file = directory / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Author source differs from pinned revision: {file}')
    return directory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name, spec in json.loads(LOCK.read_text()).items():
        dest = args.root / name
        if not dest.exists() and not args.verify_only:
            subprocess.run(['git', 'clone', 'https://github.com/' + spec['repository'] + '.git', str(dest)], check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', spec['commit']], check=True)
        verify(args.root, name)
        print(name, spec['commit'], 'verified')


if __name__ == '__main__':
    main()
