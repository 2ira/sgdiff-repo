"""Bind the immutable native evaluator before this branch's existing model package."""
import subprocess
from pathlib import Path
import experiments

REVISION = '95083da808e500c02da458e205f4b861ceb80871'


def bind_peer(peer):
    peer = Path(peer).resolve()
    revision = subprocess.check_output(['git', '-C', str(peer), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected peer checkout {REVISION}, found {revision}')
    # This feature branch also contains experiments.new_link_models. Select the
    # pinned checkout first; never silently execute this branch's newer adapters.
    peer_experiments = str(peer / 'experiments')
    experiments.__path__ = [peer_experiments, *[p for p in experiments.__path__ if p != peer_experiments]]
    from experiments.new_link_models import protocol, run
    for module in (protocol, run):
        relative = f'experiments/new_link_models/{Path(module.__file__).name}'
        source = peer / relative
        original = subprocess.check_output(['git', '-C', str(peer), 'show', f'{REVISION}:{relative}'])
        if Path(module.__file__).resolve() != source or source.read_bytes() != original:
            raise ValueError(f'Evaluator does not match pinned peer source: {source}; use a fresh Python process')
    return protocol, run
