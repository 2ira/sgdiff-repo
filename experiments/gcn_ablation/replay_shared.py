"""Replay a checkpoint produced by run_shared.py, with provenance checking."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.gcn_ablation import experiment
from experiments.gcn_ablation.shared_backend import SharedLinkModel, configure


def main():
    sys.argv[1:] = configure(sys.argv[1:], replay=True)
    sys.modules['experiment'] = experiment
    from experiments.gcn_ablation import replay
    replay.LinkModel = SharedLinkModel
    replay.main()


if __name__ == '__main__':
    main()
