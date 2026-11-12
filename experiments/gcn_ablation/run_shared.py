"""Run the archived experiment protocol using the repository's shared GCN."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.gcn_ablation import experiment
from experiments.gcn_ablation.shared_backend import SharedLinkModel, configure


def main():
    sys.argv[1:] = configure(sys.argv[1:])
    experiment.LinkModel = SharedLinkModel
    experiment.main()


if __name__ == '__main__':
    main()
