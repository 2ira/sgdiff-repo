"""Training / evaluation engine."""

from .batch import PairSubgraphDataset, SubgraphBatch, collate_subgraphs
from .evaluator import evaluate_link_prediction, score_pairs
from .trainer import Trainer

__all__ = [
    "PairSubgraphDataset",
    "SubgraphBatch",
    "Trainer",
    "collate_subgraphs",
    "evaluate_link_prediction",
    "score_pairs",
]
