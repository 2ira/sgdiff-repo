"""Data layer: datasets, splits, sub-graph extraction and structural attacks."""

from .attack import (
    EA_MAX_NODES,
    apply_attack,
    default_community,
    dice_attack,
    embedding_attack,
    normalize_attack,
    random_perturbation,
    supports_attack,
)
from .datasets import DATASETS, available_datasets, describe_dataset, load_graph, make_synthetic
from .graph import GraphData
from .splits import LinkSplit, make_link_split, subsample_pairs
from .subgraph import Subgraph, SubgraphExtractor, extract_enclosing_subgraph

__all__ = [
    "DATASETS",
    "EA_MAX_NODES",
    "GraphData",
    "LinkSplit",
    "Subgraph",
    "SubgraphExtractor",
    "apply_attack",
    "available_datasets",
    "default_community",
    "describe_dataset",
    "dice_attack",
    "embedding_attack",
    "extract_enclosing_subgraph",
    "load_graph",
    "make_link_split",
    "make_synthetic",
    "normalize_attack",
    "random_perturbation",
    "subsample_pairs",
    "supports_attack",
]
