"""Data layer: datasets, splits, sub-graph extraction and structural attacks."""

from .attack import dice_attack, random_perturbation
from .datasets import DATASETS, available_datasets, describe_dataset, load_graph, make_synthetic
from .graph import GraphData
from .splits import LinkSplit, make_link_split, subsample_pairs
from .subgraph import Subgraph, SubgraphExtractor, extract_enclosing_subgraph

__all__ = [
    "DATASETS",
    "GraphData",
    "LinkSplit",
    "Subgraph",
    "SubgraphExtractor",
    "available_datasets",
    "describe_dataset",
    "dice_attack",
    "extract_enclosing_subgraph",
    "load_graph",
    "make_link_split",
    "make_synthetic",
    "random_perturbation",
    "subsample_pairs",
]
