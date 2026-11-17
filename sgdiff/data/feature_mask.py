"""Shared structural (whole-row) feature observation mask."""

import numpy as np


def observed_nodes(num_nodes: int, missing_rate: float, seed: int) -> np.ndarray:
    if not 0 <= missing_rate < 1:
        raise ValueError("feature_missing_rate must be in [0, 1)")
    observed = np.ones(num_nodes, dtype=bool)
    missing = np.random.default_rng(seed).choice(
        num_nodes, int(num_nodes * missing_rate), replace=False
    )
    observed[missing] = False
    return observed
