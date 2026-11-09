"""SGDiff: reproduction of "Sub-graph Based Diffusion Model for Link Prediction".

Public surface
--------------
``sgdiff.api``
    High-level functions: :func:`sgdiff.api.train`, ``predict``, ``evaluate``.
``sgdiff.data``
    Dataset loaders, splits, enclosing sub-graph extraction, structural attacks.
``sgdiff.models``
    ``SGDiff`` (structure + feature diffusion + fusion) and the baselines.
``sgdiff.engine``
    Batching, training loop and evaluation helpers.
"""

from .version import __version__

__all__ = ["__version__"]
