"""Logging configuration shared by CLI entry points."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str = "sgdiff", level: str | None = None) -> logging.Logger:
    """Return a namespaced logger configured once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("sgdiff")
        root.addHandler(handler)
        root.setLevel(os.environ.get("SGDIFF_LOG_LEVEL", "INFO"))
        root.propagate = False
        _CONFIGURED = True
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
