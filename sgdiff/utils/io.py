"""Small JSON / filesystem helpers used across the pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Serialise ``obj`` to JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=_default)
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load a JSON document from ``path``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)
