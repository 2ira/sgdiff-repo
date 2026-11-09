"""Dataset registry and loaders.

Supported sources
-----------------
``planetoid``
    Cora / Citeseer / Pubmed citation networks (pickled files from the
    Planetoid repository, fetched through a mirror-aware downloader).
``konect``
    KONECT tab-separated edge lists (e.g. ``dimacs10-netscience`` -> NS).
``mtx``
    Matrix-Market files (e.g. ``USAir97.mtx`` from networkrepository.com).
``mat`` / ``tu`` / ``edgelist``
    Local files for datasets that cannot be redistributed (SEAL ``Router``).
``synthetic``
    Small planted-partition graph used by the unit tests / smoke runs.
"""

from __future__ import annotations

import io
import pickle
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .download import download_file
from .graph import GraphData

LOGGER = get_logger(__name__)

PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_FILES = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")
KONECT_BASE = "http://konect.cc/files"
NR_BASE = "https://nrvis.com/download/data"


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a benchmark graph (statistics from paper Table 3)."""

    name: str
    kind: str
    source: str
    domain: str
    num_nodes: int
    num_edges: int  # undirected edge count
    avg_degree: float
    has_features: bool
    auto: bool = True
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "cora": DatasetSpec("cora", "planetoid", "ind.cora", "Citation", 2708, 5278, 3.89, True),
    "citeseer": DatasetSpec("citeseer", "planetoid", "ind.citeseer", "Citation", 3327, 4614, 2.77, True),
    "pubmed": DatasetSpec("pubmed", "planetoid", "ind.pubmed", "Citation", 19717, 44327, 4.49, True),
    "usair": DatasetSpec(
        "usair", "mtx", f"{NR_BASE}/misc/USAir97.zip", "Transportation", 332, 2126, 12.81, False
    ),
    "ns": DatasetSpec(
        "ns", "konect", "dimacs10-netscience", "Collaboration", 1589, 2742, 3.45, False
    ),
    "router": DatasetSpec(
        "router",
        "mat",
        "Router.mat",
        "Transportation",
        5022,
        12516,
        2.49,
        False,
        auto=False,
        note="SEAL data bundle is not redistributable; place Router.mat under data/raw/router/.",
    ),
    "synthetic": DatasetSpec("synthetic", "synthetic", "planted-partitions", "Synthetic", 400, 3200, 16.0, True),
}


def available_datasets() -> list[str]:
    """Names accepted by :func:`load_graph`."""
    return sorted(DATASETS)


def describe_dataset(name: str) -> dict:
    """Return the paper statistics of ``name`` as a plain dictionary."""
    spec = _spec(name)
    return {
        "name": spec.name,
        "domain": spec.domain,
        "num_nodes": spec.num_nodes,
        "num_edges": spec.num_edges,
        "avg_degree": spec.avg_degree,
        "has_features": spec.has_features,
        "auto_download": spec.auto,
        "note": spec.note,
    }


# --------------------------------------------------------------------------- #
# Planetoid
# --------------------------------------------------------------------------- #
def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _read_index_file(path: Path) -> np.ndarray:
    indices: list[int] = []
    for line in path.read_text().split():
        if "-" in line:
            lo, hi = line.split("-")
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(line))
    return np.asarray(indices, dtype=np.int64)


def _sparse_to_dense(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def load_planetoid(name: str, root: str | Path, download: bool = True) -> GraphData:
    """Load a Planetoid citation network and convert it to :class:`GraphData`."""
    spec = _spec(name)
    raw_dir = ensure_dir(Path(root) / "raw" / spec.name)

    paths = {}
    for suffix in PLANETOID_FILES:
        url = f"{PLANETOID_BASE}/{spec.source}.{suffix}"
        dest = raw_dir / f"{spec.source}.{suffix}"
        if download:
            download_file(url, dest, min_bytes=10)
        elif not dest.exists():
            raise FileNotFoundError(f"missing {dest}; run with download=True")
        paths[suffix] = dest

    allx = _sparse_to_dense(_load_pickle(paths["allx"])).astype(np.float32)
    tx = _sparse_to_dense(_load_pickle(paths["tx"])).astype(np.float32)
    graph = _load_pickle(paths["graph"])
    test_idx = _read_index_file(paths["test.index"])

    num_nodes = int(max(allx.shape[0], int(test_idx.max()) + 1, max(graph) + 1))
    features = np.zeros((num_nodes, allx.shape[1]), dtype=np.float32)
    features[: allx.shape[0]] = allx
    sorted_idx = np.sort(test_idx)
    order = np.argsort(test_idx)  # Planetoid stores test rows in file order
    features[sorted_idx] = tx[order]

    edges = [(int(u), int(v)) for u, neigh in graph.items() for v in neigh]
    if not edges:
        raise ValueError(f"{name}: empty edge list after parsing")

    labels = None
    try:
        ally = _sparse_to_dense(_load_pickle(paths["ally"]))
        ty = _sparse_to_dense(_load_pickle(paths["ty"]))
        labels = np.zeros(num_nodes, dtype=np.int64)
        labels[: allx.shape[0]] = ally.argmax(axis=1)
        labels[sorted_idx] = ty[order].argmax(axis=1)
    except Exception:  # noqa: BLE001 - labels are optional for link prediction
        LOGGER.debug("%s: node labels unavailable", name)

    return GraphData(
        name=spec.name,
        num_nodes=num_nodes,
        edges=np.asarray(edges, dtype=np.int64),
        x=features,
        y=labels,
        meta={"source": "planetoid", "domain": spec.domain},
    )


# --------------------------------------------------------------------------- #
# KONECT / Matrix-Market / local files
# --------------------------------------------------------------------------- #
def load_konect(name: str, root: str | Path, download: bool = True) -> GraphData:
    """Load a KONECT network from its ``download.tsv.*`` archive."""
    spec = _spec(name)
    raw_dir = ensure_dir(Path(root) / "raw" / spec.name)
    archive = raw_dir / f"{spec.source}.tar.bz2"
    if download:
        download_file(
            f"{KONECT_BASE}/download.tsv.{spec.source}.tar.bz2",
            archive,
            mirrors=("",),
            min_bytes=100,
        )
    elif not archive.exists():
        raise FileNotFoundError(f"missing {archive}")

    edges: list[tuple[int, int]] = []
    with tarfile.open(archive, "r:bz2") as tf:
        member = next(m for m in tf.getmembers() if Path(m.name).name.startswith("out."))
        handle = tf.extractfile(member)
        assert handle is not None
        for line in io.TextIOWrapper(handle, encoding="utf-8"):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]) - 1, int(parts[1]) - 1
            if u != v:
                edges.append((u, v))

    # KONECT omits isolated nodes from ``out.*``: trust the registry count when larger.
    num_nodes = max(int(max(max(u, v) for u, v in edges)) + 1, spec.num_nodes)
    return GraphData(
        name=spec.name,
        num_nodes=num_nodes,
        edges=np.asarray(edges, dtype=np.int64),
        meta={"source": f"konect:{spec.source}", "domain": spec.domain},
    )


def parse_mtx(text: str) -> np.ndarray:
    """Parse a Matrix-Market coordinate file into 0-based ``(E, 2)`` edges."""
    edges: list[tuple[int, int]] = []
    seen_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        parts = stripped.split()
        if not seen_header:
            seen_header = True  # "n_rows n_cols nnz"
            continue
        edges.append((int(parts[0]) - 1, int(parts[1]) - 1))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def load_mtx_zip(url: str, name: str, root: str | Path, download: bool = True) -> GraphData:
    """Download a networkrepository.com zip and load its first ``.mtx`` member."""
    spec = _spec(name)
    raw_dir = ensure_dir(Path(root) / "raw" / spec.name)
    archive = raw_dir / f"{Path(url).name}"
    if download:
        download_file(url, archive, min_bytes=100)
    elif not archive.exists():
        raise FileNotFoundError(f"missing {archive}")

    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".mtx"))
        text = zf.read(member).decode("utf-8", errors="replace")
    edges = parse_mtx(text)
    num_nodes = max(int(edges.max()) + 1, spec.num_nodes)
    return GraphData(
        name=spec.name,
        num_nodes=num_nodes,
        edges=edges,
        meta={"source": url, "domain": spec.domain},
    )


def load_mat(path: str | Path, name: Optional[str] = None) -> GraphData:
    """Load a SEAL-style ``.mat`` file (keys: ``net``/``network``/``A``)."""
    import scipy.io as sio
    import scipy.sparse as sp

    path = Path(path)
    mat = sio.loadmat(str(path))
    key = next((k for k in ("net", "network", "A", "adj", "adjacency") if k in mat), None)
    if key is None:
        raise KeyError(f"{path}: expected one of net/network/A/adj/adjacency, got {list(mat)}")

    adj = mat[key]
    if sp.issparse(adj):
        coo = adj.tocoo()
        edges = np.stack([coo.row, coo.col], axis=1)
        num_nodes = adj.shape[0]
    else:
        adj = np.asarray(adj)
        num_nodes = adj.shape[0]
        rows, cols = np.nonzero(adj)
        edges = np.stack([rows, cols], axis=1)
    return GraphData(
        name=name or path.stem,
        num_nodes=int(num_nodes),
        edges=np.asarray(edges, dtype=np.int64),
        meta={"source": str(path)},
    )


def load_edgelist(path: str | Path, name: Optional[str] = None, one_based: bool = False) -> GraphData:
    """Load a whitespace-separated ``u v`` edge list."""
    path = Path(path)
    edges: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "%")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        u, v = int(float(parts[0])), int(float(parts[1]))
        if one_based:
            u, v = u - 1, v - 1
        if u != v:
            edges.append((u, v))
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return GraphData(
        name=name or path.stem,
        num_nodes=int(arr.max()) + 1,
        edges=arr,
        meta={"source": str(path)},
    )


def load_tu(directory: str | Path, name: Optional[str] = None) -> GraphData:
    """Load a TUDataset folder containing ``<name>_A.txt`` (1-based ids)."""
    directory = Path(directory)
    candidates = list(directory.glob("*_A.txt"))
    if not candidates:
        raise FileNotFoundError(f"{directory}: no *_A.txt (TUDataset) file found")
    return load_edgelist(candidates[0], name=name or directory.name, one_based=True)


# --------------------------------------------------------------------------- #
# Synthetic benchmark and unified entry point
# --------------------------------------------------------------------------- #
def make_synthetic(
    num_nodes: int = 400,
    num_clusters: int = 4,
    p_in: float = 0.08,
    p_out: float = 0.002,
    num_features: int = 32,
    seed: int = 0,
) -> GraphData:
    """Planted-partition graph with informative node features (for tests)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_clusters, size=num_nodes)

    edges: set[tuple[int, int]] = set()
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            p = p_in if labels[u] == labels[v] else p_out
            if rng.random() < p:
                edges.add((u, v))

    features = rng.normal(0.0, 0.1, size=(num_nodes, num_features)).astype(np.float32)
    features[np.arange(num_nodes), labels % num_features] += 2.0
    return GraphData(
        name="synthetic",
        num_nodes=num_nodes,
        edges=np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2),
        x=features,
        y=labels,
        meta={"source": "synthetic", "num_clusters": num_clusters},
    )


def load_graph(
    name: str,
    root: str | Path = "data",
    download: bool = True,
    use_cache: bool = True,
    **kwargs,
) -> GraphData:
    """Load (and cache) a benchmark graph by name.

    Parameters
    ----------
    name:
        One of :func:`available_datasets`.
    root:
        Directory holding ``raw/`` and ``processed/``.
    download:
        Allow network access when the raw files are missing.
    use_cache:
        Reuse ``<root>/processed/<name>.npz`` when available.
    kwargs:
        ``path`` for ``mat``/``tu``/``edgelist`` datasets and generator options
        for ``synthetic``.
    """
    spec = _spec(name)
    root = Path(root)
    cache_path = root / "processed" / f"{spec.name}.npz"
    if use_cache and cache_path.exists():
        LOGGER.info("loading cached %s", cache_path)
        return _load_cache(cache_path)

    if spec.kind == "planetoid":
        graph = load_planetoid(name, root, download=download)
    elif spec.kind == "konect":
        graph = load_konect(name, root, download=download)
    elif spec.kind == "mtx":
        graph = load_mtx_zip(spec.source, name, root, download=download)
    elif spec.kind == "synthetic":
        synthetic_kwargs = {k: v for k, v in kwargs.items() if k != "path"}
        graph = make_synthetic(**synthetic_kwargs)
    elif spec.kind in {"mat", "tu", "edgelist"}:
        raw_dir = root / "raw" / spec.name
        path = kwargs.get("path")
        if path is None:
            path = raw_dir / spec.source
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{name}: expected a local file at {path}. "
                    f"{spec.note or 'Provide one with path=... or place it under data/raw/'}"
                )
        if spec.kind == "mat":
            graph = load_mat(path, name=spec.name)
        elif spec.kind == "tu":
            graph = load_tu(path, name=spec.name)
        else:
            graph = load_edgelist(path, name=spec.name)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown dataset kind: {spec.kind}")

    LOGGER.info(
        "%s: %d nodes, %d edges, features=%s",
        graph.name,
        graph.num_nodes,
        graph.num_edges,
        graph.num_features if graph.has_features else "none",
    )
    if use_cache:
        _save_cache(graph, cache_path)
    return graph


def _save_cache(graph: GraphData, path: Path) -> None:
    import json

    ensure_dir(path.parent)
    payload = {
        "name": graph.name,
        "num_nodes": graph.num_nodes,
        "edges": graph.edges,
        "meta": np.asarray([json.dumps({**graph.meta, "spec": describe_dataset(graph.name)
                                        if graph.name in DATASETS else {}})]),
    }
    if graph.x is not None:
        payload["x"] = graph.x
    if graph.y is not None:
        payload["y"] = graph.y
    np.savez_compressed(path, **payload)


def _load_cache(path: Path) -> GraphData:
    import json

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"][0])) if "meta" in data else {}
        return GraphData(
            name=str(data["name"]),
            num_nodes=int(data["num_nodes"]),
            edges=data["edges"],
            x=data["x"] if "x" in data else None,
            y=data["y"] if "y" in data else None,
            meta=meta,
        )


def _spec(name: str) -> DatasetSpec:
    key = str(name).lower()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Available: {available_datasets()}")
    return DATASETS[key]
