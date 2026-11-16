"""Training-free spectral and walk-based link-prediction baselines.

The implementations follow the classical constructions covered by DATA8002:

* Katz sums walks of increasing length with geometrically decaying weight;
* the spectral baseline uses low-frequency normalised-Laplacian coordinates;
* DeepWalk learns skip-gram embeddings from uniform random walks;
* Node2Vec replaces uniform transitions with the standard second-order ``p/q``
  bias (here fixed to the local, BFS-like ``p=1, q=2`` setting).

Every method scores the exact candidate pairs supplied by the shared harness.
No node labels, held-out edge labels or raw node attributes are used.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from scipy import sparse


SPECTRAL_WALK_MODELS = ("katz", "spectral", "deepwalk", "node2vec")


def _pairs(pairs) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return values[:, 0], values[:, 1]


def _orthonormalise(values: np.ndarray) -> np.ndarray:
    """Modified Gram-Schmidt using elementwise reductions instead of BLAS."""

    q = np.zeros_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        vector = values[:, column].astype(np.float64, copy=True)
        for previous in range(column):
            vector -= float(np.sum(q[:, previous] * vector)) * q[:, previous]
        length = float(np.sqrt(np.sum(vector * vector)))
        if length < 1e-12:
            vector[:] = 0.0
            vector[column % len(vector)] = 1.0
            for previous in range(column):
                vector -= float(np.sum(q[:, previous] * vector)) * q[:, previous]
            length = float(np.sqrt(np.sum(vector * vector)))
        q[:, column] = vector / max(length, 1e-12)
    return q


def _leading_subspace(matrix: sparse.spmatrix, dimensions: int, iterations: int = 60) -> np.ndarray:
    """Deterministic orthogonal iteration for a symmetric sparse matrix.

    The implementation intentionally avoids dense LAPACK: on some Windows
    environments torch and NumPy load incompatible OpenMP runtimes.  Sparse
    matrix-vector products plus explicit Gram-Schmidt are portable and enough
    for a fixed classical baseline.
    """

    n = int(matrix.shape[0])
    dimensions = min(max(int(dimensions), 1), n)
    rng = np.random.default_rng(0)
    q = _orthonormalise(rng.standard_normal((n, dimensions)))
    for _ in range(int(iterations)):
        columns = [np.asarray(matrix @ q[:, j]).ravel() for j in range(dimensions)]
        q = _orthonormalise(np.column_stack(columns))
    return q


def katz_scores(
    graph,
    pairs,
    max_walk_length: int = 5,
    spectral_scale: float = 0.85,
) -> np.ndarray:
    """Score pairs with a convergent Katz walk sum truncated at five hops.

    ``beta = spectral_scale / max_degree`` satisfies the Katz convergence
    condition because ``rho(A) <= max_degree`` on every undirected graph.
    """

    u, v = _pairs(pairs)
    if not len(u):
        return np.zeros(0, dtype=np.float64)
    adjacency = graph.adjacency().astype(np.float64).tocsr()
    if not adjacency.nnz:
        return np.zeros(len(u), dtype=np.float64)
    degree_bound = float(np.max(np.asarray(adjacency.sum(axis=1)).ravel()))
    if degree_bound <= 0:
        return np.zeros(len(u), dtype=np.float64)
    beta = float(spectral_scale) / degree_bound

    result = np.zeros(len(u), dtype=np.float64)
    for source in np.unique(u):
        selected = u == source
        walk = np.zeros(graph.num_nodes, dtype=np.float64)
        walk[int(source)] = 1.0
        total = np.zeros(graph.num_nodes, dtype=np.float64)
        weight = 1.0
        for _ in range(int(max_walk_length)):
            walk = np.asarray(adjacency @ walk).ravel()
            weight *= beta
            total += weight * walk
        result[selected] = total[v[selected]]
    return result


def spectral_embedding(graph, dimensions: int = 32) -> np.ndarray:
    """Low-frequency normalised-Laplacian embedding.

    The smallest eigenvectors of ``L = I - D^-1/2 A D^-1/2`` are the largest
    eigenvectors of the symmetric normalised adjacency.  We compute the latter
    for numerical efficiency and omit one trivial leading coordinate.
    """

    n = int(graph.num_nodes)
    if n <= 2:
        return np.zeros((n, 1), dtype=np.float64)
    adjacency = graph.adjacency().astype(np.float64).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    normalised = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    # ARPACK requires k < n; one additional leading vector is discarded.
    keep = min(int(dimensions), n - 2)
    # Shift eigenvalues from [-1, 1] to [0, 1].  Orthogonal iteration then
    # orders the subspace by largest algebraic normalised-adjacency eigenvalue,
    # which is equivalent to the lowest Laplacian frequencies.
    shifted = 0.5 * (sparse.eye(n, format="csr") + normalised.tocsr())
    vectors = _leading_subspace(shifted, keep + 1)
    embedding = vectors[:, 1 : keep + 1]
    if not embedding.shape[1]:
        return np.zeros((n, 1), dtype=np.float64)
    embedding[degree == 0] = 0.0
    return embedding


def _cosine_pair_scores(embedding: np.ndarray, pairs) -> np.ndarray:
    u, v = _pairs(pairs)
    norm = np.sqrt(np.sum(embedding * embedding, axis=1, keepdims=True))
    unit = embedding / np.maximum(norm, 1e-12)
    return np.sum(unit[u] * unit[v], axis=1)


def spectral_scores(graph, pairs, dimensions: int = 32) -> np.ndarray:
    """Cosine similarity of classical Laplacian spectral coordinates."""

    return _cosine_pair_scores(spectral_embedding(graph, dimensions), pairs)


def _walks(
    graph,
    seed: int,
    walks_per_node: int,
    walk_length: int,
    p: float,
    q: float,
) -> list[list[str]]:
    """Generate deterministic DeepWalk/Node2Vec training sentences."""

    rng = np.random.default_rng(int(seed))
    adjacency = graph.adjacency().tocsr()
    neighbours = [adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                  for i in range(graph.num_nodes)]
    neighbour_sets = [set(row.tolist()) for row in neighbours]
    sentences: list[list[str]] = []
    for _ in range(int(walks_per_node)):
        for start in rng.permutation(graph.num_nodes):
            walk = [int(start)]
            while len(walk) < int(walk_length):
                current = walk[-1]
                candidates = neighbours[current]
                if not len(candidates):
                    break
                if len(walk) == 1 or (p == 1.0 and q == 1.0):
                    nxt = int(candidates[rng.integers(len(candidates))])
                else:
                    previous = walk[-2]
                    weights = np.empty(len(candidates), dtype=np.float64)
                    for index, candidate in enumerate(candidates):
                        candidate = int(candidate)
                        if candidate == previous:
                            weights[index] = 1.0 / p
                        elif candidate in neighbour_sets[previous]:
                            weights[index] = 1.0
                        else:
                            weights[index] = 1.0 / q
                    weights /= weights.sum()
                    nxt = int(rng.choice(candidates, p=weights))
                walk.append(nxt)
            sentences.append([str(node) for node in walk])
    return sentences


def _stable_hash(value) -> int:
    return int.from_bytes(hashlib.sha1(str(value).encode("utf-8")).digest()[:4], "little")


def random_walk_embedding(
    graph,
    method: str,
    seed: int = 0,
    dimensions: int = 64,
    walks_per_node: int = 5,
    walk_length: int = 20,
    window: int = 5,
    epochs: int = 5,
) -> np.ndarray:
    """Learn a deterministic single-worker DeepWalk or Node2Vec embedding."""

    if method not in ("deepwalk", "node2vec"):
        raise ValueError(f"unknown random-walk method: {method}")
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            f"{method} requires the optional 'gensim' dependency; "
            "install with pip install -e '.[investigation]'"
        ) from exc
    p, q = (1.0, 1.0) if method == "deepwalk" else (1.0, 2.0)
    sentences = _walks(graph, seed, walks_per_node, walk_length, p, q)
    model = Word2Vec(
        sentences=sentences,
        vector_size=int(dimensions),
        window=int(window),
        min_count=0,
        sg=1,
        negative=5,
        workers=1,
        seed=int(seed),
        epochs=int(epochs),
        sorted_vocab=1,
        hashfxn=_stable_hash,
    )
    embedding = np.vstack([model.wv[str(node)] for node in range(graph.num_nodes)]).astype(np.float64)
    embedding[graph.degrees() == 0] = 0.0
    return embedding


def random_walk_scores(
    graph,
    pairs,
    method: str,
    seed: int = 0,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """Cosine link scores from a DeepWalk/Node2Vec embedding."""

    digest = hashlib.sha1(np.ascontiguousarray(graph.edges, dtype=np.int64).tobytes()).hexdigest()
    key = (method, int(seed), graph.num_nodes, digest)
    if cache is not None and key in cache:
        embedding = cache[key]
    else:
        embedding = random_walk_embedding(graph, method, seed=seed)
        if cache is not None:
            cache[key] = embedding
    return _cosine_pair_scores(embedding, pairs)
