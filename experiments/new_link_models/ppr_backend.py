"""Array backend for the author's Andersen push, preserving LIFO push order.

The reference in LPFormer uses dictionaries and ``v not in queue`` list scans.
This replaces membership scans with boolean arrays, with identical arithmetic,
thresholds, neighbor order and first-visit output order. Tests compare against
the pinned author function, including float64 values and isolated vertices.
"""
import numba
import numpy as np


@numba.njit(parallel=True)
def calc_ppr(indptr, indices, out_degree, alpha, eps):
    n = len(out_degree)
    neighbors = [np.empty(0, dtype=np.int64) for _ in range(n)]
    weights = [np.empty(0, dtype=np.float64) for _ in range(n)]
    threshold = alpha * eps
    for source_unsigned in numba.prange(n):
        source = np.int64(source_unsigned)
        residual = np.zeros(n, dtype=np.float64)
        probability = np.zeros(n, dtype=np.float64)
        queued = np.zeros(n, dtype=np.bool_)
        visited = np.zeros(n, dtype=np.bool_)
        stack = np.empty(n, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        size, count = 1, 0
        stack[0] = source
        queued[source] = True
        residual[source] = alpha
        while size:
            size -= 1
            u = stack[size]
            queued[u] = False
            value = residual[u]
            if not visited[u]:
                order[count] = u
                count += 1
                visited[u] = True
            probability[u] += value
            residual[u] = 0.
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                residual[v] += (1 - alpha) * value / out_degree[u]
                if residual[v] >= threshold * out_degree[v] and not queued[v]:
                    stack[size] = v
                    size += 1
                    queued[v] = True
        neighbors[source] = order[:count].copy()
        weights[source] = probability[order[:count]].copy()
    return neighbors, weights
