"""
sp4lp/paths.py
==============================================================================
Shortest-path extraction for SP4LP.

Two things changed with respect to the previous version, and the first one is
the reason the Cora numbers collapsed.

1) TARGET-EDGE MASKING IS NOW PER-PAIR, NOT PER-SOURCE.
   The old `compute_paths_grouped(..., remove_own_source_edges=True)` grouped
   the query pairs by source `u` and then deleted, in one go, EVERY edge
   (u, v_j) that appeared as a query target of `u`, before running a single BFS.
   That is much more aggressive than removing the target edge: for a training
   positive (u, v) it also deletes (u, w) for every other training positive of
   u.  On a graph with any clustering this annihilates exactly the signal the
   model is supposed to learn.  If u-w-v is a triangle and both (u,v) and (u,w)
   are training positives, the correct masked shortest path for (u,v) is
   [u, w, v] (a common neighbour!), but the old code removed (u,w) as well and
   returned either a long detour or the disconnected fallback [u, v].

   Measured on synthetic graphs with Cora-like clustering, the old grouping
   turned ~53% of training positives that should be 3-node common-neighbour
   paths into something else, and doubled the "disconnected" fallback rate.
   Validation and test positives are unaffected (they are not in the graph, so
   nothing is masked), so training and evaluation saw two different worlds.

   The new implementation masks ONLY the edge (u, v) whose own path is being
   computed.  Pairs whose endpoints are not adjacent need no masking at all and
   are still resolved with one grouped BFS per source, so the cost stays close
   to the old one.

2) PATHS ARE STORED IN A RAGGED, VECTORISED CONTAINER (`PathStore`).
   The old dict-of-lists needed a Python loop plus a tuple hash per pair on
   every batch.  With 500 HeaRT negatives per positive that dominated
   evaluation time.  `PathStore` keeps a flat int32 buffer plus offsets and
   gathers a whole batch with numpy fancy indexing.

Convention (unchanged, and faithful to the paper): a path INCLUDES both
endpoints, and if u and v are in different connected components we fall back to
the synthetic 2-node path [u, v] -- Appendix D of the paper does the same.
`PathStore` additionally records a `reachable` flag per pair so the model can
tell a genuine fallback apart from a real path, instead of having to guess.
------------------------------------------------------------------------------
"""

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "build_adj",
    "compute_paths",
    "PathStore",
    "path_uses_edge",
    "audit_paths",
]


# ============================================================================ #
#  Graph representation
# ============================================================================ #
def build_adj(edge_index, num_nodes):
    """
    Adjacency as a list of sorted, de-duplicated Python lists.

    Python lists beat numpy slices here: BFS touches neighbours one at a time,
    and scalar indexing into a numpy array is slower than iterating a list.

    edge_index : LongTensor [2, E] (either or both directions), or an iterable
                 of (u, v) pairs.
    """
    num_nodes = int(num_nodes)
    if torch.is_tensor(edge_index):
        ei = edge_index
        if ei.dim() == 2 and ei.size(0) == 2:
            src = ei[0].tolist()
            dst = ei[1].tolist()
        else:
            src = ei[:, 0].tolist()
            dst = ei[:, 1].tolist()
    else:
        pairs = list(edge_index)
        src = [int(a) for a, _ in pairs]
        dst = [int(b) for _, b in pairs]

    adj = [set() for _ in range(num_nodes)]
    for a, b in zip(src, dst):
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return [sorted(s) for s in adj]


# ============================================================================ #
#  BFS primitives
# ============================================================================ #
def _bfs_parents(adj, source, targets, banned_edge=None, max_dist=None):
    """
    BFS from `source`, recording parent pointers, stopping as soon as every node
    in `targets` has been reached (or `max_dist` hops have been explored).

    banned_edge : optional (a, b) pair that BFS must pretend does not exist.
                  Used to mask a training positive's own edge.
    """
    parent = {source: -1}
    remaining = set(targets)
    remaining.discard(source)
    if not remaining:
        return parent

    if banned_edge is None:
        ba = bb = -1
    else:
        ba, bb = banned_edge

    frontier = [source]
    depth = 0
    while frontier:
        if max_dist is not None and depth >= max_dist:
            break
        nxt = []
        for a in frontier:
            for b in adj[a]:
                if b in parent:
                    continue
                if (a == ba and b == bb) or (a == bb and b == ba):
                    continue
                parent[b] = a
                if b in remaining:
                    remaining.discard(b)
                    if not remaining:
                        return parent
                nxt.append(b)
        frontier = nxt
        depth += 1
    return parent


def _reconstruct(parent, source, target):
    """Rebuild source -> target from parent pointers, or None if unreachable."""
    if target not in parent:
        return None
    path = [target]
    while path[-1] != source:
        p = parent[path[-1]]
        if p < 0:
            break
        path.append(p)
    path.reverse()
    return path if path[0] == source else None


def compute_paths(adj, pairs, mask_target_edge=False, max_dist=None,
                  desc="shortest paths", show_progress=True,
                  component_ids=None):
    """
    Shortest node-paths for many (u, v) pairs.

    mask_target_edge : when True, the path for a pair (u, v) is computed on the
        graph MINUS the single edge (u, v) -- and nothing else is removed.  Use
        this for training positives, whose edge is present in the message-passing
        graph.  Validation and test positives are already absent from the graph,
        and negatives are non-edges, so they need `False`.

    max_dist : optional BFS depth cap.  Pairs farther than this are reported as
        unreachable and get the [u, v] fallback.  `None` means exact.

    Returns (paths, reachable): `paths` maps (u, v) -> list of node ids
    (endpoints included), `reachable` maps (u, v) -> bool.
    """
    pairs = [(int(u), int(v)) for u, v in pairs]

    # Split the work: pairs that need masking (endpoints adjacent) must be done
    # one BFS at a time; everything else amortises over a shared source.
    plain_by_src = defaultdict(list)
    masked = []
    for u, v in pairs:
        if mask_target_edge and v in _as_set(adj, u):
            masked.append((u, v))
        else:
            plain_by_src[u].append(v)

    paths = {}
    reachable = {}

    it = plain_by_src.items()
    if show_progress:
        it = tqdm(it, total=len(plain_by_src), desc=f"{desc} [grouped]")
    for u, targets in it:
        parent = _bfs_parents(adj, u, set(targets), max_dist=max_dist)
        for v in targets:
            p = _reconstruct(parent, u, v)
            if p is None:
                paths[(u, v)] = [u, v]
                reachable[(u, v)] = False
            else:
                paths[(u, v)] = p
                reachable[(u, v)] = (True if component_ids is None else
                                      component_ids[u] == component_ids[v])

    it2 = masked
    if show_progress and masked:
        it2 = tqdm(masked, desc=f"{desc} [masked]")
    for u, v in it2:
        parent = _bfs_parents(adj, u, {v}, banned_edge=(u, v), max_dist=max_dist)
        p = _reconstruct(parent, u, v)
        if p is None:
            paths[(u, v)] = [u, v]
            reachable[(u, v)] = False
        else:
            paths[(u, v)] = p
            reachable[(u, v)] = (True if component_ids is None else
                                  component_ids[u] == component_ids[v])

    return paths, reachable


def _as_set(adj, u):
    """Membership test on a sorted neighbour list without building a set."""
    lst = adj[u]
    # small lists: linear scan is fine and allocation-free
    return lst if len(lst) < 32 else set(lst)


# ============================================================================ #
#  Ragged storage
# ============================================================================ #
class PathStore:
    """
    Flat, vectorised container for a set of paths.

    Layout
    ------
    key    : dict (u, v) -> row index
    offs   : int64 [R + 1] prefix offsets into `data`
    data   : int32 [sum of path lengths] concatenated node ids
    reach  : bool  [R] False for the synthetic [u, v] fallback

    `gather(pairs)` returns the concatenated node ids and per-row lengths for a
    batch of pairs, built with numpy fancy indexing rather than a Python loop.
    """

    def __init__(self, key, offs, data, reach, feat=None):
        self.key = key
        self.offs = offs
        self.data = data
        self.reach = reach
        self.feat = feat

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dicts(cls, paths, reachable, max_len=None):
        keys = list(paths.keys())
        key = {k: i for i, k in enumerate(keys)}
        lengths = np.empty(len(keys), dtype=np.int64)
        chunks = []
        for i, k in enumerate(keys):
            p = paths[k]
            if max_len is not None and len(p) > max_len:
                p = _truncate_path(p, max_len)
            lengths[i] = len(p)
            chunks.append(np.asarray(p, dtype=np.int32))
        offs = np.zeros(len(keys) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offs[1:])
        data = (np.concatenate(chunks) if chunks
                else np.zeros(0, dtype=np.int32))
        reach = np.fromiter((bool(reachable[k]) for k in keys),
                            dtype=bool, count=len(keys))
        store = cls(key, offs, data, reach)
        store._keys_in_order = keys
        return store

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.key)

    def __contains__(self, pair):
        return (int(pair[0]), int(pair[1])) in self.key

    def rows_for(self, pairs_np):
        """pairs_np: int64 [B, 2] -> int64 [B] row indices (KeyError if unseen)."""
        key = self.key
        return np.fromiter(
            (key[(int(u), int(v))] for u, v in pairs_np),
            dtype=np.int64, count=len(pairs_np))

    def gather(self, pairs_np):
        """
        pairs_np : int64 [B, 2]
        returns  : (flat int64 [sum L], lengths int64 [B], reach bool [B])
        """
        rows = self.rows_for(pairs_np)
        starts = self.offs[rows]
        lengths = self.offs[rows + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return (np.zeros(0, dtype=np.int64), lengths, self.reach[rows])
        # ragged expansion: repeat each start, then add a per-row running index
        rep_start = np.repeat(starts, lengths)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
        sel = rep_start + within
        flat = self.data[sel].astype(np.int64, copy=False)
        if getattr(self, 'feat', None) is None:
            return flat, lengths, self.reach[rows]
        n_paths = np.ones(len(rows), dtype=np.int64)
        return (flat, lengths, n_paths, self.reach[rows],
                self.feat[sel].astype(np.int64, copy=False))

    # ------------------------------------------------------------------ #
    def stats(self):
        lengths = np.diff(self.offs)
        if len(lengths) == 0:
            return {}
        vals, counts = np.unique(lengths, return_counts=True)
        return {
            "n": int(len(lengths)),
            "mean_nodes": float(lengths.mean()),
            "unreachable_frac": float((~self.reach).mean()),
            "hist": {int(v): int(c) for v, c in zip(vals, counts)},
        }


def _truncate_path(path, max_len):
    """Keep the head and the tail; the middle of a long path matters least."""
    if len(path) <= max_len:
        return path
    head = (max_len + 1) // 2
    tail = max_len - head
    return list(path[:head]) + list(path[-tail:])


# ============================================================================ #
#  Audits
# ============================================================================ #
def path_uses_edge(path, u, v):
    """True if (u, v) appears as a consecutive step of `path`."""
    for a, b in zip(path[:-1], path[1:]):
        if (a == u and b == v) or (a == v and b == u):
            return True
    return False


def audit_paths(paths, reachable, adj=None, name="", must_not_use_own_edge=True):
    """
    Sanity-check a computed path set.  Raises AssertionError on a real leak.

    Returns a small report dict, so `check_setup.py` can print it.
    """
    n_leak = 0
    n_fallback = 0
    for (u, v), p in paths.items():
        if not reachable[(u, v)]:
            n_fallback += 1
            continue
        assert p[0] == u and p[-1] == v, f"{name}: malformed path for {(u, v)}"
        assert len(set(p)) == len(p), f"{name}: non-simple path for {(u, v)}"
        if must_not_use_own_edge and path_uses_edge(p, u, v):
            n_leak += 1
    if n_leak:
        raise AssertionError(
            f"{name}: {n_leak} supervision paths traverse their own target edge")
    return {"name": name, "n": len(paths), "fallback": n_fallback}
