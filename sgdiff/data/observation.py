"""Matched-budget observation degradation, shared by every model."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .graph import GraphData


def graph_fingerprint(graph):
    h = hashlib.sha256(np.ascontiguousarray(graph.edges).tobytes())
    if graph.x is not None:
        h.update(np.ascontiguousarray(graph.x).tobytes())
    return h.hexdigest()[:20]


@dataclass(frozen=True)
class Condition:
    name: str
    mechanism: str = "none"
    edge_ratio: float = 0.0
    feature_ratio: float = 0.0


def conditions():
    out = [Condition("clean")]
    for mechanism in ("random_delete", "degree_delete"):
        for r in (0.2, 0.4):
            out.append(Condition(f"{mechanism}_{int(100*r)}", mechanism, r))
    for mechanism in ("random_flip", "dice"):
        for r in (0.1, 0.25, 0.5):
            out.append(Condition(f"{mechanism}_{int(100*r)}", mechanism, r))
    for r in (0.2, 0.4):
        out.append(Condition(f"feature_mask_{int(100*r)}", feature_ratio=r))
    out += [Condition("joint_random_40", "random_delete", .4, .4),
            Condition("joint_degree_40", "degree_delete", .4, .4)]
    return out


def masked_node_ids(num_nodes, ratio, seed):
    return np.random.default_rng(seed + 100003).permutation(num_nodes)[:round(ratio * num_nodes)]


def degrade(graph, condition, seed, forbidden_pairs=()):
    """Delete exactly round(r*E) edges, or flip that many unique edge states.

    Degree weights 1/(min(d_u,d_v)+1) use the ORIGINAL visible graph. A fixed
    weighted random order gives nested deletion sets across rates. Feature
    masking zeros exact numbers of whole node rows before frozen preprocessing.
    DICE uses node classes only as privileged attack information, never as a
    predictor input. Query pairs cannot be inserted into the context.
    """
    r = condition.edge_ratio
    if not 0 <= r <= 1 or not 0 <= condition.feature_ratio <= 1:
        raise ValueError("ratios must be in [0,1]")
    rng = np.random.default_rng(seed)
    budget = int(round(r * graph.num_edges))
    method = condition.mechanism
    n_del = budget if method.endswith("delete") else budget // 2
    if method == "none":
        n_del = 0
    eligible = np.arange(graph.num_edges)
    labels = graph.y
    if method == "dice":
        if labels is None or len(np.unique(labels)) < 2:
            raise ValueError("DICE requires at least two explicit node classes")
        eligible = eligible[labels[graph.edges[:, 0]] == labels[graph.edges[:, 1]]]
    if n_del > len(eligible):
        raise ValueError("insufficient eligible edges for requested DICE deletion budget")
    if method == "degree_delete":
        deg = graph.degrees()
        weights = 1.0 / (np.minimum(deg[graph.edges[:, 0]], deg[graph.edges[:, 1]]) + 1)
        priority = -np.log(rng.uniform(np.finfo(float).tiny, 1, graph.num_edges)) / weights
        order = np.argsort(priority, kind="stable")
    else:
        order = rng.permutation(eligible)
    dropped = graph.edges[order[:n_del]]
    out = graph.drop_edges(dropped)
    n_ins = budget - n_del if method in ("random_flip", "dice") else 0
    forbidden = graph.edge_set() | {tuple(sorted(map(int, p))) for p in forbidden_pairs}
    added = []
    attempts = 0
    while len(added) < n_ins and attempts < max(10000, 100 * n_ins):
        attempts += 1
        u, v = map(int, rng.integers(graph.num_nodes, size=2))
        edge = (min(u, v), max(u, v))
        if u == v or edge in forbidden or (method == "dice" and labels[u] == labels[v]):
            continue
        forbidden.add(edge)
        added.append(edge)
    if len(added) != n_ins:
        raise RuntimeError("could not realize exact insertion budget")
    if added:
        out = out.add_edges(np.asarray(added))
    x = None if graph.x is None else graph.x.copy()
    masked = masked_node_ids(graph.num_nodes, condition.feature_ratio, seed)
    nonzero_before = 0
    if len(masked):
        if x is None:
            raise ValueError("feature masking requires attributes")
        nonzero_before = int(np.count_nonzero(x[masked]))
        x[masked] = 0
    out = GraphData(graph.name, graph.num_nodes, out.edges, x, graph.y, dict(graph.meta))
    info = dict(condition=condition.name, mechanism=method, requested_ratio=r,
                budget=budget, deleted=len(dropped), inserted=len(added),
                masked_nodes=len(masked), masked_nonzero_entries=nonzero_before,
                clean_edges=graph.num_edges, observed_edges=out.num_edges,
                graph_fingerprint=graph_fingerprint(out))
    return out, info
