"""Tests for the targeted perturbation probes (``sgdiff.data.perturb``)."""

import numpy as np
import pytest

from sgdiff.data.datasets import make_synthetic
from sgdiff.data.perturb import (
    PERTURBATIONS,
    delete_edges,
    insert_edges,
    shuffle_features,
)


@pytest.fixture(scope="module")
def graph():
    return make_synthetic(num_nodes=80, num_features=8, seed=0)


def _keys(edges, num_nodes):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return set((np.minimum(e[:, 0], e[:, 1]) * num_nodes + np.maximum(e[:, 0], e[:, 1])).tolist())


def test_delete_random_hits_the_budget_and_keeps_the_subgraph(graph):
    out = delete_edges(graph, 0.5, "random", seed=0)
    assert out.num_edges == graph.num_edges - int(round(0.5 * graph.num_edges))
    assert _keys(out.edges, out.num_nodes) <= _keys(graph.edges, graph.num_nodes)
    assert out.num_nodes == graph.num_nodes
    assert out.meta["perturb"]["kind"] == "del_random"


def test_delete_does_not_mutate_the_input(graph):
    before = np.array(graph.edges, copy=True)
    delete_edges(graph, 0.5, "random", seed=0)
    insert_edges(graph, 0.5, "random", seed=0)
    shuffle_features(graph, seed=0)
    assert np.array_equal(graph.edges, before)


def test_hub_deletion_prefers_high_degree_edges():
    """Hub deletion must actually be degree-biased, otherwise the probe is empty."""
    from sgdiff.data.graph import GraphData

    hub = 0
    star = [(hub, v) for v in range(1, 41)]  # 40 edges on the hub
    chain = [(v, v + 1) for v in range(1, 40)]  # 39 edges that avoid the hub
    graph = GraphData(name="hub", num_nodes=61, edges=np.asarray(star + chain, dtype=np.int64))
    hub_edges = {frozenset(e) for e in star}

    def hub_retention(out):
        kept = {frozenset(e.tolist()) for e in np.asarray(out.edges)}
        return len(kept & hub_edges) / len(hub_edges)

    random_del = hub_retention(delete_edges(graph, 0.5, "random", seed=0))
    hub_del = hub_retention(delete_edges(graph, 0.5, "hub", seed=0))
    assert hub_del < random_del - 0.2


def test_insert_random_adds_the_budget_without_duplicates_or_self_loops(graph):
    out = insert_edges(graph, 0.25, "random", seed=0)
    assert out.num_edges == graph.num_edges + int(round(0.25 * graph.num_edges))
    e = np.asarray(out.edges)
    assert np.all(e[:, 0] != e[:, 1])
    keys = _keys(out.edges, out.num_nodes)
    assert len(keys) == len(e)  # de-duplicated
    assert _keys(graph.edges, graph.num_nodes) <= keys  # nothing was removed


def test_homophilous_and_heterophilous_insertions_respect_the_labels(graph):
    """Only the *inserted* edges must obey the label constraint."""
    y = np.asarray(graph.y)
    existing = _keys(graph.edges, graph.num_nodes)

    def added_edges(mode):
        out = np.asarray(insert_edges(graph, 0.2, mode, seed=0).edges)
        keep = [
            (u, v)
            for u, v in out.tolist()
            if min(u, v) * graph.num_nodes + max(u, v) not in existing
        ]
        assert len(keep) == int(round(0.2 * graph.num_edges))
        return np.asarray(keep, dtype=np.int64)

    homo = added_edges("homo")
    hetero = added_edges("hetero")
    assert np.mean(y[homo[:, 0]] == y[homo[:, 1]]) == 1.0
    assert np.mean(y[hetero[:, 0]] != y[hetero[:, 1]]) == 1.0


def test_insertion_modes_need_labels():
    from sgdiff.data.graph import GraphData

    bare = GraphData(name="bare", num_nodes=6, edges=np.array([[0, 1], [1, 2], [2, 3]]))
    with pytest.raises(ValueError):
        insert_edges(bare, 0.5, "homo")
    with pytest.raises(ValueError):
        insert_edges(bare, 0.5, "hetero")


def test_unknown_modes_raise(graph):
    with pytest.raises(ValueError):
        delete_edges(graph, 0.5, "wat")
    with pytest.raises(ValueError):
        insert_edges(graph, 0.5, "wat")


def test_feature_shuffle_keeps_structure_and_destroys_the_feature_signal(graph):
    out = shuffle_features(graph, seed=0)
    assert np.array_equal(np.asarray(out.edges), np.asarray(graph.edges))
    # rows are a permutation -> the multiset of row norms is unchanged
    assert np.allclose(np.sort(np.linalg.norm(out.x, axis=1)), np.sort(np.linalg.norm(graph.x, axis=1)))
    assert not np.allclose(out.x, graph.x)
    # node 0's feature no longer points at its own label
    y = np.asarray(graph.y)
    width = graph.x.shape[1]
    own = np.mean(np.argmax(graph.x, axis=1) == y % width)
    permuted = np.mean(np.argmax(out.x, axis=1) == y % width)
    assert permuted < own


def test_feature_shuffle_needs_features():
    from sgdiff.data.graph import GraphData

    bare = GraphData(name="bare", num_nodes=5, edges=np.array([[0, 1], [1, 2]]))
    with pytest.raises(ValueError):
        shuffle_features(bare)


def test_perturbation_plan_is_callable_and_labelled(graph):
    assert set(PERTURBATIONS) == {
        "del@50%",
        "del_hub@50%",
        "ins@50%",
        "ins_hub@50%",
        "ins_homo@50%",
        "ins_hetero@50%",
        "feature_shuffle",
    }
    for name, fn in PERTURBATIONS.items():
        out = fn(graph, 0)
        assert out is not graph
        assert out.meta["perturb"] is not None, name


def test_probe_harness_runs_end_to_end(tmp_path):
    """The script wiring: train once, probe, heuristics, ablations, diagnostics."""
    import importlib.util
    from pathlib import Path

    from sgdiff.config import load_config

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_probe.py"
    spec = importlib.util.spec_from_file_location("run_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = load_config(
        None,
        [
            "data.name=synthetic",
            "data.root=" + str(tmp_path / "data"),
            "data.max_train_pairs=64",
            "data.max_eval_pairs=24",
            "subgraph.max_nodes_per_hop=6",
            "train.device=cpu",
        ],
    )
    result = module.run_probe(
        cfg,
        models=["gcn", "sgdiff"],
        variants=["del@50%", "ins_hub@50%", "ins_hetero@50%"],
        seeds=[0],
        max_pairs=20,
        node_pair_epochs=4,
        subgraph_epochs=2,
        sgdiff_epochs=1,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        write=False,
        quiet=True,
        verbose=False,
    )
    assert result["variants_used"] == ["clean", "del@50%", "ins_hub@50%", "ins_hetero@50%"]
    for name in ("gcn", "sgdiff"):
        per_variant = result["summary"]["models"][name]
        for variant in result["variants_used"]:
            assert 0.0 <= per_variant[variant]["auc"]["mean"] <= 1.0
        # every probe is scored against the model's own clean column
        assert per_variant["del@50%"]["drop_auc"]["n"] == 1
    # ablations exist for both kinds; the fusion branches only for SGDiff
    assert "features_zeroed" in result["summary"]["ablations"]["gcn"]
    assert "structure_only" in result["summary"]["ablations"]["sgdiff"]
    assert "feature_only" in result["summary"]["ablations"]["sgdiff"]
    assert "drnl_permuted" not in result["summary"]["ablations"]["gcn"]
    # cheap passes ran and produced one row per probe graph
    assert set(result["subgraphs"]) == {"clean", "del@50%", "ins_hub@50%", "ins_hetero@50%"}
    assert "cn" in result["summary"]["heuristics"]["clean"]
    assert result["alignment"]["sgdiff"]["cn"]["n"] == 1
