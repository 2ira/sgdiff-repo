"""Checks for scientific validity: ties, leakage, budgets, frozen evidence."""
import numpy as np
import torch
import pytest
pytest.importorskip("sklearn")
pytest.importorskip("torch_geometric")
from sklearn.metrics import average_precision_score

from sgdiff.data.datasets import make_synthetic
from sgdiff.data.graph import GraphData
from sgdiff.data.splits import make_link_split
from sgdiff.data.observation import Condition, degrade
from sgdiff.data.subgraph import extract_enclosing_subgraph
from sgdiff.engine.observation_robustness import heuristic_scores
from sgdiff.models.baselines.sp4lp import SP4LPBaseline, path_store
from sgdiff.engine.baseline_trainer import _edge_index
from sgdiff.utils.metrics import average_precision, hit_at_k


def test_tied_ap_matches_sklearn_and_ignores_pair_layout():
    rng = np.random.default_rng(17)
    labels = np.r_[np.ones(30), np.zeros(30)]
    for scores in (np.zeros(60), rng.integers(0, 4, 60)):
        expected = average_precision_score(labels, scores)
        for _ in range(10):
            order = rng.permutation(60)
            assert np.isclose(average_precision(labels[order], scores[order]), expected)
    assert average_precision(labels, np.zeros(60)) == .5
    # Upstream retains input-order tie breaking for its legacy pooled hit@k.
    # This contribution leaves that shared metric unchanged; AUROC/AP are primary.


def test_negatives_are_disjoint_and_never_positive():
    graph = make_synthetic(num_nodes=100, num_features=8)
    split = make_link_split(graph)
    sets = [{tuple(p) for p in getattr(split, f"{s}_neg")} for s in ("train","val","test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert not set.union(*sets) & graph.edge_set()


def test_deletions_are_matched_nested_and_feature_scores_invariant():
    graph = make_synthetic(num_nodes=100, num_features=8)
    pairs = np.asarray([[0,1],[10,11],[20,21]])
    clean = heuristic_scores(graph, pairs)
    for mechanism in ("random_delete", "degree_delete"):
        low, a = degrade(graph, Condition("low", mechanism, .2), 44)
        high, b = degrade(graph, Condition("high", mechanism, .4), 44)
        assert a["deleted"] == round(.2*graph.num_edges)
        assert b["deleted"] == round(.4*graph.num_edges)
        assert high.edge_set() <= low.edge_set() <= graph.edge_set()
        np.testing.assert_array_equal(clean["feature_dot"], heuristic_scores(high,pairs)["feature_dot"])


def test_flip_budget_cannot_reveal_queries():
    graph = GraphData("toy", 6, np.array([[0,1],[1,2],[2,3],[3,4]]),
                      np.ones((6,2)), np.array([0,0,0,1,1,1]))
    forbidden = [[0,5],[1,4],[2,4]]
    for mechanism in ("random_flip", "dice"):
        out, meta = degrade(graph, Condition("attack",mechanism,1.), 3, forbidden)
        assert len(out.edge_set() ^ graph.edge_set()) == meta["budget"]
        assert not out.edge_set() & {tuple(p) for p in forbidden}


def test_drnl_uses_distances_beyond_extraction_radius():
    # u--w--v with a branch attached only to u. w must have DRNL 2,
    # endpoints 1, branch unreachable from v when u is removed -> 0.
    g = GraphData("toy",4,np.array([[0,2],[2,1],[0,3]]))
    sub = extract_enclosing_subgraph(g.neighbors(),0,1,hops=1)
    labels = dict(zip(sub.nodes,sub.labels))
    assert labels == {0:1,1:1,2:2,3:0}


def test_sp4lp_hides_query_edge_and_is_symmetric():
    torch.set_num_threads(2)
    g = GraphData("toy",4,np.array([[0,1],[0,2],[1,2],[2,3]]),np.arange(12,dtype=np.float32).reshape(4,3))
    pairs = np.asarray([[0,1],[1,0],[0,3]])
    store = path_store(g,pairs)
    flat,lengths,reach = store.gather(pairs)
    assert lengths[0] == 3 and reach.all() # direct target edge is hidden
    model = SP4LPBaseline(3,hidden_dim=16).eval()
    model.store = store
    with torch.no_grad():
        z = model.encode(torch.from_numpy(g.x),_edge_index(g,"cpu"),4)
        scores = model.score_pairs(z,torch.from_numpy(pairs))
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores[0],scores[1],atol=1e-6)


def test_validation_context_hides_all_validation_targets():
    from sgdiff.config import load_config
    from sgdiff.engine.factory import prepare_datasets
    graph=make_synthetic(num_nodes=100,num_features=8)
    split=make_link_split(graph)
    datasets=prepare_datasets(graph,split,load_config())
    context=datasets["val"].extractor.graph.edge_set()
    assert not context & {tuple(p) for p in split.val_pos}
    assert not context & {tuple(p) for p in split.test_pos}


def test_exploratory_isolation_control_matches_budget_without_new_isolates():
    from scripts.run_isolation_control import without_new_isolates
    graph=GraphData("toy",6,np.asarray([[0,1],[0,2],[1,2],[1,3],[2,3],[3,4]]))
    out=without_new_isolates(graph,.5,17)
    assert graph.num_edges-out.num_edges==3
    np.testing.assert_array_equal(graph.degrees()==0,out.degrees()==0)


def test_legacy_dice_uses_classes_even_when_graph_is_connected():
    from sgdiff.data.attack import dice_attack
    graph=GraphData("connected",8,np.asarray([
        [0,1],[1,2],[2,3],[0,3],[4,5],[5,6],[6,7],[4,7],[3,4]]),
        y=np.asarray([0,0,0,0,1,1,1,1]))
    out=dice_attack(graph,ratio=.5,seed=2)
    added=out.edge_set()-graph.edge_set()
    deleted=graph.edge_set()-out.edge_set()
    assert len(added)+len(deleted)==round(.5*graph.num_edges)
    assert added and all(graph.y[u]!=graph.y[v] for u,v in added)
    assert all(graph.y[u]==graph.y[v] for u,v in deleted)
