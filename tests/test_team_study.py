import numpy as np

from sgdiff.data.datasets import make_synthetic
from sgdiff.data.observation import degrade, masked_node_ids
from sgdiff.engine.team_study import COMPONENTS, PRIMARY, study_plan


def test_study_shared_budget_and_mask():
    graph = make_synthetic(num_nodes=150, seed=0)
    forbidden = graph.edges[:10]
    clean = graph.drop_edges(forbidden)
    plan = {c.name: c for c in study_plan()}
    assert len(plan) == 15 and len(PRIMARY) == 8 and set(COMPONENTS) <= set(PRIMARY)
    for pct in (25, 50):
        rate = pct / 100
        for mechanism in ("degree_delete", "random_delete", "dice", "random_flip"):
            g, info = degrade(clean, plan[f"{mechanism}_{pct}"], 0, forbidden)
            assert not set(map(tuple, forbidden)).intersection(g.edge_set())
            assert len(g.edge_set() ^ clean.edge_set()) == round(rate * clean.num_edges)
            assert info["inserted"] + info["deleted"] == info["budget"]
            if mechanism.endswith("_delete"):
                joint, _ = degrade(clean, plan[f"joint_{mechanism}_{pct}"], 0, forbidden)
                np.testing.assert_array_equal(g.edges, joint.edges)
                idx = masked_node_ids(clean.num_nodes, rate, 0)
                np.testing.assert_array_equal(joint.x[idx], 0)
                kept = np.delete(np.arange(clean.num_nodes), idx)
                np.testing.assert_array_equal(joint.x[kept], clean.x[kept])


def test_reference_budget_and_candidate_sets_match_existing_team_benchmark():
    import json
    from pathlib import Path

    from sgdiff.data.splits import make_link_split
    from sgdiff.engine.factory import prepare_datasets
    from sgdiff.engine.team_study import settings, study_datasets

    root = Path(__file__).resolve().parents[1]
    cfg = settings("cora", 0, "reference", "data", "cpu")
    for folder in ("line-c-fsd/cuda", "part2-aligned/cuda-local"):
        archived = root / "results" / folder / "robustness_cora_observed_cuda.json"
        protocol = json.loads(archived.read_text())["protocol"]
        actual = cfg.to_dict()
        actual["train"]["device"] = protocol["resolved_config"]["train"]["device"]
        assert actual == protocol["resolved_config"]

    graph = make_synthetic(num_nodes=600)
    split = make_link_split(graph, seed=0)
    ds = study_datasets(graph, split, cfg, "reference")
    original = prepare_datasets(graph, split, cfg)
    for part in ("train", "val"):
        np.testing.assert_array_equal(ds[part].pairs, original[part].pairs)
        np.testing.assert_array_equal(ds[part].labels, original[part].labels)
    for label, pairs in [(1, split.test_pos), (0, split.test_neg)]:
        np.testing.assert_array_equal(ds["test"].pairs[ds["test"].labels == label], pairs[:500])
    assert len(ds["test"]) == 1000


def test_paper_rq3_configuration_and_native_cohort_scores():
    import torch
    from sgdiff.config import load_config
    from sgdiff.data.splits import make_link_split
    from sgdiff.engine.evaluator import score_pairs
    from sgdiff.engine.factory import build_model
    from sgdiff.engine.team_study import settings, sgdiff_streams, study_datasets

    cfg = settings("cora", 0, "paper-rq3", "data", "cpu")
    original = load_config("configs/sgdiff_cora.yaml")
    original.train.device = "cpu"
    assert cfg.to_dict() == original.to_dict()
    # Exercise the native positive/negative RNG boundaries with an uneven batch.
    graph = make_synthetic(num_nodes=80, num_features=8)
    split = make_link_split(graph, seed=0)
    ds = study_datasets(graph, split, cfg, "paper-rq3")
    cohorts = (split.test_pos[:5], split.test_neg[:5])
    cfg.structure.num_mc_samples = cfg.feature.num_mc_samples = 2
    torch.manual_seed(7)
    model = build_model(cfg, ds["train"].feature_dim).eval()
    view = ds["test"].extractor
    expected = np.concatenate([score_pairs(model, view, p, 3, "cpu", seed=11) for p in cohorts])
    scores = sgdiff_streams(model, view, cohorts, 3, "cpu", 11)
    np.testing.assert_array_equal(expected, scores["sgdiff"])
