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
