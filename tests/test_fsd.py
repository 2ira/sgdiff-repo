import unittest
from unittest.mock import patch

import numpy as np
import torch

from sgdiff.data.datasets import make_synthetic
from sgdiff.data.feature_mask import observed_nodes
from sgdiff.data.splits import make_link_split
from sgdiff.engine.baseline_trainer import evaluate_node_pair_baseline, train_node_pair_baseline
from sgdiff.models.baselines.fsd import FSDGAE, propagate_features


class TestFSD(unittest.TestCase):
    def test_sparse_matches_dense_author_stage1_operations(self):
        # Direct dense transcription of the author's structural stage1 update.
        a = np.zeros((6, 6), dtype=np.float32)
        for u, v in ((0, 1), (1, 2), (1, 3), (3, 4), (4, 5)):
            a[u, v] = a[v, u] = 1
        x = np.arange(12, dtype=np.float32).reshape(6, 2) + 1
        observed = np.array([True, False, False, False, True, False])
        out = x.copy()
        out[~observed] = 0
        previous, visited = out.copy(), observed.copy()
        hop, old_unknown = 1, -1
        while True:
            selected = visited.copy()
            for _ in range(hop):
                selected |= a @ selected.astype(np.float32) > 0
            unknown = int((~visited).sum())
            if unknown == old_unknown:
                break
            old_unknown = unknown
            sub = a * selected[:, None] * selected[None, :]
            degree = sub.sum(1)
            inv = np.zeros_like(degree)
            inv[degree > 0] = degree[degree > 0] ** -0.5
            weights = (inv[:, None] * sub * inv[None, :]) ** 1.2
            rowsum = weights.sum(1, keepdims=True)
            np.divide(weights, rowsum, out=weights, where=rowsum > 0)
            for _ in range(4):
                out = weights @ out
                out[selected] += 0.1 * previous[selected]
                out[observed] = x[observed]
            previous[selected] = out[selected]
            previous[selected] /= np.linalg.norm(previous[selected])
            visited, hop = selected, hop + 1
        actual = propagate_features(x, np.array(a.nonzero()), observed, iterations=4)
        np.testing.assert_allclose(actual, out, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(actual[observed], x[observed])

    def test_complete_features_are_identity_for_all_components(self):
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        edges = np.array([[0, 1], [1, 0]])
        for method in ("none", "fp", "fsd"):
            np.testing.assert_array_equal(
                propagate_features(x, edges, np.ones(6, dtype=bool), method), x)

    def test_mask_is_fixed_without_changing_global_rng(self):
        np.random.seed(3)
        expected = np.random.random()
        np.random.seed(3)
        mask = observed_nodes(100, 0.4, 7)
        self.assertEqual(mask.sum(), 60)
        np.testing.assert_array_equal(mask, observed_nodes(100, 0.4, 7))
        self.assertEqual(np.random.random(), expected)

    def test_validation_does_not_see_validation_or_test_edges(self):
        graph = make_synthetic(num_nodes=40, num_features=4, seed=4)
        split = make_link_split(graph, seed=4)
        model = FSDGAE(4)
        contexts = []

        def evaluate(*args):
            edge_index = args[2].cpu().numpy().T
            contexts.append({tuple(sorted(e)) for e in edge_index})
            return evaluate_node_pair_baseline(*args)

        with patch("sgdiff.engine.baseline_trainer.evaluate_node_pair_baseline", side_effect=evaluate):
            train_node_pair_baseline(model, graph, split, epochs=10, device="cpu")
        val = {tuple(e) for e in split.val_pos}
        test = {tuple(e) for e in split.test_pos}
        self.assertFalse(contexts[0] & (val | test))
        self.assertFalse(contexts[-1] & test)
        self.assertTrue(val <= contexts[-1])


if __name__ == "__main__":
    unittest.main()
