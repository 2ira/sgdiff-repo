"""Per-query scores shared by the historical attack harness and the team study."""
import numpy as np
import torch

from .batch import PairSubgraphDataset
from .baseline_trainer import _edge_index
from .course_baselines import score_course_model
from .evaluator import _evaluation_context, _loader, score_pairs
from .factory import build_extractor
from ..models.baselines.fsd import propagate_features

@torch.no_grad()
def sgdiff_streams(sg, extractor, cohorts, batch_size, device, seed):
    """Score both ablations from the same MC draws as their full prediction."""
    parts = {key: [] for key in ("sgdiff", "sgdiff_structure", "sgdiff_feature")}
    for pairs in cohorts:
        ds = PairSubgraphDataset(pairs, np.zeros(len(pairs), dtype=np.int64), extractor)
        # Native RQ3 resets the scoring RNG once per cohort (score_pairs).
        with _evaluation_context(sg, seed):
            for batch in _loader(ds, batch_size):
                out = sg(batch.to(device))
                feat, struct = out["feature_scores"], out["structure_scores"]
                logits = (out["logits"], sg.fusion(torch.zeros_like(feat), struct),
                          sg.fusion(feat, torch.zeros_like(struct)))
                for key, value in zip(parts, logits):
                    parts[key].append((value[:, 1] - value[:, 0]).cpu().numpy())
    return {key: np.concatenate(values) for key, values in parts.items()}


@torch.no_grad()
def frozen_scores(name, model, cfg, graph, pairs, features, observed, device, seed,
                  cohort_boundary=None):
    """Return aligned 1-D arrays. The caller supplies one frozen graph/feature view."""
    model.eval()
    if name in ("sgdiff", "pencil"):
        extractor = build_extractor(graph, cfg, seed=cfg.data.seed + 1)
        bs = cfg.train.eval_batch_size
        if name == "pencil":
            return {name: score_pairs(model, extractor, pairs, bs, device, seed=seed)}
        cohorts = (pairs,) if cohort_boundary is None else (pairs[:cohort_boundary], pairs[cohort_boundary:])
        return sgdiff_streams(model, extractor, cohorts, bs, device, seed)
    if name in ("gae", "fsd"):
        edges = _edge_index(graph, device)
        if name == "fsd" and not observed.all():
            features = propagate_features(features, edges.cpu().numpy(), observed, method="fsd", gamma=1.2)
        x = torch.as_tensor(features, device=device)
        z = model.encoder(x, edges, len(x))
        return {name: model.score_pairs(z, torch.as_tensor(pairs, device=device)).cpu().numpy()}
    return {name: score_course_model(model, graph, pairs, features, device)}
