"""Course heuristics and SP4LP on the benchmark's actual shared query arrays."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline_trainer import _edge_index, node_feature_matrix
from ..utils.metrics import summarize_scores
from ..utils.seed import resolve_device
from ..models.baselines.spectral_walk import (
    KATZ_CONFIGS,
    SPECTRAL_WALK_MODELS,
    katz_scores,
    random_walk_scores,
    spectral_scores,
)

CLASSICAL = ('cn', 'aa', 'pa', 'feature_dot', 'feature_cosine', 'fusion_lr') + SPECTRAL_WALK_MODELS
PATH_MODELS = ('sp4lp', 'sp4lp_endpoints', 'sp4lp_featureless')


def heuristic_scores(graph, pairs, remove_target=False):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    u, v = pairs.T
    a = graph.adjacency()
    degrees = graph.degrees().astype(float)
    overlap = a[u].multiply(a[v])
    cn = np.asarray(overlap.sum(1)).ravel()
    weights = np.zeros(graph.num_nodes)
    weights[degrees > 1] = 1 / np.log(degrees[degrees > 1])
    aa = np.asarray(overlap.multiply(weights).sum(1)).ravel()
    present = np.asarray(a[u, v]).ravel() if remove_target else 0
    pa = (degrees[u]-present) * (degrees[v]-present)
    if graph.has_features:
        x = graph.x
        dot = np.einsum('ij,ij->i', x[u], x[v])
        norm = np.linalg.norm(x[u], axis=1) * np.linalg.norm(x[v], axis=1)
        cosine = dot / np.maximum(norm, 1e-12)
    else:
        dot = cosine = np.zeros(len(pairs))
    return dict(cn=cn, aa=aa, pa=pa, feature_dot=dot, feature_cosine=cosine)


def fusion_inputs(scores):
    dot = scores['feature_dot']
    return np.column_stack([np.log1p(scores[k]) for k in ('cn','aa','pa')] +
                           [scores['feature_cosine'], np.sign(dot)*np.log1p(np.abs(dot))])


class ClassicalBaseline(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.scaler = self.regressor = None
        self.seed = 0
        self.embedding_cache = {}
        self.learned_parameters = 0

    def fit(self, graph, pairs, labels, seed=0):
        self.seed = int(seed)
        if self.name == 'fusion_lr':
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            values = fusion_inputs(heuristic_scores(graph, pairs, remove_target=True))
            self.scaler = StandardScaler().fit(values)
            self.regressor = LogisticRegression(C=1., max_iter=1000, random_state=0).fit(
                self.scaler.transform(values), labels)
        return self

    def scores(self, graph, pairs):
        if self.name in KATZ_CONFIGS:
            max_walk_length, spectral_scale = KATZ_CONFIGS[self.name]
            return katz_scores(
                graph,
                pairs,
                max_walk_length=max_walk_length,
                spectral_scale=spectral_scale,
            )
        if self.name == 'spectral':
            return spectral_scores(graph, pairs)
        if self.name in ('deepwalk', 'node2vec'):
            self.learned_parameters = graph.num_nodes * 64
            return random_walk_scores(
                graph, pairs, self.name, seed=self.seed, cache=self.embedding_cache,
            )
        values = heuristic_scores(graph, pairs)
        if self.name == 'fusion_lr':
            return self.regressor.decision_function(self.scaler.transform(fusion_inputs(values)))
        return values[self.name]


def build_course_model(name, feature_dim):
    if name in CLASSICAL:
        return ClassicalBaseline(name)
    # Optional PyG dependency is imported only when a path model is requested.
    from ..models.baselines.sp4lp import SP4LPBaseline
    return SP4LPBaseline(feature_dim, endpoints_only=name == 'sp4lp_endpoints',
                         featureless=name == 'sp4lp_featureless')


@torch.no_grad()
def score_course_model(model, graph, pairs, features=None, device='cpu'):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty(0, dtype=np.float32)
    if isinstance(model, ClassicalBaseline):
        return model.scores(graph, pairs)
    model.eval()
    device = next(model.parameters()).device
    model.set_context(graph, pairs)  # Always recompute paths on THIS graph.
    x = torch.as_tensor(node_feature_matrix(graph) if features is None else features,
                        dtype=torch.float32, device=device)
    z = model.encode(x, _edge_index(graph, device), graph.num_nodes)
    return np.concatenate([model.score_pairs(z, torch.as_tensor(pairs[i:i+512],
        device=device)).cpu().numpy() for i in range(0, len(pairs), 512)])


def fit_course_model(spec, cfg, graph, datasets, seed, epochs, device):
    torch.manual_seed(seed)
    train = datasets['train']
    context = train.extractor.graph
    features = node_feature_matrix(context)
    model = build_course_model(spec.name, features.shape[1])
    if isinstance(model, ClassicalBaseline):
        model.fit(context, train.pairs, train.labels, seed=seed)
    else:
        model.to(resolve_device(str(device)))
        dev = next(model.parameters()).device
        x = torch.as_tensor(features, device=dev)
        edges = _edge_index(context, dev)
        model.set_context(context, train.pairs)
        train_store = model.store
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        best, state, bad = -np.inf, None, 0
        for epoch in range(max(epochs, 1)):
            model.train()
            model.store = train_store
            opt.zero_grad()
            z = model.encode(x, edges, context.num_nodes)
            order = np.random.default_rng(seed+epoch).permutation(len(train))
            for start in range(0,len(order),512):
                idx = order[start:start+512]
                logits = model.score_pairs(z, torch.as_tensor(train.pairs[idx], device=dev))
                loss = F.binary_cross_entropy_with_logits(logits,
                    torch.as_tensor(train.labels[idx], dtype=torch.float32, device=dev))
                (loss * len(idx)/len(train)).backward(retain_graph=start+512 < len(order))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            if (epoch+1)%5 == 0 or epoch+1 == epochs:
                val = datasets['val']
                scores = score_course_model(model,val.extractor.graph,val.pairs,
                                             node_feature_matrix(val.extractor.graph))
                auc = summarize_scores(val.labels,scores)['auc']
                if auc > best:
                    best, bad, state = auc, 0, copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                if bad >= 6:
                    break
        if state is not None:
            model.load_state_dict(state)
    test = datasets['test']
    scores = score_course_model(model,test.extractor.graph,test.pairs,
                               node_feature_matrix(test.extractor.graph))
    metrics = summarize_scores(test.labels,scores)
    count = sum(p.numel() for p in model.parameters())
    if isinstance(model, ClassicalBaseline) and model.regressor is not None:
        count = model.regressor.coef_.size + model.regressor.intercept_.size
    elif isinstance(model, ClassicalBaseline):
        count = model.learned_parameters
    return model, metrics, count
