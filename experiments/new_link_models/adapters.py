"""Thin adapters around verified, unmodified author model classes."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import digest, cache_lock, save_npz
from .sources import verify

_MODULES = {}
VARIANTS = {'ncnc': ['full', 'no_completion', 'no_cn'],
            'lpformer': ['full', 'uniform_attention', 'no_ppr_encoding'],
            'refined_gae': ['full', 'no_norm', 'dot_decoder']}


def _load(name, file):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def settings(model, dataset):
    if model == 'ncnc':
        return dict(hidden=256, dropout=.05 if dataset == 'cora' else .55, lr=.0043 if dataset == 'cora' else .0085,
                    predictor_lr=.0024 if dataset == 'cora' else .0078, batch_size=512,
                    author_batch_size=1152 if dataset == 'cora' else 384,
                    weight_decay=0., decay=1., epochs=100,
                    predictor_edge_dropout=.4 if dataset == 'cora' else 0.,
                    train_completion_degree=-1, test_completion_degree=-1)
    if model == 'lpformer':
        return dict(hidden=128 if dataset == 'cora' else 256, dropout=.1, lr=.005, batch_size=512, author_batch_size=1024,
                    weight_decay=0., decay=.975 if dataset == 'cora' else .95, epochs=100,
                    ppr_alpha=.15, ppr_eps=1e-7)
    if model == 'refined_gae':
        return dict(hidden=1024, dropout=.6, lr=.005 if dataset == 'cora' else .001,
                    batch_size=512, author_batch_size=2048 if dataset == 'cora' else 4096,
                    weight_decay=.01 if dataset == 'cora' else 0.,
                    decay=1., epochs=100)
    raise ValueError(model)


def sparse_context(graph, device):
    from torch_sparse import SparseTensor
    edges = torch.tensor(graph.edges, device=device)
    ei = torch.cat([edges.T, edges.T.flip(0)], dim=1)
    adj = SparseTensor.from_edge_index(ei, sparse_sizes=(graph.num_nodes, graph.num_nodes)).to_symmetric()
    coo = adj.to_torch_sparse_coo_tensor().coalesce()
    return {'graph': graph, 'adj': adj, 'mask': coo.bool().int(), 'x': torch.tensor(graph.x, device=device)}


def ppr_matrix(graph, source, cache, alpha=.15, eps=1e-7):
    """Reference-equivalent Andersen push, cached by exact graph hash."""
    sha = digest(graph.edges)
    path = Path(cache) / f'ppr_{sha}_a{alpha}_e{eps}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache_lock(path):
        if not path.exists():
            from .ppr_backend import calc_ppr
            a = graph.adjacency()
            neighbors, weights = calc_ppr(a.indptr.astype(np.int64), a.indices.astype(np.int64),
                                          np.diff(a.indptr).astype(np.int64), alpha, eps)
            rows = np.repeat(np.arange(graph.num_nodes), [len(v) for v in neighbors])
            cols = np.concatenate([np.asarray(v, dtype=np.int64) for v in neighbors])
            values = np.concatenate([np.asarray(v, dtype=np.float32) for v in weights])
            save_npz(path, row=rows, col=cols, value=values, graph_sha256=sha)
    with np.load(path) as f:
        if str(f['graph_sha256']) != sha:
            raise RuntimeError('PPR cache graph mismatch')
        return torch.sparse_coo_tensor(torch.tensor(np.vstack([f['row'], f['col']])),
                                       torch.tensor(f['value']), (graph.num_nodes, graph.num_nodes)).coalesce()


class NCNC(nn.Module):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        super().__init__()
        source = verify(source_root, 'ncn')
        if 'ncn_model' not in _MODULES:
            original = sys.modules.get('utils')
            sys.modules['utils'] = _load('ncn_utils', source / 'utils.py')
            try:
                module = _load('ncn_model', source / 'model.py')
            finally:
                if original is None:
                    sys.modules.pop('utils', None)
                else:
                    sys.modules['utils'] = original
        module = _MODULES['ncn_model']
        cfg = settings('ncnc', dataset)
        h = cfg['hidden']
        self.encoder = module.GCN(feature_dim, h, h, 1, .05 if dataset == 'cora' else .75,
                                  ln=True, conv_fn='puregcn', jk=True, xdropout=.7 if dataset == 'cora' else .4,
                                  taildropout=.3 if dataset == 'cora' else 0.)
        options = dict(ln=True, use_xlin=True, tailact=True, twolayerlin=dataset != 'cora',
                       edrop=cfg['predictor_edge_dropout'])
        if variant == 'no_cn':
            self.predictor = module.CN0LinkPredictor(h, h, 1, 3, cfg['dropout'], **options)
        else:
            self.predictor = module.IncompleteCN1Predictor(
                h, h, 1, 3, cfg['dropout'], alpha=1. if dataset == 'cora' else .4,
                scale=4.3 if dataset == 'cora' else 6.5, offset=2.8 if dataset == 'cora' else 4.4,
                pt=.75, depth=0 if variant == 'no_completion' else 1,
                trainresdeg=cfg['train_completion_degree'], testresdeg=cfg['test_completion_degree'], **options)
        self.device = torch.device(device)
        self.to(device)

    def context(self, graph):
        return sparse_context(graph, self.device)

    def score(self, ctx, pairs):
        h = self.encoder(ctx['x'], ctx['adj'])
        return self.predictor(h, ctx['adj'], pairs.T).reshape(-1)

    def parameter_groups(self, cfg):
        return [{'params': self.encoder.parameters(), 'lr': cfg['lr']},
                {'params': self.predictor.parameters(), 'lr': cfg['predictor_lr']}]


class LPFormer(nn.Module):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        super().__init__()
        self.source = verify(source_root, 'lpformer')
        src = str(self.source / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from models.link_transformer import LinkTransformer
        from models.other_models import mlp_score
        cfg = settings('lpformer', dataset)
        self.device, self.cache = torch.device(device), cache
        args = dict(dim=cfg['hidden'], num_heads=1, gnn_layers=1, trans_layers=1,
                    thresh_cn=0., thresh_1hop=.01, thresh_non1hop=.01 if dataset == 'cora' else 1.,
                    gnn_drop=.1, dropout=.1, att_drop=.1, feat_drop=.1,
                    residual=False, layer_norm=True, relu=True, gcn_cache=False)
        self.core = LinkTransformer(args, {'x': torch.empty(1, feature_dim)}, device=self.device)
        self.predictor = mlp_score(self.core.out_dim, self.core.out_dim, 1, 3, dropout=.1)
        self.variant = variant
        if variant == 'uniform_attention':
            for layer in self.core.att_layers:
                nn.init.zeros_(layer.att.att)
                layer.att.att.requires_grad_(False)
        if variant == 'no_ppr_encoding':
            # Keep the graph-derived candidate mask unchanged; ablate only PE values.
            original = self.core.get_pos_encodings
            self.core.get_pos_encodings = lambda *a, **kw: original(*a, **kw) * 0
        self.to(device)

    def context(self, graph):
        ctx = sparse_context(graph, self.device)
        ctx['ppr'] = ppr_matrix(graph, self.source, self.cache).to(self.device)
        ctx['sha256'] = digest(graph.edges)
        return ctx

    def score(self, ctx, pairs, propagation=None):
        data = dict(x=ctx['x'], adj_t=ctx['adj'], adj_mask=ctx['mask'], ppr=ctx['ppr'],
                    degree=torch.from_numpy(ctx['graph'].degrees()).to(self.device))
        self.core.data = data
        h = self.core(pairs.T, adj_prop=None if propagation is None else propagation['adj'],
                      adj_mask=None if propagation is None else propagation['mask'])
        # Reuse author head parameters; expose pre-sigmoid logits for stable BCE.
        for lin in self.predictor.lins[:-1]:
            h = F.dropout(F.relu(lin(h)), p=self.predictor.dropout, training=self.training)
        return self.predictor.lins[-1](h).reshape(-1)

    def parameter_groups(self, cfg):
        return [{'params': self.parameters(), 'lr': cfg['lr']}]


class ExactGraphConv(nn.Module):
    """DGL GraphConv(norm='both') for a pre-normalized sparse adjacency.

    Reuses the author's actual parameters and projection/aggregation order.
    The adapter is checked against the CPU DGL operator for outputs/gradients;
    it lets the unchanged author LightGCN run on CUDA without a DGL CUDA wheel.
    """
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.out_features = original._out_feats

    def forward(self, normalized_adjacency, features, edge_weight=None):
        if edge_weight is not None:
            raise ValueError('The shared protocol uses unweighted graphs')
        if features.shape[-1] > self.out_features and self.weight is not None:
            out = torch.sparse.mm(normalized_adjacency, features @ self.weight)
        else:
            out = torch.sparse.mm(normalized_adjacency, features)
            if self.weight is not None:
                out = out @ self.weight
        return out if self.bias is None else out + self.bias


class RefinedGAE(nn.Module):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        super().__init__()
        self.source = verify(source_root, 'refined_gae')
        module = _load('refined_author_model', self.source / 'model.py')
        cfg = settings('refined_gae', dataset)
        self.device = torch.device(device)
        self.encoder = module.LightGCN(feature_dim, cfg['hidden'], prop_step=4, dropout=.6, norm=True)
        self.encoder.conv1 = ExactGraphConv(self.encoder.conv1)
        self.encoder.conv2 = ExactGraphConv(self.encoder.conv2)
        self.predictor = module.DotPredictor() if variant == 'dot_decoder' else module.Hadamard_MLPPredictor(
            cfg['hidden'], .6, layer=4 if dataset == 'cora' else 2, res=dataset == 'cora', norm=True,
            act='silu' if dataset == 'cora' else 'relu')
        if variant == 'no_norm':
            # Preserve dropout, which the author encoder otherwise couples to norm=True.
            self.encoder.norms = nn.ModuleList(nn.Identity() for _ in self.encoder.norms)
            if getattr(self.predictor, 'norm', False):
                self.predictor.norms = nn.ModuleList(nn.Identity() for _ in self.predictor.norms)
        self.to(device)

    def context(self, graph):
        e = torch.tensor(graph.edges, device=self.device)
        loops = torch.arange(graph.num_nodes, device=self.device)
        row = torch.cat([e[:, 0], e[:, 1], loops])
        col = torch.cat([e[:, 1], e[:, 0], loops])
        deg = torch.bincount(row, minlength=graph.num_nodes).float()
        weights = deg[row].rsqrt() * deg[col].rsqrt()
        normalized = torch.sparse_coo_tensor(torch.stack([row, col]), weights,
                                             (graph.num_nodes, graph.num_nodes)).coalesce()
        return dict(graph=graph, adjacency=normalized, x=torch.tensor(graph.x, device=self.device))

    def score(self, ctx, pairs):
        h = self.encoder(ctx['adjacency'], ctx['x'])
        return self.predictor(h[pairs[:, 0]], h[pairs[:, 1]]).reshape(-1)

    def parameter_groups(self, cfg):
        return [{'params': self.parameters(), 'lr': cfg['lr']}]


ADAPTERS = {'ncnc': NCNC, 'lpformer': LPFormer, 'refined_gae': RefinedGAE}
