"""Pinned author PubMed recipes, isolated from the earlier Cora/Citeseer runtime."""
import sys

import torch
from torch import nn

from experiments.new_link_models.adapters import NCNC, LPFormer, RefinedGAE, ExactGraphConv, _MODULES, _load, ppr_matrix, sparse_context
from experiments.new_link_models.protocol import digest
from experiments.new_link_models.sources import verify


def pubmed_settings(model):
    common = dict(batch_size=512, epochs=100, decay=1., weight_decay=0.)
    if model == 'ncnc':
        return dict(common, hidden=256, dropout=.05, lr=.0097, predictor_lr=.002,
                    author_batch_size=2048, predictor_edge_dropout=0.,
                    train_completion_degree=-1, test_completion_degree=-1,
                    author_recipe='ncn/README.md Pubmed: xdp.3 gnndp.1 pt.5 scale5.3 offset.5 alpha.3')
    if model == 'lpformer':
        return dict(common, hidden=128, dropout=.3, lr=.001, author_batch_size=1024,
                    ppr_alpha=.15, ppr_eps=1e-5, weight_decay_author=.0001,
                    author_recipe='lpformer/scripts/replicate_existing.sh Pubmed') | {'weight_decay': .0001}
    if model == 'refined_gae':
        return dict(common, hidden=512, dropout=.4, lr=.001, author_batch_size=4096,
                    propagation_steps=2, activation='gelu', exp_initial_weights=True,
                    author_recipe='refined_gae/ReadMe.md PubMed')
    raise ValueError(model)


class PubmedNCNC(NCNC):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        nn.Module.__init__(self)
        source = verify(source_root, 'ncn')
        if 'ncn_model' not in _MODULES:
            original = sys.modules.get('utils')
            sys.modules['utils'] = _load('ncn_utils', source / 'utils.py')
            try:
                _load('ncn_model', source / 'model.py')
            finally:
                if original is None:
                    sys.modules.pop('utils', None)
                else:
                    sys.modules['utils'] = original
        module = _MODULES['ncn_model']
        self.encoder = module.GCN(feature_dim, 256, 256, 1, .1, ln=True, conv_fn='puregcn',
                                  jk=True, xdropout=.3, taildropout=0.)
        options = dict(ln=True, use_xlin=True, tailact=True, twolayerlin=False, edrop=0.)
        self.predictor = module.CN0LinkPredictor(256, 256, 1, 3, .05, **options) if variant == 'no_cn' else (
            module.IncompleteCN1Predictor(256, 256, 1, 3, .05, alpha=.3, scale=5.3, offset=.5,
                pt=.5, depth=0 if variant == 'no_completion' else 1,
                trainresdeg=-1, testresdeg=-1, **options))
        self.device = torch.device(device)
        self.to(device)


class PubmedLPFormer(LPFormer):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        nn.Module.__init__(self)
        self.source = verify(source_root, 'lpformer')
        src = str(self.source / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from models.link_transformer import LinkTransformer
        from models.other_models import mlp_score
        self.device, self.cache = torch.device(device), cache
        args = dict(dim=128, num_heads=1, gnn_layers=1, trans_layers=1, thresh_cn=0.,
                    thresh_1hop=.01, thresh_non1hop=.01, gnn_drop=.3, dropout=.3,
                    att_drop=.3, feat_drop=.3, residual=False, layer_norm=True, relu=True, gcn_cache=False)
        self.core = LinkTransformer(args, {'x': torch.empty(1, feature_dim)}, device=self.device)
        self.predictor = mlp_score(self.core.out_dim, self.core.out_dim, 1, 3, dropout=.3)
        self.variant = variant
        if variant == 'uniform_attention':
            for layer in self.core.att_layers:
                nn.init.zeros_(layer.att.att)
                layer.att.att.requires_grad_(False)
        if variant == 'no_ppr_encoding':
            original = self.core.get_pos_encodings
            self.core.get_pos_encodings = lambda *a, **kw: original(*a, **kw) * 0
        self.to(device)

    def context(self, graph):
        context = sparse_context(graph, self.device)
        context['ppr'] = ppr_matrix(graph, self.source, self.cache, eps=1e-5).to(self.device)
        context['sha256'] = digest(graph.edges)
        return context


class PubmedRefinedGAE(RefinedGAE):
    def __init__(self, source_root, dataset, feature_dim, variant, device, cache):
        nn.Module.__init__(self)
        self.source = verify(source_root, 'refined_gae')
        module = _load('refined_author_model', self.source / 'model.py')
        self.device = torch.device(device)
        self.encoder = module.LightGCN(feature_dim, 512, prop_step=2, dropout=.4, exp=True, norm=True)
        self.encoder.conv1 = ExactGraphConv(self.encoder.conv1)
        self.encoder.conv2 = ExactGraphConv(self.encoder.conv2)
        self.predictor = module.DotPredictor() if variant == 'dot_decoder' else module.Hadamard_MLPPredictor(
            512, .4, layer=2, res=False, norm=True, act='gelu')
        if variant == 'no_norm':
            self.encoder.norms = nn.ModuleList(nn.Identity() for _ in self.encoder.norms)
            if getattr(self.predictor, 'norm', False):
                self.predictor.norms = nn.ModuleList(nn.Identity() for _ in self.predictor.norms)
        self.to(device)


PUBMED_ADAPTERS = {'ncnc': PubmedNCNC, 'lpformer': PubmedLPFormer, 'refined_gae': PubmedRefinedGAE}
