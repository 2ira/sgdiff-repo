"""
sp4lp/models.py
==============================================================================
SP4LP = GNN node encoder -> sequence model over the shortest path -> MLP rho.

This follows Equation (8) of the paper,

    SP4LP((u,v), G) = rho( GNN(u), GNN(v), AGG({ phi(GNN(u_i))_i : path }) )

with phi a sequence model, AGG selecting the first BFS shortest path, and rho
an MLP.  Everything below is an instantiation of that equation -- nothing here
changes what the model is, only how expressive each component is allowed to be.

What is stronger than the previous version
------------------------------------------
* GNN backbone: optional batch norm, residual connections and jumping-knowledge
  concatenation, plus a linear input projection.  A 3-layer GCN without any of
  these is hard to train on Cora at hidden=256.
* Path tokens carry a distance encoding: each node u_i on the path gets an
  embedding of its distance to u, of its distance to v, and of the total path
  length, on top of its GNN embedding.  This is the labelling trick of Zhang
  et al. / Li et al. applied inside phi.  It costs three small embedding tables
  and strictly increases what phi can separate: without it a Transformer sees a
  bag of vectors plus absolute positions, which cannot express "this node is the
  midpoint" or "this node is adjacent to the target".
* Symmetrisation is exact.  The path is encoded forwards and backwards and the
  two encodings are summed, so the score of (u,v) equals the score of (v,u);
  the distance-to-source and distance-to-target labels swap under reversal, so
  the sum is genuinely symmetric rather than approximately so.
* rho sees the path length and a reachability flag explicitly.  The paper's own
  ablation shows path length alone is worth ~14 MRR on Cora, so it should not
  have to be recovered from the pooled sequence.
* The predictor returns LOGITS.  Training uses BCE-with-logits instead of
  log(sigmoid(x)) computed by hand, which was the previous source of gradient
  underflow once scores saturated.

Everything is opt-out: set `use_distance_encoding=False`, `gnn_residual=False`,
`gnn_bn=False`, `use_len_feature=False` to recover the plain formulation.
------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
    PYG_AVAILABLE = True
except Exception:                                        # pragma: no cover
    PYG_AVAILABLE = False


# ============================================================================ #
#  Node encoders
# ============================================================================ #
class MLPEncoder(nn.Module):
    """Graph-free backbone (HeaRT's MLP baseline; also handy without PyG)."""

    def __init__(self, in_ch, hid, out, num_layers, dropout, bn=False):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = ([in_ch, out] if num_layers == 1
                else [in_ch] + [hid] * (num_layers - 1) + [out])
        for i in range(len(dims) - 1):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))
            if bn and i < len(dims) - 2:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
            else:
                self.norms.append(nn.Identity())

    def reset_parameters(self):
        for l in self.lins:
            l.reset_parameters()

    def forward(self, x, edge_index=None):
        for lin, norm in zip(self.lins[:-1], self.norms[:-1]):
            x = F.dropout(F.relu(norm(lin(x))), p=self.dropout,
                          training=self.training)
        return self.lins[-1](x)


class GNNEncoder(nn.Module):
    """gcn / sage / gat / gin, with optional BN, residual and jumping knowledge."""

    def __init__(self, name, in_ch, hid, out, num_layers, dropout,
                 gat_heads=1, bn=False, residual=False, jk=False,
                 input_proj=True, layer_mix=False):
        super().__init__()
        assert PYG_AVAILABLE, "torch_geometric is required for GNN backbones"
        self.name = name
        self.dropout = dropout
        self.residual = residual
        self.jk = jk
        self.layer_mix = layer_mix
        self.num_layers = num_layers

        # A linear projection first means every conv works at a uniform width,
        # which is what makes residual connections and JK well defined.
        self.proj = nn.Linear(in_ch, hid) if input_proj else nn.Identity()
        conv_in = hid if input_proj else in_ch

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur = conv_in
        widths = []
        for _ in range(num_layers):
            conv, cur = self._make(name, cur, hid, gat_heads)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(cur) if bn else nn.Identity())
            widths.append(cur)

        if layer_mix and len(set(widths)) != 1:
            raise ValueError('learned layer mixing requires equal GNN widths')
        self.layer_logits = (nn.Parameter(torch.zeros(len(widths)))
                             if layer_mix else None)
        readout_in = widths[-1] if layer_mix else (
            sum(widths) if jk else widths[-1])
        self.out_lin = nn.Linear(readout_in, out)

    @staticmethod
    def _make(name, a, b, gat_heads):
        """Returns (module, actual_output_dim)."""
        if name == 'gcn':
            return GCNConv(a, b, cached=False), b
        if name == 'sage':
            return SAGEConv(a, b), b
        if name == 'gat':
            heads = max(int(gat_heads), 1)
            per_head = max(b // heads, 1)
            return GATConv(a, per_head, heads=heads, concat=True), per_head * heads
        if name == 'gin':
            mlp = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b))
            return GINConv(mlp, train_eps=True), b
        raise ValueError(f"unknown gnn '{name}'")

    def reset_parameters(self):
        for c in self.convs:
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        for m in (self.proj, self.out_lin):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x, edge_index):
        x = self.proj(x)
        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == x.shape:
                h = h + x
            x = h
            hs.append(h)
        if self.layer_mix:
            weights = torch.softmax(self.layer_logits, dim=0)
            z = sum(w * h for w, h in zip(weights, hs))
        else:
            z = torch.cat(hs, dim=-1) if self.jk else x
        return self.out_lin(z)


class PaperGNNEncoder(nn.Module):
    """GCN/SAGE/GAT/GIN stack used by HeaRT and described in the paper.

    Unlike :class:`GNNEncoder`, this has exactly ``num_layers`` message-passing
    layers: there is no extra input projection or output linear layer, and the
    final convolution is not followed by an activation/dropout.  Keeping this
    as a separate style makes comparisons with older enhanced configurations
    reproducible instead of silently changing their architecture.
    """

    def __init__(self, name, in_ch, hid, out, num_layers, dropout,
                 gat_heads=1):
        super().__init__()
        assert PYG_AVAILABLE, "torch_geometric is required for GNN backbones"
        self.dropout = dropout
        self.convs = nn.ModuleList()

        def make(a, b, last=False):
            if name == 'gcn':
                return GCNConv(a, b, cached=False)
            if name == 'sage':
                return SAGEConv(a, b)
            if name == 'gat':
                heads = 1 if last else max(int(gat_heads), 1)
                per_head = b if last else max(b // heads, 1)
                return GATConv(a, per_head, heads=heads, concat=not last)
            if name == 'gin':
                mlp = nn.Sequential(nn.Linear(a, b), nn.ReLU(),
                                    nn.Linear(b, b))
                return GINConv(mlp, train_eps=True)
            raise ValueError(f"unknown gnn '{name}'")

        if num_layers == 1:
            self.convs.append(make(in_ch, out, last=True))
        else:
            self.convs.append(make(in_ch, hid))
            for _ in range(num_layers - 2):
                self.convs.append(make(hid, hid))
            self.convs.append(make(hid, out, last=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def build_encoder(cfg, in_ch):
    name = cfg['gnn_type']
    hid = cfg['hidden_dim']
    if name in ('mlp', 'none'):
        return MLPEncoder(in_ch, hid, hid, cfg['gnn_layers'], cfg['dropout'],
                          bn=cfg.get('gnn_bn', False))
    if cfg.get('encoder_style', 'enhanced') == 'paper':
        return PaperGNNEncoder(
            name, in_ch, hid, hid, cfg['gnn_layers'], cfg['dropout'],
            gat_heads=cfg.get('gat_heads', 1))
    return GNNEncoder(
        name, in_ch, hid, hid, cfg['gnn_layers'], cfg['dropout'],
        gat_heads=cfg.get('gat_heads', 1), bn=cfg.get('gnn_bn', False),
        residual=cfg.get('gnn_residual', False), jk=cfg.get('gnn_jk', False),
        layer_mix=cfg.get('gnn_layer_mix', False))


# ============================================================================ #
#  Path tokenisation
# ============================================================================ #
class PathTokenizer(nn.Module):
    """
    Turn a ragged batch of paths into a padded token tensor.

    Input (all on device):
      h        [N, D]   node embeddings from the GNN
      nodes    [T]      concatenated path node ids
      row      [T]      which path each entry belongs to
      pos      [T]      index of the entry inside its own path
      lengths  [B]      path lengths in nodes

    Output:
      tokens   [B, L, D]  padded, distance-encoded
      pad_mask [B, L]     True where padded
    and, when `symmetrise`, the same for the reversed paths, stacked along the
    batch dimension so the sequence model runs once on [2B, L, D].
    """

    def __init__(self, dim, max_len=32, use_distance_encoding=True,
                 use_endpoint_flag=True, dropout=0.0, use_node_feats=False,
                 normalise=True):
        super().__init__()
        # Feature strutturali per nodo: grado e vicini in comune con i due
        # estremi. Cambiano lungo il cammino, quindi arricchiscono i token
        # invece di offrire uno scalare globale su cui rho possa collassare.
        self.use_nf = use_node_feats
        if use_node_feats:
            self.f_deg = nn.Embedding(64, dim)
            self.f_cnu = nn.Embedding(32, dim)
            self.f_cnv = nn.Embedding(32, dim)
            for emb in (self.f_deg, self.f_cnu, self.f_cnv):
                nn.init.normal_(emb.weight, std=0.02)
        self.dim = dim
        self.max_len = max_len
        self.use_de = use_distance_encoding
        self.use_ep = use_endpoint_flag
        if use_distance_encoding:
            self.d_src = nn.Embedding(max_len + 1, dim)
            self.d_dst = nn.Embedding(max_len + 1, dim)
            self.d_len = nn.Embedding(max_len + 1, dim)
            for emb in (self.d_src, self.d_dst, self.d_len):
                nn.init.normal_(emb.weight, std=0.02)
        if use_endpoint_flag:
            self.ep = nn.Embedding(2, dim)
            nn.init.normal_(self.ep.weight, std=0.02)
        self.norm = nn.LayerNorm(dim) if normalise else nn.Identity()
        self.drop = nn.Dropout(dropout) if normalise else nn.Identity()

    def forward(self, h, nodes, row, pos, lengths, symmetrise=True, feat=None):
        B = lengths.size(0)
        L = int(lengths.max().item()) if B else 1
        D, cl = self.dim, self.max_len

        emb = h.index_select(0, nodes)                       # [T, D]
        row_len = lengths.index_select(0, row)               # [T]
        rev_pos = row_len - 1 - pos                          # [T]

        nf = None
        if self.use_nf and feat is not None:
            # cn con u e cn con v si scambiano quando il cammino si legge al
            # contrario: il termine resta simmetrico come il resto
            nf_fwd = (self.f_deg(feat[:, 0]) + self.f_cnu(feat[:, 1])
                      + self.f_cnv(feat[:, 2]))
            nf_bwd = (self.f_deg(feat[:, 0]) + self.f_cnu(feat[:, 2])
                      + self.f_cnv(feat[:, 1]))
            nf = (nf_fwd, nf_bwd)

        def one_orientation(p, which=0):
            """p = the index each token takes in this reading direction."""
            feat_t = emb
            if nf is not None:
                feat_t = feat_t + nf[which]
            if self.use_de:
                d_from_u = p.clamp(max=cl)                   # distance to start
                d_to_v = (row_len - 1 - p).clamp(max=cl)     # distance to end
                total = (row_len - 1).clamp(max=cl)          # path length
                feat_t = (feat_t + self.d_src(d_from_u)
                          + self.d_dst(d_to_v) + self.d_len(total))
            if self.use_ep:
                is_ep = ((p == 0) | (p == row_len - 1)).long()
                feat_t = feat_t + self.ep(is_ep)
            out = h.new_zeros(B, L, D)
            out.index_put_((row, p), feat_t, accumulate=False)
            return self.drop(self.norm(out))

        toks = [one_orientation(pos, 0)]
        if symmetrise:
            toks.append(one_orientation(rev_pos, 1))

        ar = torch.arange(L, device=h.device)
        pad_mask = ar[None, :] >= lengths[:, None]
        if symmetrise:
            return torch.cat(toks, 0), torch.cat([pad_mask, pad_mask], 0)
        return toks[0], pad_mask


# ============================================================================ #
#  Sequence models (phi)
#  Interface: forward(tokens [B,L,D], pad_mask [B,L]) -> [B, D]
# ============================================================================ #
def _masked_mean(x, pad_mask):
    keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
    return (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)


def _masked_max(x, pad_mask):
    v = x.masked_fill(pad_mask.unsqueeze(-1), float('-inf')).max(1).values
    return torch.where(pad_mask.all(1, keepdim=True), torch.zeros_like(v), v)


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                   nn.Linear(dim, 1))

    def forward(self, x, pad_mask):
        s = self.score(x).squeeze(-1).masked_fill(pad_mask, float('-inf'))
        w = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
        return (x * w.unsqueeze(-1)).sum(1)


class TransformerPhi(nn.Module):
    """Pre-norm Transformer encoder with a multi-view pooled readout."""

    def __init__(self, dim, ff, layers=1, heads=4, dropout=0.0, rich_pool=True):
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation='gelu', norm_first=True, batch_first=True)
        # PyTorch 2.2's experimental nested-tensor CUDA path fails with
        # ``invalid configuration argument`` on large evaluation batches.
        # Dense padded tensors are stable and are already cheap after length
        # bucketing in engine._score_pairs.
        self.encoder = nn.TransformerEncoder(
            enc, num_layers=layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)
        self.rich_pool = rich_pool
        if rich_pool:
            self.pool = AttnPool(dim)
            self.proj = nn.Sequential(
                nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim * 2, dim))

    def forward(self, x, pad_mask):
        z = self.out_norm(self.encoder(x, src_key_padding_mask=pad_mask))
        if not self.rich_pool:
            return _masked_mean(z, pad_mask)
        B = z.size(0)
        idx = torch.arange(B, device=z.device)
        last = (~pad_mask).sum(1).clamp_min(1) - 1
        ends = 0.5 * (z[:, 0] + z[idx, last])
        return self.proj(torch.cat(
            [self.pool(z, pad_mask), _masked_mean(z, pad_mask),
             _masked_max(z, pad_mask), ends], -1))


class HistoricalRichTransformerPhi(nn.Module):
    """Rich full-path Transformer recovered from the earlier experiments.

    The cached path already contains ``u`` and ``v``.  This historical variant
    additionally prepends/appends typed endpoint tokens, adds learned absolute
    positions, and combines attentive/mean/max/endpoint pooling.  Keeping this
    behaviour behind an explicit model option makes the reproduction exact
    without changing the default SP4LP architecture.
    """

    uses_explicit_endpoints = True

    def __init__(self, dim, ff, layers=4, heads=4, dropout=0.1,
                 max_len=1000):
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.position_embedding = nn.Embedding(max_len + 2, dim)
        self.endpoint_embedding = nn.Embedding(2, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation='gelu', norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dim)
        self.pool = AttnPool(dim)
        self.summary_proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim))

    def forward(self, path_embeds, pad_mask, src_repr, dst_repr):
        batch_size, path_len, _ = path_embeds.shape
        src = (src_repr.unsqueeze(1)
               + self.endpoint_embedding.weight[0].view(1, 1, -1))
        dst = (dst_repr.unsqueeze(1)
               + self.endpoint_embedding.weight[1].view(1, 1, -1))
        sequence = torch.cat([src, path_embeds, dst], dim=1)
        endpoint_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=path_embeds.device)
        full_mask = torch.cat([endpoint_mask, pad_mask, endpoint_mask], dim=1)
        positions = torch.arange(path_len + 2, device=path_embeds.device)
        sequence = sequence + self.position_embedding(positions).unsqueeze(0)
        sequence = self.input_dropout(self.input_norm(sequence))
        encoded = self.output_norm(
            self.encoder(sequence, src_key_padding_mask=full_mask))
        ends = 0.5 * (encoded[:, 0] + encoded[:, -1])
        summary = torch.cat([
            self.pool(encoded, full_mask),
            _masked_mean(encoded, full_mask),
            _masked_max(encoded, full_mask),
            ends,
        ], dim=-1)
        return self.summary_proj(summary)


class PaperTransformerPhi(nn.Module):
    """Transformer implementation specified in the paper appendix.

    It uses learned absolute positional embeddings, PyTorch's post-norm/ReLU
    encoder layer, masked-mean pooling and a final layer norm.  The richer
    pre-norm/GELU multi-view encoder remains available as ``transformer``.
    """

    def __init__(self, dim, ff, layers=1, heads=4, dropout=0.0, max_len=32):
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation='relu', norm_first=False, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            enc, num_layers=layers, enable_nested_tensor=False)
        self.pos = nn.Embedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.max_len = max_len

    def forward(self, x, pad_mask):
        length = x.size(1)
        if length > self.max_len:
            raise ValueError(
                f'path length {length} exceeds max_path_len={self.max_len}')
        pos = torch.arange(length, device=x.device)
        z = self.encoder(x + self.pos(pos).unsqueeze(0),
                         src_key_padding_mask=pad_mask)
        return self.norm(_masked_mean(z, pad_mask))


class RNNPhi(nn.Module):
    def __init__(self, kind, dim, hidden, layers=1, bidir=False, dropout=0.0,
                 pool='last'):
        super().__init__()
        cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.kind = kind
        self.bidir = bidir
        self.pool_mode = pool
        self.rnn = cls(dim, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidir,
                       dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(out, dim) if out != dim else nn.Identity()
        if pool in ('attn', 'rich'):
            self.out_norm = nn.LayerNorm(dim)
            self.pool = AttnPool(dim)
        if pool == 'rich':
            self.pool_proj = nn.Sequential(
                nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim * 2, dim))
        elif pool not in ('last', 'attn'):
            raise ValueError(f"unknown recurrent pooling '{pool}'")

    def forward(self, x, pad_mask):
        lengths = (~pad_mask).sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, state = self.rnn(packed)
        h = state[0] if self.kind == 'lstm' else state
        h = torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1]
        if self.pool_mode == 'last':
            return self.proj(h)
        seq, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1))
        seq = self.out_norm(self.proj(seq))
        attn = self.pool(seq, pad_mask)
        if self.pool_mode == 'attn':
            return attn
        B = seq.size(0)
        idx = torch.arange(B, device=seq.device)
        last = (~pad_mask).sum(1).clamp_min(1) - 1
        ends = 0.5 * (seq[:, 0] + seq[idx, last])
        return self.pool_proj(torch.cat(
            [attn, _masked_mean(seq, pad_mask), _masked_max(seq, pad_mask),
             ends], -1))


class SumPhi(nn.Module):
    """Injective sum aggregator of Xu et al. -- the paper's third phi."""

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim))

    def forward(self, x, pad_mask):
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        return self.rho((self.phi(x) * keep).sum(1))


class MeanPhi(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x, pad_mask):
        return self.mlp(_masked_mean(x, pad_mask))


def build_phi(cfg):
    name = cfg['seq_model'].lower()
    dim = cfg['hidden_dim']
    ff = cfg.get('path_hidden_dim', dim)
    layers = cfg.get('seq_layers', 1)
    drop = cfg.get('dropout', 0.0)
    if name == 'transformer':
        return TransformerPhi(dim, ff, layers, cfg.get('seq_heads', 4), drop,
                              rich_pool=cfg.get('rich_pool', True))
    if name in ('transformer_historical', 'historical_rich'):
        return HistoricalRichTransformerPhi(
            dim, ff, layers, cfg.get('seq_heads', 4), drop,
            max_len=cfg.get('max_path_len', 1000))
    if name in ('transformer_paper', 'transformer_simple'):
        return PaperTransformerPhi(
            dim, ff, layers, cfg.get('seq_heads', 4), drop,
            max_len=cfg.get('max_path_len', 32))
    if name == 'lstm':
        return RNNPhi('lstm', dim, ff, layers, False, drop,
                      cfg.get('rnn_pool', 'last'))
    if name == 'bilstm':
        return RNNPhi('lstm', dim, ff, layers, True, drop,
                      cfg.get('rnn_pool', 'last'))
    if name == 'gru':
        return RNNPhi('gru', dim, ff, layers, False, drop,
                      cfg.get('rnn_pool', 'last'))
    if name == 'bigru':
        return RNNPhi('gru', dim, ff, layers, True, drop,
                      cfg.get('rnn_pool', 'last'))
    if name == 'sum':
        return SumPhi(dim, ff, drop)
    if name == 'mean':
        return MeanPhi(dim, ff, drop)
    raise ValueError(f"unknown sequence model '{name}'")


# ============================================================================ #
#  rho + full model
# ============================================================================ #
class Predictor(nn.Module):
    """MLP rho over the endpoint pair representation and the path encoding."""

    def __init__(self, dim, hidden, layers=2, dropout=0.0, pair_op='hadamard',
                 use_path=True, use_len_feature=True, max_len=32,
                 len_mode='embed', fusion='concat', path_aux=False,
                 branch_dropout=0.0, overlap_branch=False,
                 pair_struct=False):
        super().__init__()
        self.pair_op = pair_op
        self.use_path = use_path
        self.use_len = use_len_feature
        self.fusion = fusion
        self.path_aux = path_aux
        self.branch_dropout = float(branch_dropout)
        self.overlap_branch = bool(overlap_branch)
        self.pair_struct = bool(pair_struct)
        # Evaluation-only intervention used by the branch audit.  It is never
        # enabled during training and introduces no parameters.
        self.ablate = None
        # 'embed'  : tabella discreta a max_len+2 righe. Espressiva ma
        #            memorizzabile: rho puo' impararla come lookup e ignorare
        #            tutto il resto (e' cio' che faceva collassare il modello).
        # 'scalar' : due numeri continui, 1/(1+d) e d normalizzata, piu' il
        #            flag di raggiungibilita'. rho puo' pesarli ma non puo'
        #            memorizzare un ordinamento arbitrario dei bucket, quindi
        #            la lunghezza contribuisce senza poter dominare da sola.
        self.len_mode = len_mode

        n = {'hadamard': 1, 'hadamard_sum': 2, 'hadamard_absdiff': 2,
             'concat': 2}[pair_op]
        pair_dim = dim * n
        if fusion == 'gated':
            if not use_path:
                raise ValueError("gated fusion requires the path branch")
            # Put endpoint and path evidence on comparable scales, then learn
            # a feature-wise interpolation.  rho still receives only the GNN
            # endpoints and the encoded shortest path from Eq. (8).
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.pair_proj = nn.Linear(pair_dim, dim)
            self.path_proj = nn.Linear(dim, dim)
            self.gate = nn.Linear(dim * 2, dim)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            in_dim = dim * 4  # pair, path, gated mixture, interaction
        elif fusion in ('concat', 'residual'):
            in_dim = pair_dim + (dim if use_path else 0)
        else:
            raise ValueError(f"unknown fusion '{fusion}'")
        # Senza questa, h_path puo' crescere senza limiti e schiacciare il
        # contributo degli estremi: rho degenera a leggere solo il path.
        self.path_norm = nn.LayerNorm(dim) if use_path else nn.Identity()
        if use_len_feature:
            if len_mode == 'scalar':
                in_dim += 3                       # 1/(1+d), d/max_len, reach
            else:
                self.len_emb = nn.Embedding(max_len + 2, 32)
                nn.init.normal_(self.len_emb.weight, std=0.02)
                in_dim += 32 + 1
        if self.overlap_branch:
            # Mean GNN representation of all common neighbours plus a smooth
            # cardinality feature.  These are exactly the internal nodes of
            # all length-two shortest paths, so this remains an Eq. 8 path-set
            # aggregation rather than an external test-time heuristic.
            self.overlap_norm = nn.LayerNorm(dim)
            in_dim += dim + 1
        if self.pair_struct:
            self.struct_proj = nn.Sequential(
                nn.Linear(8, dim), nn.GELU(), nn.LayerNorm(dim))
            in_dim += dim
        self.max_len = max_len

        dims = [in_dim] + [hidden] * (layers - 1) + [1]
        self.lins = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.dropout = dropout
        if fusion == 'residual':
            self.endpoint_scale = nn.Parameter(torch.tensor(1.0))
            self.path_scale = nn.Parameter(torch.tensor(1.0))
            self.residual_path_head = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1))
        if path_aux:
            self.path_head = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1))

    def reset_parameters(self):
        for l in self.lins:
            l.reset_parameters()

    def forward(self, h_u, h_v, h_path, lengths=None, reach=None,
                return_aux=False, h_overlap=None, overlap_count=None,
                pair_feat=None):
        if self.pair_op == 'hadamard':
            z = h_u * h_v
        elif self.pair_op == 'hadamard_sum':
            z = torch.cat([h_u * h_v, h_u + h_v], -1)
        elif self.pair_op == 'hadamard_absdiff':
            z = torch.cat([h_u * h_v, (h_u - h_v).abs()], -1)
        else:
            z = torch.cat([h_u + h_v, (h_u - h_v).abs()], -1)

        if self.ablate == 'endpoints':
            z = torch.zeros_like(z)
        if self.ablate == 'path' and h_path is not None:
            h_path = torch.zeros_like(h_path)

        # Training-only modality dropout: every example keeps at least one of
        # endpoint-GNN or path evidence, but a fraction must be solved without
        # the dominant endpoint shortcut.  Inference always uses both.
        pair_keep = path_keep = None
        if self.training and self.branch_dropout > 0 and h_path is not None:
            draw = torch.rand(z.size(0), device=z.device)
            half = self.branch_dropout * 0.5
            pair_keep = (draw >= half).to(z.dtype).unsqueeze(-1)
            path_keep = ((draw < half) | (draw >= self.branch_dropout)).to(
                h_path.dtype).unsqueeze(-1)
            z = z * pair_keep
            h_path = h_path * path_keep

        if self.fusion == 'gated':
            pair = self.pair_proj(self.pair_norm(z))
            path = self.path_proj(self.path_norm(h_path))
            gate = torch.sigmoid(self.gate(torch.cat([pair, path], -1)))
            mixed = gate * pair + (1.0 - gate) * path
            parts = [pair, path, mixed, pair * path]
        else:
            parts = [z]
            if self.use_path and h_path is not None:
                parts.append(self.path_norm(h_path))
        if self.use_len:
            steps = (lengths - 1).clamp(0, self.max_len)
            if self.len_mode == 'scalar':
                d = steps.to(z.dtype)
                # per le coppie sconnesse la distanza e' infinita: 1/(1+d)->0
                inv = torch.where(reach, 1.0 / (1.0 + d),
                                  torch.zeros_like(d))
                nrm = torch.where(reach, d / float(self.max_len),
                                  torch.ones_like(d))
                parts.append(torch.stack(
                    [inv, nrm, reach.to(z.dtype)], dim=-1))
            else:
                steps = torch.where(reach, steps,
                                    torch.full_like(steps, self.max_len + 1))
                parts.append(self.len_emb(steps))
                parts.append(reach.to(z.dtype).unsqueeze(-1))
        if self.overlap_branch:
            parts.append(self.overlap_norm(h_overlap))
            parts.append(torch.log1p(overlap_count.to(z.dtype)).unsqueeze(-1))
        if self.pair_struct:
            if pair_feat is None:
                raise RuntimeError('pair_struct model requires cached pair features')
            parts.append(self.struct_proj(pair_feat.to(z.dtype)))
        z = torch.cat(parts, -1)

        for lin in self.lins[:-1]:
            z = F.dropout(F.relu(lin(z)), p=self.dropout, training=self.training)
        logits = self.lins[-1](z).squeeze(-1)
        if self.fusion == 'residual':
            endpoint_score = (
                F.normalize(h_u, dim=-1) * F.normalize(h_v, dim=-1)).sum(-1)
            path_score = self.residual_path_head(
                self.path_norm(h_path)).squeeze(-1)
            if self.ablate == 'endpoints':
                endpoint_score = torch.zeros_like(endpoint_score)
            if self.ablate == 'path':
                path_score = torch.zeros_like(path_score)
            logits = (logits + self.endpoint_scale * endpoint_score
                      + self.path_scale * path_score)
        if return_aux:
            if not self.path_aux:
                raise RuntimeError('path auxiliary head is not enabled')
            aux = self.path_head(self.path_norm(h_path)).squeeze(-1)
            return logits, aux
        return logits                                      # LOGITS


class HistoricalPairContextPredictor(nn.Module):
    """Endpoint/path gated predictor used by the paper-like historical run."""

    overlap_branch = False

    def __init__(self, dim, hidden, layers=2, dropout=0.2):
        super().__init__()
        self.ablate = None
        self.path_aux = False
        self.path_norm = nn.LayerNorm(dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim))
        self.path_gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim),
            nn.Sigmoid())
        dims = [dim * 3] + [hidden] * max(layers - 1, 0) + [1]
        self.lins = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def reset_parameters(self):
        for module in (self.pair_mlp, self.path_gate, self.lins):
            for child in module.modules():
                if child is not module and hasattr(child, 'reset_parameters'):
                    child.reset_parameters()

    def forward(self, h_u, h_v, h_path, lengths=None, reach=None,
                return_aux=False, h_overlap=None, overlap_count=None,
                pair_feat=None):
        del lengths, reach, h_overlap, overlap_count, pair_feat
        if return_aux:
            raise RuntimeError('historical predictor has no auxiliary head')
        pair_features = torch.cat(
            [h_u, h_v, h_u * h_v, (h_u - h_v).abs()], dim=-1)
        pair = self.pair_mlp(pair_features)
        path = self.path_norm(h_path)
        if self.ablate == 'endpoints':
            pair = torch.zeros_like(pair)
        if self.ablate == 'path':
            path = torch.zeros_like(path)
        gate = self.path_gate(torch.cat([pair, path], dim=-1))
        gated_path = gate * path
        out = torch.cat([pair, gated_path, pair * gated_path], dim=-1)
        for lin in self.lins[:-1]:
            out = F.dropout(F.gelu(lin(out)), p=self.dropout,
                            training=self.training)
        return self.lins[-1](out).squeeze(-1)


class SP4LP(nn.Module):
    """
    encode(x, edge_index) -> node embeddings, computed once per forward pass.
    score(h, pairs, batch) -> logits, where `batch` is a PathBatch (see engine).
    """

    def __init__(self, encoder, tokenizer, phi, predictor, symmetrise=True,
                 path_agg='mean', path_conditioner=None,
                 raw_feature_weight=0.0):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.phi = phi
        self.predictor = predictor
        self.symmetrise = symmetrise
        self.path_agg = path_agg
        self.path_conditioner = path_conditioner
        self.raw_feature_weight = float(raw_feature_weight)
        self._raw_x = None
        path_dim = int(predictor.path_norm.normalized_shape[0])
        self.path_attention = (nn.Linear(path_dim, 1, bias=False)
                               if path_agg == 'attention' else None)

    def reset_parameters(self):
        for m in (self.encoder, self.predictor):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def encode(self, x, edge_index):
        # Fixed semantic residual: it preserves useful input-feature geometry
        # that deep message passing may oversmooth.  It uses neither held-out
        # edges nor labels and remains exactly symmetric in the endpoints.
        if self.raw_feature_weight != 0.0:
            self._raw_x = F.normalize(x.detach(), dim=-1, eps=1e-12)
        return self.encoder(x, edge_index)

    def score(self, h, pairs, batch, return_aux=False):
        h_u = h.index_select(0, pairs[:, 0])
        h_v = h.index_select(0, pairs[:, 1])
        tokens, pad_mask = self.tokenizer(
            h, batch.nodes, batch.row, batch.pos, batch.lengths,
            symmetrise=self.symmetrise, feat=getattr(batch, 'feat', None))
        if self.path_conditioner is not None:
            pair = torch.cat([h_u * h_v, (h_u - h_v).abs()], dim=-1)
            cond = self.path_conditioner(pair)
            cond = cond.index_select(0, batch.pair_of_path)
            if self.symmetrise:
                cond = torch.cat([cond, cond], dim=0)
            scale, shift = cond.chunk(2, dim=-1)
            tokens = tokens * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
            tokens = tokens + shift.unsqueeze(1)
        if getattr(self.phi, 'uses_explicit_endpoints', False):
            path_u = h_u.index_select(0, batch.pair_of_path)
            path_v = h_v.index_select(0, batch.pair_of_path)
            if self.symmetrise:
                path_u, path_v = (torch.cat([path_u, path_v], dim=0),
                                  torch.cat([path_v, path_u], dim=0))
            enc = self.phi(tokens, pad_mask, path_u, path_v)
        else:
            enc = self.phi(tokens, pad_mask)
        if self.symmetrise:
            half = enc.size(0) // 2
            enc = enc[:half] + enc[half:]

        # AGG dell'Eq. (8): quando una coppia ha piu' cammini minimi, le loro
        # codifiche vengono aggregate in una sola. Con un cammino per coppia
        # l'operazione e' l'identita', quindi il comportamento precedente e'
        # esattamente il caso particolare.
        B = batch.n_pairs
        if enc.size(0) == B:
            h_path = enc
        else:
            idx = batch.pair_of_path
            if self.path_agg == 'max':
                h_path = enc.new_full((B, enc.size(1)), float('-inf'))
                h_path = h_path.index_reduce(0, idx, enc, 'amax',
                                             include_self=False)
            elif self.path_agg == 'attention':
                raw = self.path_attention(enc).squeeze(-1)
                peak = raw.new_full((B,), float('-inf'))
                peak = peak.index_reduce(0, idx, raw, 'amax',
                                         include_self=False)
                weight = torch.exp(raw - peak.index_select(0, idx))
                denom = raw.new_zeros(B).index_add_(0, idx, weight)
                h_path = enc.new_zeros(B, enc.size(1))
                h_path.index_add_(0, idx, enc * weight.unsqueeze(1))
                h_path = h_path / denom.clamp_min(1e-12).unsqueeze(1)
            else:
                h_path = enc.new_zeros(B, enc.size(1))
                h_path.index_add_(0, idx, enc)
                if self.path_agg == 'mean':
                    cnt = enc.new_zeros(B).index_add_(
                        0, idx, enc.new_ones(enc.size(0)))
                    h_path = h_path / cnt.clamp_min(1.0).unsqueeze(1)
        h_overlap = overlap_count = None
        if self.predictor.overlap_branch:
            # For every enumerated u-w-v shortest path, collect w.  When all
            # shortest paths are cached this is precisely the common-neighbour
            # set, with no validation/test edge required.
            starts = torch.cumsum(batch.lengths, 0) - batch.lengths
            is_two_hop = batch.lengths.eq(3)
            owners = batch.pair_of_path[is_two_hop]
            mids = batch.nodes[(starts[is_two_hop] + 1)]
            h_overlap = h.new_zeros((batch.n_pairs, h.size(1)))
            overlap_count = h.new_zeros(batch.n_pairs)
            if mids.numel():
                h_overlap.index_add_(0, owners, h.index_select(0, mids))
                overlap_count.index_add_(0, owners, h.new_ones(mids.numel()))
                h_overlap = h_overlap / overlap_count.clamp_min(1).unsqueeze(1)
        out = self.predictor(h_u, h_v, h_path, batch.pair_lengths,
                             batch.reach, return_aux=return_aux,
                             h_overlap=h_overlap,
                             overlap_count=overlap_count,
                             pair_feat=getattr(batch, 'pair_feat', None))
        if self.raw_feature_weight != 0.0:
            if self._raw_x is None:
                raise RuntimeError('encode must be called before score')
            raw_u = self._raw_x.index_select(0, pairs[:, 0])
            raw_v = self._raw_x.index_select(0, pairs[:, 1])
            residual = self.raw_feature_weight * (raw_u * raw_v).sum(-1)
            out = ((out[0] + residual, out[1]) if return_aux
                   else out + residual)
        return out

    def forward(self, x, edge_index, pairs, batch):
        return self.score(self.encode(x, edge_index), pairs, batch)


# ---------------------------------------------------------------------------- #
def build_sp4lp(cfg):
    """
    cfg keys
    --------
    input_dim, gnn_type, gnn_layers, hidden_dim, dropout,
    gat_heads, gnn_bn, gnn_residual, gnn_jk,
    seq_model, seq_layers, seq_heads, path_hidden_dim, rich_pool,
    max_path_len, use_distance_encoding, use_endpoint_flag, symmetrise,
    predictor_hidden, predictor_layers, pair_op, use_len_feature
    """
    dim = cfg['hidden_dim']
    max_len = cfg.get('max_path_len', 32)
    enc = build_encoder(cfg, cfg['input_dim'])
    historical = bool(cfg.get('historical_rich', False))
    tok = PathTokenizer(dim, max_len,
                        cfg.get('use_distance_encoding', True),
                        cfg.get('use_endpoint_flag', True),
                        cfg.get('dropout', 0.0),
                        cfg.get('use_node_feats', False),
                        normalise=not historical)
    phi = build_phi(cfg)
    if historical:
        pred = HistoricalPairContextPredictor(
            dim, cfg.get('predictor_hidden', dim),
            cfg.get('predictor_layers', 2), cfg.get('dropout', 0.0))
    else:
        pred = Predictor(dim, cfg.get('predictor_hidden', dim),
                         cfg.get('predictor_layers', 2),
                         cfg.get('dropout', 0.0),
                         cfg.get('pair_op', 'hadamard'), True,
                         cfg.get('use_len_feature', True), max_len,
                         cfg.get('len_mode', 'embed'),
                         cfg.get('fusion', 'concat'),
                         cfg.get('path_aux', False),
                         cfg.get('branch_dropout', 0.0),
                         cfg.get('overlap_branch', False),
                         cfg.get('pair_struct', False))
    conditioner = None
    if cfg.get('condition_path', False):
        conditioner = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim * 2))
        # Begin as the unconditioned model and let validation-supervised
        # training introduce endpoint context gradually.
        nn.init.zeros_(conditioner[-1].weight)
        nn.init.zeros_(conditioner[-1].bias)
    return SP4LP(enc, tok, phi, pred, cfg.get('symmetrise', True),
                 cfg.get('path_agg', 'mean'), conditioner,
                 cfg.get('raw_feature_weight', 0.0))
