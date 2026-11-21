from torch import nn

from sgdiff.models.baselines.sp4lp import SP4LPBaseline
from sgdiff.vendor.sp4lp.models import build_sp4lp, build_phi


def configuration(dataset, profile):
    standardize = profile.endswith("_std")
    profile = profile.removesuffix("_std")
    cfg = dict(
        gnn_type="gcn",
        gnn_layers=1,
        hidden_dim=64,
        dropout=0.1,
        encoder_style="paper",
        seq_model="transformer_paper",
        seq_layers=1,
        seq_heads=4,
        path_hidden_dim=128,
        max_path_len=32,
        use_distance_encoding=False,
        use_endpoint_flag=False,
        symmetrise=True,
        predictor_hidden=64,
        predictor_layers=3,
        pair_op="hadamard",
        use_len_feature=False,
        fusion="concat",
        lr=0.001,
        weight_decay=0.0,
    )
    if profile == "paper128":
        cfg.update(hidden_dim=128, predictor_hidden=256, seq_layers=2, path_dropout=0.1)
        if dataset == "pubmed":
            cfg.update(gnn_type="sage", gnn_layers=2)
        else:
            cfg.update(dropout=0.5)
        if dataset == "citeseer":
            cfg.update(seq_model="lstm", predictor_layers=2, lr=0.01, weight_decay=1e-7)
    elif profile == "release256":
        cfg.update(hidden_dim=256, predictor_hidden=256)
    elif profile != "full64":
        raise ValueError(profile)
    if standardize:
        cfg["standardize"] = True
    return cfg


class FullSP4LP(SP4LPBaseline):
    def __init__(self, feature_dim, config, variant="full"):
        nn.Module.__init__(self)
        self.endpoints_only = variant == "endpoints"
        self.featureless = variant == "featureless"
        self.settings = dict(config, input_dim=1 if self.featureless else feature_dim)
        self.model = build_sp4lp(self.settings)
        if "path_dropout" in config:
            self.model.phi = build_phi(dict(self.settings, dropout=config["path_dropout"]))
            self.model.tokenizer.drop.p = config["path_dropout"]
        if self.endpoints_only:
            dims = (
                [config["hidden_dim"]]
                + [config["predictor_hidden"]] * (config["predictor_layers"] - 1)
                + [1]
            )
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.extend([nn.ReLU(), nn.Dropout(config["dropout"])])
            self.endpoint_head = nn.Sequential(*layers)
            self.model.tokenizer = nn.Identity()
            self.model.phi = nn.Identity()
            self.model.predictor = nn.Identity()
        self.store = None


def input_graph(graph, config):
    if not config.get("standardize", False):
        return graph
    from sgdiff.engine.baseline_trainer import node_feature_matrix
    from sgdiff.data.graph import GraphData

    return GraphData(
        graph.name, graph.num_nodes, graph.edges, x=node_feature_matrix(graph), y=graph.y
    )
