"""HL-GNN (KDD 2024) implementation without torch_sparse/torch_scatter dependencies."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Parameter, functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import coalesce

VARIANTS = ("full", "fixed_weights", "shallow_k2")
PROFILES = ("readme_exact", "compact512")


def configuration(dataset: str, profile: str) -> dict:
    """Return pre-specified official-family hyperparameter configurations."""
    base = {
        "cora": dict(mlp_num_layers=3, hidden_channels=8192, dropout=0.5, K=20, alpha=0.2, init="RWR"),
        "citeseer": dict(mlp_num_layers=2, hidden_channels=8192, dropout=0.5, K=20, alpha=0.2, init="RWR"),
        "pubmed": dict(mlp_num_layers=3, hidden_channels=512, dropout=0.6, K=20, alpha=0.2, init="KI"),
    }
    if dataset not in base:
        raise ValueError(f"Unsupported dataset: {dataset}")
    cfg = dict(base[dataset], lr=1e-3, weight_decay=0.0)
    if profile == "readme_exact":
        return cfg
    if profile == "compact512":
        cfg["hidden_channels"] = 512
        cfg["weight_decay"] = 1e-4
        return cfg
    raise ValueError(f"Unsupported profile: {profile}")


def build_norm_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Build GCN-normalized sparse COO adjacency equivalent to ToSparseTensor + gcn_norm."""
    if edge_index.numel() > 0:
        edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes, reduce="any")
    edge_index, edge_weight = gcn_norm(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        add_self_loops=True,
        dtype=torch.float32,
    )
    return torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes), device=device
    ).coalesce()


def compute_init_temp(K: int, init: str, alpha: float) -> torch.Tensor:
    """Compute exact analytical KDD 2024 HL-GNN hop weights."""
    if init == "RWR":
        temp = alpha * (1.0 - alpha) ** np.arange(K + 1)
        temp[-1] = (1.0 - alpha) ** K
        temp = temp / np.sum(np.abs(temp))
    elif init == "KI":
        temp = alpha ** np.arange(K + 1)
    elif init == "SGC":
        temp = np.zeros(K + 1, dtype=np.float32)
        temp[int(alpha)] = 1.0
    else:
        raise ValueError(f"Unsupported init: {init}")
    return torch.tensor(temp, dtype=torch.float32)


class HLGNNEncoder(nn.Module):
    """Official HL-GNN encoder: Linear(F, F) + K-step gcn_norm propagation with hop weights."""

    def __init__(self, in_dim: int, K: int, init: str, alpha: float, dropout: float, learn_temp: bool = True):
        super().__init__()
        self.K = int(K)
        self.init = init
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.learn_temp = bool(learn_temp)
        self.lin1 = Linear(in_dim, in_dim)
        self.temp = Parameter(compute_init_temp(self.K, self.init, self.alpha), requires_grad=self.learn_temp)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        norm_adj = build_norm_adj(edge_index, num_nodes, x.device)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = torch.sparse.mm(norm_adj, x)
            hidden = hidden + self.temp[k + 1] * x
        return hidden


class LinkPredictor(nn.Module):
    """Official Hadamard-MLP link predictor outputting raw logits."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        self.lins = nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, 1))
        self.dropout = float(dropout)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x).squeeze(-1)


class HLGNNLinkModel(nn.Module):
    """HL-GNN full model and two component ablations (fixed_weights, shallow_k2)."""

    def __init__(self, feature_dim: int, config: dict, variant: str = "full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.config = dict(config)
        K = 2 if variant == "shallow_k2" else int(config["K"])
        learn_temp = variant != "fixed_weights"
        self.encoder = HLGNNEncoder(
            in_dim=feature_dim,
            K=K,
            init=config["init"],
            alpha=config["alpha"],
            dropout=config["dropout"],
            learn_temp=learn_temp,
        )
        self.predictor = LinkPredictor(
            in_channels=feature_dim,
            hidden_channels=int(config["hidden_channels"]),
            num_layers=int(config["mlp_num_layers"]),
            dropout=float(config["dropout"]),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        return self.encoder(x, edge_index, num_nodes)

    def score_pairs(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return self.predictor(z[pairs[:, 0]], z[pairs[:, 1]])
