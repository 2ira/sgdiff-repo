"""Typed experiment configuration (YAML <-> dataclasses)."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    name: str = "cora"
    root: str = "data"
    val_ratio: float = 0.05
    test_ratio: float = 0.15
    neg_ratio: float = 1.0
    train_ratio: float = 1.0  # 1% train-size study -> 0.01
    max_train_pairs: Optional[int] = None
    max_eval_pairs: Optional[int] = 1000
    download: bool = True
    seed: int = 0


@dataclass
class SubgraphConfig:
    hops: int = 1  # Table 4: k = 1 for every dataset
    max_nodes_per_hop: Optional[int] = None  # Table 4: ns (None = -1, no sampling)
    max_nodes_total: Optional[int] = None  # compute safety cap (paper: unbounded)
    max_label: int = 16
    feature_mode: str = "auto"  # auto | attribute | structural
    remove_target_edge: bool = True


@dataclass
class StructureConfig:
    arch: str = "transformer"  # paper Sec. A.3 / Table 4
    node_attn: int = 256
    edge_attn: int = 64
    global_attn: int = 64
    node_mlp: int = 256
    edge_mlp: int = 128
    global_mlp: int = 128
    heads: int = 8
    num_layers: int = 2
    aux_ce_weight: float = 1.0  # "sum of node and feature cross-entropy"
    hidden_dim: int = 64  # only used by arch="edge_gnn"
    edge_dim: int = 16
    num_steps: int = 20
    beta_start: float = 0.0001
    beta_end: float = 0.2
    time_dim: int = 32
    dropout: float = 0.1
    normalize_score: bool = True
    num_mc_samples: Optional[int] = None
    weight: float = 1.0


@dataclass
class FeatureConfig:
    enabled: bool = True  # False for USAir / NS / Router (Table 4: N/A)
    hidden_dim: int = 256
    num_layers: int = 2
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_dim: int = 32
    dropout: float = 0.1
    arch: str = "gcn"
    num_mc_samples: Optional[int] = None
    weight: float = 1.0


@dataclass
class ModelConfig:
    fusion_learnable: bool = True
    init_eta: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 30
    pretrain_epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 1e-3
    fusion_lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int = 10
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    run_dir: str = "runs"
    run_name: Optional[str] = None
    eval_every: int = 1
    save_checkpoint: bool = True
    freeze_diffusion: bool = False  # true = stage 2 trains only eta1/eta2/delta
    finetune_mc_samples: int = 2


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    subgraph: SubgraphConfig = field(default_factory=SubgraphConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------- conversion
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        sections = {f.name: f for f in fields(cls)}
        unknown = set(payload) - set(sections)
        if unknown:
            raise KeyError(f"unknown config sections: {sorted(unknown)}")
        kwargs = {}
        for name, f in sections.items():
            section_payload = payload.get(name, {}) or {}
            section_cls = f.default_factory  # type: ignore[misc]
            allowed = {sf.name for sf in fields(section_cls)}
            bad = set(section_payload) - allowed
            if bad:
                raise KeyError(f"unknown keys in '{name}': {sorted(bad)}")
            kwargs[name] = section_cls(**section_payload)
        return cls(**kwargs)

    def apply_overrides(self, overrides: list[str]) -> "ExperimentConfig":
        """Apply ``section.key=value`` overrides in place and return ``self``."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like section.key=value, got '{item}'")
            path, raw = item.split("=", 1)
            parts = path.split(".")
            if len(parts) != 2:
                raise ValueError(f"override must look like section.key=value, got '{item}'")
            section, key = parts
            if not hasattr(self, section) or not hasattr(getattr(self, section), key):
                raise KeyError(f"unknown config key '{path}'")
            current = getattr(getattr(self, section), key)
            setattr(getattr(self, section), key, _coerce(raw, current))
        return self


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def load_config(path: str | Path = None, overrides: Optional[list[str]] = None) -> ExperimentConfig:
    """Load a YAML config (or defaults) and apply ``section.key=value`` overrides."""
    payload: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    cfg = ExperimentConfig.from_dict(payload)
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    return path
