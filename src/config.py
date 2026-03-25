"""YAML configuration loader for LOB Microstructure project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    exchange: str = "bybit"
    symbol: str = "BTCUSDT"
    snapshot_interval_ms: int = 100
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    lob_depth: int = 40
    format: str = "parquet"


@dataclass
class PreprocessingConfig:
    filter_type: str = "savitzky_golay"
    sg_window: int = 21
    sg_polyorder: int = 3
    kalman_q: float = 0.001
    kalman_r: float = 0.1
    normalize: bool = True
    normalize_window: int = 1000


@dataclass
class LabelingConfig:
    scheme: str = "binary"
    horizon_ms: int = 500
    horizon_steps: int = 5
    ternary_epsilon: float = 0.0001


@dataclass
class FeatureConfig:
    depth_levels: list[int] = field(default_factory=lambda: [5, 10, 40])
    sequence_length: int = 1
    include_raw_lob: bool = True
    include_imbalance: bool = True
    include_spread: bool = True
    include_weighted_mid: bool = True


@dataclass
class ModelConfig:
    type: str = "xgboost"
    seed: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    early_stopping_rounds: int = 50


@dataclass
class TrainingConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    n_snapshots: int = 100000


@dataclass
class CostConfig:
    enabled: bool = True
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.5
    slippage_bps: float = 1.0


@dataclass
class OutputConfig:
    results_dir: str = "results"
    reports_dir: str = "reports"
    plots_format: str = "png"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dict_to_dataclass(section: dict[str, Any], cls: type) -> Any:
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in section.items() if k in known})


def load_config(path: str | Path) -> Config:
    """Load a YAML config file, resolving inheritance."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if "inherit" in raw:
        base_path = path.parent / raw.pop("inherit")
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _merge_dict(base_raw, raw)

    return Config(
        data=_dict_to_dataclass(raw.get("data", {}), DataConfig),
        preprocessing=_dict_to_dataclass(raw.get("preprocessing", {}), PreprocessingConfig),
        labeling=_dict_to_dataclass(raw.get("labeling", {}), LabelingConfig),
        features=_dict_to_dataclass(raw.get("features", {}), FeatureConfig),
        model=_dict_to_dataclass(raw.get("model", {}), ModelConfig),
        training=_dict_to_dataclass(raw.get("training", {}), TrainingConfig),
        cost=_dict_to_dataclass(raw.get("cost", {}), CostConfig),
        output=_dict_to_dataclass(raw.get("output", {}), OutputConfig),
    )
