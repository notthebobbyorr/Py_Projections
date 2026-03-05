from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GlobalConfig:
    recency_weights: list[float] = field(default_factory=lambda: [5.0, 4.0, 3.0])
    recency_weights_by_metric: dict[str, list[float]] = field(default_factory=dict)
    default_k: float = 200.0
    uncertainty_draws: int = 2000
    seed: int = 7
    cov_blend_global: float = 0.6
    local_k: int = 200
    local_min_k: int = 40
    uncertainty_c: float = 120.0
    uncertainty_d: float = 20.0
    mlb_level_id: int = 1
    projection_version: str = "v1"
    output_dir: str = "projection_outputs"
    backtest_uncertainty_draws: int = 300
    min_age_samples: int = 20
    park_factors_path: str = "park_data.parquet"
    park_window_years: int = 3
    home_park_game_share: float = 0.5
    bp_hitting_rates_path: str = "projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet"


@dataclass
class DatasetConfig:
    name: str
    regressed_path: str
    base_path: str
    id_col: str
    name_col: str
    season_col: str
    level_col: str
    age_col: str
    exposure_col: str
    fallback_exposure_col: str | None = None
    hand_col: str | None = None
    team_col: str | None = None
    extra_numeric_cols: list[str] = field(default_factory=list)


@dataclass
class MetricConfig:
    include: list[str] = field(default_factory=list)
    include_extra: list[str] = field(default_factory=list)
    bounds: dict[str, list[float]] = field(default_factory=dict)
    k_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class ProjectionConfig:
    global_cfg: GlobalConfig
    datasets: dict[str, DatasetConfig]
    metrics: dict[str, MetricConfig]


def _to_global(cfg: dict[str, Any]) -> GlobalConfig:
    return GlobalConfig(**cfg)


def _to_dataset(name: str, cfg: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(name=name, **cfg)


def _to_metric(cfg: dict[str, Any]) -> MetricConfig:
    return MetricConfig(
        include=list(cfg.get("include", [])),
        include_extra=list(cfg.get("include_extra", [])),
        bounds=dict(cfg.get("bounds", {})),
        k_overrides=dict(cfg.get("k_overrides", {})),
    )


def load_config(path: str | Path) -> ProjectionConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    global_cfg = _to_global(dict(raw.get("global", {})))
    datasets_raw = dict(raw.get("datasets", {}))
    metrics_raw = dict(raw.get("metrics", {}))
    datasets = {k: _to_dataset(k, dict(v)) for k, v in datasets_raw.items()}
    metrics = {k: _to_metric(dict(v)) for k, v in metrics_raw.items()}
    return ProjectionConfig(global_cfg=global_cfg, datasets=datasets, metrics=metrics)


def resolve_metric_settings(
    dataset_name: str,
    regressed_cols: list[str],
    available_cols: list[str],
    cfg: ProjectionConfig,
) -> tuple[list[str], dict[str, float], dict[str, tuple[float, float]]]:
    metric_cfg = cfg.metrics.get(dataset_name, MetricConfig())
    inferred = sorted(c for c in regressed_cols if c.endswith("_reg"))
    if metric_cfg.include:
        include = [c for c in metric_cfg.include if c in available_cols]
    else:
        include = inferred
    extra = [c for c in metric_cfg.include_extra if c in available_cols]
    include = list(dict.fromkeys([*include, *extra]))
    bounds: dict[str, tuple[float, float]] = {}
    for metric, bound_pair in metric_cfg.bounds.items():
        if not isinstance(bound_pair, list) or len(bound_pair) != 2:
            continue
        bounds[metric] = (float(bound_pair[0]), float(bound_pair[1]))
    k_overrides = {k: float(v) for k, v in metric_cfg.k_overrides.items()}
    return include, k_overrides, bounds
