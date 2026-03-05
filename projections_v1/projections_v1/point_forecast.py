from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import GlobalConfig


@dataclass
class AgeDeltaModel:
    by_age: dict[str, dict[int, float]]
    global_delta: dict[str, float]

    def delta(self, metric: str, age: float | int | None) -> float:
        if age is None or not np.isfinite(float(age)):
            return float(self.global_delta.get(metric, 0.0))
        age_int = int(round(float(age)))
        per_age = self.by_age.get(metric, {})
        if age_int in per_age:
            return float(per_age[age_int])
        if per_age:
            nearest = min(per_age.keys(), key=lambda k: abs(k - age_int))
            return float(per_age[nearest])
        return float(self.global_delta.get(metric, 0.0))


def build_age_delta_model(
    player_season_df: pd.DataFrame,
    id_col: str,
    season_col: str,
    age_col: str,
    metric_cols: list[str],
    min_age_samples: int = 20,
) -> AgeDeltaModel:
    if player_season_df.empty:
        return AgeDeltaModel(by_age={}, global_delta={m: 0.0 for m in metric_cols})

    frame = player_season_df[[id_col, season_col, age_col, *metric_cols]].copy()
    frame = frame.sort_values([id_col, season_col])
    next_frame = frame.groupby(id_col, dropna=False).shift(-1)
    next_season = next_frame[season_col]
    season_gap = pd.to_numeric(next_season - frame[season_col], errors="coerce")
    valid_gap = season_gap > 0

    rows = []
    for metric in metric_cols:
        delta_per_year = pd.to_numeric(next_frame[metric] - frame[metric], errors="coerce") / season_gap
        temp = pd.DataFrame(
            {
                "metric": metric,
                "age": pd.to_numeric(frame[age_col], errors="coerce"),
                "delta": delta_per_year,
                "valid_gap": valid_gap,
            }
        )
        temp = temp[temp["valid_gap"]].drop(columns="valid_gap")
        rows.append(temp)
    long = pd.concat(rows, ignore_index=True)
    long = long[np.isfinite(long["delta"]) & np.isfinite(long["age"])]
    if long.empty:
        return AgeDeltaModel(by_age={}, global_delta={m: 0.0 for m in metric_cols})

    long["age"] = long["age"].round().astype(int)
    global_delta = (
        long.groupby("metric", dropna=False)["delta"]
        .mean(numeric_only=True)
        .reindex(metric_cols)
        .fillna(0.0)
        .to_dict()
    )
    age_counts = long.groupby(["metric", "age"], dropna=False)["delta"].size().rename("n").reset_index()
    age_mean = long.groupby(["metric", "age"], dropna=False)["delta"].mean(numeric_only=True).rename("delta").reset_index()
    age_table = age_mean.merge(age_counts, on=["metric", "age"], how="left")
    age_table = age_table[age_table["n"] >= int(min_age_samples)]
    by_age: dict[str, dict[int, float]] = {}
    for metric, grp in age_table.groupby("metric", dropna=False):
        by_age[str(metric)] = {int(a): float(d) for a, d in zip(grp["age"], grp["delta"])}
    return AgeDeltaModel(by_age=by_age, global_delta={str(k): float(v) for k, v in global_delta.items()})


def _metric_base_name(metric: str) -> str:
    if metric.endswith("_reg"):
        return metric[:-4]
    return metric


def _metric_n_col(metric: str) -> str:
    return f"{_metric_base_name(metric)}_n"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def _build_recency_map(weights: list[float] | tuple[float, ...]) -> dict[int, float]:
    out: dict[int, float] = {}
    for lag, w in enumerate(weights):
        wf = float(pd.to_numeric(w, errors="coerce"))
        if np.isfinite(wf) and wf > 0:
            out[int(lag)] = wf
    return out


def _clip_value(value: float, metric: str, bounds: dict[str, tuple[float, float]]) -> float:
    if metric not in bounds or not np.isfinite(value):
        return value
    low, high = bounds[metric]
    return float(np.clip(value, low, high))


def project_next_season(
    player_season_df: pd.DataFrame,
    metric_cols: list[str],
    id_col: str,
    name_col: str,
    season_col: str,
    level_col: str,
    age_col: str,
    age_source_col: str,
    exposure_col: str,
    global_cfg: GlobalConfig,
    k_overrides: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    source_season: int | None = None,
    passthrough_cols: list[str] | None = None,
) -> pd.DataFrame:
    if player_season_df.empty:
        return pd.DataFrame()
    k_overrides = k_overrides or {}
    bounds = bounds or {}

    work = player_season_df.copy()
    work[season_col] = pd.to_numeric(work[season_col], errors="coerce")
    work = work[work[season_col].notna()].copy()
    work[season_col] = work[season_col].astype(int)
    if source_season is None:
        source_season = int(work[season_col].max())

    history = work[work[season_col] <= source_season].copy()
    source_rows = history[history[season_col] == source_season].copy()
    if source_rows.empty:
        return pd.DataFrame()

    season_means = history.groupby(season_col, dropna=False)[metric_cols].mean(numeric_only=True)
    overall_means = history[metric_cols].mean(numeric_only=True)

    age_model = build_age_delta_model(
        history,
        id_col=id_col,
        season_col=season_col,
        age_col=age_col,
        metric_cols=metric_cols,
        min_age_samples=global_cfg.min_age_samples,
    )

    recency_map = _build_recency_map(global_cfg.recency_weights)
    if not recency_map:
        recency_map = {0: 5.0, 1: 4.0, 2: 3.0}
    metric_recency_maps: dict[str, dict[int, float]] = {}
    for metric, weights in (global_cfg.recency_weights_by_metric or {}).items():
        if not isinstance(weights, (list, tuple)):
            continue
        m_map = _build_recency_map(weights)
        if m_map:
            metric_recency_maps[str(metric)] = m_map
    passthrough_cols = passthrough_cols or []
    output_rows: list[dict[str, Any]] = []

    for _, src in source_rows.iterrows():
        pid = src[id_col]
        player_hist = history[history[id_col] == pid].copy()
        if player_hist.empty:
            continue
        player_hist["lag"] = source_season - player_hist[season_col]
        player_hist["recency_w"] = player_hist["lag"].map(recency_map)
        player_hist = player_hist[player_hist["recency_w"].notna()].copy()
        if player_hist.empty:
            continue

        row: dict[str, Any] = {
            id_col: pid,
            name_col: src.get(name_col),
            "level_id_source": src.get(level_col),
            "source_season": int(source_season),
            "target_season": int(source_season + 1),
            "projection_version": global_cfg.projection_version,
            "age_used": float(src.get(age_col)) if np.isfinite(pd.to_numeric(src.get(age_col), errors="coerce")) else np.nan,
            "age_source": src.get(age_source_col, "unknown"),
            exposure_col: float(pd.to_numeric(src.get(exposure_col), errors="coerce") or 0.0),
        }
        for col in passthrough_cols:
            if col in src.index:
                row[col] = src.get(col)

        for metric in metric_cols:
            metric_map = metric_recency_maps.get(metric, recency_map)
            rec_w = player_hist["lag"].map(metric_map)
            if rec_w.notna().sum() <= 0:
                rec_w = player_hist["recency_w"]
            mean_recent = _weighted_mean(player_hist[metric], rec_w)
            if not np.isfinite(mean_recent):
                mean_recent = float(season_means.get(metric, pd.Series(dtype=float)).get(source_season, np.nan))
            if not np.isfinite(mean_recent):
                mean_recent = float(overall_means.get(metric, np.nan))
            if not np.isfinite(mean_recent):
                mean_recent = 0.0

            n_col = _metric_n_col(metric)
            if n_col in player_hist.columns:
                n_eff = _weighted_mean(player_hist[n_col], rec_w)
            else:
                n_eff = _weighted_mean(player_hist[exposure_col], rec_w)
            if not np.isfinite(n_eff):
                n_eff = 0.0

            prior = float(season_means.get(metric, pd.Series(dtype=float)).get(source_season, np.nan))
            if not np.isfinite(prior):
                prior = float(overall_means.get(metric, np.nan))
            if not np.isfinite(prior):
                prior = mean_recent

            k_val = float(k_overrides.get(metric, global_cfg.default_k))
            regressed = (n_eff / (n_eff + k_val)) * mean_recent + (k_val / (n_eff + k_val)) * prior
            regressed += age_model.delta(metric, row["age_used"])
            regressed = _clip_value(regressed, metric, bounds)

            row[f"source_{metric}"] = float(pd.to_numeric(src.get(metric), errors="coerce")) if metric in src else np.nan
            row[f"n_eff_{metric}"] = float(n_eff)
            row[f"{metric}_proj_p50"] = float(regressed)

        output_rows.append(row)

    if not output_rows:
        return pd.DataFrame()
    return pd.DataFrame(output_rows)
