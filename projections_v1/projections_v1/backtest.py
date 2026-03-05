from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GlobalConfig
from .point_forecast import project_next_season
from .uncertainty import apply_uncertainty_bands, build_transition_deltas


def run_rolling_backtest(
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
    dataset_name: str = "dataset",
    metric_scale_factors: dict[str, float] | None = None,
) -> pd.DataFrame:
    if player_season_df.empty:
        return pd.DataFrame()
    k_overrides = k_overrides or {}
    bounds = bounds or {}
    metric_scale_factors = metric_scale_factors or {}

    seasons = sorted(pd.to_numeric(player_season_df[season_col], errors="coerce").dropna().astype(int).unique().tolist())
    if len(seasons) < 4:
        return pd.DataFrame()

    metric_rows: list[dict[str, float | int | str]] = []
    start_idx = 3  # ensure enough history for recency signal.
    for i in range(start_idx, len(seasons)):
        target_season = seasons[i]
        source_season = target_season - 1
        train = player_season_df[player_season_df[season_col] <= source_season].copy()
        actual = player_season_df[player_season_df[season_col] == target_season].copy()
        if train.empty or actual.empty:
            continue

        point = project_next_season(
            player_season_df=train,
            metric_cols=metric_cols,
            id_col=id_col,
            name_col=name_col,
            season_col=season_col,
            level_col=level_col,
            age_col=age_col,
            age_source_col=age_source_col,
            exposure_col=exposure_col,
            global_cfg=global_cfg,
            k_overrides=k_overrides,
            bounds=bounds,
            source_season=source_season,
        )
        if point.empty:
            continue
        transitions = build_transition_deltas(train, id_col=id_col, season_col=season_col, metric_cols=metric_cols)
        point = apply_uncertainty_bands(
            point,
            transitions_df=transitions,
            metric_cols=metric_cols,
            draws=int(global_cfg.backtest_uncertainty_draws),
            seed=int(global_cfg.seed + target_season),
            global_weight=float(global_cfg.cov_blend_global),
            local_k=int(global_cfg.local_k),
            local_min_k=int(global_cfg.local_min_k),
            uncertainty_c=float(global_cfg.uncertainty_c),
            uncertainty_d=float(global_cfg.uncertainty_d),
            bounds=bounds,
        )
        for metric, factor in metric_scale_factors.items():
            mult = float(pd.to_numeric(factor, errors="coerce"))
            if metric not in metric_cols or not np.isfinite(mult) or mult == 1.0:
                continue
            pcols = [f"{metric}_proj_p20", f"{metric}_proj_p25", f"{metric}_proj_p50", f"{metric}_proj_p75", f"{metric}_proj_p80"]
            for col in pcols:
                if col in point.columns:
                    point[col] = pd.to_numeric(point[col], errors="coerce") * mult

        merged = point.merge(actual[[id_col, *metric_cols]], on=id_col, how="inner", suffixes=("", "_actual"))
        if merged.empty:
            continue

        for metric in metric_cols:
            pred = pd.to_numeric(merged[f"{metric}_proj_p50"], errors="coerce")
            low = pd.to_numeric(merged[f"{metric}_proj_p25"], errors="coerce")
            high = pd.to_numeric(merged[f"{metric}_proj_p75"], errors="coerce")
            obs = pd.to_numeric(merged[metric], errors="coerce")
            valid = np.isfinite(pred) & np.isfinite(obs)
            if not valid.any():
                continue

            err = obs[valid] - pred[valid]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            band_valid = valid & np.isfinite(low) & np.isfinite(high)
            if band_valid.any():
                band_hit = float(((obs[band_valid] >= low[band_valid]) & (obs[band_valid] <= high[band_valid])).mean())
            else:
                band_hit = np.nan
            pred_mean = float(np.mean(pred[valid]))
            actual_mean = float(np.mean(obs[valid]))
            mean_error = float(np.mean(err))

            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "target_season": int(target_season),
                    "n": int(valid.sum()),
                    "mae": mae,
                    "rmse": rmse,
                    "coverage_25_75": float(band_hit),
                    "pred_mean": pred_mean,
                    "actual_mean": actual_mean,
                    "mean_error": mean_error,
                }
            )

    if not metric_rows:
        return pd.DataFrame()
    detail = pd.DataFrame(metric_rows)
    def _wavg(grp: pd.DataFrame, col: str) -> float:
        vals = pd.to_numeric(grp[col], errors="coerce")
        w = pd.to_numeric(grp["n"], errors="coerce")
        mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=w[mask]))

    rows: list[dict[str, float | int | str]] = []
    for (dataset, metric), grp in detail.groupby(["dataset", "metric"], dropna=False):
        pred_mean = _wavg(grp, "pred_mean")
        actual_mean = _wavg(grp, "actual_mean")
        if np.isfinite(pred_mean) and pred_mean != 0.0 and np.isfinite(actual_mean):
            actual_over_pred = float(actual_mean / pred_mean)
        else:
            actual_over_pred = np.nan
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n": int(pd.to_numeric(grp["n"], errors="coerce").fillna(0).sum()),
                "mae": _wavg(grp, "mae"),
                "rmse": _wavg(grp, "rmse"),
                "coverage_25_75": _wavg(grp, "coverage_25_75"),
                "pred_mean": pred_mean,
                "actual_mean": actual_mean,
                "mean_error": _wavg(grp, "mean_error"),
                "actual_over_pred": actual_over_pred,
                "seasons": int(pd.to_numeric(grp["target_season"], errors="coerce").nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["dataset", "metric"]).reset_index(drop=True)
