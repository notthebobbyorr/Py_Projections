from __future__ import annotations

import numpy as np
import pandas as pd


def apply_simple_mlb_equivalency(
    df: pd.DataFrame,
    metric_cols: list[str],
    season_col: str = "season",
    level_col: str = "level_id",
    mlb_level_id: int = 1,
) -> pd.DataFrame:
    """
    Translate each metric into simple season-contextual MLB-equivalent space.

    For non-MLB rows:
    1) z-score within season+level
    2) rescale to season MLB mean/std

    MLB rows are pass-through.
    """
    out = df.copy()
    if out.empty:
        return out

    for metric in metric_cols:
        if metric not in out.columns:
            continue
        metric_vals = pd.to_numeric(out[metric], errors="coerce")

        level_mom = (
            out[[season_col, level_col, metric]]
            .assign(**{metric: metric_vals})
            .groupby([season_col, level_col], dropna=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "__lvl_mean", "std": "__lvl_std"})
        )
        mlb_mom = (
            level_mom[level_mom[level_col] == mlb_level_id][[season_col, "__lvl_mean", "__lvl_std"]]
            .rename(columns={"__lvl_mean": "__mlb_mean", "__lvl_std": "__mlb_std"})
            .copy()
        )

        work = out[[season_col, level_col]].copy()
        work["__metric"] = metric_vals
        work = work.merge(level_mom, on=[season_col, level_col], how="left")
        work = work.merge(mlb_mom, on=season_col, how="left")

        lvl_std = pd.to_numeric(work["__lvl_std"], errors="coerce")
        mlb_std = pd.to_numeric(work["__mlb_std"], errors="coerce")
        num = work["__metric"].to_numpy() - work["__lvl_mean"].to_numpy()
        den = lvl_std.to_numpy()
        z = np.zeros_like(num, dtype=float)
        valid = den > 0
        z[valid] = num[valid] / den[valid]
        translated = work["__mlb_mean"].to_numpy() + z * mlb_std.to_numpy()
        translated = np.where(np.isfinite(translated), translated, work["__metric"].to_numpy())
        translated = np.where(
            work[level_col].to_numpy() == mlb_level_id,
            work["__metric"].to_numpy(),
            translated,
        )

        out[metric] = translated
    return out
