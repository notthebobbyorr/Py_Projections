from __future__ import annotations

import numpy as np
import pandas as pd


def infer_and_impute_age(
    df: pd.DataFrame,
    player_col: str,
    season_col: str,
    level_col: str,
    age_col: str = "baseball_age",
    group_cols: list[str] | None = None,
    default_age: float = 27.0,
) -> pd.DataFrame:
    out = df.copy()
    out["age_used"] = pd.to_numeric(out.get(age_col), errors="coerce")
    out["age_source"] = np.where(out["age_used"].notna(), "observed", None)

    for _, idx in out.groupby(player_col, dropna=False).groups.items():
        player_rows = out.loc[idx].copy()
        known = player_rows[player_rows["age_used"].notna()]
        if known.empty:
            continue
        known_seasons = pd.to_numeric(known[season_col], errors="coerce").to_numpy(dtype=float)
        known_ages = pd.to_numeric(known["age_used"], errors="coerce").to_numpy(dtype=float)

        missing_mask = player_rows["age_used"].isna()
        missing_idx = player_rows.index[missing_mask]
        for row_idx in missing_idx:
            target_season = float(out.at[row_idx, season_col])
            est = known_ages + (target_season - known_seasons)
            est = est[np.isfinite(est)]
            if est.size == 0:
                continue
            out.at[row_idx, "age_used"] = float(np.median(est))
            out.at[row_idx, "age_source"] = "player_inferred"

    group_cols = group_cols or [level_col]
    fillable = out["age_used"].isna()
    if fillable.any():
        med = (
            out.groupby(group_cols, dropna=False)["age_used"]
            .median(numeric_only=True)
            .rename("__group_med")
            .reset_index()
        )
        out = out.merge(med, on=group_cols, how="left")
        impute_mask = out["age_used"].isna() & out["__group_med"].notna()
        out.loc[impute_mask, "age_used"] = out.loc[impute_mask, "__group_med"]
        out.loc[impute_mask, "age_source"] = "group_imputed"
        out = out.drop(columns="__group_med")

    remaining = out["age_used"].isna()
    if remaining.any():
        global_median = float(out["age_used"].median(skipna=True))
        if not np.isfinite(global_median):
            global_median = default_age
        out.loc[remaining, "age_used"] = global_median
        out.loc[remaining, "age_source"] = "global_imputed"

    out["age_used"] = pd.to_numeric(out["age_used"], errors="coerce").round(1)
    return out
