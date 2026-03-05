from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetConfig


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_parquet(path)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sanitize_weights(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    out = out.clip(lower=0.0)
    return out


def aggregate_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    weight_col: str,
    numeric_cols: Iterable[str],
    metadata_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if weight_col not in out.columns:
        out[weight_col] = 1.0
    out[weight_col] = _sanitize_weights(out[weight_col])

    numeric_cols = [c for c in numeric_cols if c in out.columns and c not in key_cols and c != weight_col]
    meta_cols = [c for c in (metadata_cols or []) if c in out.columns and c not in key_cols]
    if not numeric_cols and not meta_cols:
        return out.drop_duplicates(subset=key_cols).copy()

    grouped = out.groupby(key_cols, dropna=False)
    agg = grouped[[weight_col]].sum().rename(columns={weight_col: "__w_sum"}).reset_index()
    agg[weight_col] = agg["__w_sum"]

    for col in numeric_cols:
        temp = out[[*key_cols, col, weight_col]].copy()
        col_num = pd.to_numeric(temp[col], errors="coerce")
        valid = col_num.notna() & np.isfinite(col_num)
        temp["__col_num"] = col_num
        temp["__w_eff"] = np.where(valid, temp[weight_col], 0.0)
        temp["__w_col"] = np.where(valid, col_num * temp[weight_col], 0.0)
        weighted_sum = (
            temp.groupby(key_cols, dropna=False)["__w_col"].sum().rename("__w_num").reset_index()
        )
        weighted_den = (
            temp.groupby(key_cols, dropna=False)["__w_eff"].sum().rename("__w_den").reset_index()
        )
        fallback_mean = (
            temp.groupby(key_cols, dropna=False)["__col_num"]
            .mean()
            .rename("__fallback")
            .reset_index()
        )
        merged = weighted_sum.merge(weighted_den, on=key_cols, how="left")
        merged = merged.merge(fallback_mean, on=key_cols, how="left")
        merged[col] = np.where(
            merged["__w_den"] > 0.0,
            merged["__w_num"] / merged["__w_den"],
            merged["__fallback"],
        )
        agg = agg.merge(merged[[*key_cols, col]], on=key_cols, how="left")

    if meta_cols:
        selector = out.sort_values(weight_col, ascending=False).drop_duplicates(key_cols)
        agg = agg.merge(selector[[*key_cols, *meta_cols]], on=key_cols, how="left")

    return agg.drop(columns=["__w_sum"], errors="ignore")


def merge_base_and_regressed(
    base_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    keys = [cfg.id_col, cfg.season_col, cfg.level_col]
    reg_numeric = [c for c in reg_df.columns if c.endswith("_reg") or c.endswith("_n")]
    reg_numeric += [c for c in reg_df.columns if c.endswith("_raw") and c != "reg_prop"]
    reg_numeric = sorted(set(c for c in reg_numeric if c in reg_df.columns))

    reg_weight_col = cfg.exposure_col if cfg.exposure_col in reg_df.columns else ""
    if not reg_weight_col:
        n_cols = [c for c in reg_df.columns if c.endswith("_n")]
        if n_cols:
            reg_df = reg_df.copy()
            reg_df["__reg_weight"] = reg_df[n_cols].mean(axis=1, numeric_only=True)
            reg_weight_col = "__reg_weight"
        else:
            reg_df = reg_df.copy()
            reg_df["__reg_weight"] = 1.0
            reg_weight_col = "__reg_weight"

    reg_meta = [cfg.name_col] if cfg.name_col in reg_df.columns else []
    reg_dedup = aggregate_duplicates(
        reg_df,
        key_cols=keys,
        weight_col=reg_weight_col,
        numeric_cols=reg_numeric,
        metadata_cols=reg_meta,
    )

    base_numeric = [
        c
        for c in [cfg.exposure_col, cfg.fallback_exposure_col, cfg.age_col, *cfg.extra_numeric_cols]
        if c and c in base_df.columns
    ]
    base_meta = [
        c
        for c in [cfg.name_col, cfg.hand_col, cfg.team_col]
        if c and c in base_df.columns
    ]
    base_weight_col = cfg.exposure_col if cfg.exposure_col in base_df.columns else ""
    if not base_weight_col and cfg.fallback_exposure_col and cfg.fallback_exposure_col in base_df.columns:
        base_weight_col = cfg.fallback_exposure_col
    if not base_weight_col:
        base_df = base_df.copy()
        base_df["__base_weight"] = 1.0
        base_weight_col = "__base_weight"

    base_dedup = aggregate_duplicates(
        base_df,
        key_cols=keys,
        weight_col=base_weight_col,
        numeric_cols=base_numeric,
        metadata_cols=base_meta,
    )

    keep_base_cols = [*keys, *base_numeric, *base_meta]
    keep_base_cols = [c for c in keep_base_cols if c in base_dedup.columns]
    merged = reg_dedup.merge(base_dedup[keep_base_cols], on=keys, how="left", suffixes=("", "_base"))

    if cfg.exposure_col not in merged.columns:
        merged[cfg.exposure_col] = np.nan
    if cfg.fallback_exposure_col and cfg.fallback_exposure_col in merged.columns:
        merged[cfg.exposure_col] = merged[cfg.exposure_col].fillna(merged[cfg.fallback_exposure_col])
    merged[cfg.exposure_col] = pd.to_numeric(merged[cfg.exposure_col], errors="coerce")
    merged[cfg.exposure_col] = merged[cfg.exposure_col].fillna(0.0)
    return merged


def build_player_season_table(
    merged_df: pd.DataFrame,
    cfg: DatasetConfig,
    metric_cols: list[str],
) -> pd.DataFrame:
    if merged_df.empty:
        return merged_df.copy()
    id_col = cfg.id_col
    season_col = cfg.season_col
    level_col = cfg.level_col
    exposure_col = cfg.exposure_col
    keys = [id_col, season_col]

    df = merged_df.copy()
    df[exposure_col] = _sanitize_weights(df[exposure_col])
    if cfg.name_col not in df.columns:
        df[cfg.name_col] = df[id_col].astype(str)
    if cfg.age_col not in df.columns:
        df[cfg.age_col] = np.nan

    n_cols = []
    for metric in metric_cols:
        base_name = metric[:-4] if metric.endswith("_reg") else metric
        candidate = f"{base_name}_n"
        if candidate in df.columns:
            n_cols.append(candidate)

    numeric_cols = [*metric_cols, *sorted(set(n_cols)), exposure_col, cfg.age_col]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    meta_cols = [cfg.name_col, level_col]
    if cfg.hand_col and cfg.hand_col in df.columns:
        meta_cols.append(cfg.hand_col)
    if cfg.team_col and cfg.team_col in df.columns:
        meta_cols.append(cfg.team_col)

    out = aggregate_duplicates(
        df,
        key_cols=keys,
        weight_col=exposure_col,
        numeric_cols=numeric_cols,
        metadata_cols=meta_cols,
    )
    out = out.rename(columns={level_col: "level_id_source"})
    return out


def validate_projection_schema(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    metric_cols: list[str],
) -> list[str]:
    required = [
        id_col,
        name_col,
        "source_season",
        "target_season",
        "projection_version",
        "age_used",
        "age_source",
        "volatility_index",
    ]
    for metric in metric_cols:
        required.extend(
            [
                f"{metric}_proj_p20",
                f"{metric}_proj_p25",
                f"{metric}_proj_p50",
                f"{metric}_proj_p75",
                f"{metric}_proj_p80",
                f"{metric}_proj_spread",
            ]
        )
    missing = [c for c in required if c not in df.columns]
    return missing
