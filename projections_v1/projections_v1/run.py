from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .age import infer_and_impute_age
from .backtest import run_rolling_backtest
from .config import ProjectionConfig, load_config, resolve_metric_settings
from .equivalency import apply_simple_mlb_equivalency
from .io import (
    build_player_season_table,
    ensure_dir,
    merge_base_and_regressed,
    read_parquet,
    validate_projection_schema,
)
from .point_forecast import project_next_season
from .uncertainty import apply_uncertainty_bands, build_transition_deltas

PITCHER_STUFF_QMATCH_BLEND = 0.50
PITCHER_STUFF_QMATCH_REF_LEVEL_ID = 1
PITCHER_STUFF_QMATCH_MIN_REF_ROWS = 40
PITCHER_STUFF_QMATCH_MIN_PROJECTED_IP = 50.0
PITCHER_PRIMARY_POSITION_COL = "lb2_primary_position"
PITCHER_PRIMARY_POSITION_TOKEN = "P"


def _weighted_rate(group: pd.DataFrame, rate_col: str, weight_col: str) -> float:
    rates = pd.to_numeric(group[rate_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    mask = rates.notna() & weights.notna() & np.isfinite(rates) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(rates[mask], weights=weights[mask]))


def _rank_pct(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() < 5:
        return pd.Series(0.5, index=series.index, dtype=float)
    return vals.rank(method="average", pct=True).fillna(0.5).clip(0.01, 0.99)


def _clip_series(values: pd.Series, metric: str, bounds: dict[str, tuple[float, float]]) -> pd.Series:
    if metric not in bounds:
        return values
    low, high = bounds[metric]
    return pd.to_numeric(values, errors="coerce").clip(lower=low, upper=high)


def _apply_pitcher_stuff_blended_quantile_match(
    projections: pd.DataFrame,
    *,
    base_df: pd.DataFrame,
    season_col: str,
    level_col: str,
    source_season: int,
    blend_weight: float = PITCHER_STUFF_QMATCH_BLEND,
    min_projected_ip: float = PITCHER_STUFF_QMATCH_MIN_PROJECTED_IP,
) -> pd.DataFrame:
    out = projections.copy()
    if out.empty or "stuff_raw_proj_p50" not in out.columns:
        return out

    if {"stuff_raw", season_col, level_col}.issubset(set(base_df.columns)):
        ref = base_df.copy()
        ref["stuff_raw"] = pd.to_numeric(ref["stuff_raw"], errors="coerce")
        ref[season_col] = pd.to_numeric(ref[season_col], errors="coerce")
        ref[level_col] = pd.to_numeric(ref[level_col], errors="coerce")
        ref = ref[
            ref["stuff_raw"].notna()
            & np.isfinite(ref["stuff_raw"])
            & ref[season_col].eq(float(source_season))
            & ref[level_col].eq(float(PITCHER_STUFF_QMATCH_REF_LEVEL_ID))
        ].copy()
    else:
        ref = pd.DataFrame(columns=["stuff_raw"])
    if len(ref) < int(PITCHER_STUFF_QMATCH_MIN_REF_ROWS):
        return out

    eligible = pd.Series(True, index=out.index)
    if "level_id_source" in out.columns:
        src_level = pd.to_numeric(out["level_id_source"], errors="coerce")
        eligible = src_level.eq(float(PITCHER_STUFF_QMATCH_REF_LEVEL_ID))
    applied_position_filter = False
    if (
        "pitcher_mlbid" in out.columns
        and {"pitcher_mlbid", PITCHER_PRIMARY_POSITION_COL}.issubset(set(base_df.columns))
    ):
        pos_map = base_df[["pitcher_mlbid", PITCHER_PRIMARY_POSITION_COL]].copy()
        pos_map["pitcher_mlbid"] = pd.to_numeric(pos_map["pitcher_mlbid"], errors="coerce").astype("Int64")
        pos_map[PITCHER_PRIMARY_POSITION_COL] = (
            pos_map[PITCHER_PRIMARY_POSITION_COL].fillna("").astype(str).str.strip().str.upper()
        )
        pitcher_ids = set(
            pos_map.loc[
                pos_map[PITCHER_PRIMARY_POSITION_COL].eq(PITCHER_PRIMARY_POSITION_TOKEN)
                & pos_map["pitcher_mlbid"].notna(),
                "pitcher_mlbid",
            ].astype("int64").tolist()
        )
        if pitcher_ids:
            proj_ids = pd.to_numeric(out["pitcher_mlbid"], errors="coerce").astype("Int64")
            eligible = eligible & proj_ids.isin(pitcher_ids)
            applied_position_filter = True
    if not applied_position_filter:
        ip_col = "IP_proj_p50" if "IP_proj_p50" in out.columns else ("IP_marcel_proj_p50" if "IP_marcel_proj_p50" in out.columns else None)
        if ip_col is not None:
            ip_vals = pd.to_numeric(out[ip_col], errors="coerce")
            eligible = eligible & ip_vals.ge(float(min_projected_ip))
    if not bool(eligible.any()):
        return out

    x = pd.to_numeric(out.loc[eligible, "stuff_raw_proj_p50"], errors="coerce")
    valid = x.notna() & np.isfinite(x)
    if int(valid.sum()) < 5:
        return out

    rank = x.loc[valid].rank(method="average", pct=True).clip(0.001, 0.999)
    ref_vals = pd.to_numeric(ref["stuff_raw"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref_vals) < int(PITCHER_STUFF_QMATCH_MIN_REF_ROWS):
        return out
    matched = pd.Series(np.nan, index=x.index, dtype=float)
    matched.loc[rank.index] = np.quantile(ref_vals, rank.to_numpy(dtype=float))

    w = float(np.clip(blend_weight, 0.0, 1.0))
    new_p50 = x.copy()
    new_p50.loc[valid] = ((1.0 - w) * x.loc[valid]) + (w * matched.loc[valid])
    delta = new_p50 - x

    out.loc[eligible, "stuff_raw_proj_p50"] = new_p50
    for pct in ("p20", "p25", "p75", "p80", "p600"):
        col = f"stuff_raw_proj_{pct}"
        if col not in out.columns:
            continue
        cur = pd.to_numeric(out.loc[eligible, col], errors="coerce")
        out.loc[eligible, col] = cur + delta

    col20 = "stuff_raw_proj_p20" if "stuff_raw_proj_p20" in out.columns else None
    col25 = "stuff_raw_proj_p25" if "stuff_raw_proj_p25" in out.columns else None
    col75 = "stuff_raw_proj_p75" if "stuff_raw_proj_p75" in out.columns else None
    col80 = "stuff_raw_proj_p80" if "stuff_raw_proj_p80" in out.columns else None
    c50 = pd.to_numeric(out.loc[eligible, "stuff_raw_proj_p50"], errors="coerce")
    if col20:
        c20 = pd.to_numeric(out.loc[eligible, col20], errors="coerce")
    else:
        c20 = None
    if col25:
        c25 = pd.to_numeric(out.loc[eligible, col25], errors="coerce")
    else:
        c25 = None
    if col75:
        c75 = pd.to_numeric(out.loc[eligible, col75], errors="coerce")
    else:
        c75 = None
    if col80:
        c80 = pd.to_numeric(out.loc[eligible, col80], errors="coerce")
    else:
        c80 = None
    if c20 is not None and c25 is not None:
        c25 = np.maximum(c25, c20)
    if c25 is not None:
        c50 = np.maximum(c50, c25)
    if c75 is not None:
        c75 = np.maximum(c75, c50)
    if c80 is not None and c75 is not None:
        c80 = np.maximum(c80, c75)
    if c20 is not None and col20:
        out.loc[eligible, col20] = c20
    if c25 is not None and col25:
        out.loc[eligible, col25] = c25
    out.loc[eligible, "stuff_raw_proj_p50"] = c50
    if c75 is not None and col75:
        out.loc[eligible, col75] = c75
    if c80 is not None and col80:
        out.loc[eligible, col80] = c80
    if col25 and col75 and "stuff_raw_proj_spread" in out.columns:
        out.loc[eligible, "stuff_raw_proj_spread"] = (
            pd.to_numeric(out.loc[eligible, col75], errors="coerce")
            - pd.to_numeric(out.loc[eligible, col25], errors="coerce")
        )
    return out


def _derive_conservative_metric_factors(
    backtest_df: pd.DataFrame,
    *,
    target_metrics: list[str],
    min_rows: float = 1200.0,
    min_seasons: int = 3,
    max_down: float = 0.18,
) -> dict[str, float]:
    factors = {metric: 1.0 for metric in target_metrics}
    if backtest_df.empty or "metric" not in backtest_df.columns:
        return factors

    work = backtest_df.copy()
    for metric in target_metrics:
        rows = work[work["metric"].astype(str) == str(metric)]
        if rows.empty:
            continue
        row = rows.sort_values("n", ascending=False).iloc[0]
        n = float(pd.to_numeric(row.get("n"), errors="coerce"))
        seasons = float(pd.to_numeric(row.get("seasons"), errors="coerce"))
        ratio = pd.to_numeric(row.get("actual_over_pred"), errors="coerce")
        if not np.isfinite(ratio):
            pred_mean = float(pd.to_numeric(row.get("pred_mean"), errors="coerce"))
            actual_mean = float(pd.to_numeric(row.get("actual_mean"), errors="coerce"))
            if np.isfinite(pred_mean) and pred_mean != 0.0 and np.isfinite(actual_mean):
                ratio = actual_mean / pred_mean
        if not np.isfinite(ratio) or ratio >= 1.0:
            continue
        conf_n = float(np.clip(n / max(float(min_rows), 1.0), 0.0, 1.0)) if np.isfinite(n) else 0.0
        conf_s = float(np.clip(seasons / max(float(min_seasons), 1.0), 0.0, 1.0)) if np.isfinite(seasons) else 0.0
        confidence = conf_n * conf_s
        down = float(np.clip((1.0 - float(ratio)) * confidence, 0.0, float(max_down)))
        factors[metric] = float(np.clip(1.0 - down, 1.0 - float(max_down), 1.0))
    return factors


def _has_nontrivial_metric_factors(metric_factors: dict[str, float], tol: float = 1e-8) -> bool:
    for factor in metric_factors.values():
        val = float(pd.to_numeric(factor, errors="coerce"))
        if np.isfinite(val) and abs(val - 1.0) > float(tol):
            return True
    return False


def _apply_metric_scale_factors(
    projections: pd.DataFrame,
    *,
    metric_factors: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = projections.copy()
    if out.empty or not metric_factors:
        return out

    pct_order = [20, 25, 50, 75, 80]
    for metric, factor in metric_factors.items():
        mult = float(pd.to_numeric(factor, errors="coerce"))
        if not np.isfinite(mult) or mult <= 0.0 or abs(mult - 1.0) < 1e-8:
            continue

        present = [p for p in pct_order if f"{metric}_proj_p{p}" in out.columns]
        if not present:
            continue

        prev = None
        for pct in pct_order:
            col = f"{metric}_proj_p{pct}"
            if col not in out.columns:
                continue
            vals = pd.to_numeric(out[col], errors="coerce") * mult
            vals = _clip_series(vals, metric, bounds).clip(lower=0.0)
            if prev is not None:
                vals = np.maximum(vals, prev)
            out[col] = vals
            prev = vals

        if 25 in present and 75 in present:
            out[f"{metric}_proj_spread"] = (
                pd.to_numeric(out[f"{metric}_proj_p75"], errors="coerce")
                - pd.to_numeric(out[f"{metric}_proj_p25"], errors="coerce")
            )
        elif 20 in present and 80 in present:
            out[f"{metric}_proj_spread"] = (
                pd.to_numeric(out[f"{metric}_proj_p80"], errors="coerce")
                - pd.to_numeric(out[f"{metric}_proj_p20"], errors="coerce")
            )
    return out


def _compute_hitter_recent_anchors(
    player_season_df: pd.DataFrame,
    *,
    id_col: str,
    season_col: str,
    level_col: str,
    team_col: str | None,
    source_season: int,
    mlb_level_id: int,
    park_data_path: Path,
    seasons_back: int = 3,
) -> pd.DataFrame:
    if (
        player_season_df.empty
        or not team_col
        or team_col not in player_season_df.columns
        or not park_data_path.exists()
    ):
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    la20_col = None
    for col in ["LA_gte_20_reg", "LA_gte_20"]:
        if col in player_season_df.columns:
            la20_col = col
            break
    damage_col = None
    for col in ["damage_rate_reg", "damage_rate"]:
        if col in player_season_df.columns:
            damage_col = col
            break
    if "PA" not in player_season_df.columns or "bbe" not in player_season_df.columns:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])
    if la20_col is None and damage_col is None:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    park = pd.read_parquet(park_data_path)
    if la20_col is not None and {"la_gte_20_bbe", "HR_per_LA_gte_20_pct"}.issubset(set(park.columns)):
        rate_col = "HR_per_LA_gte_20_pct"
        weight_col = "la_gte_20_bbe"
        use_metric_col = la20_col
    elif {"damage_bbe", "HR_per_damage_BBE_pct"}.issubset(set(park.columns)) and damage_col is not None:
        rate_col = "HR_per_damage_BBE_pct"
        weight_col = "damage_bbe"
        use_metric_col = damage_col
    else:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])
    required = {"home_team", "season", "level_id", weight_col, rate_col}
    if not required.issubset(set(park.columns)):
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    park = park.copy()
    park["season"] = pd.to_numeric(park["season"], errors="coerce")
    park["level_id"] = pd.to_numeric(park["level_id"], errors="coerce")
    park = park[park["season"].notna() & park["level_id"].notna()].copy()
    park["season"] = park["season"].astype(int)
    park["level_id"] = park["level_id"].astype(int)
    park = park[park["level_id"] == int(mlb_level_id)].copy()
    if park.empty:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    min_season = int(source_season - int(seasons_back) + 1)
    park = park[(park["season"] >= min_season) & (park["season"] <= int(source_season))].copy()
    if park.empty:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])
    park["home_team_norm"] = park["home_team"].map(_normalize_team_code)

    park_team_rates = (
        park.groupby(["season", "home_team_norm"], dropna=False).apply(
            lambda g: _weighted_rate(g, rate_col, weight_col)
        )
        .rename("hr_rate")
        .reset_index()
    )
    park_lg_rates = (
        park.groupby("season", dropna=False)
        .apply(lambda g: _weighted_rate(g, rate_col, weight_col))
        .rename("lg_hr_rate")
        .reset_index()
    )

    hist = player_season_df.copy()
    hist[season_col] = pd.to_numeric(hist[season_col], errors="coerce")
    hist[level_col] = pd.to_numeric(hist[level_col], errors="coerce")
    hist["PA"] = pd.to_numeric(hist["PA"], errors="coerce")
    hist["bbe"] = pd.to_numeric(hist["bbe"], errors="coerce")
    hist[use_metric_col] = pd.to_numeric(hist[use_metric_col], errors="coerce")
    hist = hist[
        hist[season_col].notna()
        & hist[level_col].notna()
        & (hist[level_col].astype(int) == int(mlb_level_id))
        & (hist[season_col].astype(int) >= min_season)
        & (hist[season_col].astype(int) <= int(source_season))
    ].copy()
    if hist.empty:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    hist["season_i"] = hist[season_col].astype(int)
    hist["team_norm"] = hist[team_col].map(_normalize_team_code)
    hist = hist.merge(
        park_team_rates.rename(columns={"season": "season_i", "home_team_norm": "team_norm"}),
        on=["season_i", "team_norm"],
        how="left",
    )
    hist = hist.merge(
        park_lg_rates.rename(columns={"season": "season_i"}),
        on="season_i",
        how="left",
    )
    hist["hr_rate"] = hist["hr_rate"].fillna(hist["lg_hr_rate"])

    lag = int(source_season) - hist["season_i"]
    hist["rec_w"] = lag.map({0: 5.0, 1: 4.0, 2: 3.0}).fillna(0.0)
    hist = hist[hist["rec_w"] > 0].copy()
    if hist.empty:
        return pd.DataFrame(columns=[id_col, "recent_pa_total", "recent_pa_wavg", "recent_hr_pa_anchor"])

    hist["hr_pa_anchor"] = (
        (hist[use_metric_col].clip(lower=0.0) / 100.0)
        * (hist["bbe"].clip(lower=0.0) / hist["PA"].replace(0.0, np.nan))
        * (hist["hr_rate"].clip(lower=0.0) / 100.0)
    )
    hist["hr_pa_anchor"] = hist["hr_pa_anchor"].replace([np.inf, -np.inf], np.nan)
    hist["rec_pa_w"] = hist["rec_w"] * hist["PA"].clip(lower=0.0).fillna(0.0)

    def _agg(grp: pd.DataFrame) -> pd.Series:
        pa_total = float(pd.to_numeric(grp["PA"], errors="coerce").fillna(0.0).sum())
        w = pd.to_numeric(grp["rec_w"], errors="coerce").fillna(0.0)
        pa = pd.to_numeric(grp["PA"], errors="coerce")
        pa_wavg = float(np.average(pa[w > 0], weights=w[w > 0])) if (w > 0).any() else 0.0

        hrpa = pd.to_numeric(grp["hr_pa_anchor"], errors="coerce")
        w_hr = pd.to_numeric(grp["rec_pa_w"], errors="coerce").fillna(0.0)
        mask = hrpa.notna() & (w_hr > 0)
        if mask.any():
            hr_pa_anchor = float(np.average(hrpa[mask], weights=w_hr[mask]))
        else:
            hr_pa_anchor = np.nan
        return pd.Series(
            {
                "recent_pa_total": pa_total,
                "recent_pa_wavg": pa_wavg,
                "recent_hr_pa_anchor": hr_pa_anchor,
            }
        )

    anchors = hist.groupby(id_col, dropna=False).apply(_agg).reset_index()
    return anchors


def _apply_hitter_pa_role_floor(
    projections: pd.DataFrame,
    *,
    id_col: str,
    anchors_df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    min_recent_pa: float = 1200.0,
    p50_floor_factor: float = 0.90,
) -> pd.DataFrame:
    out = projections.copy()
    if out.empty or anchors_df.empty or "PA_proj_p50" not in out.columns:
        return out
    out = out.merge(anchors_df[[id_col, "recent_pa_total", "recent_pa_wavg"]], on=id_col, how="left")

    eligible = (
        pd.to_numeric(out["recent_pa_total"], errors="coerce").fillna(0.0) >= float(min_recent_pa)
    ) & pd.to_numeric(out["recent_pa_wavg"], errors="coerce").notna()
    if not eligible.any():
        return out.drop(columns=["recent_pa_total", "recent_pa_wavg"], errors="ignore")

    target_p50 = pd.to_numeric(out["recent_pa_wavg"], errors="coerce") * float(p50_floor_factor)
    pa20 = pd.to_numeric(out["PA_proj_p20"], errors="coerce") if "PA_proj_p20" in out.columns else None
    pa25 = pd.to_numeric(out["PA_proj_p25"], errors="coerce")
    pa50 = pd.to_numeric(out["PA_proj_p50"], errors="coerce")
    pa75 = pd.to_numeric(out["PA_proj_p75"], errors="coerce")
    pa80 = pd.to_numeric(out["PA_proj_p80"], errors="coerce") if "PA_proj_p80" in out.columns else None
    delta = (target_p50 - pa50).clip(lower=0.0)
    delta = delta.where(eligible, 0.0)

    if pa20 is not None:
        pa20 = pa20 + (0.55 * delta)
    pa25 = pa25 + (0.75 * delta)
    pa50 = pa50 + delta
    pa75 = pa75 + (1.10 * delta)
    if pa80 is not None:
        pa80 = pa80 + (1.20 * delta)

    if pa20 is not None:
        pa20 = _clip_series(pa20, "PA", bounds).clip(lower=0.0)
    pa25 = _clip_series(pa25, "PA", bounds).clip(lower=0.0)
    pa50 = _clip_series(pa50, "PA", bounds).clip(lower=0.0)
    pa75 = _clip_series(pa75, "PA", bounds).clip(lower=0.0)
    if pa80 is not None:
        pa80 = _clip_series(pa80, "PA", bounds).clip(lower=0.0)
    if pa20 is not None:
        pa25 = np.maximum(pa25, pa20)
    pa50 = np.maximum(pa50, pa25)
    pa75 = np.maximum(pa75, pa50)
    if pa80 is not None:
        pa80 = np.maximum(pa80, pa75)
        out["PA_proj_p80"] = pa80
    if pa20 is not None:
        out["PA_proj_p20"] = pa20
    out["PA_proj_p25"] = pa25
    out["PA_proj_p50"] = pa50
    out["PA_proj_p75"] = pa75
    out["PA_proj_spread"] = pa75 - pa25
    return out.drop(columns=["recent_pa_total", "recent_pa_wavg"], errors="ignore")


def _apply_hitter_playing_time_adjustments(
    projections: pd.DataFrame,
    *,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = projections.copy()
    if out.empty:
        return out

    damage_col = None
    for col in ["damage_rate_reg_proj_p50", "damage_rate_proj_p50"]:
        if col in out.columns:
            damage_col = col
            break
    if damage_col is None:
        return out

    cve_col = None
    for col in ["contact_vs_avg_reg_proj_p50", "CVE_reg_proj_p50", "CVE_proj_p50"]:
        if col in out.columns:
            cve_col = col
            break
    pull_col = None
    for col in ["pull_FB_pct_reg_proj_p50", "pull_FB_pct_proj_p50"]:
        if col in out.columns:
            pull_col = col
            break
    seager_col = None
    for col in ["SEAGER_reg_proj_p50", "SEAGER_proj_p50"]:
        if col in out.columns:
            seager_col = col
            break

    dmg_pct = _rank_pct(out[damage_col])
    cve_pct = _rank_pct(out[cve_col]) if cve_col else pd.Series(0.5, index=out.index, dtype=float)
    pull_pct = _rank_pct(out[pull_col]) if pull_col else pd.Series(0.5, index=out.index, dtype=float)
    seager_pct = _rank_pct(out[seager_col]) if seager_col else pd.Series(0.5, index=out.index, dtype=float)
    dmg_cve_high = dmg_pct * cve_pct
    pull_seager_high = pull_pct * seager_pct
    all_high = dmg_pct * cve_pct * pull_pct * seager_pct
    dmg_cve_low = (1.0 - dmg_pct) * (1.0 - cve_pct)
    pull_seager_low = (1.0 - pull_pct) * (1.0 - seager_pct)
    all_low = (1.0 - dmg_pct) * (1.0 - cve_pct) * (1.0 - pull_pct) * (1.0 - seager_pct)
    both_high = 0.65 * dmg_cve_high + 0.35 * pull_seager_high
    both_low = 0.65 * dmg_cve_low + 0.35 * pull_seager_low

    # Heavily weight damage/CVE, then add pull+SEAGER with interaction terms.
    raw_score = (
        (0.45 * dmg_pct)
        + (0.20 * cve_pct)
        + (0.20 * pull_pct)
        + (0.15 * seager_pct)
        + (0.35 * both_high)
        + (0.15 * all_high)
        - (0.35 * both_low)
        - (0.12 * all_low)
    )
    centered = raw_score - 0.5
    scale = (1.0 + (0.42 * centered) + (0.24 * (both_high - both_low)) + (0.10 * (all_high - all_low))).clip(0.65, 1.40)

    for metric in ["PA", "bbe"]:
        c20 = f"{metric}_proj_p20"
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        c80 = f"{metric}_proj_p80"
        cs = f"{metric}_proj_spread"
        if c25 not in out.columns or c50 not in out.columns or c75 not in out.columns:
            continue

        p20 = pd.to_numeric(out[c20], errors="coerce") if c20 in out.columns else pd.to_numeric(out[c25], errors="coerce")
        p25 = pd.to_numeric(out[c25], errors="coerce") * scale
        p50 = pd.to_numeric(out[c50], errors="coerce") * scale
        p75 = pd.to_numeric(out[c75], errors="coerce") * scale
        p80 = pd.to_numeric(out[c80], errors="coerce") if c80 in out.columns else pd.to_numeric(out[c75], errors="coerce")
        p20 = p20 * scale
        p80 = p80 * scale

        # Pull P50 toward the upper tail for high-skill hitters and toward the lower tail for low-low hitters.
        pull_up = (
            (0.20 * dmg_pct)
            + (0.10 * cve_pct)
            + (0.08 * pull_pct)
            + (0.07 * seager_pct)
            + (0.14 * both_high)
            + (0.10 * all_high)
        )
        pull_down = (0.24 * both_low) + (0.08 * all_low)
        p50 = p50 + (pull_up * (p75 - p50)) - (pull_down * (p50 - p25))

        # Modestly reshape tails to reflect role/security effects.
        p20 = p20 + ((0.03 * dmg_pct) * (p25 - p20)) - (((0.08 * both_low) + (0.04 * all_low)) * (p25 - p20))
        p25 = p25 + (
            ((0.06 * dmg_pct) + (0.03 * pull_pct) + (0.03 * seager_pct)) * (p50 - p25)
        ) - (((0.16 * both_low) + (0.08 * all_low)) * (p50 - p25))
        p75 = p75 + (((0.10 * both_high) + (0.08 * all_high)) * (p75 - p50))
        p80 = p80 + (((0.06 * both_high) + (0.06 * all_high)) * (p80 - p75))

        p20 = _clip_series(p20, metric, bounds)
        p25 = _clip_series(p25, metric, bounds)
        p50 = _clip_series(p50, metric, bounds)
        p75 = _clip_series(p75, metric, bounds)
        p80 = _clip_series(p80, metric, bounds)

        p20 = p20.clip(lower=0.0)
        p25 = p25.clip(lower=0.0)
        p50 = p50.clip(lower=0.0)
        p75 = p75.clip(lower=0.0)
        p80 = p80.clip(lower=0.0)

        p25 = np.maximum(p25, p20)
        p50 = np.maximum(p50, p25)
        p75 = np.maximum(p75, p50)
        p80 = np.maximum(p80, p75)

        out[c20] = p20
        out[c25] = p25
        out[c50] = p50
        out[c75] = p75
        out[c80] = p80
        out[cs] = p75 - p25

    return out


def _apply_hr_power_calibration(
    projections: pd.DataFrame,
    *,
    id_col: str,
    anchors_df: pd.DataFrame | None = None,
    min_recent_pa: float = 1200.0,
    p50_floor_factor: float = 0.90,
    p75_floor_factor: float = 1.05,
) -> pd.DataFrame:
    out = projections.copy()
    required = ["HR_proj_p25", "HR_proj_p50", "HR_proj_p75"]
    if out.empty or not all(c in out.columns for c in required):
        return out

    feature_candidates = [
        ("damage_rate_reg_proj_p50", 0.45),
        ("damage_rate_proj_p50", 0.45),
        ("LA_gte_20_reg_proj_p50", 0.20),
        ("LA_gte_20_proj_p50", 0.20),
        ("EV90th_reg_proj_p50", 0.25),
        ("EV90th_proj_p50", 0.25),
        ("max_EV_reg_proj_p50", 0.20),
        ("max_EV_proj_p50", 0.20),
        ("SEAGER_reg_proj_p50", 0.10),
        ("SEAGER_proj_p50", 0.10),
    ]
    used: list[tuple[str, float]] = []
    seen_roots: set[str] = set()
    for col, weight in feature_candidates:
        root = col.replace("_reg", "")
        if col in out.columns and root not in seen_roots:
            used.append((col, weight))
            seen_roots.add(root)
    if not used:
        return out

    raw_weight_sum = sum(w for _, w in used)
    if raw_weight_sum <= 0:
        return out
    norm_weights = [(c, w / raw_weight_sum) for c, w in used]

    power_score = pd.Series(0.0, index=out.index, dtype=float)
    for col, w in norm_weights:
        power_score += _rank_pct(out[col]) * w

    # Positive-only uplift for elite power skill; stronger for upper tails.
    elite = (power_score - 0.55).clip(lower=0.0)
    super_elite = (power_score - 0.85).clip(lower=0.0)
    mult_25 = (1.0 + (0.08 * elite) + (0.15 * super_elite)).clip(1.00, 1.10)
    mult_50 = (1.0 + (0.20 * elite) + (0.35 * super_elite)).clip(1.00, 1.25)
    mult_75 = (1.0 + (0.38 * elite) + (0.75 * super_elite)).clip(1.00, 1.50)

    out["HR_proj_p25"] = pd.to_numeric(out["HR_proj_p25"], errors="coerce") * mult_25
    out["HR_proj_p50"] = pd.to_numeric(out["HR_proj_p50"], errors="coerce") * mult_50
    out["HR_proj_p75"] = pd.to_numeric(out["HR_proj_p75"], errors="coerce") * mult_75

    if anchors_df is not None and not anchors_df.empty and "PA_proj_p50" in out.columns and "PA_proj_p75" in out.columns:
        a = anchors_df[[id_col, "recent_pa_total", "recent_hr_pa_anchor"]].copy()
        out = out.merge(a, on=id_col, how="left")
        recent_pa = pd.to_numeric(out["recent_pa_total"], errors="coerce").fillna(0.0)
        anchor = pd.to_numeric(out["recent_hr_pa_anchor"], errors="coerce")
        eligible = (recent_pa >= float(min_recent_pa)) & anchor.notna() & np.isfinite(anchor)
        hr_floor_50 = (
            pd.to_numeric(out["PA_proj_p50"], errors="coerce").clip(lower=0.0)
            * anchor.clip(lower=0.0)
            * float(p50_floor_factor)
        )
        hr_floor_75 = (
            pd.to_numeric(out["PA_proj_p75"], errors="coerce").clip(lower=0.0)
            * anchor.clip(lower=0.0)
            * float(p75_floor_factor)
        )
        out.loc[eligible, "HR_proj_p50"] = np.maximum(
            pd.to_numeric(out.loc[eligible, "HR_proj_p50"], errors="coerce"),
            pd.to_numeric(hr_floor_50.loc[eligible], errors="coerce"),
        )
        out.loc[eligible, "HR_proj_p75"] = np.maximum(
            pd.to_numeric(out.loc[eligible, "HR_proj_p75"], errors="coerce"),
            pd.to_numeric(hr_floor_75.loc[eligible], errors="coerce"),
        )
        out = out.drop(columns=["recent_pa_total", "recent_hr_pa_anchor"], errors="ignore")

    p25 = pd.to_numeric(out["HR_proj_p25"], errors="coerce").clip(lower=0.0)
    p50 = pd.to_numeric(out["HR_proj_p50"], errors="coerce").clip(lower=0.0)
    p75 = pd.to_numeric(out["HR_proj_p75"], errors="coerce").clip(lower=0.0)
    p50 = np.maximum(p50, p25)
    p75 = np.maximum(p75, p50)
    out["HR_proj_p25"] = p25
    out["HR_proj_p50"] = p50
    out["HR_proj_p75"] = p75
    out["HR_proj_spread"] = p75 - p25
    return out


def _normalize_team_code(value: Any, known_codes: set[str] | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    parts = [p.strip() for p in re.split(r"[|,/;]+", text) if p.strip()]
    if not parts:
        return None
    if known_codes:
        for code in reversed(parts):
            if code in known_codes:
                return code
    return parts[-1]


def _derive_hitter_primary_position(
    merged_df: pd.DataFrame,
    *,
    id_col: str,
    season_col: str,
    exposure_col: str,
) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame(columns=[id_col, season_col, "position"])

    pos_map = {
        "is_C": "C",
        "is_X1B": "1B",
        "is_X2B": "2B",
        "is_X3B": "3B",
        "is_SS": "SS",
        "is_OF": "OF",
        "is_UT": "UT",
        "is_P": "P",
    }
    available = [c for c in pos_map if c in merged_df.columns]
    if not available:
        return pd.DataFrame(columns=[id_col, season_col, "position"])

    work = merged_df.copy()
    if exposure_col not in work.columns:
        work["__pos_weight"] = 1.0
        exposure_col = "__pos_weight"
    cols = [id_col, season_col, exposure_col, *available]
    work = work[cols].copy()
    work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
    work[season_col] = pd.to_numeric(work[season_col], errors="coerce")
    work = work[work[id_col].notna() & work[season_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=[id_col, season_col, "position"])
    work[id_col] = work[id_col].astype("int64")
    work[season_col] = work[season_col].astype("int64")
    exposure = pd.to_numeric(work[exposure_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    long_frames: list[pd.DataFrame] = []
    for col in available:
        flag = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        score = exposure * flag
        long_frames.append(
            pd.DataFrame(
                {
                    id_col: work[id_col].to_numpy(),
                    season_col: work[season_col].to_numpy(),
                    "position": pos_map[col],
                    "score": score.to_numpy(),
                }
            )
        )
    long = pd.concat(long_frames, ignore_index=True)
    long = long[long["score"] > 0].copy()
    if long.empty:
        return pd.DataFrame(columns=[id_col, season_col, "position"])

    agg = (
        long.groupby([id_col, season_col, "position"], dropna=False)["score"]
        .sum(numeric_only=True)
        .reset_index()
    )
    best = agg.sort_values(
        by=[id_col, season_col, "score", "position"],
        ascending=[True, True, False, True],
    ).drop_duplicates(subset=[id_col, season_col], keep="first")
    return best[[id_col, season_col, "position"]]


def _apply_hitter_park_adjustments(
    projections: pd.DataFrame,
    *,
    source_season: int,
    park_data_path: Path,
    mlb_level_id: int,
    window_years: int,
    team_col: str | None,
) -> pd.DataFrame:
    out = projections.copy()
    for stat in ["HR", "HR_damage", "HR_non_damage", "XBH", "H"]:
        for pct in [25, 50, 75]:
            out[f"{stat}_proj_p{pct}"] = np.nan
        out[f"{stat}_proj_spread"] = np.nan

    if out.empty or not park_data_path.exists() or not team_col or team_col not in out.columns:
        return out

    available_pcts = [p for p in [20, 25, 50, 75, 80] if f"bbe_proj_p{p}" in out.columns]
    if not available_pcts:
        return out
    la20_metric = "LA_gte_20_reg"
    if "LA_gte_20_reg_proj_p50" not in out.columns and "LA_gte_20_proj_p50" in out.columns:
        la20_metric = "LA_gte_20"
    if f"{la20_metric}_proj_p50" not in out.columns:
        return out
    la_lte_0_metric = "LA_lte_0_reg"
    if "LA_lte_0_reg_proj_p50" not in out.columns and "LA_lte_0_proj_p50" in out.columns:
        la_lte_0_metric = "LA_lte_0"
    if f"{la_lte_0_metric}_proj_p50" not in out.columns:
        return out
    damage_metric = "damage_rate_reg"
    if "damage_rate_reg_proj_p50" not in out.columns and "damage_rate_proj_p50" in out.columns:
        damage_metric = "damage_rate"
    if f"{damage_metric}_proj_p50" not in out.columns:
        return out

    park = pd.read_parquet(park_data_path)
    required = {
        "park_mlbid",
        "home_team",
        "season",
        "level_id",
        "damage_bbe",
        "non_damage_la_gte_20_bbe",
        "la_lte_0_bbe",
        "la_0_to_20_bbe",
        "la_gte_20_bbe",
        "bbe_total",
        "HR_per_damage_BBE_pct",
        "HR_per_non_damage_LA_gte_20_BBE_pct",
        "XBH_per_damage_BBE_pct",
        "XBH_per_BBE_pct",
        "Hits_per_LA_lte_0_pct",
        "Hits_per_LA_0_to_20_pct",
        "Hits_per_LA_gte_20_pct",
        "Hits_per_BBE_pct",
    }
    if not required.issubset(set(park.columns)):
        return out

    park = park.copy()
    park["season"] = pd.to_numeric(park["season"], errors="coerce")
    park["level_id"] = pd.to_numeric(park["level_id"], errors="coerce")
    park = park[park["season"].notna() & park["level_id"].notna()].copy()
    park["season"] = park["season"].astype(int)
    park["level_id"] = park["level_id"].astype(int)
    park = park[park["level_id"] == int(mlb_level_id)].copy()
    if park.empty:
        return out

    max_season = int(park["season"].max())
    src_season = int(min(source_season, max_season))
    park = park[park["season"] == src_season].copy()
    if park.empty:
        min_season = int(src_season - int(window_years) + 1)
        park = park[(park["season"] >= min_season) & (park["season"] <= src_season)].copy()
    if park.empty:
        return out

    park["home_team_norm"] = park["home_team"].map(lambda x: _normalize_team_code(x))
    park = park[park["home_team_norm"].notna()].copy()
    if park.empty:
        return out

    def _rate_series(grp: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "hr_damage_rate": _weighted_rate(grp, "HR_per_damage_BBE_pct", "damage_bbe"),
                "hr_non_damage_rate": _weighted_rate(
                    grp,
                    "HR_per_non_damage_LA_gte_20_BBE_pct",
                    "non_damage_la_gte_20_bbe",
                ),
                "xbh_damage_rate": _weighted_rate(grp, "XBH_per_damage_BBE_pct", "damage_bbe"),
                "xbh_bbe_rate": _weighted_rate(grp, "XBH_per_BBE_pct", "bbe_total"),
                "h_la_lte_0_rate": _weighted_rate(grp, "Hits_per_LA_lte_0_pct", "la_lte_0_bbe"),
                "h_la_0_to_20_rate": _weighted_rate(grp, "Hits_per_LA_0_to_20_pct", "la_0_to_20_bbe"),
                "h_la_gte_20_rate": _weighted_rate(grp, "Hits_per_LA_gte_20_pct", "la_gte_20_bbe"),
                "h_rate": _weighted_rate(grp, "Hits_per_BBE_pct", "bbe_total"),
            }
        )

    rate_cols = [
        "HR_per_damage_BBE_pct",
        "HR_per_non_damage_LA_gte_20_BBE_pct",
        "XBH_per_damage_BBE_pct",
        "XBH_per_BBE_pct",
        "Hits_per_LA_lte_0_pct",
        "Hits_per_LA_0_to_20_pct",
        "Hits_per_LA_gte_20_pct",
        "Hits_per_BBE_pct",
        "damage_bbe",
        "non_damage_la_gte_20_bbe",
        "la_lte_0_bbe",
        "la_0_to_20_bbe",
        "la_gte_20_bbe",
        "bbe_total",
    ]
    park_overall_rates = park.groupby(["park_mlbid"], dropna=False)[rate_cols].apply(_rate_series).reset_index()
    lg_overall = _rate_series(park)

    map_source = park[park["season"] == src_season].copy()
    if map_source.empty:
        map_source = park[park["season"] == int(park["season"].max())].copy()
    team_to_park = (
        map_source.groupby("home_team_norm", dropna=False)["park_mlbid"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)
        .to_dict()
    )
    known_teams = set(team_to_park.keys())

    park_lookup = {
        row["park_mlbid"]: {
            "hr_damage_rate": float(row["hr_damage_rate"]),
            "hr_non_damage_rate": float(row["hr_non_damage_rate"]),
            "xbh_damage_rate": float(row["xbh_damage_rate"]),
            "xbh_bbe_rate": float(row["xbh_bbe_rate"]),
            "h_la_lte_0_rate": float(row["h_la_lte_0_rate"]),
            "h_la_0_to_20_rate": float(row["h_la_0_to_20_rate"]),
            "h_la_gte_20_rate": float(row["h_la_gte_20_rate"]),
            "h_rate": float(row["h_rate"]),
        }
        for _, row in park_overall_rates.iterrows()
    }
    lg_default = {
        "hr_damage_rate": float(lg_overall["hr_damage_rate"]),
        "hr_non_damage_rate": float(lg_overall["hr_non_damage_rate"]),
        "xbh_damage_rate": float(lg_overall["xbh_damage_rate"]),
        "xbh_bbe_rate": float(lg_overall["xbh_bbe_rate"]),
        "h_la_lte_0_rate": float(lg_overall["h_la_lte_0_rate"]),
        "h_la_0_to_20_rate": float(lg_overall["h_la_0_to_20_rate"]),
        "h_la_gte_20_rate": float(lg_overall["h_la_gte_20_rate"]),
        "h_rate": float(lg_overall["h_rate"]),
    }
    if not np.isfinite(lg_default["h_rate"]):
        return out
    lg_la_lte_0_rate = (
        lg_default["h_la_lte_0_rate"]
        if np.isfinite(lg_default["h_la_lte_0_rate"])
        else lg_default["h_rate"]
    )

    contact_col = None
    for col in ["contact_vs_avg_reg_proj_p50", "contact_vs_avg_proj_p50", "z_con_reg_proj_p50", "z_con_proj_p50"]:
        if col in out.columns:
            contact_col = col
            break
    zcon_col = None
    for col in ["z_con_reg_proj_p50", "z_con_proj_p50"]:
        if col in out.columns:
            zcon_col = col
            break
    ld_col = None
    for col in ["LD_pct_reg_proj_p50", "LD_pct_proj_p50"]:
        if col in out.columns:
            ld_col = col
            break
    contact_skill = _rank_pct(out[contact_col]) if contact_col else pd.Series(0.5, index=out.index, dtype=float)
    zcon_skill = _rank_pct(out[zcon_col]) if zcon_col else pd.Series(0.5, index=out.index, dtype=float)
    ld_skill = _rank_pct(out[ld_col]) if ld_col else pd.Series(0.5, index=out.index, dtype=float)
    damage_skill = _rank_pct(out[f"{damage_metric}_proj_p50"])
    la20_skill = _rank_pct(out[f"{la20_metric}_proj_p50"])
    hit_skill = (0.40 * contact_skill) + (0.35 * zcon_skill) + (0.25 * ld_skill)
    power_skill = ((0.65 * damage_skill) + (0.35 * la20_skill)).clip(0.0, 1.0)
    singles_skill = (hit_skill * (1.0 - power_skill)).clip(0.0, 1.0)
    mid_mult = (0.82 + (0.45 * hit_skill)).clip(0.75, 1.35)
    low_mult = (0.78 + (0.60 * hit_skill)).clip(0.70, 1.45)
    high_mult = (0.90 + (0.25 * hit_skill)).clip(0.80, 1.20)
    mid_singles_boost = (1.00 + (0.15 * singles_skill)).clip(1.00, 1.15)
    low_singles_boost = (1.00 + (0.25 * singles_skill)).clip(1.00, 1.25)
    high_singles_boost = (1.00 + (0.20 * singles_skill)).clip(1.00, 1.20)

    team_series = out[team_col]
    for pct in available_pcts:
        bbe_series = pd.to_numeric(out[f"bbe_proj_p{pct}"], errors="coerce")
        damage_series = pd.to_numeric(out[f"{damage_metric}_proj_p{pct}"], errors="coerce")
        la20_series = pd.to_numeric(out[f"{la20_metric}_proj_p{pct}"], errors="coerce")
        la_lte_0_series = pd.to_numeric(out[f"{la_lte_0_metric}_proj_p{pct}"], errors="coerce")
        hr_damage_vals = []
        hr_non_damage_vals = []
        hr_vals = []
        xbh_vals = []
        h_vals = []
        for idx in out.index:
            team_code = _normalize_team_code(team_series.get(idx), known_teams)
            park_id = team_to_park.get(team_code)

            park_rates = None
            if park_id is not None and np.isfinite(float(park_id)):
                park_rates = park_lookup.get(park_id)
            if park_rates is None:
                park_rates = lg_default
            park_rates = dict(park_rates)
            for key, default_val in lg_default.items():
                if not np.isfinite(float(pd.to_numeric(park_rates.get(key), errors="coerce"))):
                    park_rates[key] = default_val

            bbe = float(bbe_series.get(idx))
            damage_rate = float(damage_series.get(idx))
            la20_pct = float(la20_series.get(idx))
            la_lte_0_pct = float(la_lte_0_series.get(idx))
            if not np.isfinite(bbe) or not np.isfinite(damage_rate):
                hr_damage_vals.append(np.nan)
                hr_non_damage_vals.append(np.nan)
                hr_vals.append(np.nan)
                xbh_vals.append(np.nan)
                h_vals.append(np.nan)
                continue

            bbe = max(0.0, bbe)
            damage_bbe = bbe * max(0.0, damage_rate) / 100.0
            la20_pct = max(0.0, la20_pct) if np.isfinite(la20_pct) else 0.0
            la_lte_0_pct = max(0.0, la_lte_0_pct) if np.isfinite(la_lte_0_pct) else 0.0
            la_mid_pct = float(np.clip(100.0 - la20_pct - la_lte_0_pct, 0.0, 100.0))
            la20_bbe = bbe * la20_pct / 100.0
            non_damage_share = float(np.clip(1.0 - (max(0.0, damage_rate) / 100.0), 0.0, 1.0))
            non_damage_la20_bbe = la20_bbe * non_damage_share
            la_lte_0_bbe = bbe * la_lte_0_pct / 100.0
            la_mid_bbe = bbe * la_mid_pct / 100.0

            hr_damage = max(
                0.0,
                damage_bbe * max(0.0, park_rates.get("hr_damage_rate", np.nan)) / 100.0,
            )
            hr_non_damage = max(
                0.0,
                non_damage_la20_bbe * max(0.0, park_rates.get("hr_non_damage_rate", np.nan)) / 100.0,
            )
            hr_total = hr_damage + hr_non_damage

            xbh_rate = park_rates.get("xbh_damage_rate", np.nan)
            if not np.isfinite(xbh_rate):
                xbh_rate = park_rates.get("xbh_bbe_rate", np.nan)
            h_mid_rate = park_rates.get("h_la_0_to_20_rate", np.nan)
            h_gte_20_rate = park_rates.get("h_la_gte_20_rate", np.nan)
            if not np.isfinite(h_mid_rate):
                h_mid_rate = park_rates.get("h_rate", np.nan)
            if not np.isfinite(h_gte_20_rate):
                h_gte_20_rate = park_rates.get("h_rate", np.nan)
            idx_mid_mult = float(mid_mult.get(idx, 1.0)) * float(mid_singles_boost.get(idx, 1.0))
            idx_low_mult = float(low_mult.get(idx, 1.0)) * float(low_singles_boost.get(idx, 1.0))
            idx_high_mult = float(high_mult.get(idx, 1.0)) * float(high_singles_boost.get(idx, 1.0))
            h_mid_rate = min(95.0, max(0.0, h_mid_rate * idx_mid_mult)) if np.isfinite(h_mid_rate) else np.nan
            h_gte_20_rate = min(95.0, max(0.0, h_gte_20_rate * idx_high_mult)) if np.isfinite(h_gte_20_rate) else np.nan
            h_lte_0_rate = min(95.0, max(0.0, lg_la_lte_0_rate * idx_low_mult))

            h_mid = (
                max(0.0, la_mid_bbe * max(0.0, h_mid_rate) / 100.0)
                if np.isfinite(h_mid_rate)
                else np.nan
            )
            h_lte_0 = max(0.0, la_lte_0_bbe * max(0.0, h_lte_0_rate) / 100.0)
            h_gte_20 = (
                max(0.0, la20_bbe * max(0.0, h_gte_20_rate) / 100.0)
                if np.isfinite(h_gte_20_rate)
                else np.nan
            )
            h_total = np.nansum([h_mid, h_lte_0, h_gte_20])

            hr_damage_vals.append(hr_damage)
            hr_non_damage_vals.append(hr_non_damage)
            hr_vals.append(hr_total if np.isfinite(hr_total) else np.nan)
            xbh_vals.append(max(0.0, damage_bbe * max(0.0, xbh_rate) / 100.0) if np.isfinite(xbh_rate) else np.nan)
            h_vals.append(h_total if np.isfinite(h_total) else np.nan)

        out[f"HR_damage_proj_p{pct}"] = hr_damage_vals
        out[f"HR_non_damage_proj_p{pct}"] = hr_non_damage_vals
        out[f"HR_proj_p{pct}"] = hr_vals
        out[f"XBH_proj_p{pct}"] = xbh_vals
        out[f"H_proj_p{pct}"] = h_vals

    pct_order = [20, 25, 50, 75, 80]
    for stat in ["HR", "HR_damage", "HR_non_damage", "XBH", "H"]:
        prev = None
        present: list[int] = []
        for p in pct_order:
            col = f"{stat}_proj_p{p}"
            if col not in out.columns:
                continue
            series = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
            if prev is not None:
                series = np.maximum(series, prev)
            out[col] = series
            prev = series
            present.append(p)
        if 25 in present and 75 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p75"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p25"], errors="coerce")
            )
        elif 20 in present and 80 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p80"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p20"], errors="coerce")
            )
    return out


def _add_hitter_xbh_from_h_columns(projections: pd.DataFrame) -> pd.DataFrame:
    out = projections.copy()
    pct_order = [20, 25, 50, 75, 80]
    present: list[int] = []
    prev = None
    for pct in pct_order:
        h_col = f"H_proj_p{pct}"
        rate_col = f"XBH_per_H_pct_proj_p{pct}"
        out_col = f"XBH_from_H_proj_p{pct}"
        if h_col not in out.columns or rate_col not in out.columns:
            continue
        h_vals = pd.to_numeric(out[h_col], errors="coerce")
        rate_vals = pd.to_numeric(out[rate_col], errors="coerce")
        vals = (h_vals * rate_vals / 100.0).clip(lower=0.0)
        if prev is not None:
            vals = np.maximum(vals, prev)
        out[out_col] = vals
        prev = vals
        present.append(pct)

    if 25 in present and 75 in present:
        out["XBH_from_H_proj_spread"] = (
            pd.to_numeric(out["XBH_from_H_proj_p75"], errors="coerce")
            - pd.to_numeric(out["XBH_from_H_proj_p25"], errors="coerce")
        )
    elif 20 in present and 80 in present:
        out["XBH_from_H_proj_spread"] = (
            pd.to_numeric(out["XBH_from_H_proj_p80"], errors="coerce")
            - pd.to_numeric(out["XBH_from_H_proj_p20"], errors="coerce")
        )
    return out


def _join_bp_hitting_rates(
    merged_df: pd.DataFrame,
    *,
    bp_path: Path,
    id_col: str,
    season_col: str,
    level_col: str,
) -> pd.DataFrame:
    out = merged_df.copy()
    for metric in ["XBH_per_H_pct", "SO_per_PA_pct", "BB_per_PA_pct"]:
        if metric not in out.columns:
            out[metric] = np.nan
        n_col = f"{metric}_n"
        if n_col not in out.columns:
            out[n_col] = np.nan

    if out.empty or not bp_path.exists():
        return out

    bp = pd.read_parquet(bp_path)
    required = {
        "mlbid",
        "season",
        "level_id",
        "xbh_per_h_pct",
        "xbh_hits_den",
        "strikeout_rate",
        "walk_rate",
        "stolen_bases_agg",
        "caught_stealing_agg",
        "plate_appearances_agg",
    }
    if not required.issubset(set(bp.columns)):
        return out

    work = bp.copy()
    work["mlbid"] = pd.to_numeric(work["mlbid"], errors="coerce")
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["level_id"] = pd.to_numeric(work["level_id"], errors="coerce")
    work = work[work["mlbid"].notna() & work["season"].notna() & work["level_id"].notna()].copy()
    if work.empty:
        return out
    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work["level_id"] = work["level_id"].astype("int64")
    pa = pd.to_numeric(work["plate_appearances_agg"], errors="coerce")
    sb = pd.to_numeric(work["stolen_bases_agg"], errors="coerce")
    cs = pd.to_numeric(work["caught_stealing_agg"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        work["SB_per_PA_pct"] = np.where(pa > 0, 100.0 * sb / pa, np.nan)
        work["SBA_per_PA_pct"] = np.where(pa > 0, 100.0 * (sb + cs) / pa, np.nan)

    specs: list[tuple[str, str, str]] = [
        ("XBH_per_H_pct", "xbh_per_h_pct", "xbh_hits_den"),
        ("SO_per_PA_pct", "strikeout_rate", "plate_appearances_agg"),
        ("BB_per_PA_pct", "walk_rate", "plate_appearances_agg"),
        ("SB_per_PA_pct", "SB_per_PA_pct", "plate_appearances_agg"),
        ("SBA_per_PA_pct", "SBA_per_PA_pct", "plate_appearances_agg"),
    ]
    rows: list[dict[str, Any]] = []
    for (mlbid, season, level_id), grp in work.groupby(["mlbid", "season", "level_id"], dropna=False):
        row: dict[str, Any] = {"mlbid": int(mlbid), "season": int(season), "level_id": int(level_id)}
        for metric, rate_col, den_col in specs:
            rates = pd.to_numeric(grp[rate_col], errors="coerce")
            dens = pd.to_numeric(grp[den_col], errors="coerce")
            mask = rates.notna() & dens.notna() & np.isfinite(rates) & np.isfinite(dens) & (dens > 0)
            if mask.any():
                row[metric] = float(np.average(rates[mask], weights=dens[mask]))
                row[f"{metric}_n"] = float(dens[mask].sum())
            else:
                finite = rates[np.isfinite(rates)]
                row[metric] = float(finite.mean()) if not finite.empty else np.nan
                row[f"{metric}_n"] = np.nan
        rows.append(row)
    if not rows:
        return out

    bp_agg = pd.DataFrame(rows).rename(
        columns={"mlbid": id_col, "season": season_col, "level_id": level_col}
    )
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce")
    out[season_col] = pd.to_numeric(out[season_col], errors="coerce")
    out[level_col] = pd.to_numeric(out[level_col], errors="coerce")
    out = out.merge(bp_agg, on=[id_col, season_col, level_col], how="left", suffixes=("", "_bp"))

    for metric in ["XBH_per_H_pct", "SO_per_PA_pct", "BB_per_PA_pct", "SB_per_PA_pct", "SBA_per_PA_pct"]:
        bp_col = f"{metric}_bp"
        if bp_col in out.columns:
            out[metric] = pd.to_numeric(out[bp_col], errors="coerce")
        out[metric] = pd.to_numeric(out[metric], errors="coerce").clip(lower=0.0, upper=100.0)
        n_col = f"{metric}_n"
        bp_n_col = f"{n_col}_bp"
        if bp_n_col in out.columns:
            out[n_col] = pd.to_numeric(out[bp_n_col], errors="coerce")
        out[n_col] = pd.to_numeric(out[n_col], errors="coerce").clip(lower=0.0)

    drop_cols = [c for c in out.columns if c.endswith("_bp")]
    return out.drop(columns=drop_cols, errors="ignore")


def _apply_hitter_bp_rate_equivalency(
    merged_df: pd.DataFrame,
    *,
    season_col: str,
    level_col: str,
    mlb_level_id: int,
) -> pd.DataFrame:
    out = merged_df.copy()
    if out.empty:
        return out

    bp_rate_metrics = [
        "XBH_per_H_pct",
        "SO_per_PA_pct",
        "BB_per_PA_pct",
        "SB_per_PA_pct",
        "SBA_per_PA_pct",
    ]
    present = [m for m in bp_rate_metrics if m in out.columns]
    if not present:
        return out

    out = apply_simple_mlb_equivalency(
        out,
        metric_cols=present,
        season_col=season_col,
        level_col=level_col,
        mlb_level_id=int(mlb_level_id),
    )
    for metric in present:
        out[metric] = pd.to_numeric(out[metric], errors="coerce").clip(lower=0.0, upper=100.0)
    return out


def _append_bp_only_hitter_rows(
    merged_df: pd.DataFrame,
    *,
    bp_path: Path,
    id_col: str,
    season_col: str,
    level_col: str,
    name_col: str,
    exposure_col: str,
    age_col: str,
    team_col: str | None,
) -> pd.DataFrame:
    out = merged_df.copy()
    if not bp_path.exists():
        return out

    bp = pd.read_parquet(bp_path)
    required = {"mlbid", "season", "level_id", "plate_appearances_agg"}
    if not required.issubset(set(bp.columns)):
        return out

    work = bp.copy()
    work["mlbid"] = pd.to_numeric(work["mlbid"], errors="coerce")
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["level_id"] = pd.to_numeric(work["level_id"], errors="coerce")
    work["plate_appearances_agg"] = pd.to_numeric(work["plate_appearances_agg"], errors="coerce")
    work = work[
        work["mlbid"].notna()
        & work["season"].notna()
        & work["level_id"].notna()
        & work["plate_appearances_agg"].notna()
        & (work["plate_appearances_agg"] > 0.0)
    ].copy()
    if work.empty:
        return out

    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work["level_id"] = work["level_id"].astype("int64")

    def _agg(g: pd.DataFrame) -> pd.Series:
        pa = pd.to_numeric(g["plate_appearances_agg"], errors="coerce")
        pa = pa.where(np.isfinite(pa), np.nan).dropna()
        pa_val = float(pa.max()) if not pa.empty else np.nan
        result: dict[str, Any] = {"bp_pa": pa_val}

        if "baseball_age" in g.columns:
            age = pd.to_numeric(g["baseball_age"], errors="coerce")
            w = pd.to_numeric(g["plate_appearances_agg"], errors="coerce")
            mask = age.notna() & w.notna() & np.isfinite(age) & np.isfinite(w) & (w > 0)
            if mask.any():
                result["bp_age"] = float(np.average(age[mask], weights=w[mask]))
            else:
                result["bp_age"] = np.nan
        if "player_display_text" in g.columns:
            names = g["player_display_text"].dropna().astype(str)
            result["bp_name"] = names.iloc[0] if not names.empty else np.nan
        if team_col and "team_abbreviation" in g.columns:
            teams = g["team_abbreviation"].dropna().astype(str)
            result["bp_team"] = teams.iloc[0] if not teams.empty else np.nan
        return pd.Series(result)

    bp_rows = (
        work.groupby(["mlbid", "season", "level_id"], dropna=False)
        .apply(_agg)
        .reset_index()
        .rename(columns={"mlbid": id_col, "season": season_col, "level_id": level_col})
    )
    if bp_rows.empty:
        return out

    existing = out[[id_col, season_col, level_col]].copy()
    existing["__has"] = 1
    bp_rows = bp_rows.merge(existing, on=[id_col, season_col, level_col], how="left")
    bp_rows = bp_rows[bp_rows["__has"].isna()].copy()
    bp_rows = bp_rows.drop(columns=["__has"], errors="ignore")
    if bp_rows.empty:
        return out

    add = pd.DataFrame(index=bp_rows.index)
    add[id_col] = bp_rows[id_col]
    add[season_col] = bp_rows[season_col]
    add[level_col] = bp_rows[level_col]
    add[exposure_col] = pd.to_numeric(bp_rows.get("bp_pa"), errors="coerce").clip(lower=0.0)
    if age_col:
        add[age_col] = pd.to_numeric(bp_rows.get("bp_age"), errors="coerce")
    add[name_col] = bp_rows.get("bp_name", np.nan)
    if team_col:
        add[team_col] = bp_rows.get("bp_team", np.nan)

    for col in out.columns:
        if col not in add.columns:
            add[col] = np.nan
    for col in add.columns:
        if col not in out.columns:
            out[col] = np.nan
    add = add[out.columns]
    return pd.concat([out, add], axis=0, ignore_index=True, sort=False)


def _estimate_lg_non_hr_xbh_tb(
    *,
    bp_path: Path,
    source_season: int,
    mlb_level_id: int,
    recency_weights: list[float],
) -> float:
    default_mu = 2.075
    if not bp_path.exists():
        return default_mu

    bp = pd.read_parquet(bp_path)
    needed = {"season", "level_id", "doubles_agg", "triples_agg"}
    if not needed.issubset(set(bp.columns)):
        return default_mu

    bp = bp.copy()
    bp["season"] = pd.to_numeric(bp["season"], errors="coerce")
    bp["level_id"] = pd.to_numeric(bp["level_id"], errors="coerce")
    bp = bp[bp["season"].notna() & bp["level_id"].notna()].copy()
    if bp.empty:
        return default_mu
    bp["season"] = bp["season"].astype(int)
    bp["level_id"] = bp["level_id"].astype(int)
    bp = bp[bp["level_id"] == int(mlb_level_id)].copy()
    if bp.empty:
        return default_mu

    doubles = pd.to_numeric(bp["doubles_agg"], errors="coerce").fillna(0.0).clip(lower=0.0)
    triples = pd.to_numeric(bp["triples_agg"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bp["non_hr_xbh"] = doubles + triples
    bp["non_hr_xbh_tb"] = (2.0 * doubles) + (3.0 * triples)
    season_agg = bp.groupby("season", dropna=False)[["non_hr_xbh", "non_hr_xbh_tb"]].sum(numeric_only=True)
    denom = season_agg["non_hr_xbh"]
    season_mu = pd.Series(np.nan, index=season_agg.index, dtype=float)
    valid = denom > 0
    season_mu.loc[valid] = season_agg.loc[valid, "non_hr_xbh_tb"] / denom.loc[valid]
    season_mu = season_mu.replace([np.inf, -np.inf], np.nan).dropna()
    if season_mu.empty:
        return default_mu

    weight_map = {lag: float(w) for lag, w in enumerate(recency_weights)}
    vals: list[float] = []
    wts: list[float] = []
    for season, mu in season_mu.items():
        lag = int(source_season) - int(season)
        if lag in weight_map:
            vals.append(float(mu))
            wts.append(weight_map[lag])
    if vals and sum(wts) > 0:
        return float(np.average(np.array(vals, dtype=float), weights=np.array(wts, dtype=float)))
    return float(season_mu.iloc[-1])


def _add_hitter_plate_outcome_and_slash_columns(
    projections: pd.DataFrame,
    *,
    lg_non_hr_xbh_tb_mu: float,
) -> pd.DataFrame:
    out = projections.copy()
    pct_order = [20, 25, 50, 75, 80]
    for pct in pct_order:
        pa_col = f"PA_proj_p{pct}"
        bb_rate_col = f"BB_per_PA_pct_proj_p{pct}"
        so_rate_col = f"SO_per_PA_pct_proj_p{pct}"
        sb_rate_col = f"SB_per_PA_pct_proj_p{pct}"
        sba_rate_col = f"SBA_per_PA_pct_proj_p{pct}"
        h_col = f"H_proj_p{pct}"
        hr_col = f"HR_proj_p{pct}"
        xbh_col = f"XBH_from_H_proj_p{pct}" if f"XBH_from_H_proj_p{pct}" in out.columns else f"XBH_proj_p{pct}"

        pa = pd.to_numeric(out[pa_col], errors="coerce") if pa_col in out.columns else None
        bb_rate = pd.to_numeric(out[bb_rate_col], errors="coerce") if bb_rate_col in out.columns else None
        so_rate = pd.to_numeric(out[so_rate_col], errors="coerce") if so_rate_col in out.columns else None
        sb_rate = pd.to_numeric(out[sb_rate_col], errors="coerce") if sb_rate_col in out.columns else None
        sba_rate = pd.to_numeric(out[sba_rate_col], errors="coerce") if sba_rate_col in out.columns else None
        h = pd.to_numeric(out[h_col], errors="coerce") if h_col in out.columns else None
        hr = pd.to_numeric(out[hr_col], errors="coerce") if hr_col in out.columns else None
        xbh = pd.to_numeric(out[xbh_col], errors="coerce") if xbh_col in out.columns else None

        if pa is None:
            continue
        if bb_rate is not None:
            bb = (pa * bb_rate / 100.0).clip(lower=0.0)
            out[f"BB_proj_p{pct}"] = bb
        else:
            bb = pd.to_numeric(out.get(f"BB_proj_p{pct}"), errors="coerce")
        if so_rate is not None:
            so = (pa * so_rate / 100.0).clip(lower=0.0)
            out[f"SO_proj_p{pct}"] = so
        if sb_rate is not None:
            sb = (pa * sb_rate / 100.0).clip(lower=0.0)
            out[f"SB_proj_p{pct}"] = sb
        else:
            sb = pd.to_numeric(out.get(f"SB_proj_p{pct}"), errors="coerce")
        if sba_rate is not None:
            sba = (pa * sba_rate / 100.0).clip(lower=0.0)
            if sb is not None:
                sba = np.maximum(sba, sb)
            out[f"SBA_proj_p{pct}"] = sba
        else:
            sba = pd.to_numeric(out.get(f"SBA_proj_p{pct}"), errors="coerce")
        if sb is not None and sba is not None:
            out[f"CS_proj_p{pct}"] = (sba - sb).clip(lower=0.0)
        if bb is None:
            bb = pd.Series(np.nan, index=out.index, dtype=float)

        ab = (pa - bb).clip(lower=0.0)
        out[f"AB_proj_p{pct}"] = ab
        if h is not None:
            safe_ab = ab.where(ab > 0.0, np.nan)
            out[f"AVG_proj_p{pct}"] = (h / safe_ab).clip(lower=0.0, upper=1.0)
            out[f"OBP_proj_p{pct}"] = ((h + bb) / pa.where(pa > 0.0, np.nan)).clip(lower=0.0, upper=1.0)
        if hr is not None and xbh is not None:
            non_hr_xbh = (xbh - hr).clip(lower=0.0)
            out[f"non_HR_XBH_proj_p{pct}"] = non_hr_xbh
            singles = ((h - xbh) if h is not None else pd.Series(np.nan, index=out.index, dtype=float)).clip(lower=0.0)
            out[f"1B_proj_p{pct}"] = singles
            safe_ab = ab.where(ab > 0.0, np.nan)
            out[f"SLG_proj_p{pct}"] = (
                (singles + (4.0 * hr) + (float(lg_non_hr_xbh_tb_mu) * non_hr_xbh)) / safe_ab
            ).clip(lower=0.0, upper=4.0)
        obp_col = f"OBP_proj_p{pct}"
        slg_col = f"SLG_proj_p{pct}"
        if obp_col in out.columns and slg_col in out.columns:
            out[f"OPS_proj_p{pct}"] = (
                pd.to_numeric(out[obp_col], errors="coerce")
                + pd.to_numeric(out[slg_col], errors="coerce")
            ).clip(lower=0.0, upper=5.0)

    for stat in ["BB", "SO", "SB", "SBA", "CS", "AB", "1B", "XBH_from_H", "non_HR_XBH"]:
        prev = None
        present: list[int] = []
        for p in pct_order:
            col = f"{stat}_proj_p{p}"
            if col not in out.columns:
                continue
            s = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
            if prev is not None:
                s = np.maximum(s, prev)
            out[col] = s
            prev = s
            present.append(p)
        if 25 in present and 75 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p75"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p25"], errors="coerce")
            )
        elif 20 in present and 80 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p80"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p20"], errors="coerce")
            )

    for stat in ["AVG", "OBP", "SLG", "OPS"]:
        present = [p for p in pct_order if f"{stat}_proj_p{p}" in out.columns]
        if 25 in present and 75 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p75"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p25"], errors="coerce")
            )
        elif 20 in present and 80 in present:
            out[f"{stat}_proj_spread"] = (
                pd.to_numeric(out[f"{stat}_proj_p80"], errors="coerce")
                - pd.to_numeric(out[f"{stat}_proj_p20"], errors="coerce")
            )

    out["lg_non_hr_xbh_tb_mu"] = float(lg_non_hr_xbh_tb_mu)
    return out


def _add_hitter_xbh_per_h_history_metric(
    merged_df: pd.DataFrame,
    *,
    season_col: str,
    team_col: str | None,
    park_data_path: Path,
    mlb_level_id: int,
) -> pd.DataFrame:
    out = merged_df.copy()
    if out.empty or not team_col or team_col not in out.columns or not park_data_path.exists():
        return out

    damage_col = None
    for col in ["damage_rate", "damage_rate_reg"]:
        if col in out.columns:
            damage_col = col
            break
    la20_col = None
    for col in ["LA_gte_20", "LA_gte_20_reg"]:
        if col in out.columns:
            la20_col = col
            break
    la_lte0_col = None
    for col in ["LA_lte_0", "LA_lte_0_reg"]:
        if col in out.columns:
            la_lte0_col = col
            break
    if damage_col is None or la20_col is None or la_lte0_col is None or "bbe" not in out.columns:
        return out

    park = pd.read_parquet(park_data_path)
    required = {
        "home_team",
        "season",
        "level_id",
        "damage_bbe",
        "XBH_per_damage_BBE_pct",
        "la_lte_0_bbe",
        "la_0_to_20_bbe",
        "la_gte_20_bbe",
        "Hits_per_LA_lte_0_pct",
        "Hits_per_LA_0_to_20_pct",
        "Hits_per_LA_gte_20_pct",
    }
    if not required.issubset(set(park.columns)):
        return out

    park = park.copy()
    park["season"] = pd.to_numeric(park["season"], errors="coerce")
    park["level_id"] = pd.to_numeric(park["level_id"], errors="coerce")
    park = park[park["season"].notna() & park["level_id"].notna()].copy()
    park["season"] = park["season"].astype(int)
    park["level_id"] = park["level_id"].astype(int)
    park = park[park["level_id"] == int(mlb_level_id)].copy()
    if park.empty:
        return out

    park["home_team_norm"] = park["home_team"].map(_normalize_team_code)
    park = park[park["home_team_norm"].notna()].copy()
    if park.empty:
        return out

    def _team_rates(grp: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "xbh_damage_rate": _weighted_rate(grp, "XBH_per_damage_BBE_pct", "damage_bbe"),
                "h_lte0_rate": _weighted_rate(grp, "Hits_per_LA_lte_0_pct", "la_lte_0_bbe"),
                "h_mid_rate": _weighted_rate(grp, "Hits_per_LA_0_to_20_pct", "la_0_to_20_bbe"),
                "h_hi_rate": _weighted_rate(grp, "Hits_per_LA_gte_20_pct", "la_gte_20_bbe"),
            }
        )

    team_rates = (
        park.groupby(["season", "home_team_norm"], dropna=False)
        .apply(_team_rates)
        .reset_index()
        .rename(columns={"season": "season_i", "home_team_norm": "team_norm"})
    )
    lg_rates = (
        park.groupby("season", dropna=False)
        .apply(_team_rates)
        .reset_index()
        .rename(
            columns={
                "season": "season_i",
                "xbh_damage_rate": "lg_xbh_damage_rate",
                "h_lte0_rate": "lg_h_lte0_rate",
                "h_mid_rate": "lg_h_mid_rate",
                "h_hi_rate": "lg_h_hi_rate",
            }
        )
    )

    out["season_i"] = pd.to_numeric(out[season_col], errors="coerce")
    out = out[out["season_i"].notna()].copy()
    if out.empty:
        return merged_df
    out["season_i"] = out["season_i"].astype(int)
    known_codes = set(team_rates["team_norm"].dropna().astype(str).unique().tolist())
    out["team_norm"] = out[team_col].map(lambda x: _normalize_team_code(x, known_codes))

    out = out.merge(team_rates, on=["season_i", "team_norm"], how="left")
    out = out.merge(lg_rates, on="season_i", how="left")

    for col, lg_col in [
        ("xbh_damage_rate", "lg_xbh_damage_rate"),
        ("h_lte0_rate", "lg_h_lte0_rate"),
        ("h_mid_rate", "lg_h_mid_rate"),
        ("h_hi_rate", "lg_h_hi_rate"),
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[lg_col] = pd.to_numeric(out[lg_col], errors="coerce")
        out[col] = out[col].fillna(out[lg_col]).clip(lower=0.0, upper=100.0)

    bbe = pd.to_numeric(out["bbe"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dmg = pd.to_numeric(out[damage_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    la20 = pd.to_numeric(out[la20_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    la_lte0 = pd.to_numeric(out[la_lte0_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    la_mid = np.clip(100.0 - la20 - la_lte0, 0.0, 100.0)

    damage_bbe = bbe * (dmg / 100.0)
    la_lte0_bbe = bbe * (la_lte0 / 100.0)
    la_mid_bbe = bbe * (la_mid / 100.0)
    la20_bbe = bbe * (la20 / 100.0)

    xbh_est = damage_bbe * (pd.to_numeric(out["xbh_damage_rate"], errors="coerce") / 100.0)
    hits_est = (
        la_lte0_bbe * (pd.to_numeric(out["h_lte0_rate"], errors="coerce") / 100.0)
        + la_mid_bbe * (pd.to_numeric(out["h_mid_rate"], errors="coerce") / 100.0)
        + la20_bbe * (pd.to_numeric(out["h_hi_rate"], errors="coerce") / 100.0)
    )
    valid_hits = hits_est > 0
    xbh_per_h = pd.Series(np.nan, index=out.index, dtype=float)
    xbh_per_h.loc[valid_hits] = (100.0 * xbh_est.loc[valid_hits] / hits_est.loc[valid_hits]).clip(0.0, 100.0)

    out["XBH_per_H_pct"] = xbh_per_h
    out["XBH_per_H_pct_n"] = hits_est.where(valid_hits, np.nan).clip(lower=0.0)
    return out.drop(
        columns=[
            "season_i",
            "team_norm",
            "xbh_damage_rate",
            "h_lte0_rate",
            "h_mid_rate",
            "h_hi_rate",
            "lg_xbh_damage_rate",
            "lg_h_lte0_rate",
            "lg_h_mid_rate",
            "lg_h_hi_rate",
        ],
        errors="ignore",
    )


def _add_hitter_p600_columns(projections: pd.DataFrame) -> pd.DataFrame:
    out = projections.copy()
    if out.empty or "PA_proj_p50" not in out.columns:
        return out

    pa50 = pd.to_numeric(out["PA_proj_p50"], errors="coerce")
    valid = pa50 > 0.0

    target_pa = 600.0
    cumulative_stats = [
        "PA",
        "bbe",
        "AB",
        "BB",
        "SO",
        "SB",
        "SBA",
        "CS",
        "HR",
        "HR_damage",
        "HR_non_damage",
        "XBH",
        "XBH_from_H",
        "non_HR_XBH",
        "1B",
        "H",
    ]
    for stat in cumulative_stats:
        p50_col = f"{stat}_proj_p50"
        if p50_col not in out.columns:
            continue
        stat50 = pd.to_numeric(out[p50_col], errors="coerce")
        p600 = (stat50 / pa50) * target_pa
        p600 = p600.where(valid, np.nan).clip(lower=0.0)
        out[f"{stat}_proj_p600"] = p600
    return out


def _round_projection_floats(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float32", "float64", "Float64"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(int(decimals))
    return out


def _run_dataset(
    dataset_name: str,
    cfg: ProjectionConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = cfg.datasets[dataset_name]
    reg_df = read_parquet(ds.regressed_path)
    base_df = read_parquet(ds.base_path)
    available_cols = sorted(set(reg_df.columns).union(base_df.columns))
    metric_cols, k_overrides, bounds = resolve_metric_settings(
        dataset_name=dataset_name,
        regressed_cols=reg_df.columns.tolist(),
        available_cols=available_cols,
        cfg=cfg,
    )
    if not metric_cols:
        return pd.DataFrame(), pd.DataFrame()

    merged = merge_base_and_regressed(base_df=base_df, reg_df=reg_df, cfg=ds)
    eq_metrics = [m for m in metric_cols if m.endswith("_reg")]
    if eq_metrics:
        merged = apply_simple_mlb_equivalency(
            merged,
            metric_cols=eq_metrics,
            season_col=ds.season_col,
            level_col=ds.level_col,
            mlb_level_id=cfg.global_cfg.mlb_level_id,
        )
    hitter_position = pd.DataFrame(columns=[ds.id_col, ds.season_col, "position"])
    if dataset_name == "hitters":
        hitter_position = _derive_hitter_primary_position(
            base_df,
            id_col=ds.id_col,
            season_col=ds.season_col,
            exposure_col=ds.exposure_col,
        )
        bp_path = Path(getattr(cfg.global_cfg, "bp_hitting_rates_path", "projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet"))
        merged = _append_bp_only_hitter_rows(
            merged,
            bp_path=bp_path,
            id_col=ds.id_col,
            season_col=ds.season_col,
            level_col=ds.level_col,
            name_col=ds.name_col,
            exposure_col=ds.exposure_col,
            age_col=ds.age_col,
            team_col=ds.team_col,
        )
        merged = _join_bp_hitting_rates(
            merged,
            bp_path=bp_path,
            id_col=ds.id_col,
            season_col=ds.season_col,
            level_col=ds.level_col,
        )
        bp_non_null = (
            int(pd.to_numeric(merged["XBH_per_H_pct"], errors="coerce").notna().sum())
            if "XBH_per_H_pct" in merged.columns
            else 0
        )
        if bp_non_null < 500:
            merged = _add_hitter_xbh_per_h_history_metric(
                merged,
                season_col=ds.season_col,
                team_col=ds.team_col,
                park_data_path=Path(cfg.global_cfg.park_factors_path),
                mlb_level_id=int(cfg.global_cfg.mlb_level_id),
            )
        merged = _apply_hitter_bp_rate_equivalency(
            merged,
            season_col=ds.season_col,
            level_col=ds.level_col,
            mlb_level_id=int(cfg.global_cfg.mlb_level_id),
        )

        for metric, k_val in [
            ("XBH_per_H_pct", 160.0),
            ("SO_per_PA_pct", 260.0),
            ("BB_per_PA_pct", 260.0),
            ("SB_per_PA_pct", 340.0),
            ("SBA_per_PA_pct", 340.0),
        ]:
            if metric in merged.columns:
                metric_cols = list(dict.fromkeys([*metric_cols, metric]))
                bounds.setdefault(metric, (0.0, 100.0))
                k_overrides.setdefault(metric, k_val)

    player_season = build_player_season_table(merged_df=merged, cfg=ds, metric_cols=metric_cols)
    if dataset_name == "hitters" and not hitter_position.empty:
        player_season = player_season.merge(
            hitter_position[[ds.id_col, ds.season_col, "position"]],
            on=[ds.id_col, ds.season_col],
            how="left",
        )
    level_for_age = "level_id_source" if "level_id_source" in player_season.columns else ds.level_col
    group_cols = [level_for_age]
    if ds.hand_col and ds.hand_col in player_season.columns:
        group_cols.append(ds.hand_col)
    player_season = infer_and_impute_age(
        player_season,
        player_col=ds.id_col,
        season_col=ds.season_col,
        level_col=level_for_age,
        age_col=ds.age_col,
        group_cols=group_cols,
    )
    passthrough_cols = [c for c in [ds.hand_col, ds.team_col, "position"] if c and c in player_season.columns]

    projections = project_next_season(
        player_season_df=player_season,
        metric_cols=metric_cols,
        id_col=ds.id_col,
        name_col=ds.name_col,
        season_col=ds.season_col,
        level_col=level_for_age,
        age_col="age_used",
        age_source_col="age_source",
        exposure_col=ds.exposure_col,
        global_cfg=cfg.global_cfg,
        k_overrides=k_overrides,
        bounds=bounds,
        passthrough_cols=passthrough_cols,
    )
    transitions = build_transition_deltas(
        player_season,
        id_col=ds.id_col,
        season_col=ds.season_col,
        metric_cols=metric_cols,
    )
    projections = apply_uncertainty_bands(
        projections,
        transitions_df=transitions,
        metric_cols=metric_cols,
        draws=int(cfg.global_cfg.uncertainty_draws),
        seed=int(cfg.global_cfg.seed),
        global_weight=float(cfg.global_cfg.cov_blend_global),
        local_k=int(cfg.global_cfg.local_k),
        local_min_k=int(cfg.global_cfg.local_min_k),
        uncertainty_c=float(cfg.global_cfg.uncertainty_c),
        uncertainty_d=float(cfg.global_cfg.uncertainty_d),
        bounds=bounds,
    )
    if dataset_name == "pitchers" and "stuff_raw_proj_p50" in projections.columns:
        src_season_vals = pd.to_numeric(projections.get("source_season"), errors="coerce")
        if src_season_vals.notna().any():
            src_season = int(src_season_vals.max())
        else:
            src_season = int(pd.to_numeric(base_df[ds.season_col], errors="coerce").dropna().max())
        projections = _apply_pitcher_stuff_blended_quantile_match(
            projections,
            base_df=base_df,
            season_col=ds.season_col,
            level_col=ds.level_col,
            source_season=src_season,
            blend_weight=float(PITCHER_STUFF_QMATCH_BLEND),
        )
    if dataset_name == "pitchers" and "position" not in projections.columns:
        projections["position"] = "P"
    if dataset_name == "hitters":
        if "position" not in projections.columns:
            projections["position"] = "UNK"
        projections["position"] = projections["position"].fillna("UNK").astype(str)
    hitter_metric_scale_factors: dict[str, float] = {}
    hitter_pre_calibration_backtest = pd.DataFrame()
    if dataset_name == "hitters" and not projections.empty:
        hitter_pre_calibration_backtest = run_rolling_backtest(
            player_season_df=player_season,
            metric_cols=metric_cols,
            id_col=ds.id_col,
            name_col=ds.name_col,
            season_col=ds.season_col,
            level_col=level_for_age,
            age_col="age_used",
            age_source_col="age_source",
            exposure_col=ds.exposure_col,
            global_cfg=cfg.global_cfg,
            k_overrides=k_overrides,
            bounds=bounds,
            dataset_name=dataset_name,
        )
        hitter_metric_scale_factors = _derive_conservative_metric_factors(
            hitter_pre_calibration_backtest,
            target_metrics=[
                "PA",
                "bbe",
                "XBH_per_H_pct",
                "SO_per_PA_pct",
                "BB_per_PA_pct",
                "SB_per_PA_pct",
                "SBA_per_PA_pct",
            ],
            min_rows=1200.0,
            min_seasons=3,
            max_down=0.18,
        )
        src_season = int(pd.to_numeric(projections["source_season"], errors="coerce").max())
        anchors = _compute_hitter_recent_anchors(
            player_season_df=player_season,
            id_col=ds.id_col,
            season_col=ds.season_col,
            level_col=level_for_age,
            team_col=ds.team_col,
            source_season=src_season,
            mlb_level_id=int(cfg.global_cfg.mlb_level_id),
            park_data_path=Path(cfg.global_cfg.park_factors_path),
            seasons_back=3,
        )
        projections = _apply_hitter_playing_time_adjustments(
            projections,
            bounds=bounds,
        )
        projections = _apply_hitter_pa_role_floor(
            projections,
            id_col=ds.id_col,
            anchors_df=anchors,
            bounds=bounds,
            min_recent_pa=1200.0,
            p50_floor_factor=0.90,
        )
        projections = _apply_metric_scale_factors(
            projections,
            metric_factors=hitter_metric_scale_factors,
            bounds=bounds,
        )
        projections = _apply_hitter_park_adjustments(
            projections,
            source_season=src_season,
            park_data_path=Path(cfg.global_cfg.park_factors_path),
            mlb_level_id=int(cfg.global_cfg.mlb_level_id),
            window_years=int(cfg.global_cfg.park_window_years),
            team_col=ds.team_col,
        )
        projections = _add_hitter_xbh_from_h_columns(projections)
        lg_non_hr_xbh_tb_mu = _estimate_lg_non_hr_xbh_tb(
            bp_path=Path(getattr(cfg.global_cfg, "bp_hitting_rates_path", "projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet")),
            source_season=src_season,
            mlb_level_id=int(cfg.global_cfg.mlb_level_id),
            recency_weights=list(cfg.global_cfg.recency_weights),
        )
        projections = _add_hitter_plate_outcome_and_slash_columns(
            projections,
            lg_non_hr_xbh_tb_mu=lg_non_hr_xbh_tb_mu,
        )
        projections = _add_hitter_p600_columns(projections)
        # Keep raw park-adjusted HR outputs (damage + non-damage) without post-hoc HR calibration.

    projections = _round_projection_floats(projections, decimals=3)
    missing = validate_projection_schema(
        projections,
        id_col=ds.id_col,
        name_col=ds.name_col,
        metric_cols=metric_cols,
    )
    if missing:
        raise RuntimeError(f"{dataset_name} projection schema missing columns: {missing}")

    output_path = output_dir / f"{dataset_name[:-1]}_projections.parquet"
    projections.to_parquet(output_path, index=False)

    use_cached_hitter_backtest = (
        dataset_name == "hitters"
        and not hitter_pre_calibration_backtest.empty
        and not _has_nontrivial_metric_factors(hitter_metric_scale_factors)
    )
    if use_cached_hitter_backtest:
        backtest = hitter_pre_calibration_backtest.copy()
    else:
        backtest = run_rolling_backtest(
            player_season_df=player_season,
            metric_cols=metric_cols,
            id_col=ds.id_col,
            name_col=ds.name_col,
            season_col=ds.season_col,
            level_col=level_for_age,
            age_col="age_used",
            age_source_col="age_source",
            exposure_col=ds.exposure_col,
            global_cfg=cfg.global_cfg,
            k_overrides=k_overrides,
            bounds=bounds,
            dataset_name=dataset_name,
            metric_scale_factors=hitter_metric_scale_factors if dataset_name == "hitters" else None,
        )
    backtest = _round_projection_floats(backtest, decimals=3)
    return projections, backtest


def run_projection_pipeline(
    config_path: str | Path = Path("projections_v1/projection_config.yml"),
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = ensure_dir(output_dir or cfg.global_cfg.output_dir)

    hitters_proj, hitters_bt = _run_dataset("hitters", cfg, out_dir)
    pitchers_proj, pitchers_bt = _run_dataset("pitchers", cfg, out_dir)
    backtest = pd.concat([hitters_bt, pitchers_bt], ignore_index=True)
    backtest_path = out_dir / "projection_backtest_summary.parquet"
    backtest.to_parquet(backtest_path, index=False)

    return {
        "hitters_rows": int(len(hitters_proj)),
        "pitchers_rows": int(len(pitchers_proj)),
        "backtest_rows": int(len(backtest)),
        "output_dir": str(out_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run projection pipeline (v1).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("projections_v1/projection_config.yml"),
        help="Path to projection config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_projection_pipeline(config_path=args.config, output_dir=args.output_dir)
    print(
        "Projection pipeline completed:",
        f"hitters={result['hitters_rows']:,}",
        f"pitchers={result['pitchers_rows']:,}",
        f"backtest={result['backtest_rows']:,}",
        f"output_dir={result['output_dir']}",
    )


if __name__ == "__main__":
    main()
