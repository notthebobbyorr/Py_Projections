from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projections_v1.age import infer_and_impute_age
from projections_v1.config import load_config, resolve_metric_settings
from projections_v1.equivalency import apply_simple_mlb_equivalency
from projections_v1.io import build_player_season_table, merge_base_and_regressed, read_parquet
from projections_v1.point_forecast import project_next_season


KPI_ALIAS_MAP: dict[str, str] = {
    "damage_rate_reg": "damage_rate",
    "EV90th_reg": "EV90th",
    "max_EV_reg": "max_EV",
    "pull_FB_pct_reg": "pull_FB_pct",
    "LA_gte_20_reg": "LA_gte_20",
    "LA_lte_0_reg": "LA_lte_0",
    "SEAGER_reg": "SEAGER",
    "selection_skill_reg": "selection_skill",
    "hittable_pitches_taken_reg": "hittable_pitches_taken",
    "chase_reg": "chase",
    "z_con_reg": "z_con",
    "secondary_whiff_pct_reg": "secondary_whiff_pct",
    "whiffs_vs_95_reg": "whiffs_vs_95",
    "contact_vs_avg_reg": "contact_vs_avg",
}

KPI_SKILL_METRICS: list[str] = list(KPI_ALIAS_MAP.keys())

KPI_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "damage_rate_reg": (0.0, 100.0),
    "EV90th_reg": (70.0, 130.0),
    "max_EV_reg": (80.0, 130.0),
    "pull_FB_pct_reg": (0.0, 100.0),
    "LA_gte_20_reg": (0.0, 100.0),
    "LA_lte_0_reg": (0.0, 100.0),
    "SEAGER_reg": (0.0, 100.0),
    "selection_skill_reg": (0.0, 100.0),
    "hittable_pitches_taken_reg": (0.0, 100.0),
    "chase_reg": (0.0, 100.0),
    "z_con_reg": (0.0, 100.0),
    "secondary_whiff_pct_reg": (0.0, 100.0),
    "whiffs_vs_95_reg": (0.0, 100.0),
    "contact_vs_avg_reg": (-100.0, 100.0),
    "PA": (0.0, 750.0),
    "bbe": (0.0, 650.0),
}

COUNT_PREFIXES = (
    "PA",
    "AB",
    "H",
    "HR",
    "BB",
    "SO",
    "RBI",
    "Runs",
    "SB",
    "SBA",
    "CS",
    "BBE",
    "BIP",
    "SBO",
    "XBH",
    "1B",
    "2B",
    "3B",
    "HBP",
    "SF",
    "SH",
)

TRAD_RATE_METRICS: list[str] = [
    "batted_ball_rate_mlb_eq_non_ar_delta",
    "strikeout_rate_mlb_eq_non_ar_delta",
    "walk_rate_mlb_eq_non_ar_delta",
    "hit_by_pitch_rate_mlb_eq_non_ar_delta",
    "singles_rate_bbe_mlb_eq_non_ar_delta",
    "doubles_rate_bbe_mlb_eq_non_ar_delta",
    "triples_rate_bbe_mlb_eq_non_ar_delta",
    "home_run_rate_bbe_mlb_eq_non_ar_delta",
    "sac_fly_rate_bbe_mlb_eq_non_ar_delta",
    "sac_hit_rate_bbe_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta",
    "stolen_base_success_rate_mlb_eq_non_ar_delta",
    "xbh_from_h_rate_mlb_eq_non_ar_delta",
    "runs_rate_mlb_eq_non_ar_delta",
    "rbi_rate_mlb_eq_non_ar_delta",
    "babip_recalc_rate_mlb_eq_non_ar_delta",
]

TWO_STAGE_TARGET_METRICS: list[str] = [
    "PA",
    "batted_ball_rate_mlb_eq_non_ar_delta",
    "strikeout_rate_mlb_eq_non_ar_delta",
    "walk_rate_mlb_eq_non_ar_delta",
    "singles_rate_bbe_mlb_eq_non_ar_delta",
    "doubles_rate_bbe_mlb_eq_non_ar_delta",
    "triples_rate_bbe_mlb_eq_non_ar_delta",
    "home_run_rate_bbe_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
    "stolen_base_success_rate_mlb_eq_non_ar_delta",
]
TWO_STAGE_EXCLUDED_KPI_SKILLS: set[str] = {"max_EV_reg"}
TWO_STAGE_ALPHA_GRID: list[float] = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
TWO_STAGE_MODES: tuple[str, str] = ("raw", "zspace")
TWO_STAGE_ZSPACE_MIN_MLB_PA_DEFAULT: float = 200.0

COMPOSITE_METRIC_SOURCE: dict[str, str] = {
    # old advanced (adv_*)
    "strikeout_rate_mlb_eq_non_ar_delta": "old_advanced",
    "stolen_base_success_rate_mlb_eq_non_ar_delta": "old_advanced",
    # old naive (naive_*)
    "runs_rate_mlb_eq_non_ar_delta": "old_naive",
    "sac_hit_rate_bbe_mlb_eq_non_ar_delta": "old_naive",
    # updated advanced (upd_adv_*)
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": "updated_advanced",
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": "updated_advanced",
    # updated naive (upd_naive_*)
    "babip_recalc_rate_mlb_eq_non_ar_delta": "updated_naive",
    "batted_ball_rate_mlb_eq_non_ar_delta": "updated_naive",
    "doubles_rate_bbe_mlb_eq_non_ar_delta": "updated_naive",
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": "updated_naive",
    "home_run_rate_bbe_mlb_eq_non_ar_delta": "updated_naive",
    "rbi_rate_mlb_eq_non_ar_delta": "updated_naive",
    "sac_fly_rate_bbe_mlb_eq_non_ar_delta": "updated_naive",
    "singles_rate_bbe_mlb_eq_non_ar_delta": "updated_naive",
    "walk_rate_mlb_eq_non_ar_delta": "updated_naive",
    "xbh_from_h_rate_mlb_eq_non_ar_delta": "updated_naive",
}

TRAD_PREDICTIVE_PROFILE: dict[str, dict[str, float]] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 94.48},
    "strikeout_rate_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 90.40},
    "walk_rate_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 238.83},
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": {"lambda": 0.80, "k": 272.66},
    "singles_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.67, "k": 443.18},
    "doubles_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.80, "k": 1278.94},
    "triples_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.67, "k": 505.95},
    "home_run_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 175.32},
    "sac_fly_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.25, "k": 463.18},
    "sac_hit_rate_bbe_mlb_eq_non_ar_delta": {"lambda": 0.67, "k": 4213.64},
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 128.70},
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 16.88},
    "stolen_base_success_rate_mlb_eq_non_ar_delta": {"lambda": 0.00, "k": 50.92},
    "xbh_from_h_rate_mlb_eq_non_ar_delta": {"lambda": 0.67, "k": 86.49},
    "runs_rate_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 752.86},
    "rbi_rate_mlb_eq_non_ar_delta": {"lambda": 0.50, "k": 552.67},
    "babip_recalc_rate_mlb_eq_non_ar_delta": {"lambda": 0.67, "k": 528.80},
}

RARE_EVENT_EXTREME_PROFILE: dict[str, dict[str, float]] = {
    # For individual, sparse events (HR/SB), preserve strong player-specific tails
    # when there is enough recent denominator signal.
    "home_run_rate_bbe_mlb_eq_non_ar_delta": {
        "ratio_low": 0.5,
        "ratio_high": 4,
        "abs_gap": 0.01,
        "min_den": 600.0,
    },
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": {
        "ratio_low": 0.40,
        "ratio_high": 1.90,
        "abs_gap": 0.01,
        "min_den": 600.0,
    },
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": {
        "ratio_low": 0.45,
        "ratio_high": 1.80,
        "abs_gap": 0.025,
        "min_den": 300.0,
    },
}


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    n = _safe_numeric(num)
    d = _safe_numeric(den)
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    mask = n.notna() & d.notna() & np.isfinite(n) & np.isfinite(d) & (d > 0)
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def _solve_nonneg_least_squares_3(
    rows: list[tuple[np.ndarray, float, float]],
) -> np.ndarray:
    # Small active-set solve for 3 vars with x>=0 constraints.
    # rows: (a_vec[3], b_scalar, weight)
    if not rows:
        return np.zeros(3, dtype="float64")
    candidates = [
        (0, 1, 2),
        (0, 1),
        (0, 2),
        (1, 2),
        (0,),
        (1,),
        (2,),
        tuple(),
    ]
    best_x = np.zeros(3, dtype="float64")
    best_obj = np.inf
    for free in candidates:
        x = np.zeros(3, dtype="float64")
        if len(free) > 0:
            M = np.zeros((len(free), len(free)), dtype="float64")
            y = np.zeros(len(free), dtype="float64")
            for a, b, w in rows:
                af = a[list(free)]
                M += w * np.outer(af, af)
                y += w * af * b
            # small ridge for numeric stability
            M = M + (1e-10 * np.eye(len(free), dtype="float64"))
            try:
                xf = np.linalg.solve(M, y)
            except Exception:
                xf = np.linalg.lstsq(M, y, rcond=None)[0]
            if np.any(~np.isfinite(xf)) or np.any(xf < 0):
                continue
            x[list(free)] = xf
        obj = 0.0
        for a, b, w in rows:
            r = float(np.dot(a, x) - b)
            obj += float(w) * (r * r)
        if obj < best_obj:
            best_obj = obj
            best_x = x
    return np.clip(best_x, 0.0, None)


def _solve_hit_mix_rate_authoritative(
    *,
    one_t: pd.Series,
    two_t: pd.Series,
    three_t: pd.Series,
    babip_t: pd.Series,
    xbh_t: pd.Series,
    bbe: pd.Series,
    bip: pd.Series,
    hr: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = one_t.index
    out_one = pd.Series(np.nan, index=idx, dtype="float64")
    out_two = pd.Series(np.nan, index=idx, dtype="float64")
    out_three = pd.Series(np.nan, index=idx, dtype="float64")

    one_arr = _safe_numeric(one_t).to_numpy(dtype=float)
    two_arr = _safe_numeric(two_t).to_numpy(dtype=float)
    three_arr = _safe_numeric(three_t).to_numpy(dtype=float)
    bab_arr = _safe_numeric(babip_t).to_numpy(dtype=float)
    xbh_arr = _safe_numeric(xbh_t).to_numpy(dtype=float)
    bbe_arr = _safe_numeric(bbe).to_numpy(dtype=float)
    bip_arr = _safe_numeric(bip).to_numpy(dtype=float)
    hr_arr = _safe_numeric(hr).to_numpy(dtype=float)

    for i in range(len(idx)):
        rows: list[tuple[np.ndarray, float, float]] = []
        if np.isfinite(one_arr[i]) and np.isfinite(bbe_arr[i]):
            rows.append((np.array([1.0, 0.0, 0.0], dtype="float64"), float(max(0.0, one_arr[i])), 1.0))
        if np.isfinite(two_arr[i]) and np.isfinite(bbe_arr[i]):
            rows.append((np.array([0.0, 1.0, 0.0], dtype="float64"), float(max(0.0, two_arr[i])), 1.0))
        if np.isfinite(three_arr[i]) and np.isfinite(bbe_arr[i]):
            rows.append((np.array([0.0, 0.0, 1.0], dtype="float64"), float(max(0.0, three_arr[i])), 0.35))

        if np.isfinite(bab_arr[i]) and np.isfinite(bip_arr[i]) and bip_arr[i] >= 0:
            target_non_hr = float(max(0.0, bab_arr[i] * bip_arr[i]))
            rows.append((np.array([1.0, 1.0, 1.0], dtype="float64"), target_non_hr, 1.0))

        if np.isfinite(xbh_arr[i]) and np.isfinite(hr_arr[i]):
            x = float(np.clip(xbh_arr[i], 0.0, 1.0))
            # (2B+3B+HR) - x*(1B+2B+3B+HR) = 0 -> linear in [1B,2B,3B]
            a = np.array([-x, 1.0 - x, 1.0 - x], dtype="float64")
            b = float(-(1.0 - x) * max(0.0, hr_arr[i]))
            rows.append((a, b, 1.0))

        if not rows:
            v = np.array(
                [
                    max(0.0, one_arr[i]) if np.isfinite(one_arr[i]) else 0.0,
                    max(0.0, two_arr[i]) if np.isfinite(two_arr[i]) else 0.0,
                    max(0.0, three_arr[i]) if np.isfinite(three_arr[i]) else 0.0,
                ],
                dtype="float64",
            )
        else:
            v = _solve_nonneg_least_squares_3(rows)

        out_one.iat[i] = float(v[0])
        out_two.iat[i] = float(v[1])
        out_three.iat[i] = float(v[2])

    return out_one, out_two, out_three


def _load_projection(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing projection file: {path}")
    return pd.read_parquet(path)


def _build_traditional_from_historical_rates(
    *,
    historical_rates_path: Path,
    constants_path: Path,
    bp_hitting_path: Path,
    historical_ar_path: Path,
    out_projection_path: Path,
    out_age_curve_path: Path,
    metric_recency_weights_json: Path | None,
    z_coherence_mode: str,
    z_anchor_k: float,
    hr_anchor_k: float,
    z_tail_strength: float,
    coherence_mode: str,
    uncertainty_draws: int,
    seed: int,
    default_k: float,
    k_scale: float,
) -> Path:
    from build_bp_rate_projections_2026_non_ar_post_inv_coh import (
        _load_metric_recency_weights,
        build_bp_rate_projections_2026,
    )

    metric_recency_weights = _load_metric_recency_weights(metric_recency_weights_json)
    build_bp_rate_projections_2026(
        input_path=historical_rates_path,
        constants_path=constants_path,
        bp_hitting_path=bp_hitting_path,
        historical_ar_path=historical_ar_path,
        z_coherence_mode=z_coherence_mode,
        z_anchor_k=float(z_anchor_k),
        hr_anchor_k=float(hr_anchor_k),
        z_tail_strength=float(z_tail_strength),
        coherence_mode=coherence_mode,
        out_projection_path=out_projection_path,
        out_age_curve_path=out_age_curve_path,
        uncertainty_draws=int(uncertainty_draws),
        seed=int(seed),
        metric_recency_weights=metric_recency_weights,
        default_k=float(default_k),
        k_scale=float(k_scale),
    )
    return out_projection_path


def _load_mlb_pa_lookup(bp_path: Path) -> pd.DataFrame:
    if not bp_path.exists():
        raise FileNotFoundError(f"Missing BP hitting table: {bp_path}")

    bp = pd.read_parquet(bp_path)
    level_col = (
        "bp_level_id"
        if "bp_level_id" in bp.columns
        else ("level_id" if "level_id" in bp.columns else None)
    )
    pa_col = (
        "plate_appearances_agg"
        if "plate_appearances_agg" in bp.columns
        else ("PA" if "PA" in bp.columns else None)
    )
    if level_col is None or pa_col is None:
        raise ValueError(
            f"BP table must include one of bp_level_id/level_id and one of plate_appearances_agg/PA: {bp_path}"
        )
    needed = {"mlbid", "season", level_col, pa_col}
    if not needed.issubset(bp.columns):
        raise ValueError(
            f"BP table missing required columns {needed - set(bp.columns)}: {bp_path}"
        )

    work = bp[list(needed)].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    work[level_col] = _safe_numeric(work[level_col])
    work[pa_col] = _safe_numeric(work[pa_col])
    work = work[
        work["mlbid"].notna()
        & work["season"].notna()
        & work[level_col].notna()
        & work[pa_col].notna()
        & (work[level_col].astype(int) == 1)
    ].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work[pa_col] = work[pa_col].clip(lower=0.0)

    lookup = (
        work.groupby(["mlbid", "season"], dropna=False)[pa_col]
        .sum()
        .rename("mlb_pa")
        .reset_index()
    )
    return lookup


def _load_mlb_ops_lookup(bp_path: Path) -> pd.DataFrame:
    if not bp_path.exists():
        raise FileNotFoundError(f"Missing BP hitting table: {bp_path}")
    bp = pd.read_parquet(bp_path)
    level_col = (
        "bp_level_id"
        if "bp_level_id" in bp.columns
        else ("level_id" if "level_id" in bp.columns else None)
    )
    if level_col is None or "ops" not in bp.columns:
        return pd.DataFrame(columns=["mlbid", "season", "mlb_ops"])
    needed = {"mlbid", "season", level_col, "ops", "plate_appearances_agg"}
    if not needed.issubset(bp.columns):
        return pd.DataFrame(columns=["mlbid", "season", "mlb_ops"])
    work = bp[list(needed)].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    work[level_col] = _safe_numeric(work[level_col])
    work["ops"] = _safe_numeric(work["ops"])
    work["plate_appearances_agg"] = _safe_numeric(work["plate_appearances_agg"])
    work = work[
        work["mlbid"].notna()
        & work["season"].notna()
        & work[level_col].notna()
        & work["ops"].notna()
        & work["plate_appearances_agg"].notna()
        & (work[level_col].astype(int) == 1)
        & (work["plate_appearances_agg"] > 0.0)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbid", "season", "mlb_ops"])
    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    ops_lookup = (
        work.groupby(["mlbid", "season"], dropna=False)
        .apply(
            lambda g: np.average(
                _safe_numeric(g["ops"]).to_numpy(dtype=float),
                weights=_safe_numeric(g["plate_appearances_agg"]).to_numpy(dtype=float),
            )
        )
        .rename("mlb_ops")
        .reset_index()
    )
    return ops_lookup


def _load_level_pa_shares(bp_path: Path) -> pd.DataFrame:
    if not bp_path.exists():
        return pd.DataFrame(
            columns=[
                "mlbid",
                "season",
                "bp_level_id",
                "level_pa_share",
                "season_total_pa",
            ]
        )
    bp = pd.read_parquet(bp_path)
    level_col = (
        "bp_level_id"
        if "bp_level_id" in bp.columns
        else ("level_id" if "level_id" in bp.columns else None)
    )
    pa_col = (
        "plate_appearances_agg"
        if "plate_appearances_agg" in bp.columns
        else ("PA" if "PA" in bp.columns else None)
    )
    if level_col is None or pa_col is None:
        return pd.DataFrame(
            columns=[
                "mlbid",
                "season",
                "bp_level_id",
                "level_pa_share",
                "season_total_pa",
            ]
        )

    needed = {"mlbid", "season", level_col, pa_col}
    if not needed.issubset(bp.columns):
        return pd.DataFrame(
            columns=[
                "mlbid",
                "season",
                "bp_level_id",
                "level_pa_share",
                "season_total_pa",
            ]
        )

    work = bp[list(needed)].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    work[level_col] = _safe_numeric(work[level_col])
    work[pa_col] = _safe_numeric(work[pa_col]).fillna(0.0).clip(lower=0.0)
    work = work[
        work["mlbid"].notna()
        & work["season"].notna()
        & work[level_col].notna()
        & work[level_col].astype(int).isin([1, 2, 3, 4, 5])
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "mlbid",
                "season",
                "bp_level_id",
                "level_pa_share",
                "season_total_pa",
            ]
        )

    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work["bp_level_id"] = work[level_col].astype(int)
    grouped = (
        work.groupby(["mlbid", "season", "bp_level_id"], dropna=False)[pa_col]
        .sum()
        .rename("level_pa")
        .reset_index()
    )
    total = (
        grouped.groupby(["mlbid", "season"], dropna=False)["level_pa"]
        .sum()
        .rename("total_pa")
        .reset_index()
    )
    grouped = grouped.merge(total, on=["mlbid", "season"], how="left")
    grouped["level_pa_share"] = (
        _safe_divide(grouped["level_pa"], grouped["total_pa"])
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    grouped = grouped.rename(columns={"total_pa": "season_total_pa"})
    return grouped[
        ["mlbid", "season", "bp_level_id", "level_pa_share", "season_total_pa"]
    ].copy()


def _coverage_from_mlb_pa(
    proj: pd.DataFrame,
    *,
    id_col: str,
    source_season_col: str,
    mlb_lookup: pd.DataFrame,
    stability_pa_per_season: float,
    seasons_back: int = 3,
    spread_boost_max: float = 0.65,
) -> pd.DataFrame:
    out = proj.copy()
    out[id_col] = _safe_numeric(out[id_col]).astype("Int64")
    out[source_season_col] = _safe_numeric(out[source_season_col]).astype("Int64")

    lookup_series = mlb_lookup.assign(
        mlbid=_safe_numeric(mlb_lookup["mlbid"]).astype("Int64"),
        season=_safe_numeric(mlb_lookup["season"]).astype("Int64"),
        mlb_pa=_safe_numeric(mlb_lookup["mlb_pa"]).fillna(0.0),
    ).set_index(["mlbid", "season"])["mlb_pa"]

    mlb_recent = np.zeros(len(out), dtype=float)
    ids = out[id_col].to_numpy()
    src = out[source_season_col].to_numpy()
    for lag in range(int(seasons_back)):
        season_vals = src - lag
        idx = pd.MultiIndex.from_arrays([ids, season_vals], names=["mlbid", "season"])
        vals = lookup_series.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        mlb_recent += vals

    required = max(float(stability_pa_per_season), 1.0) * float(
        max(int(seasons_back), 1)
    )
    coverage = np.clip(mlb_recent / required, 0.0, 1.0)
    baseline_share = 1.0 - coverage
    spread_mult = 1.0 + (float(spread_boost_max) * baseline_share)

    out["mlb_pa_recent_3yr"] = mlb_recent
    out["mlb_pa_coverage_3yr"] = coverage
    out["baseline_fill_share_3yr"] = baseline_share
    out["variability_injection_mult"] = spread_mult
    out["variability_stability_pa_per_season"] = float(stability_pa_per_season)
    return out


def _collect_metric_bases(df: pd.DataFrame) -> list[str]:
    return sorted(c[: -len("_proj_p50")] for c in df.columns if c.endswith("_proj_p50"))


def _ensure_p25_p75(df: pd.DataFrame, *, metric_bases: list[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metric_bases:
        c20 = f"{metric}_proj_p20"
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        c80 = f"{metric}_proj_p80"
        if c50 not in out.columns:
            continue

        if c25 not in out.columns and c20 in out.columns:
            p20 = _safe_numeric(out[c20])
            p50 = _safe_numeric(out[c50])
            out[c25] = p20 + ((p50 - p20) * (1.0 / 6.0))
        if c75 not in out.columns and c80 in out.columns:
            p50 = _safe_numeric(out[c50])
            p80 = _safe_numeric(out[c80])
            out[c75] = p50 + ((p80 - p50) * (5.0 / 6.0))
    return out


def _bounds_for_metric(
    metric: str,
    *,
    mode: str,
) -> tuple[float | None, float | None]:
    if mode == "kpi":
        if metric in KPI_BOUNDS:
            return KPI_BOUNDS[metric]
        if metric in KPI_ALIAS_MAP.values():
            reg = next((k for k, v in KPI_ALIAS_MAP.items() if v == metric), metric)
            if reg in KPI_BOUNDS:
                return KPI_BOUNDS[reg]

    if metric.endswith("_mlb_eq_non_ar_delta"):
        return (0.0, 1.0)
    if "_rate_" in metric:
        return (0.0, 1.0)
    if metric in {"AVG", "OBP", "SLG", "OPS"}:
        return (0.0, 2.0)
    if metric in {"K%", "BB%"}:
        return (0.0, 100.0)
    if metric.startswith(COUNT_PREFIXES):
        return (0.0, None)
    return (None, None)


def _apply_mlb_pa_spread_injection(
    df: pd.DataFrame,
    *,
    metric_bases: list[str],
    mode: str,
) -> pd.DataFrame:
    out = df.copy()
    if "variability_injection_mult" not in out.columns:
        return out

    mult = _safe_numeric(out["variability_injection_mult"]).fillna(1.0).clip(lower=1.0)
    pct_order = [20, 25, 50, 75, 80]

    for metric in metric_bases:
        c50 = f"{metric}_proj_p50"
        c25 = f"{metric}_proj_p25"
        c75 = f"{metric}_proj_p75"
        if not all(c in out.columns for c in [c50, c25, c75]):
            continue

        p50 = _safe_numeric(out[c50])
        p25 = _safe_numeric(out[c25])
        p75 = _safe_numeric(out[c75])
        dn = (p50 - p25).clip(lower=0.0)
        up = (p75 - p50).clip(lower=0.0)
        out[c25] = p50 - (dn * mult)
        out[c75] = p50 + (up * mult)

        lo, hi = _bounds_for_metric(metric, mode=mode)
        prev = None
        for pct in pct_order:
            col = f"{metric}_proj_p{pct}"
            if col not in out.columns:
                continue
            vals = _safe_numeric(out[col])
            if lo is not None:
                vals = vals.clip(lower=float(lo))
            if hi is not None:
                vals = vals.clip(upper=float(hi))
            if prev is not None:
                vals = np.maximum(vals, prev)
            out[col] = vals
            prev = vals

        c20 = f"{metric}_proj_p20"
        c80 = f"{metric}_proj_p80"
        if c20 in out.columns and c25 in out.columns:
            out[c20] = np.minimum(_safe_numeric(out[c20]), _safe_numeric(out[c25]))
        if c80 in out.columns and c75 in out.columns:
            out[c80] = np.maximum(_safe_numeric(out[c80]), _safe_numeric(out[c75]))

        spread_col = f"{metric}_proj_spread"
        if spread_col in out.columns:
            out[spread_col] = _safe_numeric(out[c75]) - _safe_numeric(out[c25])
    return out


def _build_traditional_hist_context(
    historical_rates_path: Path,
    *,
    mlb_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if not historical_rates_path.exists():
        raise FileNotFoundError(
            f"Missing historical rates file: {historical_rates_path}"
        )
    hist = pd.read_parquet(historical_rates_path).copy()
    need = {"mlbid", "season", "plate_appearances_agg"}
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {historical_rates_path}: {missing}"
        )

    hist["mlbid"] = _safe_numeric(hist["mlbid"])
    hist["season"] = _safe_numeric(hist["season"])
    hist["plate_appearances_agg"] = _safe_numeric(hist["plate_appearances_agg"]).clip(
        lower=0.0
    )
    hist = hist[hist["mlbid"].notna() & hist["season"].notna()].copy()
    hist["mlbid"] = hist["mlbid"].astype("int64")
    hist["season"] = hist["season"].astype("int64")
    hist = hist.merge(
        mlb_lookup,
        left_on=["mlbid", "season"],
        right_on=["mlbid", "season"],
        how="left",
    )
    hist["mlb_pa"] = _safe_numeric(hist["mlb_pa"]).fillna(0.0).clip(lower=0.0)

    pa = _safe_numeric(hist["plate_appearances_agg"]).fillna(0.0).clip(lower=0.0)
    bbe = _safe_numeric(hist.get("BBE_mlb_eq_non_ar_delta", np.nan))
    h = _safe_numeric(hist.get("H_mlb_eq_non_ar_delta", np.nan))
    ab = _safe_numeric(hist.get("AB_mlb_eq_non_ar_delta", np.nan))
    so = _safe_numeric(hist.get("SO_mlb_eq_non_ar_delta", np.nan))
    hr = _safe_numeric(hist.get("HR_mlb_eq_non_ar_delta", np.nan))
    sf = _safe_numeric(hist.get("SF_mlb_eq_non_ar_delta", np.nan))

    sba_pa_rate = _safe_numeric(
        hist.get("stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta", np.nan)
    )
    sba_sbo_rate = _safe_numeric(
        hist.get("stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta", np.nan)
    )
    sba_est = (pa * sba_pa_rate).where(pa.notna() & sba_pa_rate.notna())
    sbo_est = pd.Series(np.nan, index=hist.index, dtype="float64")
    mask_sbo = sba_est.notna() & sba_sbo_rate.notna() & (sba_sbo_rate > 1e-6)
    sbo_est.loc[mask_sbo] = sba_est.loc[mask_sbo] / sba_sbo_rate.loc[mask_sbo]

    bip_est = (ab - so - hr + sf).where(
        ab.notna() & so.notna() & hr.notna() & sf.notna()
    )

    denom_map: dict[str, pd.Series] = {
        "batted_ball_rate_mlb_eq_non_ar_delta": pa,
        "strikeout_rate_mlb_eq_non_ar_delta": pa,
        "walk_rate_mlb_eq_non_ar_delta": pa,
        "hit_by_pitch_rate_mlb_eq_non_ar_delta": pa,
        "singles_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "doubles_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "triples_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "home_run_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "sac_fly_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "sac_hit_rate_bbe_mlb_eq_non_ar_delta": bbe,
        "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": pa,
        "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": sbo_est,
        "stolen_base_success_rate_mlb_eq_non_ar_delta": sba_est,
        "xbh_from_h_rate_mlb_eq_non_ar_delta": h,
        "runs_rate_mlb_eq_non_ar_delta": pa,
        "rbi_rate_mlb_eq_non_ar_delta": pa,
        "babip_recalc_rate_mlb_eq_non_ar_delta": bip_est,
    }
    for metric in TRAD_RATE_METRICS:
        hist[metric] = _safe_numeric(hist.get(metric, np.nan))
        hist[f"n_{metric}"] = _safe_numeric(denom_map.get(metric, np.nan)).clip(
            lower=0.0
        )
    keep = [
        "mlbid",
        "season",
        "mlb_pa",
        *TRAD_RATE_METRICS,
        *[f"n_{m}" for m in TRAD_RATE_METRICS],
    ]
    if "levels_rolled_up" in hist.columns:
        keep.append("levels_rolled_up")
    return hist[keep].copy()


def _add_traditional_ops_quality(
    proj: pd.DataFrame,
    *,
    mlb_lookup: pd.DataFrame,
    mlb_ops_lookup: pd.DataFrame,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty:
        return out
    if mlb_ops_lookup.empty:
        out["ops_quality_recent"] = np.nan
        out["ops_quality_z"] = 0.0
        out["quality_established_factor"] = np.clip(
            (_safe_numeric(out.get("mlb_pa_recent_3yr", 0.0)).fillna(0.0) - 200.0)
            / 900.0,
            0.0,
            1.0,
        )
        return out

    pa_series = mlb_lookup.assign(
        mlbid=_safe_numeric(mlb_lookup["mlbid"]).astype("Int64"),
        season=_safe_numeric(mlb_lookup["season"]).astype("Int64"),
        mlb_pa=_safe_numeric(mlb_lookup["mlb_pa"]).fillna(0.0),
    ).set_index(["mlbid", "season"])["mlb_pa"]
    ops_series = mlb_ops_lookup.assign(
        mlbid=_safe_numeric(mlb_ops_lookup["mlbid"]).astype("Int64"),
        season=_safe_numeric(mlb_ops_lookup["season"]).astype("Int64"),
        mlb_ops=_safe_numeric(mlb_ops_lookup["mlb_ops"]),
    ).set_index(["mlbid", "season"])["mlb_ops"]

    ids = _safe_numeric(out["mlbid"]).astype("Int64").to_numpy()
    src = _safe_numeric(out["source_season"]).astype("Int64").to_numpy()
    lag_weights = [1.0, 0.67, 0.45]
    ops_num = np.zeros(len(out), dtype=float)
    ops_den = np.zeros(len(out), dtype=float)
    for lag, w in enumerate(lag_weights):
        idx = pd.MultiIndex.from_arrays([ids, src - lag], names=["mlbid", "season"])
        pa_vals = pa_series.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ops_vals = ops_series.reindex(idx).to_numpy(dtype=float)
        w_pa = w * pa_vals
        good = np.isfinite(ops_vals) & (w_pa > 0)
        ops_num[good] += ops_vals[good] * w_pa[good]
        ops_den[good] += w_pa[good]
    ops_recent = np.full(len(out), np.nan, dtype=float)
    good_ops = ops_den > 0
    ops_recent[good_ops] = ops_num[good_ops] / ops_den[good_ops]
    out["ops_quality_recent"] = ops_recent

    ops_join = mlb_ops_lookup.merge(mlb_lookup, on=["mlbid", "season"], how="left")
    ops_join["mlb_pa"] = _safe_numeric(ops_join["mlb_pa"]).fillna(0.0).clip(lower=0.0)
    ops_join["mlb_ops"] = _safe_numeric(ops_join["mlb_ops"])
    league_rows: list[tuple[int, float, float]] = []
    for season in sorted(
        _safe_numeric(out["source_season"]).dropna().astype(int).unique().tolist()
    ):
        vals: list[float] = []
        wts: list[float] = []
        for lag, w in enumerate(lag_weights):
            s = int(season - lag)
            g = ops_join[_safe_numeric(ops_join["season"]).astype("Int64") == s]
            x = _safe_numeric(g["mlb_ops"]).to_numpy(dtype=float)
            p = _safe_numeric(g["mlb_pa"]).to_numpy(dtype=float)
            ww = w * p
            m = np.isfinite(x) & np.isfinite(ww) & (ww > 0)
            if m.any():
                vals.append(x[m])
                wts.append(ww[m])
        if not vals:
            league_rows.append((season, np.nan, np.nan))
            continue
        vx = np.concatenate(vals)
        vw = np.concatenate(wts)
        mu = float(np.average(vx, weights=vw))
        var = float(np.average((vx - mu) ** 2, weights=vw))
        sd = float(np.sqrt(max(var, 1e-8)))
        league_rows.append((season, mu, sd))
    lg = pd.DataFrame(league_rows, columns=["source_season", "ops_lg_mu", "ops_lg_sd"])
    out = out.merge(lg, on="source_season", how="left")
    out["ops_quality_z"] = (
        _safe_numeric(out["ops_quality_recent"]) - _safe_numeric(out["ops_lg_mu"])
    ) / _safe_numeric(out["ops_lg_sd"]).replace(0.0, np.nan)
    out["ops_quality_z"] = (
        _safe_numeric(out["ops_quality_z"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=-2.5, upper=2.5)
    )
    out["quality_established_factor"] = np.clip(
        (_safe_numeric(out.get("mlb_pa_recent_3yr", 0.0)).fillna(0.0) - 200.0) / 900.0,
        0.0,
        1.0,
    )
    return out


def _apply_traditional_predictive_adjustments(
    proj: pd.DataFrame,
    *,
    historical_rates_path: Path,
    bp_hitting_path: Path,
    mlb_lookup: pd.DataFrame,
    mlb_ops_lookup: pd.DataFrame,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty:
        return out
    hist = _build_traditional_hist_context(historical_rates_path, mlb_lookup=mlb_lookup)
    out["mlbid"] = _safe_numeric(out["mlbid"]).astype("Int64")
    out["source_season"] = _safe_numeric(out["source_season"]).astype("Int64")

    out = _add_traditional_ops_quality(
        out, mlb_lookup=mlb_lookup, mlb_ops_lookup=mlb_ops_lookup
    )
    ids = _safe_numeric(out["mlbid"]).fillna(-1).astype("int64").to_numpy()
    src = _safe_numeric(out["source_season"]).fillna(-1).astype("int64").to_numpy()

    level_shares = _load_level_pa_shares(bp_hitting_path)
    level_share_map: dict[int, np.ndarray] = {}
    metric_level_mu_by_season: dict[str, dict[tuple[int, int], float]] = {}
    metric_level_mu_global: dict[str, dict[int, float]] = {}
    low_sample_low_level_mask = np.zeros(len(out), dtype=bool)
    hard_level_revert_mask = np.zeros(len(out), dtype=bool)
    if not level_shares.empty:
        level_series = level_shares.assign(
            mlbid=_safe_numeric(level_shares["mlbid"]).astype("Int64"),
            season=_safe_numeric(level_shares["season"]).astype("Int64"),
            bp_level_id=_safe_numeric(level_shares["bp_level_id"]).astype("Int64"),
            level_pa_share=_safe_numeric(level_shares["level_pa_share"])
            .fillna(0.0)
            .clip(lower=0.0),
        ).set_index(["mlbid", "season", "bp_level_id"])["level_pa_share"]
        total_pa_series = (
            level_shares.assign(
                mlbid=_safe_numeric(level_shares["mlbid"]).astype("Int64"),
                season=_safe_numeric(level_shares["season"]).astype("Int64"),
                season_total_pa=_safe_numeric(level_shares["season_total_pa"])
                .fillna(0.0)
                .clip(lower=0.0),
            )
            .drop_duplicates(subset=["mlbid", "season"])
            .set_index(["mlbid", "season"])["season_total_pa"]
        )
        for lvl in [1, 2, 3, 4, 5]:
            lvl_idx = pd.MultiIndex.from_arrays(
                [ids, src, np.full(len(out), int(lvl), dtype="int64")],
                names=["mlbid", "season", "bp_level_id"],
            )
            level_share_map[int(lvl)] = (
                level_series.reindex(lvl_idx).fillna(0.0).to_numpy(dtype=float)
            )
        source_idx = pd.MultiIndex.from_arrays([ids, src], names=["mlbid", "season"])
        source_total_pa = (
            total_pa_series.reindex(source_idx).fillna(0.0).to_numpy(dtype=float)
        )
        mlb_share = level_share_map.get(1, np.zeros(len(out), dtype=float))
        mlb_recent = (
            _safe_numeric(out.get("mlb_pa_recent_3yr", 0.0))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        low_sample_low_level_mask = (
            (mlb_share < 0.05) & (mlb_recent < 30.0) & (source_total_pa < 450.0)
        )
        # For ultra-low source-season samples at non-MLB levels, ignore player input
        # and fully revert to prior-season level baseline (e.g., AAA -> AAA).
        hard_level_revert_mask = (
            (mlb_share < 0.05) & (mlb_recent < 30.0) & (source_total_pa < 30.0)
        )
        hist_level = pd.DataFrame()
        if "levels_rolled_up" in hist.columns:
            raw_levels = hist["levels_rolled_up"].astype(str).str.strip()
            # Use single-level rows to estimate true level-specific priors in MLB-equivalent space.
            single_mask = raw_levels.str.fullmatch(r"\d+")
            hist_level = hist.loc[single_mask].copy()
            hist_level["bp_level_id"] = _safe_numeric(
                raw_levels.loc[single_mask]
            ).astype("Int64")
            hist_level = hist_level[
                hist_level["bp_level_id"].notna()
                & hist_level["bp_level_id"].astype(int).isin([1, 2, 3, 4, 5])
            ].copy()
        for metric in TRAD_RATE_METRICS:
            n_col = f"n_{metric}"
            if metric not in hist_level.columns or n_col not in hist_level.columns:
                continue
            tmp = hist_level[["season", "bp_level_id", metric, n_col]].copy()
            tmp["season"] = _safe_numeric(tmp["season"]).astype("Int64")
            tmp["bp_level_id"] = _safe_numeric(tmp["bp_level_id"]).astype("Int64")
            tmp[metric] = _safe_numeric(tmp[metric])
            tmp[n_col] = _safe_numeric(tmp[n_col]).fillna(0.0).clip(lower=0.0)
            tmp = tmp[
                tmp["season"].notna()
                & tmp["bp_level_id"].notna()
                & tmp[metric].notna()
                & (tmp[n_col] > 0.0)
            ].copy()
            if tmp.empty:
                continue
            tmp["w"] = tmp[n_col]
            tmp = tmp[tmp["w"] > 0.0].copy()
            if tmp.empty:
                continue
            by_s_lvl: dict[tuple[int, int], float] = {}
            for (s, lvl), g in tmp.groupby(["season", "bp_level_id"], dropna=False):
                vals = _safe_numeric(g[metric]).to_numpy(dtype=float)
                wts = _safe_numeric(g["w"]).to_numpy(dtype=float)
                good = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
                if good.any():
                    by_s_lvl[(int(s), int(lvl))] = float(
                        np.average(vals[good], weights=wts[good])
                    )
            by_lvl: dict[int, float] = {}
            for lvl, g in tmp.groupby("bp_level_id", dropna=False):
                vals = _safe_numeric(g[metric]).to_numpy(dtype=float)
                wts = _safe_numeric(g["w"]).to_numpy(dtype=float)
                good = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
                if good.any():
                    by_lvl[int(lvl)] = float(np.average(vals[good], weights=wts[good]))
            metric_level_mu_by_season[metric] = by_s_lvl
            metric_level_mu_global[metric] = by_lvl

    sparse_zero_prior_metrics = {
        "home_run_rate_bbe_mlb_eq_non_ar_delta",
        "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
        "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta",
        "sac_hit_rate_bbe_mlb_eq_non_ar_delta",
        "triples_rate_bbe_mlb_eq_non_ar_delta",
    }
    event_lookup_by_metric: dict[str, pd.Series] = {}
    for metric in sparse_zero_prior_metrics:
        n_col = f"n_{metric}"
        if metric not in hist.columns or n_col not in hist.columns:
            continue
        ev = (
            (_safe_numeric(hist[metric]) * _safe_numeric(hist[n_col]))
            .fillna(0.0)
            .clip(lower=0.0)
        )
        s = pd.DataFrame(
            {
                "mlbid": _safe_numeric(hist["mlbid"]).astype("Int64"),
                "season": _safe_numeric(hist["season"]).astype("Int64"),
                "event": ev,
            }
        ).set_index(["mlbid", "season"])["event"]
        event_lookup_by_metric[metric] = s

    for metric in TRAD_RATE_METRICS:
        cfg = TRAD_PREDICTIVE_PROFILE.get(metric)
        if cfg is None:
            continue
        n_col = f"n_{metric}"
        if n_col not in hist.columns:
            continue
        den_series = (
            hist[["mlbid", "season", n_col]]
            .assign(
                mlbid=_safe_numeric(hist["mlbid"]).astype("Int64"),
                season=_safe_numeric(hist["season"]).astype("Int64"),
                den=_safe_numeric(hist[n_col]).fillna(0.0),
            )
            .set_index(["mlbid", "season"])["den"]
        )
        lam = float(cfg["lambda"])
        weights = [1.0, lam, lam * lam]
        n_eff = np.zeros(len(out), dtype=float)
        for lag, w in enumerate(weights):
            idx = pd.MultiIndex.from_arrays([ids, src - lag], names=["mlbid", "season"])
            n_eff += float(w) * den_series.reindex(idx).fillna(0.0).to_numpy(
                dtype=float
            )
        out[f"n_eff_pred_{metric}"] = n_eff

    mu_by_metric_season: dict[str, dict[int, float]] = {}
    mu_global: dict[str, float] = {}
    for metric in TRAD_RATE_METRICS:
        n_col = f"n_{metric}"
        if metric not in hist.columns or n_col not in hist.columns:
            continue
        tmp = hist[["season", "mlb_pa", metric, n_col]].copy()
        tmp["season"] = _safe_numeric(tmp["season"]).astype("Int64")
        tmp["mlb_pa"] = _safe_numeric(tmp["mlb_pa"]).fillna(0.0)
        tmp[metric] = _safe_numeric(tmp[metric])
        tmp[n_col] = _safe_numeric(tmp[n_col]).fillna(0.0)
        tmp = tmp[
            tmp["season"].notna()
            & tmp[metric].notna()
            & (tmp[n_col] > 0.0)
            & (tmp["mlb_pa"] > 0.0)
        ].copy()
        if tmp.empty:
            continue
        by_s: dict[int, float] = {}
        for s, g in tmp.groupby("season", dropna=False):
            vals = _safe_numeric(g[metric]).to_numpy(dtype=float)
            wts = _safe_numeric(g[n_col]).to_numpy(dtype=float)
            good = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
            if good.any():
                by_s[int(s)] = float(np.average(vals[good], weights=wts[good]))
        mu_by_metric_season[metric] = by_s
        mu_global[metric] = float(np.average(tmp[metric], weights=tmp[n_col]))

    qz = _safe_numeric(out.get("ops_quality_z", 0.0)).fillna(0.0).to_numpy(dtype=float)
    established = (
        _safe_numeric(out.get("quality_established_factor", 0.0))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    mult = (
        _safe_numeric(out.get("variability_injection_mult", 1.0))
        .fillna(1.0)
        .clip(lower=1.0)
        .to_numpy(dtype=float)
    )

    pct_order = [25, 50, 75, 80]
    for metric in TRAD_RATE_METRICS:
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        if not all(c in out.columns for c in [c25, c50, c75]):
            continue
        cfg = TRAD_PREDICTIVE_PROFILE.get(metric)
        if cfg is None:
            continue

        k = float(max(cfg["k"], 1e-6))
        n_eff = (
            _safe_numeric(out.get(f"n_eff_pred_{metric}", np.nan))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        w_base = np.clip(n_eff / (n_eff + k), 0.0, 1.0)
        pos = np.clip(qz, 0.0, None)
        neg = np.clip(-qz, 0.0, None)
        w_adj = np.clip(
            w_base + (0.14 * established * pos) - (0.12 * established * neg),
            0.02,
            0.995,
        )

        p25 = _safe_numeric(out[c25]).to_numpy(dtype=float)
        p50 = _safe_numeric(out[c50]).to_numpy(dtype=float)
        p75 = _safe_numeric(out[c75]).to_numpy(dtype=float)
        src_season = (
            _safe_numeric(out["source_season"]).fillna(0).astype(int).to_numpy()
        )
        mu_s = mu_by_metric_season.get(metric, {})
        mu0 = float(mu_global.get(metric, np.nanmean(p50)))
        mu = np.array([float(mu_s.get(int(s), mu0)) for s in src_season], dtype=float)

        # Low-sample low-level fallback: pull to prior-season level-mix baseline
        # (weighted across multiple levels when applicable) instead of MLB-wide prior.
        if metric in metric_level_mu_by_season and level_share_map:
            mu_num = np.zeros(len(out), dtype=float)
            mu_den = np.zeros(len(out), dtype=float)
            by_s_lvl = metric_level_mu_by_season.get(metric, {})
            by_lvl = metric_level_mu_global.get(metric, {})
            for lvl in [1, 2, 3, 4, 5]:
                share = level_share_map.get(int(lvl))
                if share is None:
                    continue
                mu_lvl = np.array(
                    [
                        float(
                            by_s_lvl.get(
                                (int(s), int(lvl)), by_lvl.get(int(lvl), np.nan)
                            )
                        )
                        for s in src_season
                    ],
                    dtype=float,
                )
                good = np.isfinite(mu_lvl) & (share > 0.0)
                if good.any():
                    mu_num[good] += share[good] * mu_lvl[good]
                    mu_den[good] += share[good]
            mu_level_mix = np.full(len(out), np.nan, dtype=float)
            ok = mu_den > 1e-8
            mu_level_mix[ok] = mu_num[ok] / mu_den[ok]
            # Apply level-mix prior per player whenever available, so priors reflect
            # each player's own source-season level distribution (e.g., AAA vs MLB).
            use_level_prior = np.isfinite(mu_level_mix)
            if use_level_prior.any():
                mu = np.where(use_level_prior, mu_level_mix, mu)

        # Sparse-event metrics: if recent implied events are zero, do not default to league average.
        # Keep prior at zero for low-evidence players (e.g., SBA, SH, 3B profiles).
        if metric in event_lookup_by_metric:
            ev_series = event_lookup_by_metric[metric]
            ev_sum = np.zeros(len(out), dtype=float)
            for lag in [0, 1, 2]:
                idx = pd.MultiIndex.from_arrays(
                    [ids, src - lag], names=["mlbid", "season"]
                )
                ev_sum += ev_series.reindex(idx).fillna(0.0).to_numpy(dtype=float)
            zero_recent_events = ev_sum <= 1e-10
            low_evidence = w_base <= 0.35
            use_zero_prior = zero_recent_events & low_evidence
            if use_zero_prior.any():
                mu = np.where(use_zero_prior, 0.0, mu)

        # Rare-event extreme protection: if recent observed rate is meaningfully
        # extreme (low or high) with enough denominator, pull p50 toward that
        # observed rate instead of league/level prior.
        rare_extreme_mask = np.zeros(len(out), dtype=bool)
        rare_recent_rate = np.full(len(out), np.nan, dtype=float)
        rare_recent_den = np.zeros(len(out), dtype=float)
        rare_cfg = RARE_EVENT_EXTREME_PROFILE.get(metric)
        if rare_cfg is not None and metric in event_lookup_by_metric:
            ev_series = event_lookup_by_metric[metric]
            for lag, w in enumerate(weights):
                idx = pd.MultiIndex.from_arrays(
                    [ids, src - lag], names=["mlbid", "season"]
                )
                rare_recent_den += float(w) * den_series.reindex(idx).fillna(
                    0.0
                ).to_numpy(dtype=float)
                rare_recent_rate_num = (
                    ev_series.reindex(idx).fillna(0.0).to_numpy(dtype=float)
                )
                rare_recent_rate_num = float(w) * rare_recent_rate_num
                rare_recent_rate = np.where(
                    np.isfinite(rare_recent_rate),
                    rare_recent_rate + rare_recent_rate_num,
                    rare_recent_rate_num,
                )
            valid_den = rare_recent_den > 1e-8
            rare_rate = np.full(len(out), np.nan, dtype=float)
            rare_rate[valid_den] = (
                rare_recent_rate[valid_den] / rare_recent_den[valid_den]
            )
            rare_recent_rate = rare_rate
            mu_safe = np.where(np.isfinite(mu) & (mu > 1e-10), mu, np.nan)
            ratio = np.where(np.isfinite(mu_safe), rare_recent_rate / mu_safe, np.nan)
            gap = np.abs(rare_recent_rate - mu)
            enough_den = rare_recent_den >= float(rare_cfg["min_den"])
            extreme_low = np.isfinite(ratio) & (ratio <= float(rare_cfg["ratio_low"]))
            extreme_high = np.isfinite(ratio) & (ratio >= float(rare_cfg["ratio_high"]))
            rare_extreme_mask = (
                enough_den
                & np.isfinite(rare_recent_rate)
                & np.isfinite(mu)
                & (gap >= float(rare_cfg["abs_gap"]))
                & (extreme_low | extreme_high)
            )

        # Force stronger regression for low-sample players without MLB evidence.
        if low_sample_low_level_mask.any():
            w_adj = np.where(low_sample_low_level_mask, np.minimum(w_adj, 0.05), w_adj)
        if hard_level_revert_mask.any():
            w_adj = np.where(hard_level_revert_mask, 0.0, w_adj)

        p50_new = (w_adj * p50) + ((1.0 - w_adj) * mu)
        if rare_extreme_mask.any():
            alpha = np.clip(rare_recent_den / (rare_recent_den + (2 * k)), 0.35, 0.65)
            p50_new = np.where(
                rare_extreme_mask,
                ((1.0 - alpha) * p50_new) + (alpha * rare_recent_rate),
                p50_new,
            )
        dn = np.clip(p50 - p25, 0.0, None)
        up = np.clip(p75 - p50, 0.0, None)
        asym = established * pos
        up_mult = np.clip(mult * (1.0 + (0.30 * asym)), 0.6, 2.8)
        down_mult = np.clip(
            mult * (1.0 - (0.15 * asym) + (0.08 * established * neg)), 0.55, 2.3
        )
        p25_new = p50_new - (dn * down_mult)
        p75_new = p50_new + (up * up_mult)

        out[c25] = p25_new
        out[c50] = p50_new
        out[c75] = p75_new

        lo, hi = _bounds_for_metric(metric, mode="traditional")
        prev = None
        for pct in pct_order:
            col = f"{metric}_proj_p{pct}"
            if col not in out.columns:
                continue
            vals = _safe_numeric(out[col])
            if lo is not None:
                vals = vals.clip(lower=float(lo))
            if hi is not None:
                vals = vals.clip(upper=float(hi))
            if prev is not None:
                vals = np.maximum(vals, prev)
            out[col] = vals
            prev = vals
        c20 = f"{metric}_proj_p20"
        c25_chk = f"{metric}_proj_p25"
        c80 = f"{metric}_proj_p80"
        c75_chk = f"{metric}_proj_p75"
        if c20 in out.columns and c25_chk in out.columns:
            out[c20] = np.minimum(_safe_numeric(out[c20]), _safe_numeric(out[c25_chk]))
        if c80 in out.columns and c75_chk in out.columns:
            out[c80] = np.maximum(_safe_numeric(out[c80]), _safe_numeric(out[c75_chk]))
        spread_col = f"{metric}_proj_spread"
        if spread_col in out.columns:
            out[spread_col] = _safe_numeric(out[c75]) - _safe_numeric(out[c25])

    return out


def _recompute_traditional_key_outputs(
    df: pd.DataFrame,
    *,
    preserve_rate_metrics: set[str] | None = None,
    rate_authoritative_components: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    preserve = set(preserve_rate_metrics or set())
    tags = [25, 50, 75]
    strikeout_pct_map = {25: 75, 50: 50, 75: 25}
    for pct in tags:
        pa_col = f"PA_proj_p{pct}"
        so_pct = int(strikeout_pct_map.get(int(pct), int(pct)))
        req_rates = {
            "bbe": f"batted_ball_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "so": f"strikeout_rate_mlb_eq_non_ar_delta_proj_p{so_pct}",
            "bb": f"walk_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "hbp": f"hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "s1b": f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "s2b": f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "s3b": f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "hr": f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "sf": f"sac_fly_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "sh": f"sac_hit_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "sba_pa": f"stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta_proj_p{pct}",
            "sba_sbo": f"stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta_proj_p{pct}",
            "sb_succ": f"stolen_base_success_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "xbh": f"xbh_from_h_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "runs": f"runs_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "rbi": f"rbi_rate_mlb_eq_non_ar_delta_proj_p{pct}",
            "babip": f"babip_recalc_rate_mlb_eq_non_ar_delta_proj_p{pct}",
        }
        if pa_col not in out.columns or not all(
            c in out.columns for c in req_rates.values()
        ):
            continue

        pa = _safe_numeric(out[pa_col]).clip(lower=0.0)
        bbe = (pa * _safe_numeric(out[req_rates["bbe"]])).clip(lower=0.0)
        so = (pa * _safe_numeric(out[req_rates["so"]])).clip(lower=0.0)
        bb = (pa * _safe_numeric(out[req_rates["bb"]])).clip(lower=0.0)
        hbp = (pa * _safe_numeric(out[req_rates["hbp"]])).clip(lower=0.0)
        hr = (bbe * _safe_numeric(out[req_rates["hr"]])).clip(lower=0.0)
        sf = (bbe * _safe_numeric(out[req_rates["sf"]])).clip(lower=0.0)
        sh = (bbe * _safe_numeric(out[req_rates["sh"]])).clip(lower=0.0)
        ab = (pa - (bb + hbp + sh + sf)).clip(lower=0.0)
        bip = (ab - so - hr + sf).clip(lower=0.0)
        babip_hits = (bip * _safe_numeric(out[req_rates["babip"]])).clip(lower=0.0)

        one_raw = (bbe * _safe_numeric(out[req_rates["s1b"]])).clip(lower=0.0)
        two_raw = (bbe * _safe_numeric(out[req_rates["s2b"]])).clip(lower=0.0)
        three_raw = (bbe * _safe_numeric(out[req_rates["s3b"]])).clip(lower=0.0)
        if bool(rate_authoritative_components):
            one, two, three = _solve_hit_mix_rate_authoritative(
                one_t=one_raw,
                two_t=two_raw,
                three_t=three_raw,
                babip_t=_safe_numeric(out[req_rates["babip"]]),
                xbh_t=_safe_numeric(out[req_rates["xbh"]]),
                bbe=bbe,
                bip=bip,
                hr=hr,
            )
            one = one.clip(lower=0.0)
            two = two.clip(lower=0.0)
            three = three.clip(lower=0.0)
        else:
            non_hr_raw = (one_raw + two_raw + three_raw).clip(lower=0.0)
            one = one_raw.copy()
            two = two_raw.copy()
            three = three_raw.copy()
            has_mix = non_hr_raw > 0
            one.loc[has_mix] = one_raw.loc[has_mix] * (
                babip_hits.loc[has_mix] / non_hr_raw.loc[has_mix]
            ).clip(lower=0.0)
            two.loc[has_mix] = two_raw.loc[has_mix] * (
                babip_hits.loc[has_mix] / non_hr_raw.loc[has_mix]
            ).clip(lower=0.0)
            three.loc[has_mix] = three_raw.loc[has_mix] * (
                babip_hits.loc[has_mix] / non_hr_raw.loc[has_mix]
            ).clip(lower=0.0)
            no_mix = ~has_mix
            one.loc[no_mix] = babip_hits.loc[no_mix]
            two.loc[no_mix] = 0.0
            three.loc[no_mix] = 0.0

            # Keep non-HR hit total pinned to BABIP target.
            non_hr_total = (one + two + three).clip(lower=0.0)
            one = (one + (babip_hits - non_hr_total)).clip(lower=0.0)
        hits = (one + two + three + hr).clip(lower=0.0)
        sbo = (one + bb + hbp + sh).clip(lower=0.0)
        sba_pa = (pa * _safe_numeric(out[req_rates["sba_pa"]])).clip(lower=0.0)
        sba_sbo = (sbo * _safe_numeric(out[req_rates["sba_sbo"]])).clip(lower=0.0)
        sba = sba_pa.copy()
        both = sba_pa.notna() & sba_sbo.notna()
        sba.loc[both] = (sba_pa.loc[both] + sba_sbo.loc[both]) / 2.0
        only_sbo = sba_pa.isna() & sba_sbo.notna()
        sba.loc[only_sbo] = sba_sbo.loc[only_sbo]
        sb = (sba * _safe_numeric(out[req_rates["sb_succ"]])).clip(lower=0.0)
        cs = (sba - sb).clip(lower=0.0)
        runs = (pa * _safe_numeric(out[req_rates["runs"]])).clip(lower=0.0)
        rbi = (pa * _safe_numeric(out[req_rates["rbi"]])).clip(lower=0.0)

        avg = _safe_divide(hits, ab)
        obp = _safe_divide(hits + bb + hbp, pa)
        slg = _safe_divide(one + (2.0 * two) + (3.0 * three) + (4.0 * hr), ab)
        iso = slg - avg
        ops = obp + slg

        out[f"BBE_proj_p{pct}"] = bbe
        out[f"SO_proj_p{pct}"] = so
        out[f"BB_proj_p{pct}"] = bb
        out[f"HBP_proj_p{pct}"] = hbp
        out[f"1B_proj_p{pct}"] = one
        out[f"2B_proj_p{pct}"] = two
        out[f"3B_proj_p{pct}"] = three
        out[f"HR_proj_p{pct}"] = hr
        out[f"SF_proj_p{pct}"] = sf
        out[f"SH_proj_p{pct}"] = sh
        out[f"AB_proj_p{pct}"] = ab
        out[f"BIP_proj_p{pct}"] = bip
        out[f"H_proj_p{pct}"] = hits
        out[f"SBO_proj_p{pct}"] = sbo
        out[f"SBA_proj_p{pct}"] = sba
        out[f"SB_proj_p{pct}"] = sb
        out[f"CS_proj_p{pct}"] = cs
        out[f"Runs_proj_p{pct}"] = runs
        out[f"RBI_proj_p{pct}"] = rbi
        if "singles_rate_bbe_mlb_eq_non_ar_delta" not in preserve:
            out[f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                one, bbe
            ).clip(lower=0.0, upper=1.0)
        if "doubles_rate_bbe_mlb_eq_non_ar_delta" not in preserve:
            out[f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                two, bbe
            ).clip(lower=0.0, upper=1.0)
        if "triples_rate_bbe_mlb_eq_non_ar_delta" not in preserve:
            out[f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                three, bbe
            ).clip(lower=0.0, upper=1.0)
        if "home_run_rate_bbe_mlb_eq_non_ar_delta" not in preserve:
            out[f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                hr, bbe
            ).clip(lower=0.0, upper=1.0)
        if "xbh_from_h_rate_mlb_eq_non_ar_delta" not in preserve:
            out[f"xbh_from_h_rate_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                two + three + hr, hits
            ).clip(lower=0.0, upper=1.0)
        if "babip_recalc_rate_mlb_eq_non_ar_delta" not in preserve:
            out[f"babip_recalc_rate_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
                one + two + three, bip
            ).clip(lower=0.0, upper=1.0)
        out[f"AVG_proj_p{pct}"] = avg
        out[f"OBP_proj_p{pct}"] = obp
        out[f"SLG_proj_p{pct}"] = slg
        out[f"ISO_proj_p{pct}"] = iso
        out[f"OPS_proj_p{pct}"] = ops
        out[f"K%_proj_p{pct}"] = _safe_divide(so, pa) * 100.0
        out[f"BB%_proj_p{pct}"] = _safe_divide(bb, pa) * 100.0

    for stat in ["OPS", "OBP", "SLG", "AVG", "ISO", "HR", "SB"]:
        c25 = f"{stat}_proj_p25"
        c75 = f"{stat}_proj_p75"
        spread = f"{stat}_proj_spread"
        if c25 in out.columns and c75 in out.columns:
            out[spread] = _safe_numeric(out[c75]) - _safe_numeric(out[c25])
    return out


def _apply_traditional_slg_top_end_calibration(
    df: pd.DataFrame,
    *,
    target_qualified_count: int = 18,
    qualified_pa_cutoff: float = 502.0,
    target_slg_floor: float = 0.505,
) -> pd.DataFrame:
    out = df.copy()
    needed_cols = {
        "PA_proj_p50",
        "SLG_proj_p50",
        "home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p50",
    }
    if not needed_cols.issubset(out.columns):
        return out

    pa50 = _safe_numeric(out["PA_proj_p50"]).fillna(0.0)
    slg50 = _safe_numeric(out["SLG_proj_p50"])
    qualified = (pa50 >= float(qualified_pa_cutoff)) & slg50.notna()
    if int(qualified.sum()) < 1:
        return out

    current_count = int((slg50[qualified] >= float(target_slg_floor)).sum())
    if current_count >= int(target_qualified_count):
        return out

    opsq = _safe_numeric(out.get("ops_quality_z", 0.0)).fillna(0.0)
    estab = (
        _safe_numeric(out.get("quality_established_factor", 0.0))
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    hr = _safe_numeric(out["home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p50"]).fillna(
        0.0
    )
    ab50 = _safe_numeric(out.get("AB_proj_p50", np.nan)).fillna(0.0).clip(lower=0.0)
    one50 = _safe_numeric(out.get("1B_proj_p50", np.nan)).fillna(0.0).clip(lower=0.0)
    lift_cap = np.minimum(_safe_divide(one50, ab50).fillna(0.0).clip(lower=0.0), 0.095)
    deficit = (float(target_slg_floor) - slg50).clip(lower=0.0)

    def _pct_rank(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 5:
            return pd.Series(0.5, index=s.index, dtype="float64")
        return s.rank(method="average", pct=True).fillna(0.5).clip(0.01, 0.99)

    gap_score = 1.0 - _pct_rank(deficit)
    elite_score = (
        (0.62 * _pct_rank(hr))
        + (0.28 * _pct_rank(opsq))
        + (0.20 * gap_score)
        + (0.15 * estab)
    )
    reachable = (deficit <= (lift_cap + 1e-9)) | (deficit <= 0.0)
    cand = out[qualified & reachable].copy()
    if len(cand) < int(target_qualified_count):
        cand = out[qualified].copy()
    cand = cand.assign(_elite=elite_score.loc[cand.index])
    cand = cand.sort_values("_elite", ascending=False)
    top_n = cand.head(int(target_qualified_count)).index
    if len(top_n) == 0:
        return out

    need50 = pd.Series(0.0, index=out.index, dtype="float64")
    need50.loc[top_n] = (
        float(target_slg_floor)
        - _safe_numeric(out.loc[top_n, "SLG_proj_p50"]).fillna(0.0)
    ).clip(lower=0.0, upper=0.080)
    # Keep some lift for already-strong profiles so best bats rise across tails too.
    extra_boost = (
        0.018
        * _safe_numeric(
            out.loc[top_n, "home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p50"]
        )
        .fillna(0.0)
        .rank(pct=True)
        * _safe_numeric(out.loc[top_n, "quality_established_factor"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    need50.loc[top_n] = (need50.loc[top_n] + extra_boost).clip(lower=0.0, upper=0.095)

    lift_by_pct = {
        25: need50 * 0.80,
        50: need50,
        75: need50 * 1.15,
    }

    for pct, slg_lift in lift_by_pct.items():
        req = [
            f"AB_proj_p{pct}",
            f"1B_proj_p{pct}",
            f"2B_proj_p{pct}",
            f"3B_proj_p{pct}",
            f"HR_proj_p{pct}",
            f"H_proj_p{pct}",
            f"BB_proj_p{pct}",
            f"HBP_proj_p{pct}",
            f"PA_proj_p{pct}",
            f"BBE_proj_p{pct}",
            f"SO_proj_p{pct}",
            f"SF_proj_p{pct}",
            f"SH_proj_p{pct}",
            f"BIP_proj_p{pct}",
        ]
        if not all(c in out.columns for c in req):
            continue
        idx = slg_lift[slg_lift > 0].index
        if len(idx) == 0:
            continue

        ab = _safe_numeric(out.loc[idx, f"AB_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        one = _safe_numeric(out.loc[idx, f"1B_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        two = _safe_numeric(out.loc[idx, f"2B_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        three = (
            _safe_numeric(out.loc[idx, f"3B_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        )
        hrv = _safe_numeric(out.loc[idx, f"HR_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        bb = _safe_numeric(out.loc[idx, f"BB_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        hbp = (
            _safe_numeric(out.loc[idx, f"HBP_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        )
        pa = _safe_numeric(out.loc[idx, f"PA_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        so = _safe_numeric(out.loc[idx, f"SO_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        sf = _safe_numeric(out.loc[idx, f"SF_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        sh = _safe_numeric(out.loc[idx, f"SH_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        bbe = (
            _safe_numeric(out.loc[idx, f"BBE_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        )
        bip = (
            _safe_numeric(out.loc[idx, f"BIP_proj_p{pct}"]).fillna(0.0).clip(lower=0.0)
        )

        if "quality_established_factor" in out.columns:
            estab_idx = (
                _safe_numeric(out.loc[idx, "quality_established_factor"])
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
        else:
            estab_idx = pd.Series(0.0, index=idx, dtype="float64")
        if "ops_quality_z" in out.columns:
            qz_idx = _safe_numeric(out.loc[idx, "ops_quality_z"]).fillna(0.0)
        else:
            qz_idx = pd.Series(0.0, index=idx, dtype="float64")

        lift = _safe_numeric(slg_lift.loc[idx]).fillna(0.0).clip(lower=0.0)
        tb_add = (lift * ab).clip(lower=0.0)

        # Allow SLG lift to flow into HR (not only 2B/3B) with a soft player-specific HR-rate ceiling.
        hr_rate_base = _safe_divide(hrv, bbe).fillna(0.0).clip(lower=0.0, upper=1.0)
        hr_pref = (
            0.20
            + (2.20 * hr_rate_base)
            + (0.06 * estab_idx)
            + (0.03 * np.clip(qz_idx, 0.0, None))
        ).clip(lower=0.10, upper=0.65)
        max_hr_rate = (0.15 + (2.40 * hr_rate_base)).clip(lower=0.12, upper=0.30)
        hr_room = ((bbe * max_hr_rate) - hrv).clip(lower=0.0)

        conv_hr = np.minimum((tb_add * hr_pref) / 3.0, np.minimum(one, hr_room))
        rem_tb = (tb_add - (3.0 * conv_hr)).clip(lower=0.0)

        one_left = (one - conv_hr).clip(lower=0.0)
        conv_2b = np.minimum(rem_tb, one_left)
        rem_tb = (rem_tb - conv_2b).clip(lower=0.0)
        conv_3b = np.minimum(rem_tb / 2.0, (one_left - conv_2b).clip(lower=0.0))

        one_new = (one - conv_hr - conv_2b - conv_3b).clip(lower=0.0)
        two_new = (two + conv_2b).clip(lower=0.0)
        three_new = (three + conv_3b).clip(lower=0.0)
        hr_new = (hrv + conv_hr).clip(lower=0.0)
        hits_new = (one_new + two_new + three_new + hr_new).clip(lower=0.0)

        avg = _safe_divide(hits_new, ab)
        obp = _safe_divide(hits_new + bb + hbp, pa)
        slg = _safe_divide(
            one_new + (2.0 * two_new) + (3.0 * three_new) + (4.0 * hr_new), ab
        )
        iso = slg - avg
        ops = obp + slg

        out.loc[idx, f"1B_proj_p{pct}"] = one_new
        out.loc[idx, f"2B_proj_p{pct}"] = two_new
        out.loc[idx, f"3B_proj_p{pct}"] = three_new
        out.loc[idx, f"HR_proj_p{pct}"] = hr_new
        out.loc[idx, f"H_proj_p{pct}"] = hits_new
        out.loc[idx, f"AVG_proj_p{pct}"] = avg
        out.loc[idx, f"OBP_proj_p{pct}"] = obp
        out.loc[idx, f"SLG_proj_p{pct}"] = slg
        out.loc[idx, f"ISO_proj_p{pct}"] = iso
        out.loc[idx, f"OPS_proj_p{pct}"] = ops

        out.loc[idx, f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = (
            _safe_divide(one_new, bbe).clip(lower=0.0, upper=1.0)
        )
        out.loc[idx, f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = (
            _safe_divide(two_new, bbe).clip(lower=0.0, upper=1.0)
        )
        out.loc[idx, f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = (
            _safe_divide(three_new, bbe).clip(lower=0.0, upper=1.0)
        )
        out.loc[idx, f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"] = (
            _safe_divide(hr_new, bbe).clip(lower=0.0, upper=1.0)
        )
        out.loc[idx, f"xbh_from_h_rate_mlb_eq_non_ar_delta_proj_p{pct}"] = _safe_divide(
            two_new + three_new + hr_new, hits_new
        ).clip(lower=0.0, upper=1.0)
        out.loc[idx, f"babip_recalc_rate_mlb_eq_non_ar_delta_proj_p{pct}"] = (
            _safe_divide(one_new + two_new + three_new, bip).clip(lower=0.0, upper=1.0)
        )

    for pct in [25, 50, 75]:
        c_iso = f"ISO_proj_p{pct}"
        c_slg = f"SLG_proj_p{pct}"
        c_avg = f"AVG_proj_p{pct}"
        if c_slg in out.columns and c_avg in out.columns:
            out[c_iso] = _safe_numeric(out[c_slg]) - _safe_numeric(out[c_avg])

    for stat in ["OPS", "OBP", "SLG", "AVG", "ISO", "HR", "SB"]:
        c25 = f"{stat}_proj_p25"
        c50 = f"{stat}_proj_p50"
        c75 = f"{stat}_proj_p75"
        if all(c in out.columns for c in [c25, c50, c75]):
            p25 = _safe_numeric(out[c25])
            p50 = _safe_numeric(out[c50])
            p75 = _safe_numeric(out[c75])
            p50 = np.maximum(p50, p25)
            p75 = np.maximum(p75, p50)
            out[c25] = p25
            out[c50] = p50
            out[c75] = p75
            out[f"{stat}_proj_spread"] = p75 - p25

    for pct in [25, 50, 75]:
        c_iso = f"ISO_proj_p{pct}"
        c_slg = f"SLG_proj_p{pct}"
        c_avg = f"AVG_proj_p{pct}"
        if c_slg in out.columns and c_avg in out.columns:
            out[c_iso] = _safe_numeric(out[c_slg]) - _safe_numeric(out[c_avg])
    if all(c in out.columns for c in ["ISO_proj_p25", "ISO_proj_p50", "ISO_proj_p75"]):
        p25 = _safe_numeric(out["ISO_proj_p25"])
        p50 = np.maximum(_safe_numeric(out["ISO_proj_p50"]), p25)
        p75 = np.maximum(_safe_numeric(out["ISO_proj_p75"]), p50)
        out["ISO_proj_p25"] = p25
        out["ISO_proj_p50"] = p50
        out["ISO_proj_p75"] = p75
        out["ISO_proj_spread"] = p75 - p25

    return out


def _build_traditional_metric_expert_composite(
    *,
    old_advanced: pd.DataFrame,
    old_naive: pd.DataFrame,
    updated_advanced: pd.DataFrame,
    updated_naive: pd.DataFrame,
    id_col: str = "mlbid",
    metric_source_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = updated_advanced.copy()
    if id_col not in out.columns:
        return out

    routing = dict(metric_source_map or COMPOSITE_METRIC_SOURCE)
    source_frames = {
        "old_advanced": old_advanced.copy(),
        "old_naive": old_naive.copy(),
        "updated_advanced": updated_advanced.copy(),
        "updated_naive": updated_naive.copy(),
    }
    for key, frame in list(source_frames.items()):
        if id_col not in frame.columns:
            source_frames[key] = pd.DataFrame(columns=[id_col])
            continue
        frame[id_col] = _safe_numeric(frame[id_col]).astype("Int64")
        frame = frame[frame[id_col].notna()].copy()
        frame[id_col] = frame[id_col].astype("int64")
        source_frames[key] = frame.drop_duplicates(subset=[id_col], keep="last")

    out[id_col] = _safe_numeric(out[id_col]).astype("Int64")
    out = out[out[id_col].notna()].copy()
    out[id_col] = out[id_col].astype("int64")

    for metric, src_key in routing.items():
        src = source_frames.get(src_key, pd.DataFrame(columns=[id_col]))
        if src.empty:
            continue
        src_map = src.set_index(id_col, drop=False)
        for suffix in [
            "proj_p20",
            "proj_p25",
            "proj_p50",
            "proj_p75",
            "proj_p80",
            "proj_spread",
        ]:
            col = f"{metric}_{suffix}"
            if col not in out.columns or col not in src_map.columns:
                continue
            repl = out[id_col].map(src_map[col])
            mask = repl.notna()
            if bool(mask.any()):
                out.loc[mask, col] = repl.loc[mask]

        lo, hi = _bounds_for_metric(metric, mode="traditional")
        prev = None
        for pct in [25, 50, 75]:
            col = f"{metric}_proj_p{pct}"
            if col not in out.columns:
                continue
            vals = _safe_numeric(out[col])
            if lo is not None:
                vals = vals.clip(lower=float(lo))
            if hi is not None:
                vals = vals.clip(upper=float(hi))
            if prev is not None:
                vals = np.maximum(vals, prev)
            out[col] = vals
            prev = vals
        c20 = f"{metric}_proj_p20"
        c25 = f"{metric}_proj_p25"
        c80 = f"{metric}_proj_p80"
        c75 = f"{metric}_proj_p75"
        if c20 in out.columns and c25 in out.columns:
            out[c20] = np.minimum(_safe_numeric(out[c20]), _safe_numeric(out[c25]))
        if c80 in out.columns and c75 in out.columns:
            out[c80] = np.maximum(_safe_numeric(out[c80]), _safe_numeric(out[c75]))
        spread = f"{metric}_proj_spread"
        if c25 in out.columns and c75 in out.columns:
            out[spread] = _safe_numeric(out[c75]) - _safe_numeric(out[c25])

    out = _recompute_traditional_key_outputs(
        out,
        preserve_rate_metrics=set(routing.keys()),
        rate_authoritative_components=True,
    )
    return out


def _add_kpi_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    suffixes = [
        "source",
        "n_eff",
        "proj_p20",
        "proj_p25",
        "proj_p50",
        "proj_p75",
        "proj_p80",
        "proj_p600",
        "proj_spread",
    ]
    for src_base, alias_base in KPI_ALIAS_MAP.items():
        for suffix in suffixes:
            if suffix in {"source", "n_eff"}:
                src_col = f"{suffix}_{src_base}"
                alias_col = f"{suffix}_{alias_base}"
            else:
                src_col = f"{src_base}_{suffix}"
                alias_col = f"{alias_base}_{suffix}"
            if src_col in out.columns and alias_col not in out.columns:
                out[alias_col] = out[src_col]
    return out


def _spearman_corr(pred: pd.Series, actual: pd.Series) -> float:
    p = _safe_numeric(pred)
    a = _safe_numeric(actual)
    mask = p.notna() & a.notna() & np.isfinite(p) & np.isfinite(a)
    if int(mask.sum()) < 2:
        return np.nan
    ps = p[mask]
    ac = a[mask]
    if ps.nunique(dropna=True) < 2 or ac.nunique(dropna=True) < 2:
        return np.nan
    return float(ps.corr(ac, method="spearman"))


def _is_true_flag(values: pd.Series) -> pd.Series:
    txt = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return txt.isin({"T", "TRUE", "1", "Y", "YES"})


def _two_stage_reference_season(frame: pd.DataFrame) -> pd.Series:
    source_raw = (
        frame["source_season"]
        if "source_season" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    target_raw = (
        frame["target_season"]
        if "target_season" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    source = _safe_numeric(source_raw)
    target = _safe_numeric(target_raw)
    ref = source.copy()
    fallback = target - 1.0
    ref = ref.where(ref.notna(), fallback)
    return ref


def _build_two_stage_reference_params(
    panel: pd.DataFrame,
    *,
    target_metrics: list[str],
    zspace_mlb_pa_min: float,
) -> dict[str, dict[str, float]]:
    if panel.empty:
        return {}
    ref_season = _safe_numeric(_two_stage_reference_season(panel)).astype("Int64")
    pa_min = float(max(zspace_mlb_pa_min, 0.0))
    if "mlb_pa" in panel.columns:
        pa_vals = _safe_numeric(panel["mlb_pa"]).fillna(0.0)
        ref_mask = pa_vals >= pa_min
    elif "appeared_in_MLB" in panel.columns:
        # Fallback for older panels without explicit mlb_pa.
        ref_mask = _is_true_flag(panel["appeared_in_MLB"])
    else:
        return {}
    out: dict[str, dict[str, float]] = {}
    for metric in target_metrics:
        actual_col = f"{metric}_actual"
        if actual_col not in panel.columns:
            continue
        season_rows = ref_mask & ref_season.notna()
        by_season: dict[int, dict[str, float]] = {}
        for season in sorted(ref_season.loc[season_rows].dropna().astype(int).unique().tolist()):
            season_mask = season_rows & ref_season.eq(int(season))
            vals = _safe_numeric(panel.loc[season_mask, actual_col])
            vals = vals[vals.notna() & np.isfinite(vals)]
            n = int(len(vals))
            if n == 0:
                continue
            mu = float(np.nanmedian(vals))
            sigma = float(np.nanstd(vals, ddof=0))
            if not np.isfinite(sigma) or sigma <= 1e-12:
                q75 = float(np.nanpercentile(vals, 75))
                q25 = float(np.nanpercentile(vals, 25))
                sigma = float(max((q75 - q25) / 1.349, 1e-6))
            by_season[int(season)] = {
                "mu": float(mu),
                "sigma": float(sigma),
                "n_ref": int(n),
            }
        vals_all = _safe_numeric(panel.loc[ref_mask, actual_col])
        vals_all = vals_all[vals_all.notna() & np.isfinite(vals_all)]
        if len(vals_all) == 0 and not by_season:
            continue
        mu = float(np.nanmedian(vals_all)) if len(vals_all) else np.nan
        sigma = float(np.nanstd(vals_all, ddof=0)) if len(vals_all) else np.nan
        if not np.isfinite(sigma) or sigma <= 1e-12:
            if len(vals_all):
                q75 = float(np.nanpercentile(vals_all, 75))
                q25 = float(np.nanpercentile(vals_all, 25))
                sigma = float(max((q75 - q25) / 1.349, 1e-6))
            elif by_season:
                latest = max(by_season.keys())
                sigma = float(by_season[latest]["sigma"])
                mu = float(by_season[latest]["mu"])
        out[metric] = {
            "by_season": by_season,
            "fallback": {
                "mu": float(mu) if np.isfinite(mu) else np.nan,
                "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
                "n_ref": int(len(vals_all)),
            },
            "mlb_pa_min_ref": float(pa_min),
        }
    return out


def _resolve_two_stage_reference_for_rows(
    reference: dict[str, Any],
    ref_season: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = ref_season.index
    mu_series = pd.Series(np.nan, index=idx, dtype="float64")
    sigma_series = pd.Series(np.nan, index=idx, dtype="float64")
    n_series = pd.Series(np.nan, index=idx, dtype="float64")

    by_raw = reference.get("by_season", {})
    by_season: dict[int, dict[str, float]] = {}
    if isinstance(by_raw, dict):
        for k, v in by_raw.items():
            try:
                s = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                by_season[s] = v
    season_keys = sorted(by_season.keys())

    ref_vals = _safe_numeric(ref_season).astype("Int64")
    valid_ref = ref_vals.notna()
    if season_keys:
        for s in sorted(ref_vals.loc[valid_ref].dropna().astype(int).unique().tolist()):
            use_keys = [k for k in season_keys if k <= int(s)]
            chosen = max(use_keys) if use_keys else max(season_keys)
            rec = by_season.get(chosen, {})
            mask = ref_vals.eq(int(s))
            mu_series.loc[mask] = float(pd.to_numeric(rec.get("mu"), errors="coerce"))
            sigma_series.loc[mask] = float(pd.to_numeric(rec.get("sigma"), errors="coerce"))
            n_series.loc[mask] = float(pd.to_numeric(rec.get("n_ref"), errors="coerce"))

    fb = reference.get("fallback", {}) if isinstance(reference, dict) else {}
    fb_mu = float(pd.to_numeric(fb.get("mu"), errors="coerce"))
    fb_sigma = float(pd.to_numeric(fb.get("sigma"), errors="coerce"))
    fb_n = float(pd.to_numeric(fb.get("n_ref"), errors="coerce"))
    mu_series = mu_series.where(mu_series.notna(), fb_mu)
    sigma_series = sigma_series.where(sigma_series.notna(), fb_sigma)
    n_series = n_series.where(n_series.notna(), fb_n)
    return mu_series, sigma_series, n_series


def _two_stage_skill_feature_cols_from_frame(kpi: pd.DataFrame) -> list[str]:
    cols = set(kpi.columns.tolist())
    feats: list[str] = []
    for metric in KPI_SKILL_METRICS:
        if metric in TWO_STAGE_EXCLUDED_KPI_SKILLS:
            continue
        c = f"{metric}_proj_p50"
        if c in cols:
            feats.append(c)
    return feats


def _prepare_kpi_two_stage_keys(
    kpi: pd.DataFrame,
    *,
    extra_cols: list[str],
) -> pd.DataFrame:
    if kpi.empty:
        return pd.DataFrame(columns=["mlbid", "level_id_source", "target_season", *extra_cols])

    id_col = "batter_mlbid" if "batter_mlbid" in kpi.columns else "mlbid"
    need = [id_col, "level_id_source", "target_season", *[c for c in extra_cols if c in kpi.columns]]
    work = kpi[need].copy()
    work["mlbid"] = _safe_numeric(work[id_col]).astype("Int64")
    work["level_id_source"] = _safe_numeric(work["level_id_source"]).astype("Int64")
    work["target_season"] = _safe_numeric(work["target_season"]).astype("Int64")
    work = work[
        work["mlbid"].notna() & work["level_id_source"].notna() & work["target_season"].notna()
    ].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work["level_id_source"] = work["level_id_source"].astype("int64")
    work["target_season"] = work["target_season"].astype("int64")
    if id_col != "mlbid":
        work = work.drop(columns=[id_col], errors="ignore")
    return work.drop_duplicates(
        subset=["mlbid", "level_id_source", "target_season"], keep="last"
    )


def _attach_traditional_level_from_kpi(
    traditional: pd.DataFrame,
    kpi: pd.DataFrame,
) -> pd.DataFrame:
    out = traditional.copy()
    if out.empty:
        if "level_id_source" not in out.columns:
            out["level_id_source"] = pd.Series(dtype="Int64")
        return out
    if "mlbid" not in out.columns or "target_season" not in out.columns:
        if "level_id_source" not in out.columns:
            out["level_id_source"] = pd.Series(np.nan, index=out.index, dtype="Float64")
        return out

    key_map = _prepare_kpi_two_stage_keys(kpi, extra_cols=[])
    key_map = key_map[["mlbid", "target_season", "level_id_source"]].drop_duplicates(
        subset=["mlbid", "target_season"], keep="last"
    )
    out["mlbid"] = _safe_numeric(out["mlbid"]).astype("Int64")
    out["target_season"] = _safe_numeric(out["target_season"]).astype("Int64")
    out = out.merge(
        key_map.rename(columns={"level_id_source": "__level_id_source_kpi"}),
        on=["mlbid", "target_season"],
        how="left",
    )
    if "level_id_source" in out.columns:
        out["level_id_source"] = _safe_numeric(out["level_id_source"]).astype("Float64")
        out["level_id_source"] = out["level_id_source"].fillna(
            _safe_numeric(out["__level_id_source_kpi"]).astype("Float64")
        )
    else:
        out["level_id_source"] = _safe_numeric(out["__level_id_source_kpi"]).astype("Float64")
    out = out.drop(columns=["__level_id_source_kpi"], errors="ignore")
    return out


def _eligible_two_stage_targets(
    seasons: list[int],
    *,
    train_start_season: int,
    train_end_season: int,
    min_prior_seasons: int = 3,
) -> list[int]:
    season_set = {int(s) for s in seasons}
    out: list[int] = []
    for target in sorted(season_set):
        if int(target) < int(train_start_season) or int(target) > int(train_end_season):
            continue
        source = int(target - 1)
        required = {source - lag for lag in range(int(min_prior_seasons))}
        if source in season_set and required.issubset(season_set):
            out.append(int(target))
    return out


def _load_two_stage_actual_rates(
    historical_rates_path: Path,
    *,
    target_metrics: list[str],
) -> pd.DataFrame:
    raw = pd.read_parquet(historical_rates_path).copy()
    if "mlbid" not in raw.columns or "season" not in raw.columns:
        return pd.DataFrame(columns=["mlbid", "target_season", *[f"{m}_actual" for m in target_metrics]])
    available_metrics = [m for m in target_metrics if m in raw.columns]
    if "PA" in target_metrics and "plate_appearances_agg" in raw.columns:
        available_metrics.append("plate_appearances_agg")
    out_cols = ["mlbid", "season", *available_metrics]
    out_cols = list(dict.fromkeys(out_cols))
    work = raw[out_cols].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"]).astype("Int64")
    work["season"] = _safe_numeric(work["season"]).astype("Int64")
    work = work[work["mlbid"].notna() & work["season"].notna()].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work["target_season"] = work["season"].astype("int64")
    work = work.drop(columns=["season"])
    if "plate_appearances_agg" in work.columns and "PA" in target_metrics:
        work = work.rename(columns={"plate_appearances_agg": "PA"})
    for metric in target_metrics:
        if metric in work.columns:
            work = work.rename(columns={metric: f"{metric}_actual"})
    return work


def _build_kpi_skill_rolling_projections(
    *,
    config_path: Path,
    train_start_season: int,
    train_end_season: int,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = load_config(config_path)
    if "hitters" not in cfg.datasets:
        return pd.DataFrame(), []
    ds = cfg.datasets["hitters"]
    reg_df = read_parquet(ds.regressed_path)
    base_df = read_parquet(ds.base_path)
    available_cols = sorted(set(reg_df.columns).union(base_df.columns))
    metric_cols, k_overrides, bounds = resolve_metric_settings(
        dataset_name="hitters",
        regressed_cols=reg_df.columns.tolist(),
        available_cols=available_cols,
        cfg=cfg,
    )
    skill_metrics = [
        m
        for m in KPI_SKILL_METRICS
        if m not in TWO_STAGE_EXCLUDED_KPI_SKILLS and m in metric_cols and m in available_cols
    ]
    if not skill_metrics:
        return pd.DataFrame(), []

    merged = merge_base_and_regressed(base_df=base_df, reg_df=reg_df, cfg=ds)
    eq_metrics = [m for m in skill_metrics if m.endswith("_reg")]
    if eq_metrics:
        merged = apply_simple_mlb_equivalency(
            merged,
            metric_cols=eq_metrics,
            season_col=ds.season_col,
            level_col=ds.level_col,
            mlb_level_id=cfg.global_cfg.mlb_level_id,
        )

    player_season = build_player_season_table(merged_df=merged, cfg=ds, metric_cols=skill_metrics)
    if player_season.empty:
        return pd.DataFrame(), []
    level_for_age = "level_id_source" if "level_id_source" in player_season.columns else ds.level_col
    player_season = infer_and_impute_age(
        player_season,
        player_col=ds.id_col,
        season_col=ds.season_col,
        level_col=level_for_age,
        age_col=ds.age_col,
        group_cols=[level_for_age],
    )

    seasons = (
        _safe_numeric(player_season[ds.season_col])
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    targets = _eligible_two_stage_targets(
        seasons=seasons,
        train_start_season=int(train_start_season),
        train_end_season=int(train_end_season),
        min_prior_seasons=3,
    )
    if not targets:
        return pd.DataFrame(), []

    skill_k = {k: v for k, v in k_overrides.items() if k in set(skill_metrics)}
    skill_bounds = {k: v for k, v in bounds.items() if k in set(skill_metrics)}
    rows: list[pd.DataFrame] = []
    for target in targets:
        source = int(target - 1)
        proj = project_next_season(
            player_season_df=player_season,
            metric_cols=skill_metrics,
            id_col=ds.id_col,
            name_col=ds.name_col,
            season_col=ds.season_col,
            level_col=level_for_age,
            age_col="age_used",
            age_source_col="age_source",
            exposure_col=ds.exposure_col,
            global_cfg=cfg.global_cfg,
            k_overrides=skill_k,
            bounds=skill_bounds,
            source_season=source,
            passthrough_cols=[],
        )
        if proj.empty:
            continue
        keep = [
            c
            for c in [
                ds.id_col,
                "level_id_source",
                "source_season",
                "target_season",
                *[f"{m}_proj_p50" for m in skill_metrics],
            ]
            if c in proj.columns
        ]
        rows.append(proj[keep].copy())
    if not rows:
        return pd.DataFrame(), []

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={ds.id_col: "batter_mlbid"})
    out["batter_mlbid"] = _safe_numeric(out["batter_mlbid"]).astype("Int64")
    out["level_id_source"] = _safe_numeric(out["level_id_source"]).astype("Int64")
    out["target_season"] = _safe_numeric(out["target_season"]).astype("Int64")
    out["source_season"] = _safe_numeric(out["source_season"]).astype("Int64")
    out = out[
        out["batter_mlbid"].notna()
        & out["level_id_source"].notna()
        & out["target_season"].notna()
    ].copy()
    out["batter_mlbid"] = out["batter_mlbid"].astype("int64")
    out["level_id_source"] = out["level_id_source"].astype("int64")
    out["target_season"] = out["target_season"].astype("int64")
    if "source_season" in out.columns:
        out["source_season"] = out["source_season"].astype("int64")
    out = out.drop_duplicates(
        subset=["batter_mlbid", "level_id_source", "target_season"],
        keep="last",
    )
    feature_cols = _two_stage_skill_feature_cols_from_frame(out)
    return out, feature_cols


def _build_traditional_stage1_rolling_projections(
    *,
    historical_rates_path: Path,
    constants_path: Path,
    bp_hitting_path: Path,
    historical_ar_path: Path,
    metric_recency_weights_json: Path | None,
    z_coherence_mode: str,
    z_anchor_k: float,
    hr_anchor_k: float,
    z_tail_strength: float,
    coherence_mode: str,
    uncertainty_draws: int,
    seed: int,
    default_k: float,
    k_scale: float,
    stability_pa_per_season: float,
    spread_boost_max: float,
    train_start_season: int,
    train_end_season: int,
) -> pd.DataFrame:
    from build_bp_rate_projections_2026_non_ar_post_inv_coh import (
        _load_metric_recency_weights,
        build_bp_rate_projections_2026,
    )

    raw_all = pd.read_parquet(historical_rates_path).copy()
    if "mlbid" not in raw_all.columns or "season" not in raw_all.columns:
        return pd.DataFrame()
    raw_all["mlbid"] = _safe_numeric(raw_all["mlbid"]).astype("Int64")
    raw_all["season"] = _safe_numeric(raw_all["season"]).astype("Int64")
    raw_all = raw_all[raw_all["mlbid"].notna() & raw_all["season"].notna()].copy()
    raw_all["mlbid"] = raw_all["mlbid"].astype("int64")
    raw_all["season"] = raw_all["season"].astype("int64")
    seasons = sorted(raw_all["season"].unique().tolist())
    targets = _eligible_two_stage_targets(
        seasons=seasons,
        train_start_season=int(train_start_season),
        train_end_season=int(train_end_season),
        min_prior_seasons=3,
    )
    if not targets:
        return pd.DataFrame()

    mlb_lookup = _load_mlb_pa_lookup(bp_hitting_path)
    mlb_ops_lookup = _load_mlb_ops_lookup(bp_hitting_path)
    metric_recency_weights = _load_metric_recency_weights(metric_recency_weights_json)
    hist_ar_all = pd.read_parquet(historical_ar_path).copy()
    if "season" in hist_ar_all.columns:
        hist_ar_all["season"] = _safe_numeric(hist_ar_all["season"]).astype("Int64")

    out_rows: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for target in targets:
            source = int(target - 1)
            hist_subset = raw_all[raw_all["season"] <= source].copy()
            if hist_subset.empty:
                continue
            input_tmp = tmp / f"bp_input_through_{source}.parquet"
            hist_subset.to_parquet(input_tmp, index=False)

            hist_ar_tmp = tmp / f"bp_ar_through_{source}.parquet"
            if "season" in hist_ar_all.columns:
                ar_subset = hist_ar_all[_safe_numeric(hist_ar_all["season"]) <= source].copy()
            else:
                ar_subset = hist_ar_all.copy()
            ar_subset.to_parquet(hist_ar_tmp, index=False)

            proj_tmp = tmp / f"bp_proj_for_{target}.parquet"
            age_tmp = tmp / f"bp_age_for_{target}.parquet"
            proj, _ = build_bp_rate_projections_2026(
                input_path=input_tmp,
                constants_path=constants_path,
                bp_hitting_path=bp_hitting_path,
                historical_ar_path=hist_ar_tmp,
                z_coherence_mode=z_coherence_mode,
                z_anchor_k=float(z_anchor_k),
                hr_anchor_k=float(hr_anchor_k),
                z_tail_strength=float(z_tail_strength),
                coherence_mode=coherence_mode,
                out_projection_path=proj_tmp,
                out_age_curve_path=age_tmp,
                uncertainty_draws=int(uncertainty_draws),
                seed=int(seed),
                metric_recency_weights=metric_recency_weights,
                default_k=float(default_k),
                k_scale=float(k_scale),
            )

            metric_bases = _collect_metric_bases(proj)
            proj = _coverage_from_mlb_pa(
                proj,
                id_col="mlbid",
                source_season_col="source_season",
                mlb_lookup=mlb_lookup,
                stability_pa_per_season=float(stability_pa_per_season),
                seasons_back=3,
                spread_boost_max=float(spread_boost_max),
            )
            proj = _ensure_p25_p75(proj, metric_bases=metric_bases)
            proj = _apply_mlb_pa_spread_injection(
                proj,
                metric_bases=metric_bases,
                mode="traditional",
            )
            proj = _apply_traditional_predictive_adjustments(
                proj,
                historical_rates_path=input_tmp,
                bp_hitting_path=bp_hitting_path,
                mlb_lookup=mlb_lookup,
                mlb_ops_lookup=mlb_ops_lookup,
            )
            proj = _recompute_traditional_key_outputs(proj)
            proj = _apply_traditional_slg_top_end_calibration(proj)

            keep = [
                c
                for c in [
                    "mlbid",
                    "source_season",
                    "target_season",
                    *[f"{m}_proj_p50" for m in TWO_STAGE_TARGET_METRICS],
                ]
                if c in proj.columns
            ]
            out_rows.append(proj[keep].copy())

    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows, ignore_index=True)
    out["mlbid"] = _safe_numeric(out["mlbid"]).astype("Int64")
    out["target_season"] = _safe_numeric(out["target_season"]).astype("Int64")
    out = out[out["mlbid"].notna() & out["target_season"].notna()].copy()
    out["mlbid"] = out["mlbid"].astype("int64")
    out["target_season"] = out["target_season"].astype("int64")
    return out.drop_duplicates(subset=["mlbid", "target_season"], keep="last")


def _build_two_stage_training_panel(
    *,
    kpi_rolling: pd.DataFrame,
    traditional_rolling: pd.DataFrame,
    actual_rates: pd.DataFrame,
    mlb_lookup: pd.DataFrame,
    feature_cols: list[str],
    target_metrics: list[str],
) -> pd.DataFrame:
    if kpi_rolling.empty or traditional_rolling.empty or actual_rates.empty:
        return pd.DataFrame()
    trad = _attach_traditional_level_from_kpi(traditional_rolling, kpi_rolling)
    kpi_keys = _prepare_kpi_two_stage_keys(kpi_rolling, extra_cols=feature_cols)
    panel = trad.merge(
        kpi_keys,
        on=["mlbid", "level_id_source", "target_season"],
        how="left",
    )
    panel = panel.merge(actual_rates, on=["mlbid", "target_season"], how="left")
    if not mlb_lookup.empty and {"mlbid", "season", "mlb_pa"}.issubset(mlb_lookup.columns):
        mlb_key = mlb_lookup[["mlbid", "season", "mlb_pa"]].copy()
        mlb_key["mlbid"] = _safe_numeric(mlb_key["mlbid"]).astype("Int64")
        mlb_key["season"] = _safe_numeric(mlb_key["season"]).astype("Int64")
        mlb_key["mlb_pa"] = _safe_numeric(mlb_key["mlb_pa"])
        mlb_key = mlb_key[mlb_key["mlbid"].notna() & mlb_key["season"].notna()].copy()
        mlb_key["mlbid"] = mlb_key["mlbid"].astype("int64")
        mlb_key["season"] = mlb_key["season"].astype("int64")
        mlb_key = (
            mlb_key.groupby(["mlbid", "season"], dropna=False)["mlb_pa"]
            .sum()
            .rename("mlb_pa")
            .reset_index()
        )
        panel = panel.merge(
            mlb_key.rename(columns={"season": "target_season"}),
            on=["mlbid", "target_season"],
            how="left",
        )
        panel["appeared_in_MLB"] = np.where(
            _safe_numeric(panel["mlb_pa"]).fillna(0.0) > 0.0,
            "T",
            "F",
        )
    for metric in target_metrics:
        c50 = f"{metric}_proj_p50"
        if c50 not in panel.columns:
            continue
        panel[f"{metric}_proj_p50_stage1"] = _safe_numeric(panel[c50])
    return panel


def _fit_two_stage_models(
    panel: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_metrics: list[str],
    cap_sd_mult: float,
    two_stage_mode: str = "raw",
    reference_params: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, float], pd.DataFrame]:
    if panel.empty or not feature_cols:
        return {}, {}, pd.DataFrame()
    mode = str(two_stage_mode).strip().lower()
    if mode not in TWO_STAGE_MODES:
        mode = "raw"
    ref_params = reference_params or {}

    models: dict[str, Pipeline] = {}
    caps_by_metric: dict[str, float] = {}
    diag_rows: list[dict[str, Any]] = []

    seasons = _safe_numeric(panel.get("target_season", np.nan)).astype("Int64")
    ref_season = _safe_numeric(_two_stage_reference_season(panel)).astype("Int64")
    level = _safe_numeric(panel.get("level_id_source", np.nan))
    mlb_mask = level.eq(1).fillna(False)
    X_all = panel[feature_cols].copy()

    for metric in target_metrics:
        base_col = f"{metric}_proj_p50_stage1"
        actual_col = f"{metric}_actual"
        if base_col not in panel.columns or actual_col not in panel.columns:
            continue

        base = _safe_numeric(panel[base_col])
        actual = _safe_numeric(panel[actual_col])
        ref_mu = np.nan
        ref_sigma = np.nan
        ref_n = np.nan
        ref_latest_season = np.nan
        ref_mlb_pa_min = np.nan
        base_model = base
        actual_model = actual
        if mode == "zspace":
            ref = ref_params.get(metric, {})
            mu_rows, sigma_rows, n_rows = _resolve_two_stage_reference_for_rows(
                ref, ref_season
            )
            valid_ref = (
                mu_rows.notna()
                & sigma_rows.notna()
                & np.isfinite(mu_rows)
                & np.isfinite(sigma_rows)
                & (sigma_rows > 0)
            )
            if not bool(valid_ref.any()):
                continue
            base_model = (base - mu_rows) / sigma_rows
            actual_model = (actual - mu_rows) / sigma_rows
            by_season = ref.get("by_season", {}) if isinstance(ref, dict) else {}
            if isinstance(by_season, dict) and by_season:
                try:
                    latest_key = max(int(k) for k in by_season.keys())
                    latest = by_season.get(latest_key, {})
                    ref_latest_season = float(latest_key)
                    ref_mu = float(pd.to_numeric(latest.get("mu"), errors="coerce"))
                    ref_sigma = float(pd.to_numeric(latest.get("sigma"), errors="coerce"))
                    ref_n = float(pd.to_numeric(latest.get("n_ref"), errors="coerce"))
                except Exception:
                    pass
            if not np.isfinite(ref_mu) or not np.isfinite(ref_sigma) or ref_sigma <= 0:
                fb = ref.get("fallback", {}) if isinstance(ref, dict) else {}
                ref_mu = float(pd.to_numeric(fb.get("mu"), errors="coerce"))
                ref_sigma = float(pd.to_numeric(fb.get("sigma"), errors="coerce"))
                ref_n = float(pd.to_numeric(fb.get("n_ref"), errors="coerce"))
            ref_mlb_pa_min = float(
                pd.to_numeric(ref.get("mlb_pa_min_ref"), errors="coerce")
            )
        y = actual_model - base_model
        valid = (
            mlb_mask
            & base_model.notna()
            & actual_model.notna()
            & np.isfinite(base_model)
            & np.isfinite(actual_model)
        )
        if int(valid.sum()) < 60:
            continue

        season_vals = seasons[valid & seasons.notna()].astype(int)
        unique_seasons = sorted(season_vals.unique().tolist())
        if len(unique_seasons) < 2:
            continue

        best_alpha = float(TWO_STAGE_ALPHA_GRID[0])
        best_score = -np.inf
        for alpha in TWO_STAGE_ALPHA_GRID:
            fold_scores: list[float] = []
            for fold_season in unique_seasons[1:]:
                tr_mask = valid & seasons.notna() & (seasons.astype(int) < int(fold_season))
                va_mask = valid & seasons.notna() & (seasons.astype(int) == int(fold_season))
                if int(tr_mask.sum()) < 40 or int(va_mask.sum()) < 20:
                    continue
                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=float(alpha))),
                    ]
                )
                model.fit(X_all.loc[tr_mask], y.loc[tr_mask])
                pred_resid = pd.Series(model.predict(X_all.loc[va_mask]), index=panel.index[va_mask], dtype="float64")
                resid_sd = float(np.nanstd(_safe_numeric(y.loc[tr_mask]), ddof=0))
                cap = float(max(float(cap_sd_mult) * max(resid_sd, 0.0), 1e-12))
                pred_resid = pred_resid.clip(lower=-cap, upper=cap)
                pred_corrected = base_model.loc[va_mask] + pred_resid
                fold_score = _spearman_corr(pred_corrected, actual_model.loc[va_mask])
                if np.isfinite(fold_score):
                    fold_scores.append(float(fold_score))
            alpha_score = float(np.nanmean(fold_scores)) if fold_scores else np.nan
            if np.isfinite(alpha_score) and (
                alpha_score > best_score
                or (alpha_score == best_score and float(alpha) < best_alpha)
            ):
                best_score = alpha_score
                best_alpha = float(alpha)

        final_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(best_alpha))),
            ]
        )
        final_model.fit(X_all.loc[valid], y.loc[valid])
        final_sd = float(np.nanstd(_safe_numeric(y.loc[valid]), ddof=0))
        final_cap = float(max(float(cap_sd_mult) * max(final_sd, 0.0), 1e-12))
        models[metric] = final_model
        caps_by_metric[metric] = final_cap
        diag_rows.append(
            {
                "metric": metric,
                "two_stage_mode": mode,
                "best_alpha": float(best_alpha),
                "cv_spearman": float(best_score) if np.isfinite(best_score) else np.nan,
                "train_rows_mlb": int(valid.sum()),
                "train_seasons_mlb": int(len(unique_seasons)),
                "train_resid_sd": float(final_sd),
                "resid_cap": float(final_cap),
                "feature_count": int(len(feature_cols)),
                "z_ref_mu_median_mlb_pa_ge_min_latest": (
                    float(ref_mu) if np.isfinite(ref_mu) else np.nan
                ),
                "z_ref_sigma_mlb_pa_ge_min_latest": (
                    float(ref_sigma) if np.isfinite(ref_sigma) else np.nan
                ),
                "z_ref_n_mlb_pa_ge_min_latest": (
                    float(ref_n) if np.isfinite(ref_n) else np.nan
                ),
                "z_ref_latest_season": (
                    float(ref_latest_season) if np.isfinite(ref_latest_season) else np.nan
                ),
                "z_ref_mlb_pa_min": (
                    float(ref_mlb_pa_min) if np.isfinite(ref_mlb_pa_min) else np.nan
                ),
            }
        )

    diag = pd.DataFrame(diag_rows)
    return models, caps_by_metric, diag


def _apply_two_stage_models_to_traditional(
    *,
    traditional: pd.DataFrame,
    kpi: pd.DataFrame,
    feature_cols: list[str],
    target_metrics: list[str],
    models: dict[str, Pipeline],
    caps_by_metric: dict[str, float],
    emit_p25p75: bool,
    train_window: str,
    two_stage_mode: str = "raw",
    reference_params: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    mode = str(two_stage_mode).strip().lower()
    if mode not in TWO_STAGE_MODES:
        mode = "raw"
    ref_params = reference_params or {}
    out = _attach_traditional_level_from_kpi(traditional, kpi)
    kpi_keyed = _prepare_kpi_two_stage_keys(kpi, extra_cols=feature_cols)
    out = out.merge(
        kpi_keyed,
        on=["mlbid", "level_id_source", "target_season"],
        how="left",
        suffixes=("", "_kpi"),
    )

    for c in feature_cols:
        if c in out.columns:
            out[c] = _safe_numeric(out[c])

    ref_season = _safe_numeric(_two_stage_reference_season(out)).astype("Int64")
    applied_any = np.zeros(len(out), dtype=bool)
    for metric in target_metrics:
        if metric not in models:
            continue
        c50 = f"{metric}_proj_p50"
        if c50 not in out.columns:
            continue
        base_raw = _safe_numeric(out[c50])
        out[f"{metric}_proj_p50_stage1"] = base_raw
        pred_col = f"two_stage_resid_pred_{metric}"
        app_col = f"two_stage_resid_applied_{metric}"
        out[pred_col] = np.nan
        out[app_col] = 0.0

        if not feature_cols:
            continue
        feat = out[feature_cols].copy()
        feat_num = feat.apply(_safe_numeric)
        has_signal = feat_num.notna().any(axis=1)
        base_model = base_raw.copy()
        ref_mu_rows = pd.Series(np.nan, index=out.index, dtype="float64")
        ref_sigma_rows = pd.Series(np.nan, index=out.index, dtype="float64")
        if mode == "zspace":
            ref = ref_params.get(metric, {})
            ref_mu_rows, ref_sigma_rows, _ = _resolve_two_stage_reference_for_rows(
                ref,
                ref_season,
            )
            valid_ref = (
                ref_mu_rows.notna()
                & ref_sigma_rows.notna()
                & np.isfinite(ref_mu_rows)
                & np.isfinite(ref_sigma_rows)
                & (ref_sigma_rows > 0)
            )
            if not bool(valid_ref.any()):
                continue
            base_model = (base_raw - ref_mu_rows) / ref_sigma_rows
        mask = base_model.notna() & has_signal
        if not bool(mask.any()):
            continue

        pred_resid_model = pd.Series(
            models[metric].predict(feat_num.loc[mask]),
            index=out.index[mask],
            dtype="float64",
        )
        pred_resid_raw = (
            pred_resid_model * ref_sigma_rows.loc[mask]
            if mode == "zspace"
            else pred_resid_model
        )
        out.loc[mask, pred_col] = pred_resid_raw
        cap = float(caps_by_metric.get(metric, np.inf))
        if np.isfinite(cap):
            pred_applied_model = pred_resid_model.clip(lower=-abs(cap), upper=abs(cap))
        else:
            pred_applied_model = pred_resid_model
        pred_applied_raw = (
            pred_applied_model * ref_sigma_rows.loc[mask]
            if mode == "zspace"
            else pred_applied_model
        )
        out.loc[mask, app_col] = pred_applied_raw

        if mode == "zspace":
            new_model = base_model.loc[mask] + pred_applied_model
            new_p50 = ref_mu_rows.loc[mask] + (new_model * ref_sigma_rows.loc[mask])
        else:
            new_p50 = base_raw.loc[mask] + pred_applied_model
        lo, hi = _bounds_for_metric(metric, mode="traditional")
        if lo is not None:
            new_p50 = new_p50.clip(lower=float(lo))
        if hi is not None:
            new_p50 = new_p50.clip(upper=float(hi))
        out.loc[mask, c50] = new_p50
        applied_any[mask.to_numpy()] = True

        if bool(emit_p25p75):
            c25 = f"{metric}_proj_p25"
            c75 = f"{metric}_proj_p75"
            if c25 in out.columns and c75 in out.columns:
                old50 = _safe_numeric(out[f"{metric}_proj_p50_stage1"])
                old25 = _safe_numeric(out[c25])
                old75 = _safe_numeric(out[c75])
                new50_all = _safe_numeric(out[c50])
                dn = (old50 - old25).clip(lower=0.0)
                up = (old75 - old50).clip(lower=0.0)
                new25 = new50_all - dn
                new75 = new50_all + up
                if lo is not None:
                    new25 = new25.clip(lower=float(lo))
                    new75 = new75.clip(lower=float(lo))
                if hi is not None:
                    new25 = new25.clip(upper=float(hi))
                    new75 = new75.clip(upper=float(hi))
                new25 = np.minimum(new25, new50_all)
                new75 = np.maximum(new75, new50_all)
                out[c25] = new25
                out[c75] = new75
                spread_col = f"{metric}_proj_spread"
                if spread_col in out.columns:
                    out[spread_col] = _safe_numeric(out[c75]) - _safe_numeric(out[c25])

    out["two_stage_applied"] = applied_any.astype(int)
    out["two_stage_model_version"] = f"two_stage_kpi_skills_v1_{mode}"
    out["two_stage_train_window"] = str(train_window)
    out = _recompute_traditional_key_outputs(out)
    out = _apply_traditional_slg_top_end_calibration(out)
    return out


def build_two_stage_fused_output(
    *,
    kpi: pd.DataFrame,
    traditional: pd.DataFrame,
    historical_rates_path: Path,
    constants_path: Path,
    bp_hitting_path: Path,
    historical_ar_path: Path,
    metric_recency_weights_json: Path | None,
    z_coherence_mode: str,
    z_anchor_k: float,
    hr_anchor_k: float,
    z_tail_strength: float,
    coherence_mode: str,
    uncertainty_draws: int,
    seed: int,
    default_k: float,
    k_scale: float,
    stability_pa_per_season: float,
    spread_boost_max: float,
    train_start_season: int,
    train_end_season: int,
    cap_sd_mult: float,
    emit_p25p75: bool,
    two_stage_mode: str,
    two_stage_zspace_min_mlb_pa: float,
    out_path: Path,
    model_bundle_path: Path,
    diagnostics_out: Path,
    kpi_config_path: Path = Path("projections_v1/projection_config_lb2.yml"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = str(two_stage_mode).strip().lower()
    if mode not in TWO_STAGE_MODES:
        mode = "raw"
    kpi_roll, feature_cols = _build_kpi_skill_rolling_projections(
        config_path=kpi_config_path,
        train_start_season=int(train_start_season),
        train_end_season=int(train_end_season),
    )
    trad_roll = _build_traditional_stage1_rolling_projections(
        historical_rates_path=historical_rates_path,
        constants_path=constants_path,
        bp_hitting_path=bp_hitting_path,
        historical_ar_path=historical_ar_path,
        metric_recency_weights_json=metric_recency_weights_json,
        z_coherence_mode=z_coherence_mode,
        z_anchor_k=float(z_anchor_k),
        hr_anchor_k=float(hr_anchor_k),
        z_tail_strength=float(z_tail_strength),
        coherence_mode=coherence_mode,
        uncertainty_draws=int(uncertainty_draws),
        seed=int(seed),
        default_k=float(default_k),
        k_scale=float(k_scale),
        stability_pa_per_season=float(stability_pa_per_season),
        spread_boost_max=float(spread_boost_max),
        train_start_season=int(train_start_season),
        train_end_season=int(train_end_season),
    )
    actual = _load_two_stage_actual_rates(
        historical_rates_path,
        target_metrics=TWO_STAGE_TARGET_METRICS,
    )
    mlb_lookup = _load_mlb_pa_lookup(bp_hitting_path)
    panel = _build_two_stage_training_panel(
        kpi_rolling=kpi_roll,
        traditional_rolling=trad_roll,
        actual_rates=actual,
        mlb_lookup=mlb_lookup,
        feature_cols=feature_cols,
        target_metrics=TWO_STAGE_TARGET_METRICS,
    )
    reference_params = (
        _build_two_stage_reference_params(
            panel,
            target_metrics=TWO_STAGE_TARGET_METRICS,
            zspace_mlb_pa_min=float(two_stage_zspace_min_mlb_pa),
        )
        if mode == "zspace"
        else {}
    )
    models, caps_by_metric, diag = _fit_two_stage_models(
        panel,
        feature_cols=feature_cols,
        target_metrics=TWO_STAGE_TARGET_METRICS,
        cap_sd_mult=float(cap_sd_mult),
        two_stage_mode=mode,
        reference_params=reference_params,
    )

    train_window = f"{int(train_start_season)}-{int(train_end_season)}"
    fused = _apply_two_stage_models_to_traditional(
        traditional=traditional,
        kpi=kpi,
        feature_cols=feature_cols,
        target_metrics=TWO_STAGE_TARGET_METRICS,
        models=models,
        caps_by_metric=caps_by_metric,
        emit_p25p75=bool(emit_p25p75),
        train_window=train_window,
        two_stage_mode=mode,
        reference_params=reference_params,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(out_path, index=False)

    diag_out = diag.copy()
    diag_out["feature_columns"] = ", ".join(feature_cols)
    diag_out["train_panel_rows"] = int(len(panel))
    diag_out["kpi_rolling_rows"] = int(len(kpi_roll))
    diag_out["traditional_rolling_rows"] = int(len(trad_roll))
    diag_out["train_window"] = train_window
    diag_out["zspace_mlb_pa_min"] = float(two_stage_zspace_min_mlb_pa)
    diagnostics_out_df = diag_out if not diag_out.empty else pd.DataFrame(
        [
            {
                "feature_columns": ", ".join(feature_cols),
                "train_panel_rows": int(len(panel)),
                "kpi_rolling_rows": int(len(kpi_roll)),
                "traditional_rolling_rows": int(len(trad_roll)),
                "train_window": train_window,
                "zspace_mlb_pa_min": float(two_stage_zspace_min_mlb_pa),
            }
        ]
    )
    diagnostics_out_df.to_parquet(diagnostics_out, index=False)

    bundle = {
        "model_version": f"two_stage_kpi_skills_v1_{mode}",
        "two_stage_mode": mode,
        "target_metrics": list(models.keys()),
        "feature_columns": feature_cols,
        "caps_by_metric": caps_by_metric,
        "reference_params": reference_params,
        "zspace_mlb_pa_min": float(two_stage_zspace_min_mlb_pa),
        "train_window": train_window,
        "emit_p25p75": bool(emit_p25p75),
        "models_by_metric": models,
    }
    joblib.dump(bundle, model_bundle_path)
    return fused, diagnostics_out_df


def build_sandbox_sets(
    *,
    kpi_input: Path,
    traditional_input: Path,
    updated_naive_input: Path,
    old_naive_input: Path,
    old_advanced_input: Path,
    historical_rates_path: Path,
    bp_hitting_path: Path,
    build_traditional_from_historical: bool,
    build_composite_from_metric_experts: bool,
    constants_path: Path,
    historical_ar_path: Path,
    traditional_base_out: Path,
    age_curves_out: Path,
    metric_recency_weights_json: Path | None,
    z_coherence_mode: str,
    z_anchor_k: float,
    hr_anchor_k: float,
    z_tail_strength: float,
    coherence_mode: str,
    uncertainty_draws: int,
    seed: int,
    default_k: float,
    k_scale: float,
    out_kpi: Path,
    out_traditional: Path,
    out_composite: Path,
    stability_pa_per_season: float,
    spread_boost_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mlb_lookup = _load_mlb_pa_lookup(bp_hitting_path)
    mlb_ops_lookup = _load_mlb_ops_lookup(bp_hitting_path)

    kpi = _load_projection(kpi_input)
    kpi = _coverage_from_mlb_pa(
        kpi,
        id_col="batter_mlbid",
        source_season_col="source_season",
        mlb_lookup=mlb_lookup,
        stability_pa_per_season=float(stability_pa_per_season),
        seasons_back=3,
        spread_boost_max=float(spread_boost_max),
    )
    kpi_metrics = list(dict.fromkeys([*KPI_SKILL_METRICS, "PA", "bbe"]))
    kpi = _ensure_p25_p75(kpi, metric_bases=kpi_metrics)
    kpi = _apply_mlb_pa_spread_injection(
        kpi,
        metric_bases=kpi_metrics,
        mode="kpi",
    )
    kpi = _add_kpi_alias_columns(kpi)

    traditional_source = traditional_input
    if bool(build_traditional_from_historical):
        traditional_base_out.parent.mkdir(parents=True, exist_ok=True)
        age_curves_out.parent.mkdir(parents=True, exist_ok=True)
        traditional_source = _build_traditional_from_historical_rates(
            historical_rates_path=historical_rates_path,
            constants_path=constants_path,
            bp_hitting_path=bp_hitting_path,
            historical_ar_path=historical_ar_path,
            out_projection_path=traditional_base_out,
            out_age_curve_path=age_curves_out,
            metric_recency_weights_json=metric_recency_weights_json,
            z_coherence_mode=z_coherence_mode,
            z_anchor_k=float(z_anchor_k),
            hr_anchor_k=float(hr_anchor_k),
            z_tail_strength=float(z_tail_strength),
            coherence_mode=coherence_mode,
            uncertainty_draws=int(uncertainty_draws),
            seed=int(seed),
            default_k=float(default_k),
            k_scale=float(k_scale),
        )

    traditional = _load_projection(traditional_source)
    traditional_metrics = _collect_metric_bases(traditional)
    traditional = _coverage_from_mlb_pa(
        traditional,
        id_col="mlbid",
        source_season_col="source_season",
        mlb_lookup=mlb_lookup,
        stability_pa_per_season=float(stability_pa_per_season),
        seasons_back=3,
        spread_boost_max=float(spread_boost_max),
    )
    traditional = _ensure_p25_p75(traditional, metric_bases=traditional_metrics)
    traditional = _apply_mlb_pa_spread_injection(
        traditional,
        metric_bases=traditional_metrics,
        mode="traditional",
    )
    traditional = _apply_traditional_predictive_adjustments(
        traditional,
        historical_rates_path=historical_rates_path,
        bp_hitting_path=bp_hitting_path,
        mlb_lookup=mlb_lookup,
        mlb_ops_lookup=mlb_ops_lookup,
    )
    traditional = _recompute_traditional_key_outputs(traditional)
    traditional = _apply_traditional_slg_top_end_calibration(traditional)

    composite = pd.DataFrame()
    if bool(build_composite_from_metric_experts):
        updated_naive = _load_projection(updated_naive_input)
        old_naive = (
            _load_projection(old_naive_input)
            if old_naive_input.exists()
            else updated_naive.copy()
        )
        old_advanced = (
            _load_projection(old_advanced_input)
            if old_advanced_input.exists()
            else traditional.copy()
        )
        composite = _build_traditional_metric_expert_composite(
            old_advanced=old_advanced,
            old_naive=old_naive,
            updated_advanced=traditional,
            updated_naive=updated_naive,
            id_col="mlbid",
            metric_source_map=COMPOSITE_METRIC_SOURCE,
        )

    frames_to_round = [kpi, traditional]
    if not composite.empty:
        frames_to_round.append(composite)
    for frame in frames_to_round:
        float_cols = frame.select_dtypes(
            include=["float32", "float64", "Float64"]
        ).columns
        if len(float_cols) > 0:
            frame[float_cols] = frame[float_cols].round(6)

    out_kpi.parent.mkdir(parents=True, exist_ok=True)
    out_traditional.parent.mkdir(parents=True, exist_ok=True)
    out_composite.parent.mkdir(parents=True, exist_ok=True)
    kpi.to_parquet(out_kpi, index=False)
    traditional.to_parquet(out_traditional, index=False)
    if not composite.empty:
        composite.to_parquet(out_composite, index=False)
    return kpi, traditional, composite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sandbox-ready KPI and Traditional hitter projection sets. "
            "Traditional can be generated from BP_single_source_mlb_eq_non_ar_delta "
            "with full Marcel/age-curve/z-space pipeline, then post-processed with "
            "MLB-PA variability and rare/sparse-event rules."
        )
    )
    parser.add_argument(
        "--kpi-input",
        type=Path,
        default=Path("projection_outputs/lb2_refresh/hitter_projections.parquet"),
    )
    parser.add_argument(
        "--traditional-input",
        type=Path,
        default=Path(
            "BP_rate_projections_2026_non_ar_post_inv_coh_no_z_anchor.parquet"
        ),
        help="Prebuilt Traditional projection file (used only with --use-prebuilt-traditional).",
    )
    parser.add_argument(
        "--use-prebuilt-traditional",
        action="store_true",
        help="Skip from-scratch Traditional build and use --traditional-input directly.",
    )
    parser.add_argument(
        "--naive-input",
        type=Path,
        default=Path("projection_outputs/sandbox/naive_marcel_age_projections_2026.parquet"),
        help="Updated naive projection file used in metric-expert routing (upd_naive bucket).",
    )
    parser.add_argument(
        "--old-naive-input",
        type=Path,
        default=Path("projection_outputs/sandbox/naive_marcel_age_projections_2026.parquet"),
        help="Old naive projection file used in metric-expert routing (naive bucket).",
    )
    parser.add_argument(
        "--old-advanced-input",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_projections_2026_pre_predictive.parquet"),
        help="Old advanced projection file used in metric-expert routing (adv bucket).",
    )
    parser.add_argument(
        "--no-metric-expert-composite",
        action="store_true",
        help="Disable writing the metric-expert Traditional composite output.",
    )
    parser.add_argument(
        "--historical-rates-path",
        type=Path,
        default=Path("BP_single_source_mlb_eq_non_ar_delta.parquet"),
        help="Historical BP MLB-equivalent rate table used to compute predictive n_eff and seasonal priors.",
    )
    parser.add_argument(
        "--constants-path",
        type=Path,
        default=Path("BP_data_AR_2015_2025_constants.parquet"),
        help="Constants table for from-scratch Traditional build.",
    )
    parser.add_argument(
        "--historical-ar-path",
        type=Path,
        default=Path("BP_data_AR_2015_2025.parquet"),
        help="Historical AR table for correlation/coherence profile in from-scratch Traditional build.",
    )
    parser.add_argument(
        "--bp-hitting-path",
        type=Path,
        default=Path(
            "projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet"
        ),
    )
    parser.add_argument(
        "--traditional-base-out",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_base_from_hist_2026.parquet"),
        help="Intermediate from-scratch Traditional projection output before sandbox post-processing.",
    )
    parser.add_argument(
        "--age-curves-out",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_age_curves_from_hist_2026.parquet"),
        help="Intermediate age curves output for from-scratch Traditional build.",
    )
    parser.add_argument(
        "--metric-recency-weights-json",
        type=Path,
        default=None,
        help="Optional JSON mapping metric -> [w0,w1,w2] for per-metric Marcel recency weights.",
    )
    parser.add_argument(
        "--z-coherence-mode",
        choices=["direct", "inverse", "none"],
        default="inverse",
        help="Z-space rate congruity adjustment before inverse-transform for from-scratch Traditional build.",
    )
    parser.add_argument(
        "--z-anchor-k",
        type=float,
        default=0.0,
        help="Additional z-space p50 shrinkage toward 0: n_eff/(n_eff+K).",
    )
    parser.add_argument(
        "--hr-anchor-k",
        type=float,
        default=0.0,
        help="HR-rate-only pull toward MLB-T mean: n_eff/(n_eff+K).",
    )
    parser.add_argument(
        "--z-tail-strength",
        type=float,
        default=0.0,
        help="Z-space tail compression strength (1=full, 0=off).",
    )
    parser.add_argument(
        "--coherence-mode",
        choices=["direct", "inverse", "none"],
        default="inverse",
        help="Raw-space rate congruity adjustment after inverse-transform.",
    )
    parser.add_argument("--uncertainty-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--default-k",
        type=float,
        default=200.0,
        help="Fallback regression K when a metric has no override.",
    )
    parser.add_argument(
        "--k-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to metric-specific K overrides.",
    )
    parser.add_argument(
        "--out-kpi",
        type=Path,
        default=Path("projection_outputs/sandbox/kpi_projections_2026.parquet"),
    )
    parser.add_argument(
        "--out-traditional",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_projections_2026.parquet"),
    )
    parser.add_argument(
        "--out-composite",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_composite_projections_2026.parquet"),
    )
    parser.add_argument(
        "--stability-pa-per-season",
        type=float,
        default=500.0,
        help="Per-season stability target used for MLB PA coverage (3-year window).",
    )
    parser.add_argument(
        "--spread-boost-max",
        type=float,
        default=0.65,
        help="Maximum additive spread multiplier when MLB coverage is 0 (final = 1 + value).",
    )
    parser.add_argument(
        "--build-two-stage-fused",
        action="store_true",
        help="Build two-stage KPI-adjusted traditional projections (p50 first).",
    )
    parser.add_argument(
        "--out-two-stage-traditional",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet"),
        help="Output path for two-stage fused traditional projections.",
    )
    parser.add_argument(
        "--two-stage-model-bundle",
        type=Path,
        default=Path("projection_outputs/sandbox/two_stage_models_2026.joblib"),
        help="Model bundle output path for two-stage residual models.",
    )
    parser.add_argument(
        "--two-stage-diagnostics-out",
        type=Path,
        default=Path("projection_outputs/sandbox/two_stage_diagnostics_2026.parquet"),
        help="Diagnostics parquet path for two-stage training and model selection details.",
    )
    parser.add_argument(
        "--two-stage-train-start-season",
        type=int,
        default=2018,
        help="First target season used for rolling two-stage residual training.",
    )
    parser.add_argument(
        "--two-stage-train-end-season",
        type=int,
        default=2025,
        help="Last target season used for rolling two-stage residual training.",
    )
    parser.add_argument(
        "--two-stage-cap-sd-mult",
        type=float,
        default=1.5,
        help="Residual cap multiplier (cap = multiplier * residual SD by metric).",
    )
    parser.add_argument(
        "--two-stage-mode",
        choices=list(TWO_STAGE_MODES),
        default="raw",
        help=(
            "Two-stage model space: 'raw' fits residuals in raw metric space; "
            "'zspace' fits residuals in z-space using season-specific reference parameters "
            "from MLB rows meeting the minimum MLB PA threshold, where mu=median and sigma=std."
        ),
    )
    parser.add_argument(
        "--two-stage-zspace-min-mlb-pa",
        type=float,
        default=TWO_STAGE_ZSPACE_MIN_MLB_PA_DEFAULT,
        help=(
            "Minimum MLB PA required in a reference season for z-space standardization "
            "parameter estimation."
        ),
    )
    parser.add_argument(
        "--two-stage-emit-p25p75",
        action="store_true",
        help="After p50 adjustment, reconstruct p25/p75 from original stage1 asymmetry.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    kpi, traditional, composite = build_sandbox_sets(
        kpi_input=args.kpi_input,
        traditional_input=args.traditional_input,
        updated_naive_input=args.naive_input,
        old_naive_input=args.old_naive_input,
        old_advanced_input=args.old_advanced_input,
        historical_rates_path=args.historical_rates_path,
        bp_hitting_path=args.bp_hitting_path,
        build_traditional_from_historical=not bool(args.use_prebuilt_traditional),
        build_composite_from_metric_experts=not bool(args.no_metric_expert_composite),
        constants_path=args.constants_path,
        historical_ar_path=args.historical_ar_path,
        traditional_base_out=args.traditional_base_out,
        age_curves_out=args.age_curves_out,
        metric_recency_weights_json=args.metric_recency_weights_json,
        z_coherence_mode=args.z_coherence_mode,
        z_anchor_k=float(args.z_anchor_k),
        hr_anchor_k=float(args.hr_anchor_k),
        z_tail_strength=float(args.z_tail_strength),
        coherence_mode=args.coherence_mode,
        uncertainty_draws=int(args.uncertainty_draws),
        seed=int(args.seed),
        default_k=float(args.default_k),
        k_scale=float(args.k_scale),
        out_kpi=args.out_kpi,
        out_traditional=args.out_traditional,
        out_composite=args.out_composite,
        stability_pa_per_season=float(args.stability_pa_per_season),
        spread_boost_max=float(args.spread_boost_max),
    )
    print(f"Wrote {len(kpi):,} KPI rows to {args.out_kpi}")
    print(f"Wrote {len(traditional):,} Traditional rows to {args.out_traditional}")
    if not composite.empty:
        print(
            "Wrote "
            f"{len(composite):,} Traditional composite rows to {args.out_composite}"
        )
    if bool(args.build_two_stage_fused):
        fused, diag = build_two_stage_fused_output(
            kpi=kpi,
            traditional=traditional,
            historical_rates_path=args.historical_rates_path,
            constants_path=args.constants_path,
            bp_hitting_path=args.bp_hitting_path,
            historical_ar_path=args.historical_ar_path,
            metric_recency_weights_json=args.metric_recency_weights_json,
            z_coherence_mode=args.z_coherence_mode,
            z_anchor_k=float(args.z_anchor_k),
            hr_anchor_k=float(args.hr_anchor_k),
            z_tail_strength=float(args.z_tail_strength),
            coherence_mode=args.coherence_mode,
            uncertainty_draws=int(args.uncertainty_draws),
            seed=int(args.seed),
            default_k=float(args.default_k),
            k_scale=float(args.k_scale),
            stability_pa_per_season=float(args.stability_pa_per_season),
            spread_boost_max=float(args.spread_boost_max),
            train_start_season=int(args.two_stage_train_start_season),
            train_end_season=int(args.two_stage_train_end_season),
            cap_sd_mult=float(args.two_stage_cap_sd_mult),
            emit_p25p75=bool(args.two_stage_emit_p25p75),
            two_stage_mode=str(args.two_stage_mode),
            two_stage_zspace_min_mlb_pa=float(args.two_stage_zspace_min_mlb_pa),
            out_path=args.out_two_stage_traditional,
            model_bundle_path=args.two_stage_model_bundle,
            diagnostics_out=args.two_stage_diagnostics_out,
        )
        print(
            "Wrote "
            f"{len(fused):,} Two-stage Traditional rows to {args.out_two_stage_traditional}"
        )
        print(
            "Wrote "
            f"{len(diag):,} Two-stage diagnostics rows to {args.two_stage_diagnostics_out}"
        )
        print(f"Wrote two-stage model bundle to {args.two_stage_model_bundle}")


if __name__ == "__main__":
    main()
