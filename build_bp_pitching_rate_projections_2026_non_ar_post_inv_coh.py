from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_bp_rate_projections_2026_non_ar_post_inv_coh import (
    NO_COHERENCE_UNIT_CLIP_METRICS,
    _apply_z_anchor_recenter,
    _build_metric_correlation_profile,
    _compute_age_curves,
    _derive_z_bounds,
    _from_z_space_projection,
    _to_z_space,
)
from projections_v1.age import infer_and_impute_age
from projections_v1.config import GlobalConfig
from projections_v1.point_forecast import project_next_season
from projections_v1.uncertainty import apply_uncertainty_bands, build_transition_deltas


BASE_RATE_METRICS = [
    "strikeout_rate",
    "walk_rate",
    "hit_by_pitch_rate",
    "home_run_rate",
    "babip",
    "groundball_rate",
    "flyball_rate",
    "line_drive_rate",
    "infield_fly_rate",
    "infield_hit_rate",
    "whiff_rate",
    "zone_rate",
    "first_strike_rate",
    "called_strike_rate",
    "putaway_rate",
    "o_swing_rate",
    "z_swing_rate",
    "o_contact_rate",
    "z_contact_rate",
    "contact_rate",
]

VOLUME_COMPONENT_METRICS = [
    "G_mlb_eq_non_ar_delta",
    "GS_mlb_eq_non_ar_delta",
    "IP_mlb_eq_non_ar_delta",
    "IP_per_G_mlb_eq_non_ar_delta",
    "TBF_per_IP_mlb_eq_non_ar_delta",
    "HR_per_BBE_mlb_eq_non_ar_delta",
    "W_per_IP_mlb_eq_non_ar_delta",
    "SV_per_G_mlb_eq_non_ar_delta",
    "ER_per_IP_mlb_eq_non_ar_delta",
    "whip_mlb_eq_non_ar_delta",
]

RATE_METRICS = [
    f"{m}_mlb_eq_non_ar_delta" for m in BASE_RATE_METRICS
] + VOLUME_COMPONENT_METRICS
DISPLAY_PITCHING_BASES = [
    "G",
    "GS",
    "IP",
    "TBF",
    "ERA",
    "WHIP",
    "SO",
    "W",
    "SV",
    "BB",
    "H",
    "HR",
    "HBP",
    "K%",
    "BB%",
    "BABIP",
    "Whiff%",
    "SwStr%",
    "HR/BBE%",
    "GB%",
]
MARCEL_DISPLAY_BASES = ["G_marcel", "GS_marcel", "IP_marcel", "TBF_marcel", "SV_marcel"]
ER_PER_IP_MODEL_FEATURES = [
    "strikeout_rate_mlb_eq_non_ar_delta",
    "walk_rate_mlb_eq_non_ar_delta",
    "HR_per_BBE_mlb_eq_non_ar_delta",
    "babip_mlb_eq_non_ar_delta",
    "groundball_rate_mlb_eq_non_ar_delta",
]
ER_PER_IP_MODEL_MIN_BF = 100.0
ER_PER_IP_MODEL_RIDGE = 120.0
ER_PER_IP_RECENT_WEIGHTS = (5.0, 4.0, 3.0)
ER_PER_IP_RECENT_BLEND_K = 180.0
LEAGUE_TARGET_BABIP = 0.289
LEAGUE_TARGET_HR_PER_BBE = 0.045
MARCEL_RECENCY_WEIGHTS: tuple[float, float, float] = (5.0, 4.0, 3.0)
TBF_PER_IP_MIN = 3.75
NO_CLIP_METRICS: set[str] = {
    "HR_per_BBE_mlb_eq_non_ar_delta",
    "groundball_rate_mlb_eq_non_ar_delta",
}
AR_IP_BLEND_BY_MLB_SEASONS: dict[int, float] = {
    0: 1.00,
    1: 0.70,
    2: 0.40,
}
STARTER_GS_BONUS = 3.0
RATE_BOUNDS = {
    **{m: (0.0, 1.0) for m in RATE_METRICS},
    "G_mlb_eq_non_ar_delta": (0.0, 90.0),
    "GS_mlb_eq_non_ar_delta": (0.0, 34.0),
    "IP_mlb_eq_non_ar_delta": (0.0, 220.0),
    "IP_per_G_mlb_eq_non_ar_delta": (0.0, 6.5),
    "TBF_per_IP_mlb_eq_non_ar_delta": (TBF_PER_IP_MIN, 8.0),
    "HR_per_BBE_mlb_eq_non_ar_delta": (0.0, 0.30),
    "W_per_IP_mlb_eq_non_ar_delta": (0.0, 0.30),
    "SV_per_G_mlb_eq_non_ar_delta": (0.0, 1.0),
    "ER_per_IP_mlb_eq_non_ar_delta": (0.0, 1.5),
    "whip_mlb_eq_non_ar_delta": (0.4, 3.0),
}

SKILL_DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "strikeout_rate_mlb_eq_non_ar_delta": (0.02, 0.55),
    "walk_rate_mlb_eq_non_ar_delta": (0.01, 0.30),
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": (0.00, 0.05),
    "home_run_rate_mlb_eq_non_ar_delta": (0.00, 0.14),
    "babip_mlb_eq_non_ar_delta": (0.20, 0.45),
    "groundball_rate_mlb_eq_non_ar_delta": (0.15, 0.80),
    "flyball_rate_mlb_eq_non_ar_delta": (0.08, 0.70),
    "line_drive_rate_mlb_eq_non_ar_delta": (0.08, 0.45),
    "infield_fly_rate_mlb_eq_non_ar_delta": (0.00, 0.35),
    "infield_hit_rate_mlb_eq_non_ar_delta": (0.00, 0.40),
    "whiff_rate_mlb_eq_non_ar_delta": (0.05, 0.60),
    "zone_rate_mlb_eq_non_ar_delta": (0.20, 0.65),
    "first_strike_rate_mlb_eq_non_ar_delta": (0.30, 0.85),
    "called_strike_rate_mlb_eq_non_ar_delta": (0.05, 0.35),
    "putaway_rate_mlb_eq_non_ar_delta": (0.05, 0.55),
    "o_swing_rate_mlb_eq_non_ar_delta": (0.10, 0.60),
    "z_swing_rate_mlb_eq_non_ar_delta": (0.40, 0.98),
    "o_contact_rate_mlb_eq_non_ar_delta": (0.25, 1.00),
    "z_contact_rate_mlb_eq_non_ar_delta": (0.50, 1.00),
    "contact_rate_mlb_eq_non_ar_delta": (0.40, 1.00),
    "G_mlb_eq_non_ar_delta": (0.0, 75.0),
    "GS_mlb_eq_non_ar_delta": (0.0, 34.0),
    "IP_mlb_eq_non_ar_delta": (0.0, 220.0),
    "IP_per_G_mlb_eq_non_ar_delta": (0.2, 6.5),
    "TBF_per_IP_mlb_eq_non_ar_delta": (TBF_PER_IP_MIN, 5.8),
    "HR_per_BBE_mlb_eq_non_ar_delta": (0.00, 0.16),
    "W_per_IP_mlb_eq_non_ar_delta": (0.00, 0.12),
    "SV_per_G_mlb_eq_non_ar_delta": (0.00, 0.55),
    "ER_per_IP_mlb_eq_non_ar_delta": (0.12, 0.95),
    "whip_mlb_eq_non_ar_delta": (0.5, 1.90),
}

MAX_SKILL_TAIL_SWING: dict[str, float] = {
    "strikeout_rate_mlb_eq_non_ar_delta": 0.12,
    "walk_rate_mlb_eq_non_ar_delta": 0.10,
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": 0.02,
    "home_run_rate_mlb_eq_non_ar_delta": 0.10,
    "babip_mlb_eq_non_ar_delta": 0.10,
    "groundball_rate_mlb_eq_non_ar_delta": 0.30,
    "flyball_rate_mlb_eq_non_ar_delta": 0.20,
    "line_drive_rate_mlb_eq_non_ar_delta": 0.20,
    "infield_fly_rate_mlb_eq_non_ar_delta": 0.05,
    "infield_hit_rate_mlb_eq_non_ar_delta": 0.05,
    "whiff_rate_mlb_eq_non_ar_delta": 0.2,
    "zone_rate_mlb_eq_non_ar_delta": 0.08,
    "first_strike_rate_mlb_eq_non_ar_delta": 0.08,
    "called_strike_rate_mlb_eq_non_ar_delta": 0.06,
    "putaway_rate_mlb_eq_non_ar_delta": 0.08,
    "o_swing_rate_mlb_eq_non_ar_delta": 0.08,
    "z_swing_rate_mlb_eq_non_ar_delta": 0.08,
    "o_contact_rate_mlb_eq_non_ar_delta": 0.08,
    "z_contact_rate_mlb_eq_non_ar_delta": 0.08,
    "contact_rate_mlb_eq_non_ar_delta": 0.08,
    "G_mlb_eq_non_ar_delta": 18.0,
    "GS_mlb_eq_non_ar_delta": 12.0,
    "IP_mlb_eq_non_ar_delta": 42.0,
    "IP_per_G_mlb_eq_non_ar_delta": 1.2,
    "TBF_per_IP_mlb_eq_non_ar_delta": 0.6,
    "HR_per_BBE_mlb_eq_non_ar_delta": 0.05,
    "W_per_IP_mlb_eq_non_ar_delta": 0.04,
    "SV_per_G_mlb_eq_non_ar_delta": 0.20,
    "ER_per_IP_mlb_eq_non_ar_delta": 0.40,
    "whip_mlb_eq_non_ar_delta": 0.28,
}

METRIC_K_RELAX_MULTIPLIER: dict[str, float] = {
    # Increase projection spread on these outcomes by reducing empirical shrink.
    "walk_rate_mlb_eq_non_ar_delta": 0.75,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.25,
    "babip_mlb_eq_non_ar_delta": 0.75,
    "HR_per_BBE_mlb_eq_non_ar_delta": 0.75,
    "groundball_rate_mlb_eq_non_ar_delta": 0.25,
    "whiff_rate_mlb_eq_non_ar_delta": 0.25,
}

COHERENCE_METRIC_MAP: list[tuple[str, str]] = [
    ("strikeout_rate_mlb_eq_non_ar_delta", "strikeout_rate"),
    ("walk_rate_mlb_eq_non_ar_delta", "walk_rate"),
    ("home_run_rate_mlb_eq_non_ar_delta", "home_run_rate"),
    ("whiff_rate_mlb_eq_non_ar_delta", "whiff_rate"),
    ("babip_mlb_eq_non_ar_delta", "babip"),
    ("whip_mlb_eq_non_ar_delta", "whip"),
    ("groundball_rate_mlb_eq_non_ar_delta", "groundball_rate"),
    ("flyball_rate_mlb_eq_non_ar_delta", "flyball_rate"),
    ("line_drive_rate_mlb_eq_non_ar_delta", "line_drive_rate"),
    ("ER_per_IP_mlb_eq_non_ar_delta", "er_per_ip_coh"),
]

COHERENCE_ALPHA = {
    "strikeout_rate_mlb_eq_non_ar_delta": 0.16,
    "walk_rate_mlb_eq_non_ar_delta": 0.10,
    "home_run_rate_mlb_eq_non_ar_delta": 0.12,
    "whiff_rate_mlb_eq_non_ar_delta": 0.10,
    "babip_mlb_eq_non_ar_delta": 0.12,
    "whip_mlb_eq_non_ar_delta": 0.08,
    "groundball_rate_mlb_eq_non_ar_delta": 0.10,
    "flyball_rate_mlb_eq_non_ar_delta": 0.10,
    "line_drive_rate_mlb_eq_non_ar_delta": 0.08,
    "ER_per_IP_mlb_eq_non_ar_delta": 0.10,
}

COHERENCE_MAX_DELTA = {
    "strikeout_rate_mlb_eq_non_ar_delta": 0.030,
    "walk_rate_mlb_eq_non_ar_delta": 0.020,
    "home_run_rate_mlb_eq_non_ar_delta": 0.020,
    "whiff_rate_mlb_eq_non_ar_delta": 0.020,
    "babip_mlb_eq_non_ar_delta": 0.020,
    "whip_mlb_eq_non_ar_delta": 0.060,
    "groundball_rate_mlb_eq_non_ar_delta": 0.035,
    "flyball_rate_mlb_eq_non_ar_delta": 0.035,
    "line_drive_rate_mlb_eq_non_ar_delta": 0.025,
    "ER_per_IP_mlb_eq_non_ar_delta": 0.035,
}

Z_COHERENCE_ALPHA = {
    "strikeout_rate_mlb_eq_non_ar_delta": 0.16,
    "walk_rate_mlb_eq_non_ar_delta": 0.10,
    "home_run_rate_mlb_eq_non_ar_delta": 0.12,
    "whiff_rate_mlb_eq_non_ar_delta": 0.1,
    "babip_mlb_eq_non_ar_delta": 0.08,
    "whip_mlb_eq_non_ar_delta": 0.06,
    "groundball_rate_mlb_eq_non_ar_delta": 0.08,
    "flyball_rate_mlb_eq_non_ar_delta": 0.08,
    "line_drive_rate_mlb_eq_non_ar_delta": 0.12,
    "ER_per_IP_mlb_eq_non_ar_delta": 0.03,
}

Z_COHERENCE_MAX_DELTA = {
    "strikeout_rate_mlb_eq_non_ar_delta": 0.55,
    "walk_rate_mlb_eq_non_ar_delta": 0.45,
    "home_run_rate_mlb_eq_non_ar_delta": 0.45,
    "whiff_rate_mlb_eq_non_ar_delta": 0.40,
    "babip_mlb_eq_non_ar_delta": 0.35,
    "whip_mlb_eq_non_ar_delta": 0.40,
    "groundball_rate_mlb_eq_non_ar_delta": 0.45,
    "flyball_rate_mlb_eq_non_ar_delta": 0.45,
    "line_drive_rate_mlb_eq_non_ar_delta": 0.35,
    "ER_per_IP_mlb_eq_non_ar_delta": 0.40,
}

COHERENCE_EXCLUDED_METRICS: set[str] = {
    "IP_mlb_eq_non_ar_delta",
}

# Keep non-unit metrics in native scale during coherence passes.
NO_COHERENCE_UNIT_CLIP_METRICS.update(
    {
        "G_mlb_eq_non_ar_delta",
        "GS_mlb_eq_non_ar_delta",
        "IP_mlb_eq_non_ar_delta",
        "IP_per_G_mlb_eq_non_ar_delta",
        "TBF_per_IP_mlb_eq_non_ar_delta",
        "HR_per_BBE_mlb_eq_non_ar_delta",
        "groundball_rate_mlb_eq_non_ar_delta",
        "ER_per_IP_mlb_eq_non_ar_delta",
        "whip_mlb_eq_non_ar_delta",
    }
)

PITCHER_KPI_FEATURE_CANDIDATES: dict[str, list[str]] = {
    "stuff": ["stuff", "stuff_reg", "stuff_raw_reg"],
    "swstr": ["SwStr_reg", "SwStr"],
    "csw": ["CSW_reg", "CSW"],
    "ball_pct": ["Ball_pct_reg", "Ball_pct"],
    "zone": ["Zone_reg", "Zone"],
    "z_contact": ["Z_Contact_reg", "Z_Contact"],
    "chase": ["Chase_reg", "Chase"],
    "ld_pct": ["LD_pct_reg", "LD_pct"],
    "la_lte_0": ["LA_lte_0_reg", "LA_lte_0"],
    "la_gte_20": ["LA_gte_20_reg", "LA_gte_20"],
    "fastball_velo": ["fastball_velo_reg", "fastball_velo"],
    "fa_vaa": ["FA_VAA_reg", "FA_VAA", "fa_vaa_reg", "fa_vaa"],
    "fa_usage": ["FA_Usage_reg", "FA_Usage", "fa_usage_reg", "fa_usage"],
    "bb_spin": ["BB_Spin_reg", "BB_Spin", "bb_spin_reg", "bb_spin"],
}

PITCHER_KPI_TO_RATE_WEIGHTS: dict[str, dict[str, float]] = {
    "strikeout_rate_mlb_eq_non_ar_delta": {
        "stuff": 0.24,
        "swstr": 0.34,
        "csw": 0.20,
        "z_contact": -0.22,
        "chase": 0.12,
        "ball_pct": -0.08,
        "fastball_velo": 0.10,
        "fa_vaa": 0.10,
        "fa_usage": 0.06,
        "bb_spin": 0.10,
    },
    "walk_rate_mlb_eq_non_ar_delta": {
        "ball_pct": 0.45,
        "zone": -0.22,
        "csw": -0.16,
        "chase": -0.08,
    },
    "whiff_rate_mlb_eq_non_ar_delta": {
        "stuff": 0.30,
        "swstr": 0.42,
        "z_contact": -0.24,
        "fastball_velo": 0.08,
        "fa_vaa": 0.16,
        "fa_usage": 0.10,
        "bb_spin": 0.14,
    },
    "zone_rate_mlb_eq_non_ar_delta": {"zone": 0.42, "ball_pct": -0.36},
    "first_strike_rate_mlb_eq_non_ar_delta": {
        "zone": 0.34,
        "ball_pct": -0.28,
        "csw": 0.12,
    },
    "called_strike_rate_mlb_eq_non_ar_delta": {
        "csw": 0.34,
        "zone": 0.22,
        "ball_pct": -0.18,
    },
    "o_swing_rate_mlb_eq_non_ar_delta": {"chase": 0.55, "stuff": 0.08},
    "z_contact_rate_mlb_eq_non_ar_delta": {
        "z_contact": 0.70,
        "swstr": -0.24,
        "fa_vaa": -0.10,
        "bb_spin": -0.08,
    },
    "contact_rate_mlb_eq_non_ar_delta": {
        "z_contact": 0.55,
        "swstr": -0.30,
        "fa_vaa": -0.10,
        "bb_spin": -0.08,
    },
    "groundball_rate_mlb_eq_non_ar_delta": {
        "la_lte_0": 0.44,
        "la_gte_20": -0.24,
        "ld_pct": -0.10,
    },
    "flyball_rate_mlb_eq_non_ar_delta": {"la_gte_20": 0.42, "la_lte_0": -0.24},
    "line_drive_rate_mlb_eq_non_ar_delta": {
        "ld_pct": 0.52,
        "la_gte_20": 0.14,
        "la_lte_0": -0.16,
    },
    "home_run_rate_mlb_eq_non_ar_delta": {
        "la_gte_20": 0.34,
        "la_lte_0": -0.22,
        "ld_pct": 0.16,
        "fa_usage": 0.08,
        "fa_vaa": 0.05,
    },
    "HR_per_BBE_mlb_eq_non_ar_delta": {
        "la_gte_20": 0.18,
        "la_lte_0": -0.12,
        "ld_pct": 0.08,
        "fa_usage": 0.06,
        "fa_vaa": 0.04,
    },
    # Small inverse pull: higher LA>=20 profiles generally carry lower BABIP allowed.
    "babip_mlb_eq_non_ar_delta": {
        "ld_pct": 0.30,
        "la_lte_0": 0.10,
        "la_gte_20": -0.04,
        "z_contact": 0.10,
    },
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = str(path.suffix).lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported KPI file type: {path}")


def _resolve_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_pitcher_kpi_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        df = _read_table(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    id_col = _resolve_first_col(df, ["pitcher_mlbid", "mlbid"])
    season_col = _resolve_first_col(df, ["season", "source_season"])
    if id_col is None or season_col is None:
        return pd.DataFrame()
    out = df.copy()
    if id_col != "mlbid":
        out = out.rename(columns={id_col: "mlbid"})
    if season_col != "season":
        out = out.rename(columns={season_col: "season"})
    return out


def _build_pitcher_kpi_signals(
    kpi: pd.DataFrame,
    *,
    source_season: int,
    min_ref_tbf: float = 100.0,
) -> pd.DataFrame:
    if kpi.empty:
        return pd.DataFrame(columns=["mlbid"])

    work = kpi.copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    level_col = _resolve_first_col(work, ["level_id", "bp_level_id", "level_id_source"])
    exp_col = _resolve_first_col(
        work, ["TBF", "batters_faced_agg", "tbf_agg", "IP", "innings_pitched_agg"]
    )
    if level_col is None:
        work["__kpi_level"] = 1
        level_col = "__kpi_level"
    work[level_col] = _safe_numeric(work[level_col]).fillna(1.0)
    if exp_col is None:
        work["__kpi_exp"] = 1.0
        exp_col = "__kpi_exp"
    work[exp_col] = _safe_numeric(work[exp_col]).fillna(0.0).clip(lower=0.0)
    work = work[
        (work["season"] == int(source_season))
        & work["mlbid"].notna()
        & work[level_col].isin([1, 11, 14, 16])
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbid"])
    work["mlbid"] = work["mlbid"].astype("int64")

    out = pd.DataFrame({"mlbid": sorted(work["mlbid"].unique().tolist())})
    for feature, candidates in PITCHER_KPI_FEATURE_CANDIDATES.items():
        candidate_pool = list(candidates)
        for c in candidates:
            candidate_pool.append(f"{c}_proj_p50")
            candidate_pool.append(f"source_{c}")
        src_col = _resolve_first_col(work, candidate_pool)
        if src_col is None:
            continue
        vals = _safe_numeric(work[src_col]).replace([np.inf, -np.inf], np.nan)
        tmp = work[["mlbid", level_col, exp_col]].copy()
        tmp["val"] = vals
        tmp = tmp[tmp["val"].notna()].copy()
        if tmp.empty:
            continue

        tmp["w"] = _safe_numeric(tmp[exp_col]).fillna(0.0).clip(lower=0.0)
        tmp["wx"] = tmp["w"] * tmp["val"]
        grp = tmp.groupby("mlbid", as_index=False).agg(
            wx=("wx", "sum"), w=("w", "sum"), val_mean=("val", "mean")
        )
        grp["player_val"] = np.where(
            grp["w"] > 0.0, grp["wx"] / grp["w"], grp["val_mean"]
        )

        ref = tmp[
            (tmp[level_col] == 1) & (tmp["w"] >= float(max(min_ref_tbf, 0.0)))
        ].copy()
        if len(ref) < 40:
            ref = tmp[tmp["w"] >= float(max(min_ref_tbf, 0.0))].copy()
        if len(ref) < 40:
            ref = tmp.copy()
        ref_vals = _safe_numeric(ref["val"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref_vals) < 2:
            continue
        mu = float(ref_vals.mean())
        sd = float(ref_vals.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-8:
            continue
        grp[f"kpi_z_{feature}"] = (grp["player_val"] - mu) / sd
        out = out.merge(grp[["mlbid", f"kpi_z_{feature}"]], on="mlbid", how="left")
    return out


def _apply_pitcher_kpi_skill_pull(
    proj: pd.DataFrame,
    *,
    hist_raw: pd.DataFrame,
    kpi: pd.DataFrame,
    source_season: int,
    pull_strength: float = 0.18,
    min_ref_tbf: float = 100.0,
    max_kpi_z: float = 2.5,
    reliability_k: float = 160.0,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty or kpi.empty or float(pull_strength) <= 0.0:
        return out
    ksig = _build_pitcher_kpi_signals(
        kpi, source_season=source_season, min_ref_tbf=float(min_ref_tbf)
    )
    if ksig.empty:
        return out
    out = out.merge(ksig, on="mlbid", how="left")

    h = hist_raw.copy()
    if "appeared_in_MLB_hist" in h.columns:
        mlb = h[_safe_numeric(h["appeared_in_MLB_hist"]).fillna(0) == 1].copy()
        if len(mlb) >= 500:
            h = mlb
    metric_sd_map: dict[str, float] = {}
    for metric in RATE_METRICS:
        s = _safe_numeric(h.get(metric)).replace([np.inf, -np.inf], np.nan).dropna()
        sd = float(s.std(ddof=0)) if len(s) >= 2 else np.nan
        if not np.isfinite(sd) or sd <= 1e-8:
            lo, hi = RATE_BOUNDS.get(metric, (0.0, 1.0))
            sd = max((float(hi) - float(lo)) * 0.08, 1e-3)
        metric_sd_map[metric] = sd

    strength = float(np.clip(pull_strength, 0.0, 1.0))
    z_cap = float(max(0.5, max_kpi_z))
    rel_k = float(max(reliability_k, 1e-6))
    for metric, fmap in PITCHER_KPI_TO_RATE_WEIGHTS.items():
        cols = [f"kpi_z_{f}" for f in fmap]
        present = [c for c in cols if c in out.columns]
        if metric not in RATE_METRICS or not present:
            continue
        num = pd.Series(0.0, index=out.index, dtype="float64")
        den = pd.Series(0.0, index=out.index, dtype="float64")
        for feature, weight in fmap.items():
            c = f"kpi_z_{feature}"
            if c not in out.columns:
                continue
            z = _safe_numeric(out[c]).clip(lower=-z_cap, upper=z_cap)
            valid = z.notna()
            num = num + z.fillna(0.0) * float(weight)
            den = den + valid.astype(float) * abs(float(weight))
        score = _safe_divide(num, den).fillna(0.0).clip(lower=-z_cap, upper=z_cap)
        n_col = f"n_eff_{metric}"
        n_eff = _safe_numeric(out.get(n_col, np.nan)).fillna(0.0).clip(lower=0.0)
        rel = (n_eff / (n_eff + rel_k)).clip(lower=0.0, upper=1.0)
        metric_sd = float(metric_sd_map.get(metric, 0.02))
        raw_delta = (score * metric_sd * strength * (0.35 + (0.65 * rel))).astype(
            "float64"
        )
        max_delta = float(
            np.clip(COHERENCE_MAX_DELTA.get(metric, 0.03) * 1.5, 0.003, 0.20)
        )
        # Smooth saturation avoids hard clipping plateaus at exact edge values.
        scale = max(max_delta, 1e-9)
        delta = (max_delta * np.tanh(raw_delta / scale)).astype("float64")
        for pct in [25, 50, 75]:
            col = f"{metric}_proj_p{pct}"
            if col in out.columns:
                out[col] = _safe_numeric(out[col]) + delta
    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return _safe_numeric(df[col])
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    n = _safe_numeric(num)
    d = _safe_numeric(den)
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    mask = n.notna() & d.notna() & np.isfinite(n) & np.isfinite(d) & (d > 0.0)
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def _derive_pitching_component_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()

    games = _safe_col(out, "games_agg")
    gs = _safe_col(out, "games_started_agg")
    ip = _safe_col(out, "innings_pitched_agg")
    tbf = _safe_col(out, "batters_faced_agg")
    so = _safe_col(out, "strikeouts_agg")
    bb = _safe_col(out, "walks_agg")
    hbp = _safe_col(out, "hit_by_pitch_agg")
    hr = _safe_col(out, "home_runs_agg")
    wins = _safe_col(out, "wins_agg")
    saves = _safe_col(out, "saves_agg")
    er = _safe_col(out, "earned_runs_agg")
    runs = _safe_col(out, "runs_agg")

    start_share = _safe_divide(gs.clip(lower=0.0), games.clip(lower=0.0)).fillna(0.0)
    game_basis = pd.Series(
        np.where(start_share >= 0.50, gs, games),
        index=out.index,
        dtype="float64",
    )
    bbe = (tbf - so - bb - hbp).clip(lower=0.0)
    er_like = er.where(er.notna(), runs)

    derived = {
        "G_mlb_eq_non_ar_delta": games.clip(lower=0.0),
        "GS_mlb_eq_non_ar_delta": gs.clip(lower=0.0),
        "IP_mlb_eq_non_ar_delta": ip.clip(lower=0.0),
        "IP_per_G_mlb_eq_non_ar_delta": _safe_divide(ip, game_basis),
        "TBF_per_IP_mlb_eq_non_ar_delta": _safe_divide(tbf, ip),
        "HR_per_BBE_mlb_eq_non_ar_delta": _safe_divide(hr, bbe),
        "W_per_IP_mlb_eq_non_ar_delta": _safe_divide(wins, ip),
        "SV_per_G_mlb_eq_non_ar_delta": _safe_divide(saves, games),
        "ER_per_IP_mlb_eq_non_ar_delta": _safe_divide(er_like, ip),
    }
    for metric, vals in derived.items():
        lo, hi = RATE_BOUNDS.get(metric, (0.0, 1.0))
        if metric in NO_CLIP_METRICS:
            new_vals = _safe_numeric(vals)
        else:
            new_vals = _safe_numeric(vals).clip(lower=float(lo), upper=float(hi))
        if metric in out.columns:
            if metric in NO_CLIP_METRICS:
                existing = _safe_numeric(out[metric])
            else:
                existing = _safe_numeric(out[metric]).clip(
                    lower=float(lo), upper=float(hi)
                )
            out[metric] = new_vals.where(new_vals.notna(), existing)
        else:
            out[metric] = new_vals
    return out


def _replace_er_per_ip_with_component_model(
    raw: pd.DataFrame,
    *,
    appeared_hist: pd.DataFrame | None = None,
    min_bf: float = ER_PER_IP_MODEL_MIN_BF,
    ridge: float = ER_PER_IP_MODEL_RIDGE,
) -> pd.DataFrame:
    out = raw.copy()
    target = "ER_per_IP_mlb_eq_non_ar_delta"
    req = {"mlbid", "season", target, *ER_PER_IP_MODEL_FEATURES}
    if not req.issubset(out.columns):
        return out

    mlb_flag = pd.Series(0, index=out.index, dtype="int64")
    if (
        appeared_hist is not None
        and not appeared_hist.empty
        and {"mlbid", "season", "appeared_in_MLB_hist"}.issubset(appeared_hist.columns)
    ):
        idx = out[["mlbid", "season"]].copy()
        idx["mlbid"] = _safe_numeric(idx["mlbid"])
        idx["season"] = _safe_numeric(idx["season"])
        app = appeared_hist[["mlbid", "season", "appeared_in_MLB_hist"]].copy()
        app["mlbid"] = _safe_numeric(app["mlbid"])
        app["season"] = _safe_numeric(app["season"])
        idx = idx.merge(app, on=["mlbid", "season"], how="left")
        mlb_flag = _safe_numeric(idx["appeared_in_MLB_hist"]).fillna(0).astype("int64")
    elif "bp_level_id" in out.columns:
        mlb_flag = (_safe_numeric(out["bp_level_id"]).fillna(0) == 1).astype("int64")
    else:
        return out

    train_mask = mlb_flag == 1
    if "batters_faced_agg" in out.columns:
        train_mask &= _safe_numeric(out["batters_faced_agg"]).fillna(0.0) >= float(
            max(min_bf, 0.0)
        )
    if "innings_pitched_agg" in out.columns:
        train_mask &= _safe_numeric(out["innings_pitched_agg"]).fillna(0.0) > 0.0
    for c in [*ER_PER_IP_MODEL_FEATURES, target]:
        s = _safe_numeric(out[c]).replace([np.inf, -np.inf], np.nan)
        train_mask &= s.notna()
    if int(train_mask.sum()) < 200:
        return out

    train = out.loc[train_mask, [*ER_PER_IP_MODEL_FEATURES, target]].copy()
    for c in ER_PER_IP_MODEL_FEATURES:
        if c in NO_CLIP_METRICS:
            train[c] = _safe_numeric(train[c])
        else:
            lo, hi = RATE_BOUNDS.get(c, (0.0, 1.0))
            train[c] = _safe_numeric(train[c]).clip(lower=float(lo), upper=float(hi))
    t_lo, t_hi = RATE_BOUNDS.get(target, (0.0, 1.5))
    y = (
        _safe_numeric(train[target])
        .clip(lower=float(t_lo), upper=float(t_hi))
        .to_numpy(dtype=float)
    )
    x = np.column_stack(
        [
            np.ones(len(train), dtype=float),
            *[
                _safe_numeric(train[c]).to_numpy(dtype=float)
                for c in ER_PER_IP_MODEL_FEATURES
            ],
        ]
    )
    if "innings_pitched_agg" in out.columns:
        w = (
            _safe_numeric(out.loc[train_mask, "innings_pitched_agg"])
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
    else:
        w = np.ones(len(train), dtype=float)
    if not np.isfinite(w).all() or float(np.nansum(w)) <= 0.0:
        w = np.ones(len(train), dtype=float)
    w = np.clip(w, 1e-6, None)
    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = y * sw

    lam = float(max(ridge, 0.0))
    reg = np.eye(x.shape[1], dtype=float)
    reg[0, 0] = 0.0  # do not penalize intercept
    a = (xw.T @ xw) + (lam * reg)
    b = xw.T @ yw
    try:
        beta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(a, b, rcond=None)[0]

    all_x = np.column_stack(
        [
            np.ones(len(out), dtype=float),
            *[
                _safe_numeric(out[c]).to_numpy(dtype=float)
                for c in ER_PER_IP_MODEL_FEATURES
            ],
        ]
    )
    valid_pred = np.isfinite(all_x).all(axis=1)
    pred = pd.Series(np.nan, index=out.index, dtype="float64")
    pred_vals = all_x[valid_pred] @ beta
    pred.loc[valid_pred] = pred_vals
    pred = pred.clip(lower=float(t_lo), upper=float(t_hi))

    observed = _safe_numeric(out[target]).clip(lower=float(t_lo), upper=float(t_hi))
    out["ER_per_IP_mlb_eq_non_ar_delta_observed"] = observed
    out[target] = pred.where(pred.notna(), observed)
    out["ER_per_IP_mlb_eq_non_ar_delta_model_train_n"] = int(train_mask.sum())
    out["ER_per_IP_mlb_eq_non_ar_delta_model_ridge"] = float(lam)
    return out


def _apply_projected_er_per_ip_component_blend(
    proj: pd.DataFrame,
    *,
    hist_actual: pd.DataFrame,
    source_season: int,
    min_bf: float = ER_PER_IP_MODEL_MIN_BF,
    ridge: float = ER_PER_IP_MODEL_RIDGE,
    recent_weights: tuple[float, float, float] = ER_PER_IP_RECENT_WEIGHTS,
    recent_blend_k: float = ER_PER_IP_RECENT_BLEND_K,
) -> pd.DataFrame:
    out = proj.copy()
    target = "ER_per_IP_mlb_eq_non_ar_delta"
    req_hist = {
        "mlbid",
        "season",
        "innings_pitched_agg",
        target,
        *ER_PER_IP_MODEL_FEATURES,
    }
    if out.empty or hist_actual.empty or not req_hist.issubset(hist_actual.columns):
        return out
    if "mlbid" not in out.columns:
        return out

    h = hist_actual.copy()
    h["mlbid"] = _safe_numeric(h["mlbid"])
    h["season"] = _safe_numeric(h["season"])
    h["innings_pitched_agg"] = _safe_numeric(h["innings_pitched_agg"])
    if "batters_faced_agg" in h.columns:
        h["batters_faced_agg"] = _safe_numeric(h["batters_faced_agg"])
    for c in [*ER_PER_IP_MODEL_FEATURES, target]:
        h[c] = _safe_numeric(h[c]).replace([np.inf, -np.inf], np.nan)
    h = h[h["mlbid"].notna() & h["season"].notna()].copy()
    if h.empty:
        return out
    h["mlbid"] = h["mlbid"].astype("int64")
    h["season"] = h["season"].astype("int64")

    if "appeared_in_MLB_hist" in h.columns:
        mlb_flag = _safe_numeric(h["appeared_in_MLB_hist"]).fillna(0).astype("int64")
    elif "bp_level_id" in h.columns:
        mlb_flag = (_safe_numeric(h["bp_level_id"]).fillna(0) == 1).astype("int64")
    else:
        mlb_flag = pd.Series(1, index=h.index, dtype="int64")

    train_mask = mlb_flag == 1
    if "batters_faced_agg" in h.columns:
        train_mask &= _safe_numeric(h["batters_faced_agg"]).fillna(0.0) >= float(
            max(min_bf, 0.0)
        )
    train_mask &= _safe_numeric(h["innings_pitched_agg"]).fillna(0.0) > 0.0
    for c in [*ER_PER_IP_MODEL_FEATURES, target]:
        train_mask &= _safe_numeric(h[c]).notna()
    if int(train_mask.sum()) < 200:
        return out

    train = h.loc[
        train_mask, [*ER_PER_IP_MODEL_FEATURES, target, "innings_pitched_agg"]
    ].copy()
    for c in ER_PER_IP_MODEL_FEATURES:
        if c in NO_CLIP_METRICS:
            train[c] = _safe_numeric(train[c])
        else:
            lo, hi = RATE_BOUNDS.get(c, (0.0, 1.0))
            train[c] = _safe_numeric(train[c]).clip(lower=float(lo), upper=float(hi))
    t_lo, t_hi = RATE_BOUNDS.get(target, (0.0, 1.5))
    y = (
        _safe_numeric(train[target])
        .clip(lower=float(t_lo), upper=float(t_hi))
        .to_numpy(dtype=float)
    )
    x = np.column_stack(
        [
            np.ones(len(train), dtype=float),
            *[
                _safe_numeric(train[c]).to_numpy(dtype=float)
                for c in ER_PER_IP_MODEL_FEATURES
            ],
        ]
    )
    w = (
        _safe_numeric(train["innings_pitched_agg"])
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(w).all() or float(np.nansum(w)) <= 0.0:
        w = np.ones(len(train), dtype=float)
    w = np.clip(w, 1e-6, None)
    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = y * sw

    lam = float(max(ridge, 0.0))
    reg = np.eye(x.shape[1], dtype=float)
    reg[0, 0] = 0.0
    a = (xw.T @ xw) + (lam * reg)
    b = xw.T @ yw
    try:
        beta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(a, b, rcond=None)[0]

    pid = _safe_numeric(out["mlbid"]).fillna(-1).astype("int64")
    recent_vals = pd.Series(np.nan, index=out.index, dtype="float64")
    recent_rel = pd.Series(0.0, index=out.index, dtype="float64")
    rw = tuple(float(x) for x in recent_weights)
    if len(rw) < 3:
        rw = (5.0, 4.0, 3.0)
    season_w = {
        int(source_season): rw[0],
        int(source_season) - 1: rw[1],
        int(source_season) - 2: rw[2],
    }
    recent = h[h["season"].isin(season_w.keys())][
        ["mlbid", "season", target, "innings_pitched_agg"]
    ].copy()
    recent[target] = _safe_numeric(recent[target]).clip(
        lower=float(t_lo), upper=float(t_hi)
    )
    recent["innings_pitched_agg"] = (
        _safe_numeric(recent["innings_pitched_agg"]).fillna(0.0).clip(lower=0.0)
    )
    recent = recent[
        recent[target].notna() & recent["innings_pitched_agg"].gt(0.0)
    ].copy()
    if not recent.empty:
        recent["season_w"] = _safe_numeric(recent["season"].map(season_w)).fillna(0.0)
        recent["w"] = (recent["season_w"] * recent["innings_pitched_agg"]).clip(
            lower=0.0
        )
        recent = recent[recent["w"] > 0.0].copy()
        if not recent.empty:
            recent["num"] = recent[target] * recent["w"]
            agg = recent.groupby("mlbid", dropna=False).agg(
                num=("num", "sum"), den=("w", "sum")
            )
            agg["recent_er_per_ip"] = _safe_divide(agg["num"], agg["den"]).clip(
                lower=float(t_lo), upper=float(t_hi)
            )
            agg["recent_rel"] = _safe_divide(
                agg["den"], agg["den"] + float(max(recent_blend_k, 1e-6))
            ).clip(lower=0.0, upper=1.0)
            recent_vals = _safe_numeric(pid.map(agg["recent_er_per_ip"]))
            recent_rel = (
                _safe_numeric(pid.map(agg["recent_rel"]))
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )

    for pct in [25, 50, 75]:
        feat_cols = [f"{c}_proj_p{pct}" for c in ER_PER_IP_MODEL_FEATURES]
        out_col = f"{target}_proj_p{pct}"
        if not all(c in out.columns for c in feat_cols):
            continue
        p_feats: list[np.ndarray] = []
        for c in ER_PER_IP_MODEL_FEATURES:
            pc = f"{c}_proj_p{pct}"
            vals = _safe_numeric(out[pc])
            if c not in NO_CLIP_METRICS:
                lo, hi = RATE_BOUNDS.get(c, (0.0, 1.0))
                vals = vals.clip(lower=float(lo), upper=float(hi))
            p_feats.append(vals.to_numpy(dtype=float))
        all_x = np.column_stack([np.ones(len(out), dtype=float), *p_feats])
        valid_pred = np.isfinite(all_x).all(axis=1)
        pred = pd.Series(np.nan, index=out.index, dtype="float64")
        pred_vals = all_x[valid_pred] @ beta
        pred.loc[valid_pred] = pred_vals
        pred = pred.clip(lower=float(t_lo), upper=float(t_hi))

        existing = (
            _safe_numeric(out[out_col])
            if out_col in out.columns
            else pd.Series(np.nan, index=out.index, dtype="float64")
        )
        base = pred.where(pred.notna(), existing)
        blended = ((base * (1.0 - recent_rel)) + (recent_vals * recent_rel)).where(
            recent_vals.notna(), base
        )
        out[out_col] = blended.clip(lower=float(t_lo), upper=float(t_hi))

    out["ER_per_IP_mlb_eq_non_ar_delta_component_model_train_n"] = int(train_mask.sum())
    out["ER_per_IP_mlb_eq_non_ar_delta_recent_blend_k"] = float(
        max(recent_blend_k, 1e-6)
    )
    return out


def _project_ip_from_components(proj: pd.DataFrame, *, pct: int) -> pd.Series:
    g = _safe_col(proj, f"G_mlb_eq_non_ar_delta_proj_p{pct}").clip(lower=0.0)
    gs = _safe_col(proj, f"GS_mlb_eq_non_ar_delta_proj_p{pct}").clip(lower=0.0)
    ip_per_g = _safe_col(proj, f"IP_per_G_mlb_eq_non_ar_delta_proj_p{pct}").clip(
        lower=0.0
    )
    gs = np.minimum(gs, g)
    start_share = _safe_divide(gs, g).fillna(0.0)
    basis = pd.Series(
        np.where(start_share >= 0.50, gs, g), index=proj.index, dtype="float64"
    )
    return (basis * ip_per_g).clip(lower=0.0)


def _apply_league_component_environment(
    proj: pd.DataFrame,
    *,
    target_babip: float = LEAGUE_TARGET_BABIP,
    target_hr_per_bbe: float = LEAGUE_TARGET_HR_PER_BBE,
) -> pd.DataFrame:
    out = proj.copy()
    babip_metric = "babip_mlb_eq_non_ar_delta"
    hr_metric = "HR_per_BBE_mlb_eq_non_ar_delta"
    p50_babip_col = f"{babip_metric}_proj_p50"
    p50_hr_col = f"{hr_metric}_proj_p50"
    if p50_babip_col not in out.columns or p50_hr_col not in out.columns:
        return out

    def _component_weights(pct: int) -> tuple[pd.Series, pd.Series]:
        ip = _safe_col(out, f"IP_mlb_eq_non_ar_delta_proj_p{pct}")
        if int(ip.notna().sum()) == 0:
            ip = _project_ip_from_components(out, pct=pct)
        ip = ip.clip(lower=0.0)
        tbf_per_ip = _safe_col(out, f"TBF_per_IP_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0
        )
        tbf = (ip * tbf_per_ip).clip(lower=0.0)
        k_rate = _safe_col(out, f"strikeout_rate_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        bb_rate = _safe_col(out, f"walk_rate_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        hbp_rate = _safe_col(
            out, f"hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_p{pct}"
        ).clip(lower=0.0, upper=1.0)
        hr_rate = _safe_col(out, f"{hr_metric}_proj_p{pct}").clip(lower=0.0)
        bbe = (tbf * (1.0 - k_rate - bb_rate - hbp_rate)).clip(lower=0.0)
        hr = (bbe * hr_rate).clip(lower=0.0)
        bip_opp = (bbe - hr).clip(lower=0.0)
        return bbe, bip_opp

    bbe50, bip50 = _component_weights(50)
    babip50 = _safe_numeric(out[p50_babip_col])
    hr50 = _safe_numeric(out[p50_hr_col])

    def _weighted_mean(vals: pd.Series, w: pd.Series) -> float:
        mask = vals.notna() & w.notna() & np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
        if bool(mask.any()):
            return float(np.average(vals.loc[mask], weights=w.loc[mask]))
        return float(vals.mean())

    src_babip = _weighted_mean(babip50, bip50)
    src_hr = _weighted_mean(hr50, bbe50)
    if (
        not np.isfinite(src_babip)
        or src_babip <= 0.0
        or not np.isfinite(src_hr)
        or src_hr <= 0.0
    ):
        return out

    babip_delta = float(target_babip) - float(src_babip)
    hr_delta = float(target_hr_per_bbe) - float(src_hr)

    for pct in [25, 50, 75]:
        for metric, delta in [(babip_metric, babip_delta), (hr_metric, hr_delta)]:
            c = f"{metric}_proj_p{pct}"
            if c not in out.columns:
                continue
            vals = _safe_numeric(out[c]) + float(delta)
            if metric not in NO_CLIP_METRICS:
                lo, hi = RATE_BOUNDS.get(metric, (0.0, 1.0))
                vals = vals.clip(lower=float(lo), upper=float(hi))
            out[c] = vals

    out[f"{babip_metric}_league_source_p50"] = float(src_babip)
    out[f"{babip_metric}_league_target_p50"] = float(target_babip)
    out[f"{hr_metric}_league_source_p50"] = float(src_hr)
    out[f"{hr_metric}_league_target_p50"] = float(target_hr_per_bbe)
    return out


def _apply_league_ra9_environment(
    proj: pd.DataFrame,
    *,
    hist_raw: pd.DataFrame,
    source_season: int,
) -> pd.DataFrame:
    out = proj.copy()
    req_hist = {"season", "innings_pitched_agg"}
    if not req_hist.issubset(hist_raw.columns):
        return out
    if "ER_per_IP_mlb_eq_non_ar_delta_proj_p50" not in out.columns:
        return out

    h = hist_raw.copy()
    h["season"] = _safe_numeric(h["season"])
    h["innings_pitched_agg"] = _safe_numeric(h["innings_pitched_agg"])
    run_col = (
        "runs_agg"
        if "runs_agg" in h.columns
        else ("earned_runs_agg" if "earned_runs_agg" in h.columns else None)
    )
    if run_col is None:
        return out
    h[run_col] = _safe_numeric(h[run_col])
    h = h[
        (h["season"] == int(source_season))
        & h["innings_pitched_agg"].notna()
        & np.isfinite(h["innings_pitched_agg"])
        & (h["innings_pitched_agg"] > 0.0)
        & h[run_col].notna()
        & np.isfinite(h[run_col])
        & (h[run_col] >= 0.0)
    ].copy()
    if "appeared_in_MLB_hist" in h.columns:
        h["appeared_in_MLB_hist"] = _safe_numeric(h["appeared_in_MLB_hist"]).fillna(0)
        h_mlb = h[h["appeared_in_MLB_hist"] == 1].copy()
        if len(h_mlb) >= 100:
            h = h_mlb
    if h.empty:
        return out

    ip_hist = float(h["innings_pitched_agg"].sum())
    runs_hist = float(h[run_col].sum())
    if not np.isfinite(ip_hist) or not np.isfinite(runs_hist) or ip_hist <= 0.0:
        return out
    league_ra9 = float((9.0 * runs_hist) / ip_hist)
    if not np.isfinite(league_ra9) or league_ra9 <= 0.0:
        return out

    ip_proj = _safe_col(out, "IP_marcel_proj_p50")
    if int(ip_proj.notna().sum()) == 0:
        ip_proj = _project_ip_from_components(out, pct=50)
    erip_proj = _safe_numeric(out["ER_per_IP_mlb_eq_non_ar_delta_proj_p50"]).clip(
        lower=0.0
    )
    er_proj = (ip_proj * erip_proj).clip(lower=0.0)
    ip_sum = float(pd.to_numeric(ip_proj, errors="coerce").sum())
    er_sum = float(pd.to_numeric(er_proj, errors="coerce").sum())
    if not np.isfinite(ip_sum) or not np.isfinite(er_sum) or ip_sum <= 0.0:
        return out
    proj_ra9 = float((9.0 * er_sum) / ip_sum)
    if not np.isfinite(proj_ra9) or proj_ra9 <= 0.0:
        return out

    scale = float(np.clip(league_ra9 / proj_ra9, 0.60, 1.40))
    lo, hi = RATE_BOUNDS["ER_per_IP_mlb_eq_non_ar_delta"]
    for pct in [25, 50, 75]:
        c = f"ER_per_IP_mlb_eq_non_ar_delta_proj_p{pct}"
        if c not in out.columns:
            continue
        vals = _safe_numeric(out[c]) * scale
        out[c] = vals.clip(lower=float(lo), upper=float(hi))
    out["ER_per_IP_mlb_eq_non_ar_delta_proj_spread"] = _safe_numeric(
        out.get("ER_per_IP_mlb_eq_non_ar_delta_proj_p75")
    ) - _safe_numeric(out.get("ER_per_IP_mlb_eq_non_ar_delta_proj_p25"))
    out["league_ra9_source"] = league_ra9
    out["league_ra9_projected_pre_env"] = proj_ra9
    out["league_ra9_env_scale"] = scale
    return out


def _add_pitching_display_projections(proj: pd.DataFrame) -> pd.DataFrame:
    out = proj.copy()
    if "starter_flag_marcel_proj_p50" in out.columns:
        starter_mask_anchor = (
            _safe_numeric(out["starter_flag_marcel_proj_p50"]).fillna(0.0) >= 0.5
        )
    else:
        starter_anchor = _safe_col(out, "starter_share_marcel_proj_p50").clip(
            lower=0.0, upper=1.0
        )
        starter_mask_anchor = starter_anchor >= 0.50
    for pct in [25, 50, 75]:
        g_base = _safe_col(out, f"G_mlb_eq_non_ar_delta_proj_p{pct}").clip(lower=0.0)
        gs_base = _safe_col(out, f"GS_mlb_eq_non_ar_delta_proj_p{pct}").clip(lower=0.0)
        g_marcel = _safe_col(out, f"G_marcel_proj_p{pct}").clip(lower=0.0)
        gs_marcel = _safe_col(out, f"GS_marcel_proj_p{pct}").clip(lower=0.0)
        g = g_marcel.where(g_marcel.notna(), g_base)
        gs = gs_marcel.where(gs_marcel.notna(), gs_base)
        if pct == 50 and bool(starter_mask_anchor.any()):
            # For starter-flagged pitchers, pin median games to median starts.
            g = g.where(~starter_mask_anchor, gs)
        ip_per_g = _safe_col(out, f"IP_per_G_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0
        )
        tbf_per_ip = _safe_col(out, f"TBF_per_IP_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0
        )
        ip_marcel = _safe_col(out, f"IP_marcel_proj_p{pct}").clip(lower=0.0)
        tbf_marcel = _safe_col(out, f"TBF_marcel_proj_p{pct}").clip(lower=0.0)
        hr_per_bbe = _safe_col(out, f"HR_per_BBE_mlb_eq_non_ar_delta_proj_p{pct}")
        w_per_ip = _safe_col(out, f"W_per_IP_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0
        )
        sv_per_g = _safe_col(out, f"SV_per_G_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        sv_marcel = _safe_col(out, f"SV_marcel_proj_p{pct}").clip(lower=0.0)
        er_per_ip = _safe_col(out, f"ER_per_IP_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0
        )
        k_rate = _safe_col(out, f"strikeout_rate_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        bb_rate = _safe_col(out, f"walk_rate_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        hbp_rate = _safe_col(
            out, f"hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_p{pct}"
        ).clip(lower=0.0, upper=1.0)
        babip = _safe_col(out, f"babip_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        whiff_rate = _safe_col(out, f"whiff_rate_mlb_eq_non_ar_delta_proj_p{pct}").clip(
            lower=0.0, upper=1.0
        )
        gb_rate = _safe_col(out, f"groundball_rate_mlb_eq_non_ar_delta_proj_p{pct}")

        gs = np.minimum(gs, g)
        start_share = _safe_divide(gs, g).fillna(0.0)
        game_basis = pd.Series(
            np.where(start_share >= 0.50, gs, g), index=out.index, dtype="float64"
        )
        ip_chain = (game_basis * ip_per_g).clip(lower=0.0)
        ip = ip_marcel.where(ip_marcel.notna(), ip_chain).clip(lower=0.0)
        tbf_chain = (ip * tbf_per_ip).clip(lower=0.0)
        tbf = tbf_marcel.where(tbf_marcel.notna(), tbf_chain).clip(lower=0.0)
        so = (tbf * k_rate).clip(lower=0.0)
        bb = (tbf * bb_rate).clip(lower=0.0)
        hbp = (tbf * hbp_rate).clip(lower=0.0)
        bbe = (tbf - so - bb - hbp).clip(lower=0.0)
        hr = (bbe * hr_per_bbe).clip(lower=0.0)
        bip_hits = ((bbe - hr).clip(lower=0.0) * babip).clip(lower=0.0)
        hits = (hr + bip_hits).clip(lower=0.0)
        ra = (ip * er_per_ip).clip(lower=0.0)
        er = (0.92 * ra).clip(lower=0.0)
        era = _safe_divide(9.0 * er, ip).clip(lower=0.0)
        whip = _safe_divide(bb + hits, ip).clip(lower=0.0)
        wins = (ip * w_per_ip).clip(lower=0.0)
        saves_rate = (g * sv_per_g).clip(lower=0.0)
        saves = sv_marcel.where(sv_marcel.notna(), saves_rate)
        saves = np.minimum(saves, g).clip(lower=0.0)
        # Hard starter suppression fail-safe for final displayed SV outputs.
        if bool(starter_mask_anchor.any()):
            saves = saves.where(~starter_mask_anchor, np.minimum(saves, 0.49))
        k_pct = (_safe_divide(so, tbf) * 100.0).clip(lower=0.0, upper=100.0)
        bb_pct = (_safe_divide(bb, tbf) * 100.0).clip(lower=0.0, upper=100.0)
        hr_bbe_pct = hr_per_bbe * 100.0
        whiff_pct = (whiff_rate * 100.0).clip(lower=0.0, upper=100.0)
        gb_pct = gb_rate * 100.0

        out[f"G_proj_p{pct}"] = g
        out[f"GS_proj_p{pct}"] = gs
        out[f"IP_proj_p{pct}"] = ip
        out[f"TBF_proj_p{pct}"] = tbf
        out[f"G_marcel_proj_p{pct}"] = g
        out[f"GS_marcel_proj_p{pct}"] = gs
        out[f"IP_marcel_proj_p{pct}"] = ip
        out[f"TBF_marcel_proj_p{pct}"] = tbf
        out[f"ERA_proj_p{pct}"] = era
        out[f"WHIP_proj_p{pct}"] = whip
        out[f"SO_proj_p{pct}"] = so
        out[f"W_proj_p{pct}"] = wins
        out[f"SV_proj_p{pct}"] = saves
        out[f"SV_marcel_proj_p{pct}"] = saves
        out[f"BB_proj_p{pct}"] = bb
        out[f"H_proj_p{pct}"] = hits
        out[f"HR_proj_p{pct}"] = hr
        out[f"HBP_proj_p{pct}"] = hbp
        out[f"K%_proj_p{pct}"] = k_pct
        out[f"BB%_proj_p{pct}"] = bb_pct
        out[f"BABIP_proj_p{pct}"] = babip
        out[f"HR/BBE%_proj_p{pct}"] = hr_bbe_pct
        out[f"Whiff%_proj_p{pct}"] = whiff_pct
        out[f"SwStr%_proj_p{pct}"] = whiff_pct
        out[f"GB%_proj_p{pct}"] = gb_pct

    for metric in [*DISPLAY_PITCHING_BASES, *MARCEL_DISPLAY_BASES]:
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        if not all(c in out.columns for c in [c25, c50, c75]):
            continue
        p25 = _safe_numeric(out[c25]).clip(lower=0.0)
        p50 = _safe_numeric(out[c50]).clip(lower=0.0)
        p75 = _safe_numeric(out[c75]).clip(lower=0.0)
        p50 = np.maximum(p50, p25)
        p75 = np.maximum(p75, p50)
        out[c25] = p25
        out[c50] = p50
        out[c75] = p75
        out[f"{metric}_proj_spread"] = p75 - p25
    if bool(starter_mask_anchor.any()) and all(
        c in out.columns for c in ["G_proj_p50", "GS_proj_p50"]
    ):
        out["G_proj_p50"] = _safe_numeric(out["G_proj_p50"]).where(
            ~starter_mask_anchor, _safe_numeric(out["GS_proj_p50"])
        )
    if bool(starter_mask_anchor.any()) and all(
        c in out.columns for c in ["G_marcel_proj_p50", "GS_marcel_proj_p50"]
    ):
        out["G_marcel_proj_p50"] = _safe_numeric(out["G_marcel_proj_p50"]).where(
            ~starter_mask_anchor, _safe_numeric(out["GS_marcel_proj_p50"])
        )
    return out


def _apply_marcel_games_started_games(
    proj: pd.DataFrame,
    *,
    hist_raw: pd.DataFrame,
    source_season: int,
    weights: tuple[float, float, float] = MARCEL_RECENCY_WEIGHTS,
    reg_k_games: float = 30.0,
    reg_k_start_share: float = 70.0,
    reg_k_ip: float = 90.0,
    reg_k_tbf: float = 360.0,
    reg_k_sv: float = 40.0,
) -> pd.DataFrame:
    out = proj.copy()
    tbf_per_ip_floor = float(TBF_PER_IP_MIN)
    ip_metric = "IP_mlb_eq_non_ar_delta"
    gs_hard_cap = 34.0
    ip_hard_cap = 220.0
    g_metric = "G_mlb_eq_non_ar_delta"
    gs_metric = "GS_mlb_eq_non_ar_delta"
    saves_hist_col = "saves_agg"
    ip_hist_col = "innings_pitched_agg"
    tbf_hist_col = "batters_faced_agg"
    need_hist = {"mlbid", "season", g_metric, gs_metric, ip_hist_col, tbf_hist_col}
    if not need_hist.issubset(hist_raw.columns):
        return out
    need_proj = [
        f"{g_metric}_proj_p25",
        f"{g_metric}_proj_p50",
        f"{g_metric}_proj_p75",
        f"{gs_metric}_proj_p25",
        f"{gs_metric}_proj_p50",
        f"{gs_metric}_proj_p75",
    ]
    if not all(c in out.columns for c in need_proj):
        return out

    keep_cols = [
        "mlbid",
        "season",
        g_metric,
        gs_metric,
        ip_hist_col,
        tbf_hist_col,
        "appeared_in_MLB_hist",
    ]
    has_sv_hist = saves_hist_col in hist_raw.columns
    if has_sv_hist:
        keep_cols.append(saves_hist_col)
    h = hist_raw[keep_cols].copy()
    h["mlbid"] = _safe_numeric(h["mlbid"])
    h["season"] = _safe_numeric(h["season"])
    h[g_metric] = _safe_numeric(h[g_metric]).clip(lower=0.0)
    h[gs_metric] = _safe_numeric(h[gs_metric]).clip(lower=0.0)
    h[ip_hist_col] = _safe_numeric(h[ip_hist_col]).clip(lower=0.0)
    h[tbf_hist_col] = _safe_numeric(h[tbf_hist_col]).clip(lower=0.0)
    if has_sv_hist:
        h[saves_hist_col] = _safe_numeric(h[saves_hist_col]).clip(lower=0.0)
    h["appeared_in_MLB_hist"] = _safe_numeric(h.get("appeared_in_MLB_hist", 0)).fillna(
        0
    )
    h = h[h["mlbid"].notna() & h["season"].notna()].copy()
    if h.empty:
        return out
    h["mlbid"] = h["mlbid"].astype("int64")
    h["season"] = h["season"].astype("int64")
    h[gs_metric] = np.minimum(h[gs_metric], h[g_metric])
    mlb_season_counts = (
        h[_safe_numeric(h["appeared_in_MLB_hist"]).fillna(0) == 1]
        .groupby("mlbid", dropna=False)["season"]
        .nunique()
        .to_dict()
    )

    h_mlb = h[h["appeared_in_MLB_hist"] == 1].copy()
    if len(h_mlb) >= 200:
        h_ref = h_mlb
    else:
        h_ref = h

    season_g_mean = (
        h_ref.groupby("season", dropna=False)[g_metric]
        .mean(numeric_only=True)
        .to_dict()
    )
    global_g_mean = float(_safe_numeric(h_ref[g_metric]).mean())
    if not np.isfinite(global_g_mean):
        global_g_mean = 30.0
    season_ip_mean = (
        h_ref.groupby("season", dropna=False)[ip_hist_col]
        .mean(numeric_only=True)
        .to_dict()
    )
    global_ip_mean = float(_safe_numeric(h_ref[ip_hist_col]).mean())
    if not np.isfinite(global_ip_mean):
        global_ip_mean = 70.0
    season_tbf_mean = (
        h_ref.groupby("season", dropna=False)[tbf_hist_col]
        .mean(numeric_only=True)
        .to_dict()
    )
    global_tbf_mean = float(_safe_numeric(h_ref[tbf_hist_col]).mean())
    if not np.isfinite(global_tbf_mean):
        global_tbf_mean = 300.0
    if has_sv_hist:
        season_sv_mean = (
            h_ref.groupby("season", dropna=False)[saves_hist_col]
            .mean(numeric_only=True)
            .to_dict()
        )
        global_sv_mean = float(_safe_numeric(h_ref[saves_hist_col]).mean())
        if not np.isfinite(global_sv_mean):
            global_sv_mean = 1.0
    else:
        season_sv_mean = {}
        global_sv_mean = 1.0

    h_ref_share = _safe_divide(h_ref[gs_metric], h_ref[g_metric]).clip(
        lower=0.0, upper=1.0
    )
    h_ref = h_ref.assign(_start_share=h_ref_share)
    season_share_mean = (
        h_ref.groupby("season", dropna=False)["_start_share"]
        .mean(numeric_only=True)
        .to_dict()
    )
    global_share_mean = float(_safe_numeric(h_ref["_start_share"]).mean())
    if not np.isfinite(global_share_mean):
        global_share_mean = 0.25

    lookup_cols = [g_metric, gs_metric, ip_hist_col, tbf_hist_col]
    if has_sv_hist:
        lookup_cols.append(saves_hist_col)
    lookup = h.set_index(["mlbid", "season"])[lookup_cols].to_dict("index")
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(w.sum()) <= 0.0:
        w = np.asarray([3.0, 1.0, 1.0], dtype=float)
    wsum = float(w.sum())

    g50_list: list[float] = []
    gs50_list: list[float] = []
    ip50_list: list[float] = []
    tbf50_list: list[float] = []
    sv50_list: list[float] = []
    role_share_list: list[float] = []
    reliever_flag_list: list[bool] = []
    starter_flag_list: list[bool] = []
    out_ids = _safe_numeric(out["mlbid"]).fillna(-1).astype("int64")
    ar_ip50_series = _safe_numeric(out.get(f"{ip_metric}_proj_p50"))
    for row_idx, pid in enumerate(out_ids.tolist()):
        lag_g_vals: list[float] = []
        lag_share_vals: list[float] = []
        lag_ip_vals: list[float] = []
        lag_tbf_vals: list[float] = []
        lag_sv_vals: list[float] = []
        weighted_observed_games = 0.0
        weighted_observed_ip = 0.0
        weighted_observed_tbf = 0.0
        weighted_observed_sv = 0.0
        for lag in [0, 1, 2]:
            season_i = int(source_season - lag)
            rec = lookup.get((pid, season_i))
            g_obs = (
                np.nan
                if rec is None
                else float(pd.to_numeric(rec.get(g_metric), errors="coerce"))
            )
            gs_obs = (
                np.nan
                if rec is None
                else float(pd.to_numeric(rec.get(gs_metric), errors="coerce"))
            )
            ip_obs = (
                np.nan
                if rec is None
                else float(pd.to_numeric(rec.get(ip_hist_col), errors="coerce"))
            )
            tbf_obs = (
                np.nan
                if rec is None
                else float(pd.to_numeric(rec.get(tbf_hist_col), errors="coerce"))
            )
            sv_obs = (
                np.nan
                if (rec is None or not has_sv_hist)
                else float(pd.to_numeric(rec.get(saves_hist_col), errors="coerce"))
            )

            g_fallback = float(
                pd.to_numeric(season_g_mean.get(season_i), errors="coerce")
            )
            if not np.isfinite(g_fallback):
                g_fallback = global_g_mean
            g_val = g_obs if np.isfinite(g_obs) else g_fallback
            g_val = float(np.clip(g_val, 0.0, RATE_BOUNDS[g_metric][1]))
            lag_g_vals.append(g_val)

            if np.isfinite(g_obs) and g_obs > 0.0 and np.isfinite(gs_obs):
                share_obs = float(np.clip(gs_obs / max(g_obs, 1e-9), 0.0, 1.0))
            else:
                share_obs = float(
                    pd.to_numeric(season_share_mean.get(season_i), errors="coerce")
                )
                if not np.isfinite(share_obs):
                    share_obs = global_share_mean
                share_obs = float(np.clip(share_obs, 0.0, 1.0))
            lag_share_vals.append(share_obs)

            if np.isfinite(g_obs) and g_obs > 0.0:
                weighted_observed_games += float(w[lag] * g_obs)

            ip_fallback = float(
                pd.to_numeric(season_ip_mean.get(season_i), errors="coerce")
            )
            if not np.isfinite(ip_fallback):
                ip_fallback = global_ip_mean
            ip_val = ip_obs if np.isfinite(ip_obs) else ip_fallback
            ip_val = float(np.clip(ip_val, 0.0, ip_hard_cap))
            lag_ip_vals.append(ip_val)
            if np.isfinite(ip_obs) and ip_obs > 0.0:
                weighted_observed_ip += float(w[lag] * ip_obs)

            tbf_fallback = float(
                pd.to_numeric(season_tbf_mean.get(season_i), errors="coerce")
            )
            if not np.isfinite(tbf_fallback):
                tbf_fallback = global_tbf_mean
            tbf_val = tbf_obs if np.isfinite(tbf_obs) else tbf_fallback
            tbf_val = float(np.clip(tbf_val, 0.0, 1400.0))
            lag_tbf_vals.append(tbf_val)
            if np.isfinite(tbf_obs) and tbf_obs > 0.0:
                weighted_observed_tbf += float(w[lag] * tbf_obs)

            if has_sv_hist:
                sv_fallback = float(
                    pd.to_numeric(season_sv_mean.get(season_i), errors="coerce")
                )
                if not np.isfinite(sv_fallback):
                    sv_fallback = global_sv_mean
                sv_val = sv_obs if np.isfinite(sv_obs) else sv_fallback
                sv_val = float(np.clip(sv_val, 0.0, 80.0))
                lag_sv_vals.append(sv_val)
                if np.isfinite(sv_obs) and sv_obs > 0.0:
                    weighted_observed_sv += float(w[lag] * sv_obs)

        g_raw = float(np.dot(np.asarray(lag_g_vals, dtype=float), w) / wsum)
        share_raw = float(np.dot(np.asarray(lag_share_vals, dtype=float), w) / wsum)
        ip_raw = float(np.dot(np.asarray(lag_ip_vals, dtype=float), w) / wsum)
        tbf_raw = float(np.dot(np.asarray(lag_tbf_vals, dtype=float), w) / wsum)
        sv_raw = (
            float(np.dot(np.asarray(lag_sv_vals, dtype=float), w) / wsum)
            if has_sv_hist
            else np.nan
        )

        rel_games = float(
            weighted_observed_games / (weighted_observed_games + max(reg_k_games, 1e-6))
        )
        rel_games = float(np.clip(rel_games, 0.0, 1.0))
        g50 = float((rel_games * g_raw) + ((1.0 - rel_games) * global_g_mean))
        g50 = float(np.clip(g50, RATE_BOUNDS[g_metric][0], RATE_BOUNDS[g_metric][1]))

        rel_share = float(
            weighted_observed_games
            / (weighted_observed_games + max(reg_k_start_share, 1e-6))
        )
        rel_share = float(np.clip(rel_share, 0.0, 1.0))
        share50 = float(
            (rel_share * share_raw) + ((1.0 - rel_share) * global_share_mean)
        )
        share50 = float(np.clip(share50, 0.0, 1.0))
        gs50 = float(
            np.clip(
                g50 * share50,
                RATE_BOUNDS[gs_metric][0],
                min(g50, gs_hard_cap),
            )
        )
        is_reliever = bool(share_raw <= 0.25)
        is_starter = bool(share50 >= 0.50)
        if is_reliever:
            gs50 = float(min(gs50, 0.35))
        gs50 = float(np.clip(gs50, 0.0, min(g50, gs_hard_cap)))

        # Use direct Marcel-weighted IP (no league-mean reliability shrinkage).
        ip50 = float(ip_raw)
        mlb_seasons_val = float(
            pd.to_numeric(mlb_season_counts.get(pid, 0), errors="coerce")
        )
        mlb_seasons = int(mlb_seasons_val) if np.isfinite(mlb_seasons_val) else 0
        ar_blend_w = float(AR_IP_BLEND_BY_MLB_SEASONS.get(mlb_seasons, 0.0))
        if ar_blend_w > 0.0:
            ar_ip50 = float(
                pd.to_numeric(ar_ip50_series.iloc[row_idx], errors="coerce")
            )
            if np.isfinite(ar_ip50):
                ar_ip50 = float(np.clip(ar_ip50, 0.0, ip_hard_cap))
                ip50 = float(((1.0 - ar_blend_w) * ip50) + (ar_blend_w * ar_ip50))
        ip50 = float(np.clip(ip50, 0.0, ip_hard_cap))

        # Use direct Marcel-weighted TBF (no league-mean reliability shrinkage).
        tbf50 = float(tbf_raw)
        tbf50 = float(np.clip(tbf50, 0.0, 1400.0))
        # Keep direct volume ratios in a plausible MLB band.
        tbf50 = float(
            np.clip(
                tbf50,
                max(0.0, tbf_per_ip_floor * ip50),
                max(tbf_per_ip_floor * ip50, 8.5 * ip50),
            )
        )
        if has_sv_hist:
            rel_sv = float(
                weighted_observed_sv / (weighted_observed_sv + max(reg_k_sv, 1e-6))
            )
            rel_sv = float(np.clip(rel_sv, 0.0, 1.0))
            sv50 = float((rel_sv * sv_raw) + ((1.0 - rel_sv) * global_sv_mean))
            sv50 = float(np.clip(sv50, 0.0, 80.0))
        else:
            sv50 = np.nan

        g50_list.append(g50)
        gs50_list.append(gs50)
        ip50_list.append(ip50)
        tbf50_list.append(tbf50)
        sv50_list.append(sv50)
        role_share_list.append(share50)
        reliever_flag_list.append(is_reliever)
        starter_flag_list.append(is_starter)

    old_g25 = _safe_numeric(out[f"{g_metric}_proj_p25"]).clip(lower=0.0)
    old_g50 = _safe_numeric(out[f"{g_metric}_proj_p50"]).clip(lower=0.0)
    old_g75 = _safe_numeric(out[f"{g_metric}_proj_p75"]).clip(lower=0.0)
    old_g_down = (old_g50 - old_g25).clip(lower=0.0)
    old_g_up = (old_g75 - old_g50).clip(lower=0.0)

    old_gs25 = _safe_numeric(out[f"{gs_metric}_proj_p25"]).clip(lower=0.0)
    old_gs50 = _safe_numeric(out[f"{gs_metric}_proj_p50"]).clip(lower=0.0)
    old_gs75 = _safe_numeric(out[f"{gs_metric}_proj_p75"]).clip(lower=0.0)
    old_gs_down = (old_gs50 - old_gs25).clip(lower=0.0)
    old_gs_up = (old_gs75 - old_gs50).clip(lower=0.0)
    old_ip25 = _safe_numeric(out.get("IP_mlb_eq_non_ar_delta_proj_p25")).clip(lower=0.0)
    old_ip50 = _safe_numeric(out.get("IP_mlb_eq_non_ar_delta_proj_p50")).clip(lower=0.0)
    old_ip75 = _safe_numeric(out.get("IP_mlb_eq_non_ar_delta_proj_p75")).clip(lower=0.0)
    if int(old_ip50.notna().sum()) == 0:
        old_ip25 = _project_ip_from_components(out, pct=25).clip(lower=0.0)
        old_ip50 = _project_ip_from_components(out, pct=50).clip(lower=0.0)
        old_ip75 = _project_ip_from_components(out, pct=75).clip(lower=0.0)
    old_ip_down = (old_ip50 - old_ip25).clip(lower=0.0)
    old_ip_up = (old_ip75 - old_ip50).clip(lower=0.0)
    old_tbf25 = (
        old_ip25
        * _safe_col(out, "TBF_per_IP_mlb_eq_non_ar_delta_proj_p25").clip(lower=0.0)
    ).clip(lower=0.0)
    old_tbf50 = (
        old_ip50
        * _safe_col(out, "TBF_per_IP_mlb_eq_non_ar_delta_proj_p50").clip(lower=0.0)
    ).clip(lower=0.0)
    old_tbf75 = (
        old_ip75
        * _safe_col(out, "TBF_per_IP_mlb_eq_non_ar_delta_proj_p75").clip(lower=0.0)
    ).clip(lower=0.0)
    old_tbf_down = (old_tbf50 - old_tbf25).clip(lower=0.0)
    old_tbf_up = (old_tbf75 - old_tbf50).clip(lower=0.0)
    old_sv25 = (
        old_g25
        * _safe_col(out, "SV_per_G_mlb_eq_non_ar_delta_proj_p25").clip(lower=0.0)
    ).clip(lower=0.0)
    old_sv50 = (
        old_g50
        * _safe_col(out, "SV_per_G_mlb_eq_non_ar_delta_proj_p50").clip(lower=0.0)
    ).clip(lower=0.0)
    old_sv75 = (
        old_g75
        * _safe_col(out, "SV_per_G_mlb_eq_non_ar_delta_proj_p75").clip(lower=0.0)
    ).clip(lower=0.0)
    old_sv_down = (old_sv50 - old_sv25).clip(lower=0.0)
    old_sv_up = (old_sv75 - old_sv50).clip(lower=0.0)

    g50 = pd.Series(g50_list, index=out.index, dtype="float64")
    gs50 = pd.Series(gs50_list, index=out.index, dtype="float64")
    ip50 = pd.Series(ip50_list, index=out.index, dtype="float64")
    tbf50 = pd.Series(tbf50_list, index=out.index, dtype="float64")
    sv50 = pd.Series(sv50_list, index=out.index, dtype="float64")
    gs_bonus = float(max(0.0, STARTER_GS_BONUS))
    if gs_bonus > 0.0:
        start_share50 = _safe_divide(gs50, g50).fillna(0.0)
        starter_bonus_mask = (start_share50 > 0.50) & (gs50 > 0.0) & (g50 > 0.0)
        if bool(starter_bonus_mask.any()):
            gs50_pre = gs50.copy()
            ip50_pre = ip50.copy()
            tbf50_pre = tbf50.copy()

            gs50 = gs50.where(~starter_bonus_mask, gs50 + gs_bonus).clip(
                lower=0.0, upper=gs_hard_cap
            )
            g50 = np.maximum(g50, gs50)

            tbf_per_start = (
                _safe_divide(tbf50_pre, gs50_pre).fillna(0.0).clip(lower=0.0)
            )
            tbf_per_ip = (
                _safe_divide(tbf50_pre, ip50_pre)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(tbf_per_ip_floor)
                .clip(lower=tbf_per_ip_floor, upper=8.5)
            )
            added_tbf = (gs_bonus * tbf_per_start).where(starter_bonus_mask, 0.0)
            added_ip = _safe_divide(added_tbf, tbf_per_ip).fillna(0.0)
            ip50 = (ip50 + added_ip).clip(lower=0.0, upper=ip_hard_cap)
            tbf50 = (tbf50 + added_tbf).clip(lower=0.0, upper=1400.0)
    tbf50 = np.clip(tbf50, tbf_per_ip_floor * ip50, 8.5 * ip50)

    g_down_fallback = np.maximum(2.0, 0.22 * g50)
    g_up_fallback = np.maximum(2.0, 0.22 * g50)
    g_down = old_g_down.where(old_g_down > 0.0, g_down_fallback)
    g_up = old_g_up.where(old_g_up > 0.0, g_up_fallback)
    g25 = (g50 - g_down).clip(
        lower=RATE_BOUNDS[g_metric][0], upper=RATE_BOUNDS[g_metric][1]
    )
    g75 = (g50 + g_up).clip(
        lower=RATE_BOUNDS[g_metric][0], upper=RATE_BOUNDS[g_metric][1]
    )
    g50 = np.maximum(g50, g25)
    g75 = np.maximum(g75, g50)

    gs_down_fallback = (0.26 * gs50).clip(lower=0.0)
    gs_up_fallback = (0.26 * gs50).clip(lower=0.0)
    gs_down = old_gs_down.where(old_gs_down > 0.0, gs_down_fallback)
    gs_up = old_gs_up.where(old_gs_up > 0.0, gs_up_fallback)
    gs25 = (gs50 - gs_down).clip(lower=RATE_BOUNDS[gs_metric][0], upper=gs_hard_cap)
    gs75 = (gs50 + gs_up).clip(lower=RATE_BOUNDS[gs_metric][0], upper=gs_hard_cap)
    gs50 = np.maximum(gs50, gs25)
    gs75 = np.maximum(gs75, gs50)

    reliever_mask = pd.Series(reliever_flag_list, index=out.index, dtype="bool")
    if bool(reliever_mask.any()):
        gs25 = gs25.where(~reliever_mask, 0.0)
        gs50 = gs50.where(~reliever_mask, np.minimum(gs50, 0.35))
        gs75 = gs75.where(~reliever_mask, np.minimum(gs75, 1.25))
        gs50 = np.maximum(gs50, gs25)
        gs75 = np.maximum(gs75, gs50)

    gs25 = np.minimum(gs25, g25)
    gs50 = np.minimum(gs50, g50)
    gs75 = np.minimum(gs75, g75)

    ip_down_fallback = np.maximum(8.0, 0.22 * ip50)
    ip_up_fallback = np.maximum(8.0, 0.22 * ip50)
    ip_down = old_ip_down.where(old_ip_down > 0.0, ip_down_fallback)
    ip_up = old_ip_up.where(old_ip_up > 0.0, ip_up_fallback)
    ip25 = (ip50 - ip_down).clip(lower=0.0, upper=ip_hard_cap)
    ip75 = (ip50 + ip_up).clip(lower=0.0, upper=ip_hard_cap)
    ip50 = np.maximum(ip50, ip25)
    ip75 = np.maximum(ip75, ip50)

    tbf_down_fallback = np.maximum(30.0, 0.22 * tbf50)
    tbf_up_fallback = np.maximum(30.0, 0.22 * tbf50)
    tbf_down = old_tbf_down.where(old_tbf_down > 0.0, tbf_down_fallback)
    tbf_up = old_tbf_up.where(old_tbf_up > 0.0, tbf_up_fallback)
    tbf25 = (tbf50 - tbf_down).clip(lower=0.0, upper=1400.0)
    tbf75 = (tbf50 + tbf_up).clip(lower=0.0, upper=1400.0)
    tbf50 = np.maximum(tbf50, tbf25)
    tbf75 = np.maximum(tbf75, tbf50)

    tbf25 = np.clip(tbf25, tbf_per_ip_floor * ip25, 8.5 * ip25)
    tbf50 = np.clip(tbf50, tbf_per_ip_floor * ip50, 8.5 * ip50)
    tbf75 = np.clip(tbf75, tbf_per_ip_floor * ip75, 8.5 * ip75)

    sv_down_fallback = np.maximum(0.5, 0.30 * sv50)
    sv_up_fallback = np.maximum(0.5, 0.30 * sv50)
    sv_down = old_sv_down.where(old_sv_down > 0.0, sv_down_fallback)
    sv_up = old_sv_up.where(old_sv_up > 0.0, sv_up_fallback)
    sv25 = (sv50 - sv_down).clip(lower=0.0, upper=80.0)
    sv75 = (sv50 + sv_up).clip(lower=0.0, upper=80.0)
    sv50 = np.maximum(sv50, sv25)
    sv75 = np.maximum(sv75, sv50)
    sv25 = np.minimum(sv25, g25)
    sv50 = np.minimum(sv50, g50)
    sv75 = np.minimum(sv75, g75)
    starter_mask = pd.Series(starter_flag_list, index=out.index, dtype="bool")
    if bool(starter_mask.any()):
        sv25 = sv25.where(~starter_mask, 0.0)
        sv50 = sv50.where(~starter_mask, np.minimum(sv50, 0.35))
        sv75 = sv75.where(~starter_mask, np.minimum(sv75, 1.25))
        sv50 = np.maximum(sv50, sv25)
        sv75 = np.maximum(sv75, sv50)

    out[f"{g_metric}_proj_p25"] = g25
    out[f"{g_metric}_proj_p50"] = g50
    out[f"{g_metric}_proj_p75"] = g75
    out[f"{g_metric}_proj_spread"] = g75 - g25

    out[f"{gs_metric}_proj_p25"] = gs25
    out[f"{gs_metric}_proj_p50"] = gs50
    out[f"{gs_metric}_proj_p75"] = gs75
    out[f"{gs_metric}_proj_spread"] = gs75 - gs25

    out["G_marcel_proj_p25"] = g25
    out["G_marcel_proj_p50"] = g50
    out["G_marcel_proj_p75"] = g75
    out["G_marcel_proj_spread"] = g75 - g25
    out["GS_marcel_proj_p25"] = gs25
    out["GS_marcel_proj_p50"] = gs50
    out["GS_marcel_proj_p75"] = gs75
    out["GS_marcel_proj_spread"] = gs75 - gs25
    out["IP_marcel_proj_p25"] = ip25
    out["IP_marcel_proj_p50"] = ip50
    out["IP_marcel_proj_p75"] = ip75
    out["IP_marcel_proj_spread"] = ip75 - ip25
    out["TBF_marcel_proj_p25"] = tbf25
    out["TBF_marcel_proj_p50"] = tbf50
    out["TBF_marcel_proj_p75"] = tbf75
    out["TBF_marcel_proj_spread"] = tbf75 - tbf25
    out["SV_marcel_proj_p25"] = sv25
    out["SV_marcel_proj_p50"] = sv50
    out["SV_marcel_proj_p75"] = sv75
    out["SV_marcel_proj_spread"] = sv75 - sv25
    out["starter_share_marcel_proj_p50"] = pd.Series(
        role_share_list, index=out.index, dtype="float64"
    )
    out["starter_flag_marcel_proj_p50"] = pd.Series(
        starter_flag_list, index=out.index, dtype="bool"
    )
    return out


def _apply_rate_correlation_congruity(
    proj: pd.DataFrame,
    *,
    profile: dict[str, Any] | None,
    mode: str = "direct",
    ridge: float = 0.03,
    alpha_map: dict[str, float] | None = None,
    max_delta_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty or not profile:
        return out
    mode_key = str(mode or "direct").strip().lower()
    if mode_key == "none":
        return out
    direction = -1.0 if mode_key == "inverse" else 1.0

    metrics: list[str] = list(profile.get("proj_metrics", []))
    means: dict[str, float] = dict(profile.get("means", {}))
    stds: dict[str, float] = dict(profile.get("stds", {}))
    alpha_lookup = alpha_map or COHERENCE_ALPHA
    delta_lookup = max_delta_map or COHERENCE_MAX_DELTA
    corr = np.asarray(profile.get("corr"))
    if len(metrics) < 4 or corr.size == 0:
        return out

    k = len(metrics)
    for pct in [25, 50, 75]:
        cols = [f"{m}_proj_p{pct}" for m in metrics]
        if not all(c in out.columns for c in cols):
            continue
        mat = np.column_stack(
            [_safe_numeric(out[c]).to_numpy(dtype=float) for c in cols]
        )
        valid = np.isfinite(mat).all(axis=1)
        if not valid.any():
            continue

        z = np.zeros_like(mat, dtype=float)
        for i, m in enumerate(metrics):
            sd = float(pd.to_numeric(stds.get(m), errors="coerce"))
            mu = float(pd.to_numeric(means.get(m), errors="coerce"))
            if not np.isfinite(sd) or sd <= 1e-8 or not np.isfinite(mu):
                continue
            z[:, i] = (mat[:, i] - mu) / sd

        z_new = z.copy()
        for i, m in enumerate(metrics):
            alpha = float(np.clip(alpha_lookup.get(m, 0.0), 0.0, 1.0))
            if alpha <= 0:
                continue
            others = [j for j in range(k) if j != i]
            if not others:
                continue
            r_oo = corr[np.ix_(others, others)]
            r_io = corr[i, others]
            if (
                r_oo.size == 0
                or not np.isfinite(r_oo).all()
                or not np.isfinite(r_io).all()
            ):
                continue
            reg = r_oo + (float(ridge) * np.eye(r_oo.shape[0]))
            try:
                beta = np.linalg.solve(reg, r_io)
            except np.linalg.LinAlgError:
                continue
            z_hat = z[:, others] @ beta
            z_new[:, i] = z[:, i] + (direction * alpha * (z_hat - z[:, i]))

        for i, m in enumerate(metrics):
            sd = float(pd.to_numeric(stds.get(m), errors="coerce"))
            mu = float(pd.to_numeric(means.get(m), errors="coerce"))
            if not np.isfinite(sd) or sd <= 1e-8 or not np.isfinite(mu):
                continue
            vals = (z_new[:, i] * sd) + mu
            vals = np.where(valid, vals, mat[:, i])
            max_delta = float(np.clip(delta_lookup.get(m, 0.02), 0.001, 1.0))
            vals = np.clip(vals, mat[:, i] - max_delta, mat[:, i] + max_delta)
            if m not in NO_COHERENCE_UNIT_CLIP_METRICS:
                vals = np.clip(vals, 0.0, 1.0)
            out[cols[i]] = vals
    return out


def _drop_metrics_from_coherence_profile(
    profile: dict[str, Any] | None, *, excluded_metrics: set[str]
) -> dict[str, Any] | None:
    if not profile or not excluded_metrics:
        return profile
    metrics = list(profile.get("proj_metrics", []))
    if not metrics:
        return profile
    keep_idx = [i for i, m in enumerate(metrics) if m not in excluded_metrics]
    if len(keep_idx) < 4:
        return None
    corr = np.asarray(profile.get("corr"))
    if corr.size == 0:
        return None
    try:
        corr_kept = corr[np.ix_(keep_idx, keep_idx)]
    except Exception:
        return None
    means = dict(profile.get("means", {}))
    stds = dict(profile.get("stds", {}))
    kept_metrics = [metrics[i] for i in keep_idx]
    return {
        "proj_metrics": kept_metrics,
        "means": {
            m: float(pd.to_numeric(means.get(m), errors="coerce")) for m in kept_metrics
        },
        "stds": {
            m: float(pd.to_numeric(stds.get(m), errors="coerce")) for m in kept_metrics
        },
        "corr": corr_kept,
    }


def _finalize_rate_percentiles(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = proj.copy()
    for metric in metric_cols:
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        if not all(c in out.columns for c in [c25, c50, c75]):
            continue
        if metric in NO_CLIP_METRICS:
            p25 = _safe_numeric(out[c25])
            p50 = _safe_numeric(out[c50])
            p75 = _safe_numeric(out[c75])
        else:
            lo, hi = bounds.get(metric, (0.0, 1.0))
            p25 = _safe_numeric(out[c25]).clip(lower=lo, upper=hi)
            p50 = _safe_numeric(out[c50]).clip(lower=lo, upper=hi)
            p75 = _safe_numeric(out[c75]).clip(lower=lo, upper=hi)
        p50 = np.maximum(p50, p25)
        p75 = np.maximum(p75, p50)
        out[c25] = p25
        out[c50] = p50
        out[c75] = p75
        out[f"{metric}_proj_spread"] = p75 - p25
    return out


def _apply_z_tail_shape(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    z_bounds: dict[str, tuple[float, float]],
    strength: float = 0.0,
) -> pd.DataFrame:
    out = proj.copy()
    s = float(np.clip(strength, 0.0, 2.0))
    if s <= 0.0:
        return out
    for metric in metric_cols:
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        n_col = f"n_eff_{metric}"
        if not all(c in out.columns for c in [c25, c50, c75]):
            continue

        lo, hi = z_bounds.get(metric, (-4.5, 4.5))
        p25 = _safe_numeric(out[c25])
        p50 = _safe_numeric(out[c50])
        p75 = _safe_numeric(out[c75])
        n_eff = _safe_numeric(out.get(n_col, np.nan)).fillna(0.0).clip(lower=0.0)

        down = (p50 - p25).clip(lower=0.0)
        up = (p75 - p50).clip(lower=0.0)
        info = (n_eff / (n_eff + 280.0)).clip(lower=0.0, upper=1.0)
        outlier = np.abs(p50).clip(lower=0.0, upper=4.0)
        keep_base = (0.22 + (0.42 * info)) * (1.0 / (1.0 + (0.20 * outlier)))
        keep_base = np.clip(keep_base, 0.10, 0.70)
        keep = 1.0 - ((1.0 - keep_base) * s)
        keep = np.clip(keep, 0.10, 1.0)

        new_down = down * keep
        new_up = up * keep
        p25n = (p50 - new_down).clip(lower=lo, upper=hi)
        p75n = (p50 + new_up).clip(lower=lo, upper=hi)
        p50n = p50.clip(lower=lo, upper=hi)
        p50n = np.maximum(p50n, p25n)
        p75n = np.maximum(p75n, p50n)
        out[c25] = p25n
        out[c50] = p50n
        out[c75] = p75n
        out[f"{metric}_proj_spread"] = p75n - p25n
    return out


def _build_level1_correlation_profile(
    historical_raw_path: Path,
) -> dict[str, Any] | None:
    if not historical_raw_path.exists():
        return None
    hist = pd.read_parquet(historical_raw_path)
    if "ip_coh" not in hist.columns and "innings_pitched_agg" in hist.columns:
        hist["ip_coh"] = _safe_numeric(hist["innings_pitched_agg"]).clip(lower=0.0)
    if "er_per_ip_coh" not in hist.columns:
        if "ER_per_IP_mlb_eq_non_ar_delta" in hist.columns:
            hist["er_per_ip_coh"] = _safe_numeric(hist["ER_per_IP_mlb_eq_non_ar_delta"])
        else:
            ip = _safe_numeric(hist.get("innings_pitched_agg", np.nan))
            if "earned_runs_agg" in hist.columns:
                er = _safe_numeric(hist["earned_runs_agg"])
                hist["er_per_ip_coh"] = _safe_divide(er, ip)
            elif "runs_agg" in hist.columns:
                runs = _safe_numeric(hist["runs_agg"])
                hist["er_per_ip_coh"] = _safe_divide(runs, ip)
            elif "era_mlb_eq_non_ar_delta" in hist.columns:
                hist["er_per_ip_coh"] = (
                    _safe_numeric(hist["era_mlb_eq_non_ar_delta"]) / 9.0
                )
            elif "era" in hist.columns:
                hist["er_per_ip_coh"] = _safe_numeric(hist["era"]) / 9.0
    need = ["bp_level_id"] + [h for _, h in COHERENCE_METRIC_MAP]
    if not set(need).issubset(hist.columns):
        return None
    work = hist[need].copy()
    work["bp_level_id"] = _safe_numeric(work["bp_level_id"])
    work = work[work["bp_level_id"] == 1].copy()
    if work.empty:
        return None

    proj_metrics = [p for p, _ in COHERENCE_METRIC_MAP]
    hist_cols = [h for _, h in COHERENCE_METRIC_MAP]
    x = (
        work[hist_cols]
        .apply(_safe_numeric)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )
    if len(x) < 200:
        return None
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0).replace(0.0, np.nan)
    keep = sd.notna() & np.isfinite(sd) & (sd > 1e-8)
    if int(keep.sum()) < 4:
        return None
    kept_hist_cols = [c for c in hist_cols if bool(keep.loc[c])]
    kept_proj_metrics = [
        proj_metrics[i] for i, c in enumerate(hist_cols) if c in set(kept_hist_cols)
    ]
    xk = x[kept_hist_cols]
    muk = mu[kept_hist_cols]
    sdk = sd[kept_hist_cols]
    zk = (xk - muk) / sdk
    corr = np.corrcoef(zk.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return {
        "proj_metrics": kept_proj_metrics,
        "means": {
            kept_proj_metrics[i]: float(muk.iloc[i])
            for i in range(len(kept_proj_metrics))
        },
        "stds": {
            kept_proj_metrics[i]: float(sdk.iloc[i])
            for i in range(len(kept_proj_metrics))
        },
        "corr": corr,
    }


def _build_mlb_appearance_by_season(bp_pitching_path: Path) -> pd.DataFrame:
    cols = ["mlbid", "season", "bp_level_id", "batters_faced_agg"]
    out_cols = ["mlbid", "season", "appeared_in_MLB_hist"]
    if not bp_pitching_path.exists():
        return pd.DataFrame(columns=out_cols)
    bp = pd.read_parquet(bp_pitching_path)
    if not set(cols).issubset(bp.columns):
        return pd.DataFrame(columns=out_cols)
    work = bp[cols].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    work["bp_level_id"] = _safe_numeric(work["bp_level_id"])
    work["batters_faced_agg"] = _safe_numeric(work["batters_faced_agg"])
    work = work[
        work["mlbid"].notna() & work["season"].notna() & work["bp_level_id"].notna()
    ].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work["appeared_flag"] = np.where(
        (work["bp_level_id"] == 1) & (work["batters_faced_agg"].fillna(0.0) > 0.0),
        1,
        0,
    )
    out = (
        work.groupby(["mlbid", "season"], as_index=False)["appeared_flag"]
        .max()
        .rename(columns={"appeared_flag": "appeared_in_MLB_hist"})
    )
    out["appeared_in_MLB_hist"] = out["appeared_in_MLB_hist"].fillna(0).astype(int)
    return out[out_cols]


def _build_mlb_t_z_params(
    df: pd.DataFrame,
    *,
    metric_cols: list[str],
    flag_col: str = "appeared_in_MLB_hist",
    sigma_min_batters_faced: float = 10.0,
    sigma_exposure_col: str = "innings_pitched_agg",
    mu_weight_col: str = "innings_pitched_agg",
    era_mu: float | None = None,
    era_sigma: float | None = None,
    era_metric: str = "ER_per_IP_mlb_eq_non_ar_delta",
) -> dict[str, tuple[float, float]]:
    work = df.copy()
    if flag_col in work.columns:
        flag = _safe_numeric(work[flag_col]).fillna(0)
        ref = work[flag == 1].copy()
        if len(ref) < 500:
            ref = work
    else:
        ref = work

    sigma_ref = ref.copy()
    if sigma_exposure_col in sigma_ref.columns:
        exp = _safe_numeric(sigma_ref[sigma_exposure_col]).fillna(0.0)
        sigma_ref = sigma_ref[exp >= float(max(sigma_min_batters_faced, 0.0))].copy()
    if len(sigma_ref) < 2:
        sigma_ref = ref

    mu_weights = (
        _safe_numeric(ref.get(mu_weight_col)).fillna(0.0)
        if mu_weight_col in ref.columns
        else None
    )

    params: dict[str, tuple[float, float]] = {}
    for metric in metric_cols:
        s_mu = _safe_numeric(ref.get(metric)).replace([np.inf, -np.inf], np.nan)
        s_sigma = _safe_numeric(sigma_ref.get(metric)).replace(
            [np.inf, -np.inf], np.nan
        )

        mu_valid = s_mu.notna() & np.isfinite(s_mu)
        sigma_valid = s_sigma.notna() & np.isfinite(s_sigma)
        if int(mu_valid.sum()) < 2:
            params[metric] = (0.0, 1.0)
            continue

        if mu_weights is not None:
            w = _safe_numeric(mu_weights).fillna(0.0)
            mu_mask = mu_valid & w.notna() & np.isfinite(w) & (w > 0.0)
            if int(mu_mask.sum()) >= 2 and float(w.loc[mu_mask].sum()) > 0.0:
                mu = float(np.average(s_mu.loc[mu_mask], weights=w.loc[mu_mask]))
            else:
                mu = float(s_mu.loc[mu_valid].mean())
        else:
            mu = float(s_mu.loc[mu_valid].mean())

        sigma_sample = s_sigma.loc[sigma_valid]
        if len(sigma_sample) < 2:
            sigma_sample = s_mu.loc[mu_valid]
        sigma = float(sigma_sample.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 1e-8:
            sigma = 1.0
        params[metric] = (mu, sigma)

    if era_metric in params:
        mu_cur, sigma_cur = params[era_metric]
        mu_new = float(mu_cur)
        sigma_new = float(sigma_cur)
        # ERA is provided on the 9*rate scale; convert to per-IP rate space used internally.
        if era_mu is not None and np.isfinite(float(era_mu)):
            mu_new = float(era_mu) / 9.0
        if (
            era_sigma is not None
            and np.isfinite(float(era_sigma))
            and float(era_sigma) > 0.0
        ):
            sigma_new = float(era_sigma) / 9.0
        if not np.isfinite(sigma_new) or sigma_new <= 1e-8:
            sigma_new = 1.0
        params[era_metric] = (mu_new, sigma_new)
    return params


def _derive_empirical_skill_bounds(
    hist: pd.DataFrame,
    *,
    metrics: list[str],
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    work = hist.copy()
    if "appeared_in_MLB_hist" in work.columns:
        mlb = work[_safe_numeric(work["appeared_in_MLB_hist"]).fillna(0) == 1].copy()
        if len(mlb) >= 500:
            work = mlb
    for metric in metrics:
        d_lo, d_hi = SKILL_DOMAIN_BOUNDS.get(metric, (0.0, 1.0))
        s = _safe_numeric(work.get(metric)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 200:
            bounds[metric] = (float(d_lo), float(d_hi))
            continue
        lo = float(np.nanquantile(s, q_lo))
        hi = float(np.nanquantile(s, q_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            bounds[metric] = (float(d_lo), float(d_hi))
            continue
        span = max(hi - lo, 1e-6)
        lo = float(np.clip(lo - (0.20 * span), d_lo, d_hi))
        hi = float(np.clip(hi + (0.20 * span), lo + 1e-6, d_hi))
        if metric == "babip_mlb_eq_non_ar_delta":
            # Do not impose a data-mined BABIP floor (e.g. ~.296) from historical quantiles.
            # Keep the domain lower bound and only use empirical upper shaping.
            lo = float(d_lo)
        bounds[metric] = (lo, hi)
    return bounds


def _apply_tail_swing_caps(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = proj.copy()
    for metric in metric_cols:
        if metric in NO_CLIP_METRICS:
            continue
        c25 = f"{metric}_proj_p25"
        c50 = f"{metric}_proj_p50"
        c75 = f"{metric}_proj_p75"
        if not all(c in out.columns for c in [c25, c50, c75]):
            continue
        lo, hi = bounds.get(metric, (0.0, 1.0))
        cap = float(MAX_SKILL_TAIL_SWING.get(metric, 0.08))
        span = max(float(hi - lo), 1e-6)
        cap = min(cap, 0.45 * span)
        p25 = _safe_numeric(out[c25])
        p50 = _safe_numeric(out[c50]).clip(lower=lo, upper=hi)
        p75 = _safe_numeric(out[c75])
        down = np.minimum((p50 - p25).clip(lower=0.0), cap)
        up = np.minimum((p75 - p50).clip(lower=0.0), cap)
        p25n = (p50 - down).clip(lower=lo, upper=hi)
        p75n = (p50 + up).clip(lower=lo, upper=hi)
        p50n = np.maximum(p50, p25n)
        p75n = np.maximum(p75n, p50n)
        out[c25] = p25n
        out[c50] = p50n
        out[c75] = p75n
        out[f"{metric}_proj_spread"] = p75n - p25n
    return out


def _apply_recent_source_rate_retention(
    proj: pd.DataFrame,
    *,
    hist_raw: pd.DataFrame,
    source_season: int,
    reliability_k: float = 140.0,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty or hist_raw.empty:
        return out

    metric_cfg: dict[str, tuple[float, float]] = {
        # metric: (pull_strength, max_abs_shift_in_rate_space)
        "strikeout_rate_mlb_eq_non_ar_delta": (0.95, 0.20),
        "HR_per_BBE_mlb_eq_non_ar_delta": (0.75, 0.1),
        "walk_rate_mlb_eq_non_ar_delta": (0.75, 0.1),
        "babip_mlb_eq_non_ar_delta": (0.25, 0.10),
        "groundball_rate_mlb_eq_non_ar_delta": (0.95, 0.25),
        "whiff_rate_mlb_eq_non_ar_delta": (0.95, 0.20),
    }
    active_metrics = [m for m in metric_cfg if f"{m}_proj_p50" in out.columns]
    if not active_metrics:
        return out

    cols = ["mlbid", "season", "batters_faced_agg", *active_metrics]
    if not set(cols).issubset(hist_raw.columns):
        return out
    src = hist_raw[cols].copy()
    src["mlbid"] = _safe_numeric(src["mlbid"])
    src["season"] = _safe_numeric(src["season"])
    src = src[src["mlbid"].notna() & src["season"].notna()].copy()
    src = src[src["season"].astype("int64") == int(source_season)].copy()
    if src.empty:
        return out
    src["mlbid"] = src["mlbid"].astype("int64")
    src["batters_faced_agg"] = (
        _safe_numeric(src["batters_faced_agg"]).fillna(0.0).clip(lower=0.0)
    )
    for metric in active_metrics:
        src[metric] = _safe_numeric(src[metric]).replace([np.inf, -np.inf], np.nan)
    # One row per player-season; keep highest-exposure row.
    src = (
        src.sort_values(["mlbid", "batters_faced_agg"])
        .drop_duplicates(subset=["mlbid"], keep="last")
        .set_index("mlbid")
    )

    pid = _safe_numeric(out.get("mlbid")).fillna(-1).astype("int64")
    src_tbf = (
        _safe_numeric(pid.map(src["batters_faced_agg"])).fillna(0.0).clip(lower=0.0)
    )
    rel = (src_tbf / (src_tbf + float(max(reliability_k, 1e-6)))).clip(
        lower=0.0, upper=1.0
    )

    for metric in active_metrics:
        src_vals = _safe_numeric(pid.map(src[metric]))
        if src_vals.notna().sum() == 0:
            continue
        if metric not in NO_CLIP_METRICS:
            lo, hi = RATE_BOUNDS.get(metric, (0.0, 1.0))
            src_vals = src_vals.clip(lower=float(lo), upper=float(hi))
        strength, max_shift = metric_cfg[metric]
        c50 = f"{metric}_proj_p50"
        p50 = _safe_numeric(out[c50])
        delta = ((src_vals - p50) * float(strength) * rel).fillna(0.0)
        delta = delta.clip(lower=-float(max_shift), upper=float(max_shift))
        for pct in [25, 50, 75]:
            c = f"{metric}_proj_p{pct}"
            if c in out.columns:
                out[c] = _safe_numeric(out[c]) + delta
    return out


def _build_k_overrides(constants_path: Path, k_scale: float = 1.0) -> dict[str, float]:
    if not constants_path.exists():
        return {}
    c = pd.read_parquet(constants_path)
    needed = {"stat", "bp_level_id", "K"}
    if not needed.issubset(c.columns):
        return {}
    c = c.copy()
    c["bp_level_id"] = _safe_numeric(c["bp_level_id"])
    c["K"] = _safe_numeric(c["K"])
    c = c[c["bp_level_id"] == 1]
    if c.empty:
        return {}
    med = c.groupby("stat", dropna=False)["K"].median(numeric_only=True).to_dict()
    out: dict[str, float] = {}
    for metric in RATE_METRICS:
        stat = metric.replace("_mlb_eq_non_ar_delta", "")
        kval = pd.to_numeric(med.get(stat), errors="coerce")
        if np.isfinite(kval) and float(kval) > 0.0:
            metric_scale = float(METRIC_K_RELAX_MULTIPLIER.get(metric, 1.0))
            kval_scaled = float(kval) * float(k_scale) * metric_scale
            out[metric] = float(np.clip(kval_scaled, 20.0, 20000.0))
    return out


def _build_source_season_context(
    bp_pitching_path: Path, *, source_season: int
) -> pd.DataFrame:
    out_cols = [
        "mlbid",
        f"levels_played_{source_season}",
        "appeared_in_MLB",
        "team_abbreviations",
    ]
    if not bp_pitching_path.exists():
        return pd.DataFrame(columns=out_cols)
    cols = ["mlbid", "season", "bp_level_id", "team_abbreviation", "batters_faced_agg"]
    bp = pd.read_parquet(bp_pitching_path)
    if not set(cols).issubset(bp.columns):
        return pd.DataFrame(columns=out_cols)
    work = bp[cols].copy()
    work["mlbid"] = _safe_numeric(work["mlbid"])
    work["season"] = _safe_numeric(work["season"])
    work["bp_level_id"] = _safe_numeric(work["bp_level_id"])
    work["batters_faced_agg"] = _safe_numeric(work["batters_faced_agg"])
    work["team_abbreviation"] = (
        work["team_abbreviation"].fillna("").astype(str).str.strip()
    )
    work = work[
        work["mlbid"].notna() & work["season"].notna() & work["bp_level_id"].notna()
    ].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work = work[
        (work["season"].astype("int64") == int(source_season))
        & work["bp_level_id"].astype("int64").isin([1, 2, 3, 4, 5])
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    lvl_col = f"levels_played_{source_season}"
    levels = (
        work.groupby("mlbid", dropna=False)["bp_level_id"]
        .apply(
            lambda s: "|".join(
                str(int(v))
                for v in sorted(set(_safe_numeric(s).dropna().astype(int).tolist()))
            )
        )
        .reset_index(name=lvl_col)
    )
    appeared = (
        work.groupby("mlbid", dropna=False)["bp_level_id"]
        .apply(lambda s: "T" if (_safe_numeric(s) == 1).any() else "F")
        .reset_index(name="appeared_in_MLB")
    )
    team_work = work[work["team_abbreviation"] != ""].copy()
    if team_work.empty:
        team = pd.DataFrame(columns=["mlbid", "team_abbreviations"])
    else:
        team_tot = (
            team_work.groupby(["mlbid", "team_abbreviation"], as_index=False)[
                "batters_faced_agg"
            ]
            .sum(numeric_only=True)
            .sort_values(
                ["mlbid", "batters_faced_agg", "team_abbreviation"],
                ascending=[True, False, True],
            )
        )
        team = (
            team_tot.groupby("mlbid", dropna=False)["team_abbreviation"]
            .apply(lambda s: "|".join(s.astype(str).tolist()))
            .reset_index(name="team_abbreviations")
        )
    meta = levels.merge(appeared, on="mlbid", how="outer")
    return meta.merge(team, on="mlbid", how="left")


def _load_metric_recency_weights(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metric recency weight file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "Metric recency weight file must be a JSON object of metric -> [w0,w1,w2]."
        )
    out: dict[str, list[float]] = {}
    for metric, weights in raw.items():
        if metric not in RATE_METRICS:
            continue
        if not isinstance(weights, (list, tuple)) or len(weights) < 3:
            raise ValueError(f"Metric {metric} must have at least 3 weights.")
        vals: list[float] = []
        for w in list(weights)[:3]:
            wf = float(pd.to_numeric(w, errors="coerce"))
            if not np.isfinite(wf) or wf <= 0:
                raise ValueError(
                    f"Metric {metric} has non-positive/invalid weight: {w}"
                )
            vals.append(wf)
        out[metric] = vals
    return out


def _apply_metric_population_anchor(
    proj: pd.DataFrame,
    *,
    hist: pd.DataFrame,
    metric: str,
    anchor_k: float = 220.0,
    mlb_flag_col: str = "appeared_in_MLB_hist",
) -> pd.DataFrame:
    out = proj.copy()
    if metric not in RATE_METRICS:
        return out
    k = float(max(anchor_k, 0.0))
    if k <= 0.0:
        return out
    c25 = f"{metric}_proj_p25"
    c50 = f"{metric}_proj_p50"
    c75 = f"{metric}_proj_p75"
    n_col = f"n_eff_{metric}"
    if not all(c in out.columns for c in [c25, c50, c75]):
        return out
    h = hist.copy()
    if mlb_flag_col in h.columns:
        mlb = h[_safe_numeric(h[mlb_flag_col]).fillna(0) == 1].copy()
        if len(mlb) >= 200:
            h = mlb
    mu = float(_safe_numeric(h.get(metric)).mean())
    if not np.isfinite(mu):
        return out
    p25 = _safe_numeric(out[c25])
    p50 = _safe_numeric(out[c50])
    p75 = _safe_numeric(out[c75])
    n_eff = _safe_numeric(out.get(n_col, np.nan)).fillna(0.0).clip(lower=0.0)
    w = (n_eff / (n_eff + k)).clip(lower=0.0, upper=1.0)
    p50_new = (p50 * w) + (mu * (1.0 - w))
    shift = p50_new - p50
    out[c50] = p50_new
    out[c25] = p25 + shift
    out[c75] = p75 + shift
    out[f"{metric}_anchor_mu_mlbT"] = mu
    out[f"{metric}_anchor_k"] = k
    return out


def build_bp_pitching_rate_projections_2026(
    *,
    input_path: Path,
    constants_path: Path,
    bp_pitching_path: Path,
    historical_raw_path: Path,
    z_coherence_mode: str = "inverse",
    z_anchor_k: float = 260.0,
    anchor_metric: str = "home_run_rate_mlb_eq_non_ar_delta",
    metric_anchor_k: float = 220.0,
    z_tail_strength: float = 0.50,
    coherence_mode: str = "inverse",
    out_projection_path: Path,
    out_age_curve_path: Path,
    uncertainty_draws: int = 2000,
    seed: int = 7,
    recency_weights: tuple[float, float, float] = MARCEL_RECENCY_WEIGHTS,
    metric_recency_weights: dict[str, list[float]] | None = None,
    default_k: float = 200.0,
    k_scale: float = 1.0,
    z_sigma_min_ip: float = 10.0,
    era_mu: float | None = None,
    era_sigma: float | None = None,
    pitcher_kpi_path: Path | None = None,
    kpi_pull_strength: float = 0.18,
    kpi_min_tbf: float = 100.0,
    kpi_max_z: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rw_raw = list(recency_weights)[:3]
    if len(rw_raw) < 3:
        rw_raw = list(MARCEL_RECENCY_WEIGHTS)
    rw = tuple(float(x) for x in rw_raw[:3])
    if any((not np.isfinite(float(x))) or (float(x) <= 0.0) for x in rw):
        rw = MARCEL_RECENCY_WEIGHTS

    raw = pd.read_parquet(input_path)
    raw = _derive_pitching_component_metrics(raw)
    raw_actual_er = raw.copy()
    appeared_hist = _build_mlb_appearance_by_season(bp_pitching_path)
    raw = _replace_er_per_ip_with_component_model(
        raw,
        appeared_hist=appeared_hist,
    )
    required = {
        "mlbid",
        "season",
        "batters_faced_agg",
        "baseball_age",
        "player_display_text",
    }
    miss_req = [c for c in required if c not in raw.columns]
    if miss_req:
        raise ValueError(f"Missing required columns in {input_path}: {miss_req}")
    miss_metrics = [m for m in RATE_METRICS if m not in raw.columns]
    if miss_metrics:
        raise ValueError(f"Missing required metrics in {input_path}: {miss_metrics}")
    critical_component_metrics = [
        "G_mlb_eq_non_ar_delta",
        "GS_mlb_eq_non_ar_delta",
        "IP_per_G_mlb_eq_non_ar_delta",
        "TBF_per_IP_mlb_eq_non_ar_delta",
        "HR_per_BBE_mlb_eq_non_ar_delta",
        "ER_per_IP_mlb_eq_non_ar_delta",
    ]
    missing_component_data = [
        m
        for m in critical_component_metrics
        if int(_safe_numeric(raw.get(m)).notna().sum()) == 0
    ]
    if missing_component_data:
        raise ValueError(
            "Missing component-rate source data after preprocessing for metrics: "
            f"{missing_component_data}. Ensure single-source file carries pitching count columns."
        )

    hist_raw = raw.copy()
    for c in ["mlbid", "season", "batters_faced_agg", "baseball_age"]:
        hist_raw[c] = _safe_numeric(hist_raw[c])
    hist_raw = hist_raw[hist_raw["mlbid"].notna() & hist_raw["season"].notna()].copy()
    hist_raw["mlbid"] = hist_raw["mlbid"].astype("int64")
    hist_raw["season"] = hist_raw["season"].astype("int64")
    hist_raw["level_id_source"] = 1
    hist_raw = hist_raw.rename(columns={"player_display_text": "player_name"})
    hist_raw["player_name"] = (
        hist_raw["player_name"].fillna(hist_raw["mlbid"].astype(str)).astype(str)
    )
    hist_actual = raw_actual_er.copy()
    for c in ["mlbid", "season", "batters_faced_agg", "baseball_age"]:
        if c in hist_actual.columns:
            hist_actual[c] = _safe_numeric(hist_actual[c])
    hist_actual = hist_actual[
        hist_actual["mlbid"].notna() & hist_actual["season"].notna()
    ].copy()
    hist_actual["mlbid"] = hist_actual["mlbid"].astype("int64")
    hist_actual["season"] = hist_actual["season"].astype("int64")
    kpi_frame = _load_pitcher_kpi_frame(pitcher_kpi_path)
    if not appeared_hist.empty:
        hist_raw = hist_raw.merge(appeared_hist, on=["mlbid", "season"], how="left")
        hist_actual = hist_actual.merge(
            appeared_hist, on=["mlbid", "season"], how="left"
        )
    if "appeared_in_MLB_hist" not in hist_raw.columns:
        hist_raw["appeared_in_MLB_hist"] = 0
    if "appeared_in_MLB_hist" not in hist_actual.columns:
        hist_actual["appeared_in_MLB_hist"] = 0
    hist_raw["appeared_in_MLB_hist"] = (
        _safe_numeric(hist_raw["appeared_in_MLB_hist"]).fillna(0).astype(int)
    )
    hist_actual["appeared_in_MLB_hist"] = (
        _safe_numeric(hist_actual["appeared_in_MLB_hist"]).fillna(0).astype(int)
    )

    z_params = _build_mlb_t_z_params(
        hist_raw,
        metric_cols=RATE_METRICS,
        flag_col="appeared_in_MLB_hist",
        sigma_min_batters_faced=float(z_sigma_min_ip),
        sigma_exposure_col="innings_pitched_agg",
        mu_weight_col="innings_pitched_agg",
        era_mu=era_mu,
        era_sigma=era_sigma,
    )
    hist = _to_z_space(hist_raw, metric_cols=RATE_METRICS, params=z_params)
    z_bounds = _derive_z_bounds(hist, metric_cols=RATE_METRICS)
    z_profile = _build_metric_correlation_profile(hist, metric_cols=RATE_METRICS)
    z_profile = _drop_metrics_from_coherence_profile(
        z_profile, excluded_metrics=COHERENCE_EXCLUDED_METRICS
    )

    age_curves = _compute_age_curves(
        hist,
        id_col="mlbid",
        season_col="season",
        age_col="baseball_age",
        exposure_col="batters_faced_agg",
        metrics=RATE_METRICS,
        min_pairs_per_age=30,
        smooth_window=3,
        anchor_age=27,
    )
    age_curves["value_space"] = "z"

    hist = infer_and_impute_age(
        hist,
        player_col="mlbid",
        season_col="season",
        level_col="level_id_source",
        age_col="baseball_age",
        group_cols=["level_id_source"],
    )

    gcfg = GlobalConfig(
        recency_weights=list(rw),
        recency_weights_by_metric=dict(metric_recency_weights or {}),
        default_k=float(default_k),
        uncertainty_draws=int(uncertainty_draws),
        seed=int(seed),
        cov_blend_global=0.6,
        local_k=200,
        local_min_k=40,
        uncertainty_c=120.0,
        uncertainty_d=20.0,
        projection_version="bp_pitching_eq_non_ar_post_inv_coh_zspace_2026",
        min_age_samples=20,
    )
    k_overrides = _build_k_overrides(constants_path, k_scale=float(k_scale))
    coherence_profile = (
        _build_level1_correlation_profile(historical_raw_path)
        if coherence_mode != "none"
        else None
    )
    coherence_profile = _drop_metrics_from_coherence_profile(
        coherence_profile, excluded_metrics=COHERENCE_EXCLUDED_METRICS
    )

    source_season = int(_safe_numeric(hist["season"]).max())
    src_meta = _build_source_season_context(
        bp_pitching_path, source_season=source_season
    )
    point = project_next_season(
        player_season_df=hist,
        metric_cols=RATE_METRICS,
        id_col="mlbid",
        name_col="player_name",
        season_col="season",
        level_col="level_id_source",
        age_col="age_used",
        age_source_col="age_source",
        exposure_col="batters_faced_agg",
        global_cfg=gcfg,
        k_overrides=k_overrides,
        bounds=z_bounds,
        source_season=source_season,
        passthrough_cols=[],
    )
    point = _apply_z_anchor_recenter(
        point, metric_cols=RATE_METRICS, anchor_k=float(z_anchor_k)
    )

    transitions = build_transition_deltas(
        hist, id_col="mlbid", season_col="season", metric_cols=RATE_METRICS
    )
    proj = apply_uncertainty_bands(
        point,
        transitions_df=transitions,
        metric_cols=RATE_METRICS,
        draws=int(gcfg.uncertainty_draws),
        seed=int(gcfg.seed),
        global_weight=float(gcfg.cov_blend_global),
        local_k=int(gcfg.local_k),
        local_min_k=int(gcfg.local_min_k),
        uncertainty_c=float(gcfg.uncertainty_c),
        uncertainty_d=float(gcfg.uncertainty_d),
        bounds=z_bounds,
    )

    proj = _apply_rate_correlation_congruity(
        proj,
        profile=z_profile,
        mode=z_coherence_mode,
        ridge=0.09,
        alpha_map=Z_COHERENCE_ALPHA,
        max_delta_map=Z_COHERENCE_MAX_DELTA,
    )
    proj = _finalize_rate_percentiles(proj, metric_cols=RATE_METRICS, bounds=z_bounds)
    proj = _apply_z_tail_shape(
        proj,
        metric_cols=RATE_METRICS,
        z_bounds=z_bounds,
        strength=float(z_tail_strength),
    )
    proj = _finalize_rate_percentiles(proj, metric_cols=RATE_METRICS, bounds=z_bounds)
    proj = _from_z_space_projection(proj, metric_cols=RATE_METRICS, params=z_params)
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )

    proj = _apply_rate_correlation_congruity(
        proj,
        profile=coherence_profile,
        mode=coherence_mode,
        ridge=0.03,
        alpha_map=COHERENCE_ALPHA,
        max_delta_map=COHERENCE_MAX_DELTA,
    )
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )
    proj = _apply_metric_population_anchor(
        proj,
        hist=hist_raw,
        metric=anchor_metric,
        anchor_k=float(metric_anchor_k),
        mlb_flag_col="appeared_in_MLB_hist",
    )
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )
    proj = _apply_pitcher_kpi_skill_pull(
        proj,
        hist_raw=hist_raw,
        kpi=kpi_frame,
        source_season=source_season,
        pull_strength=float(kpi_pull_strength),
        min_ref_tbf=float(kpi_min_tbf),
        max_kpi_z=float(kpi_max_z),
    )
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )

    skill_bounds = _derive_empirical_skill_bounds(hist_raw, metrics=RATE_METRICS)
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=skill_bounds
    )
    proj = _apply_tail_swing_caps(proj, metric_cols=RATE_METRICS, bounds=skill_bounds)
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=skill_bounds
    )
    proj = _apply_recent_source_rate_retention(
        proj,
        hist_raw=hist_raw,
        source_season=source_season,
    )
    proj = _apply_league_component_environment(
        proj,
        target_babip=float(LEAGUE_TARGET_BABIP),
        target_hr_per_bbe=float(LEAGUE_TARGET_HR_PER_BBE),
    )
    proj = _apply_projected_er_per_ip_component_blend(
        proj,
        hist_actual=hist_actual,
        source_season=source_season,
        recent_weights=rw,
    )
    proj = _apply_marcel_games_started_games(
        proj,
        hist_raw=hist_raw,
        source_season=source_season,
        weights=rw,
    )
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )
    proj = _apply_league_ra9_environment(
        proj, hist_raw=hist_raw, source_season=source_season
    )
    proj = _finalize_rate_percentiles(
        proj, metric_cols=RATE_METRICS, bounds=RATE_BOUNDS
    )
    proj = _add_pitching_display_projections(proj)

    if not src_meta.empty:
        proj = proj.merge(src_meta, on="mlbid", how="left")
    if "appeared_in_MLB" not in proj.columns:
        proj["appeared_in_MLB"] = "F"
    proj["appeared_in_MLB"] = proj["appeared_in_MLB"].fillna("F").astype(str)
    proj["appeared_in_MLB"] = np.where(proj["appeared_in_MLB"] == "T", "T", "F")

    keep_cols = [
        "mlbid",
        "player_name",
        "source_season",
        "target_season",
        "projection_version",
        "age_used",
        "age_source",
        "batters_faced_agg",
        f"levels_played_{source_season}",
        "appeared_in_MLB",
        "team_abbreviations",
    ]
    for metric in RATE_METRICS:
        keep_cols.extend(
            [
                f"source_{metric}",
                f"n_eff_{metric}",
                f"{metric}_proj_p25",
                f"{metric}_proj_p50",
                f"{metric}_proj_p75",
                f"{metric}_proj_spread",
            ]
        )
    for metric in DISPLAY_PITCHING_BASES:
        keep_cols.extend(
            [
                f"{metric}_proj_p25",
                f"{metric}_proj_p50",
                f"{metric}_proj_p75",
                f"{metric}_proj_spread",
            ]
        )
    for metric in MARCEL_DISPLAY_BASES:
        keep_cols.extend(
            [
                f"{metric}_proj_p25",
                f"{metric}_proj_p50",
                f"{metric}_proj_p75",
                f"{metric}_proj_spread",
            ]
        )
    keep_cols.extend(
        ["league_ra9_source", "league_ra9_projected_pre_env", "league_ra9_env_scale"]
    )
    keep_cols.extend(["starter_share_marcel_proj_p50", "starter_flag_marcel_proj_p50"])
    keep_cols.extend(
        [
            c
            for c in proj.columns
            if c.endswith("_anchor_mu_mlbT") or c.endswith("_anchor_k")
        ]
    )
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in proj.columns]))
    final_proj = proj[keep_cols].copy()

    for frame in [final_proj, age_curves]:
        fcols = frame.select_dtypes(include=["float32", "float64", "Float64"]).columns
        if len(fcols) > 0:
            frame[fcols] = frame[fcols].round(6)

    out_projection_path.parent.mkdir(parents=True, exist_ok=True)
    out_age_curve_path.parent.mkdir(parents=True, exist_ok=True)
    final_proj.to_parquet(out_projection_path, index=False)
    age_curves.to_parquet(out_age_curve_path, index=False)
    return final_proj, age_curves


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pitching age curves and 2026 p25/p50/p75 rate projections from "
            "BP_pitching_single_source_mlb_eq_non_ar_delta.parquet in MLB-T-centered z-space, "
            "inverse-transform to raw rates, then apply inverse coherence."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("BP_pitching_single_source_mlb_eq_non_ar_delta.parquet"),
    )
    parser.add_argument(
        "--constants-path",
        type=Path,
        default=Path("BP_pitching_non_ar_constants_2015_2025.parquet"),
    )
    parser.add_argument(
        "--bp-pitching-path",
        type=Path,
        default=Path(
            "projection_outputs/bp_pitching_api/bp_pitching_table_with_level_id.parquet"
        ),
    )
    parser.add_argument(
        "--historical-raw-path",
        type=Path,
        default=Path("BP_pitching_equivalents_2015_2025_non_ar_delta.parquet"),
    )
    parser.add_argument(
        "--z-coherence-mode", choices=["direct", "inverse", "none"], default="direct"
    )
    parser.add_argument("--z-anchor-k", type=float, default=260.0)
    parser.add_argument("--z-tail-strength", type=float, default=0.50)
    parser.add_argument(
        "--anchor-metric", type=str, default="home_run_rate_mlb_eq_non_ar_delta"
    )
    parser.add_argument("--metric-anchor-k", type=float, default=220.0)
    parser.add_argument(
        "--coherence-mode", choices=["direct", "inverse", "none"], default="none"
    )
    parser.add_argument(
        "--out-projections",
        type=Path,
        default=Path("BP_pitching_rate_projections_2026_non_ar_post_inv_coh.parquet"),
    )
    parser.add_argument(
        "--out-age-curves",
        type=Path,
        default=Path(
            "BP_pitching_rate_age_curves_2015_2025_non_ar_post_inv_coh.parquet"
        ),
    )
    parser.add_argument("--metric-recency-weights-json", type=Path, default=None)
    parser.add_argument(
        "--recency-weights",
        type=float,
        nargs=3,
        default=list(MARCEL_RECENCY_WEIGHTS),
        metavar=("W0", "W1", "W2"),
        help="Global Marcel recency weights [most_recent, year_minus_1, year_minus_2].",
    )
    parser.add_argument("--uncertainty-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--default-k", type=float, default=200.0)
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument(
        "--z-sigma-min-ip",
        type=float,
        default=10.0,
        help="Minimum innings pitched required in MLB-T reference sample when estimating z-space sigma.",
    )
    parser.add_argument(
        "--era-mu",
        type=float,
        default=None,
        help="Optional ERA-scale (9*ER/IP) override for MLB-T z-space mu; converted internally to per-IP.",
    )
    parser.add_argument(
        "--era-sigma",
        type=float,
        default=None,
        help="Optional ERA-scale (9*ER/IP) override for MLB-T z-space sigma; converted internally to per-IP.",
    )
    parser.add_argument(
        "--pitcher-kpi-path",
        type=Path,
        default=Path("pitchers_regressed.csv"),
        help="Optional pitcher KPI source (csv/parquet) used to directionally pull BP rates.",
    )
    parser.add_argument(
        "--kpi-pull-strength",
        type=float,
        default=0.18,
        help="Global strength for KPI-based pull of projected pitching rates (0 disables).",
    )
    parser.add_argument(
        "--kpi-min-tbf",
        type=float,
        default=100.0,
        help="Minimum TBF/IP exposure used for MLB KPI reference moments.",
    )
    parser.add_argument(
        "--kpi-max-z",
        type=float,
        default=2.5,
        help="Absolute cap on KPI z-score contribution per feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metric_recency_weights = _load_metric_recency_weights(
        args.metric_recency_weights_json
    )
    proj, ages = build_bp_pitching_rate_projections_2026(
        input_path=args.input_path,
        constants_path=args.constants_path,
        bp_pitching_path=args.bp_pitching_path,
        historical_raw_path=args.historical_raw_path,
        z_coherence_mode=args.z_coherence_mode,
        z_anchor_k=float(args.z_anchor_k),
        anchor_metric=str(args.anchor_metric),
        metric_anchor_k=float(args.metric_anchor_k),
        z_tail_strength=float(args.z_tail_strength),
        coherence_mode=args.coherence_mode,
        out_projection_path=args.out_projections,
        out_age_curve_path=args.out_age_curves,
        uncertainty_draws=int(args.uncertainty_draws),
        seed=int(args.seed),
        recency_weights=tuple(float(x) for x in list(args.recency_weights)[:3]),
        metric_recency_weights=metric_recency_weights,
        default_k=float(args.default_k),
        k_scale=float(args.k_scale),
        z_sigma_min_ip=float(args.z_sigma_min_ip),
        era_mu=(None if args.era_mu is None else float(args.era_mu)),
        era_sigma=(None if args.era_sigma is None else float(args.era_sigma)),
        pitcher_kpi_path=args.pitcher_kpi_path,
        kpi_pull_strength=float(args.kpi_pull_strength),
        kpi_min_tbf=float(args.kpi_min_tbf),
        kpi_max_z=float(args.kpi_max_z),
    )
    print(f"Wrote {len(proj):,} rows to {args.out_projections}")
    print(f"Wrote {len(ages):,} rows to {args.out_age_curves}")


if __name__ == "__main__":
    main()
