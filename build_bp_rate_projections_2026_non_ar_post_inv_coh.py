from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections_v1.age import infer_and_impute_age
from projections_v1.config import GlobalConfig
from projections_v1.point_forecast import project_next_season
from projections_v1.uncertainty import apply_uncertainty_bands, build_transition_deltas


RATE_METRICS = [
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

RATE_BOUNDS = {m: (0.0, 1.0) for m in RATE_METRICS}

SKILL_DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": (0.42, 0.80),
    "strikeout_rate_mlb_eq_non_ar_delta": (0.06, 0.42),
    "walk_rate_mlb_eq_non_ar_delta": (0.02, 0.22),
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": (0.00, 0.05),
    "singles_rate_bbe_mlb_eq_non_ar_delta": (0.12, 0.34),
    "doubles_rate_bbe_mlb_eq_non_ar_delta": (0.01, 0.12),
    "triples_rate_bbe_mlb_eq_non_ar_delta": (0.00, 0.045),
    "home_run_rate_bbe_mlb_eq_non_ar_delta": (0.000, 0.15),
    "sac_fly_rate_bbe_mlb_eq_non_ar_delta": (0.00, 0.09),
    "sac_hit_rate_bbe_mlb_eq_non_ar_delta": (0.00, 0.08),
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": (0.00, 0.11),
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": (0.00, 0.45),
    "stolen_base_success_rate_mlb_eq_non_ar_delta": (0.45, 0.95),
    "xbh_from_h_rate_mlb_eq_non_ar_delta": (0.20, 0.72),
    "runs_rate_mlb_eq_non_ar_delta": (0.03, 0.20),
    "rbi_rate_mlb_eq_non_ar_delta": (0.03, 0.20),
    "babip_recalc_rate_mlb_eq_non_ar_delta": (0.22, 0.38),
}

MAX_SKILL_TAIL_SWING: dict[str, float] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": 0.045,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.065,
    "walk_rate_mlb_eq_non_ar_delta": 0.055,
    "hit_by_pitch_rate_mlb_eq_non_ar_delta": 0.012,
    "singles_rate_bbe_mlb_eq_non_ar_delta": 0.08,
    "doubles_rate_bbe_mlb_eq_non_ar_delta": 0.05,
    "triples_rate_bbe_mlb_eq_non_ar_delta": 0.01,
    "home_run_rate_bbe_mlb_eq_non_ar_delta": 0.075,
    "sac_fly_rate_bbe_mlb_eq_non_ar_delta": 0.015,
    "sac_hit_rate_bbe_mlb_eq_non_ar_delta": 0.015,
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta": 0.025,
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta": 0.070,
    "stolen_base_success_rate_mlb_eq_non_ar_delta": 0.070,
    "xbh_from_h_rate_mlb_eq_non_ar_delta": 0.050,
    "runs_rate_mlb_eq_non_ar_delta": 0.035,
    "rbi_rate_mlb_eq_non_ar_delta": 0.035,
    "babip_recalc_rate_mlb_eq_non_ar_delta": 0.035,
}

COHERENCE_METRIC_MAP: list[tuple[str, str]] = [
    ("batted_ball_rate_mlb_eq_non_ar_delta", "batted_ball_rate"),
    ("strikeout_rate_mlb_eq_non_ar_delta", "strikeout_rate"),
    ("walk_rate_mlb_eq_non_ar_delta", "walk_rate"),
    ("babip_recalc_rate_mlb_eq_non_ar_delta", "babip_recalc_rate"),
    ("singles_rate_bbe_mlb_eq_non_ar_delta", "singles_rate_bbe"),
    ("doubles_rate_bbe_mlb_eq_non_ar_delta", "doubles_rate_bbe"),
    ("triples_rate_bbe_mlb_eq_non_ar_delta", "triples_rate_bbe"),
]

COHERENCE_ALPHA: dict[str, float] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": 0.22,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.00,
    "walk_rate_mlb_eq_non_ar_delta": 0.08,
    "babip_recalc_rate_mlb_eq_non_ar_delta": 0.18,
    "singles_rate_bbe_mlb_eq_non_ar_delta": 0.18,
    "doubles_rate_bbe_mlb_eq_non_ar_delta": 0.14,
    "triples_rate_bbe_mlb_eq_non_ar_delta": 0.08,
}

COHERENCE_MAX_DELTA_PCT_PTS: dict[str, float] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": 0.025,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.020,
    "walk_rate_mlb_eq_non_ar_delta": 0.015,
    "babip_recalc_rate_mlb_eq_non_ar_delta": 0.015,
    "singles_rate_bbe_mlb_eq_non_ar_delta": 0.015,
    "doubles_rate_bbe_mlb_eq_non_ar_delta": 0.012,
    "triples_rate_bbe_mlb_eq_non_ar_delta": 0.004,
}

Z_COHERENCE_ALPHA: dict[str, float] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": 0.18,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.00,
    "walk_rate_mlb_eq_non_ar_delta": 0.12,
    "babip_recalc_rate_mlb_eq_non_ar_delta": 0.14,
    "singles_rate_bbe_mlb_eq_non_ar_delta": 0.14,
    "doubles_rate_bbe_mlb_eq_non_ar_delta": 0.10,
    "triples_rate_bbe_mlb_eq_non_ar_delta": 0.06,
}

Z_COHERENCE_MAX_DELTA: dict[str, float] = {
    "batted_ball_rate_mlb_eq_non_ar_delta": 0.45,
    "strikeout_rate_mlb_eq_non_ar_delta": 0.45,
    "walk_rate_mlb_eq_non_ar_delta": 0.35,
    "babip_recalc_rate_mlb_eq_non_ar_delta": 0.35,
    "singles_rate_bbe_mlb_eq_non_ar_delta": 0.35,
    "doubles_rate_bbe_mlb_eq_non_ar_delta": 0.30,
    "triples_rate_bbe_mlb_eq_non_ar_delta": 0.25,
}

NO_COHERENCE_UNIT_CLIP_METRICS: set[str] = {
    "home_run_rate_bbe_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta",
}

NO_SKILL_BOUND_CLIP_METRICS: set[str] = {
    "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
    "stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta",
}


def _build_level1_correlation_profile(
    historical_ar_path: Path,
) -> dict[str, Any] | None:
    if not historical_ar_path.exists():
        return None
    need = ["bp_level_id"] + [h for _, h in COHERENCE_METRIC_MAP]
    hist = pd.read_parquet(historical_ar_path)
    if not set(need).issubset(hist.columns):
        return None

    work = hist[need].copy()
    work["bp_level_id"] = pd.to_numeric(work["bp_level_id"], errors="coerce")
    work = work[work["bp_level_id"] == 1].copy()
    if work.empty:
        return None

    proj_metrics = [p for p, _ in COHERENCE_METRIC_MAP]
    hist_cols = [h for _, h in COHERENCE_METRIC_MAP]
    x = work[hist_cols].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(x) < 200:
        return None

    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0).replace(0.0, np.nan)
    keep_mask = sd.notna() & np.isfinite(sd) & (sd > 1e-8)
    if int(keep_mask.sum()) < 4:
        return None

    kept_hist_cols = [c for c in hist_cols if bool(keep_mask.loc[c])]
    kept_proj_metrics = [
        proj_metrics[i] for i, c in enumerate(hist_cols) if c in set(kept_hist_cols)
    ]
    xk = x[kept_hist_cols].copy()
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


def _apply_rate_correlation_congruity(
    proj: pd.DataFrame,
    *,
    profile: dict[str, Any] | None,
    mode: str = "direct",
    ridge: float = 0.06,
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
    delta_lookup = max_delta_map or COHERENCE_MAX_DELTA_PCT_PTS
    corr = np.asarray(profile.get("corr"))
    if len(metrics) < 4 or corr.size == 0:
        return out

    k = len(metrics)
    for pct in [20, 50, 80]:
        cols = [f"{m}_proj_p{pct}" for m in metrics]
        if not all(c in out.columns for c in cols):
            continue

        mat = np.column_stack(
            [pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float) for c in cols]
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
            # direct: pull toward historically coherent conditional expectation
            # inverse: push away by the same modeled amount (for sensitivity testing)
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


def _safe_divide_series(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    mask = n.notna() & d.notna() & np.isfinite(n) & np.isfinite(d) & (d > 0)
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def _add_hist_ops_from_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    direct_ops_col = "OPS_mlb_eq_non_ar_delta"
    if direct_ops_col in out.columns:
        out["ops_est"] = pd.to_numeric(out[direct_ops_col], errors="coerce")
        return out
    need = [
        "plate_appearances_agg",
        "batted_ball_rate_mlb_eq_non_ar_delta",
        "walk_rate_mlb_eq_non_ar_delta",
        "hit_by_pitch_rate_mlb_eq_non_ar_delta",
        "singles_rate_bbe_mlb_eq_non_ar_delta",
        "doubles_rate_bbe_mlb_eq_non_ar_delta",
        "triples_rate_bbe_mlb_eq_non_ar_delta",
        "home_run_rate_bbe_mlb_eq_non_ar_delta",
        "sac_fly_rate_bbe_mlb_eq_non_ar_delta",
        "sac_hit_rate_bbe_mlb_eq_non_ar_delta",
    ]
    if not set(need).issubset(set(out.columns)):
        out["ops_est"] = np.nan
        return out

    pa = pd.to_numeric(out["plate_appearances_agg"], errors="coerce")
    bbe = pa * pd.to_numeric(
        out["batted_ball_rate_mlb_eq_non_ar_delta"], errors="coerce"
    )
    bb = pa * pd.to_numeric(out["walk_rate_mlb_eq_non_ar_delta"], errors="coerce")
    hbp = pa * pd.to_numeric(
        out["hit_by_pitch_rate_mlb_eq_non_ar_delta"], errors="coerce"
    )
    sf = bbe * pd.to_numeric(
        out["sac_fly_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )
    sh = bbe * pd.to_numeric(
        out["sac_hit_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )

    one_b = bbe * pd.to_numeric(
        out["singles_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )
    two_b = bbe * pd.to_numeric(
        out["doubles_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )
    three_b = bbe * pd.to_numeric(
        out["triples_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )
    hr = bbe * pd.to_numeric(
        out["home_run_rate_bbe_mlb_eq_non_ar_delta"], errors="coerce"
    )
    h = (one_b + two_b + three_b + hr).clip(lower=0.0)
    ab = (pa - (bb + hbp + sf + sh)).clip(lower=0.0)
    tb = (one_b + 2.0 * two_b + 3.0 * three_b + 4.0 * hr).clip(lower=0.0)

    obp = _safe_divide_series(h + bb + hbp, pa)
    slg = _safe_divide_series(tb, ab)
    out["ops_est"] = obp + slg
    return out


def _build_mlb_appearance_by_season(bp_hitting_path: Path) -> pd.DataFrame:
    cols = ["mlbid", "season", "bp_level_id", "plate_appearances_agg"]
    out_cols = ["mlbid", "season", "appeared_in_MLB_hist"]
    if not bp_hitting_path.exists():
        return pd.DataFrame(columns=out_cols)
    bp = pd.read_parquet(bp_hitting_path)
    if not set(cols).issubset(bp.columns):
        return pd.DataFrame(columns=out_cols)

    work = bp[cols].copy()
    work["mlbid"] = pd.to_numeric(work["mlbid"], errors="coerce")
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["bp_level_id"] = pd.to_numeric(work["bp_level_id"], errors="coerce")
    work["plate_appearances_agg"] = pd.to_numeric(
        work["plate_appearances_agg"], errors="coerce"
    )
    work = work[
        work["mlbid"].notna() & work["season"].notna() & work["bp_level_id"].notna()
    ].copy()
    work["mlbid"] = work["mlbid"].astype("int64")
    work["season"] = work["season"].astype("int64")
    work["appeared_flag"] = np.where(
        (work["bp_level_id"] == 1) & (work["plate_appearances_agg"].fillna(0.0) > 0.0),
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
) -> dict[str, tuple[float, float]]:
    work = df.copy()
    if flag_col in work.columns:
        flag = pd.to_numeric(work[flag_col], errors="coerce").fillna(0)
        ref = work[flag == 1].copy()
        if len(ref) < 500:
            ref = work
    else:
        ref = work

    params: dict[str, tuple[float, float]] = {}
    for metric in metric_cols:
        s = (
            pd.to_numeric(ref.get(metric), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(s) < 2:
            params[metric] = (0.0, 1.0)
            continue
        mu = float(s.mean())
        sigma = float(s.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 1e-8:
            sigma = 1.0
        params[metric] = (mu, sigma)
    return params


def _to_z_space(
    df: pd.DataFrame,
    *,
    metric_cols: list[str],
    params: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = df.copy()
    for metric in metric_cols:
        if metric not in out.columns:
            continue
        mu, sigma = params.get(metric, (0.0, 1.0))
        vals = pd.to_numeric(out[metric], errors="coerce")
        out[metric] = (vals - float(mu)) / float(sigma)
    return out


def _from_z_space_projection(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    params: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = proj.copy()
    suffixes = ["source", "proj_p20", "proj_p25", "proj_p50", "proj_p75", "proj_p80"]
    for metric in metric_cols:
        mu, sigma = params.get(metric, (0.0, 1.0))
        for suf in suffixes:
            col = f"{metric}_{suf}" if suf != "source" else f"source_{metric}"
            if col not in out.columns:
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = (vals * float(sigma)) + float(mu)
    return out


def _derive_z_bounds(
    df_z: pd.DataFrame,
    *,
    metric_cols: list[str],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for metric in metric_cols:
        s = (
            pd.to_numeric(df_z.get(metric), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(s) < 100:
            bounds[metric] = (-4.5, 4.5)
            continue
        lo = float(np.nanquantile(s, 0.001))
        hi = float(np.nanquantile(s, 0.999))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            bounds[metric] = (-4.5, 4.5)
            continue
        span = max(hi - lo, 1e-6)
        lo = lo - (0.20 * span)
        hi = hi + (0.20 * span)
        lo = float(np.clip(lo, -8.0, -0.25))
        hi = float(np.clip(hi, 0.25, 8.0))
        if hi <= lo:
            lo, hi = -4.5, 4.5
        bounds[metric] = (lo, hi)
    return bounds


def _build_metric_correlation_profile(
    df: pd.DataFrame,
    *,
    metric_cols: list[str],
) -> dict[str, Any] | None:
    if df.empty:
        return None
    x = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(x) < 200:
        return None
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0).replace(0.0, np.nan)
    keep = sd.notna() & np.isfinite(sd) & (sd > 1e-8)
    if int(keep.sum()) < 4:
        return None
    kept_cols = [c for c in metric_cols if bool(keep.loc[c])]
    xk = x[kept_cols].copy()
    muk = mu[kept_cols]
    sdk = sd[kept_cols]
    zk = (xk - muk) / sdk
    corr = np.corrcoef(zk.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return {
        "proj_metrics": kept_cols,
        "means": {kept_cols[i]: float(muk.iloc[i]) for i in range(len(kept_cols))},
        "stds": {kept_cols[i]: float(sdk.iloc[i]) for i in range(len(kept_cols))},
        "corr": corr,
    }


def _apply_z_anchor_recenter(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    anchor_k: float = 260.0,
) -> pd.DataFrame:
    out = proj.copy()
    k = float(max(anchor_k, 0.0))
    if k <= 0.0:
        return out
    for metric in metric_cols:
        p50_col = f"{metric}_proj_p50"
        n_col = f"n_eff_{metric}"
        if p50_col not in out.columns:
            continue
        p50 = pd.to_numeric(out[p50_col], errors="coerce")
        n_eff = (
            pd.to_numeric(out.get(n_col, np.nan), errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        # Explicit z-space re-centering toward 0 for lower-information players.
        w = (n_eff / (n_eff + k)).clip(lower=0.0, upper=1.0)
        out[p50_col] = p50 * w
    return out


def _apply_hr_population_anchor(
    proj: pd.DataFrame,
    *,
    hist: pd.DataFrame,
    metric: str = "home_run_rate_bbe_mlb_eq_non_ar_delta",
    anchor_k: float = 240.0,
    mlb_flag_col: str = "appeared_in_MLB_hist",
) -> pd.DataFrame:
    out = proj.copy()
    k = float(max(anchor_k, 0.0))
    if k <= 0.0:
        return out

    c20 = f"{metric}_proj_p20"
    c50 = f"{metric}_proj_p50"
    c80 = f"{metric}_proj_p80"
    n_col = f"n_eff_{metric}"
    req = [c20, c50, c80]
    if not all(c in out.columns for c in req):
        return out

    h = hist.copy()
    if mlb_flag_col in h.columns:
        mlb = h[pd.to_numeric(h[mlb_flag_col], errors="coerce").fillna(0) == 1].copy()
        if len(mlb) >= 200:
            h = mlb
    mu = float(pd.to_numeric(h.get(metric), errors="coerce").mean())
    if not np.isfinite(mu):
        return out

    p20 = pd.to_numeric(out[c20], errors="coerce")
    p50 = pd.to_numeric(out[c50], errors="coerce")
    p80 = pd.to_numeric(out[c80], errors="coerce")
    n_eff = (
        pd.to_numeric(out.get(n_col, np.nan), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    w = (n_eff / (n_eff + k)).clip(lower=0.0, upper=1.0)
    p50_new = (p50 * w) + (mu * (1.0 - w))
    shift = p50_new - p50
    out[c50] = p50_new
    out[c20] = p20 + shift
    out[c80] = p80 + shift
    out["HR_anchor_mu_mlbT"] = mu
    out["HR_anchor_k"] = k
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
        c20 = f"{metric}_proj_p20"
        c50 = f"{metric}_proj_p50"
        c80 = f"{metric}_proj_p80"
        n_col = f"n_eff_{metric}"
        if not all(c in out.columns for c in [c20, c50, c80]):
            continue

        lo, hi = z_bounds.get(metric, (-4.5, 4.5))
        p20 = pd.to_numeric(out[c20], errors="coerce")
        p50 = pd.to_numeric(out[c50], errors="coerce")
        p80 = pd.to_numeric(out[c80], errors="coerce")
        n_eff = (
            pd.to_numeric(out.get(n_col, np.nan), errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

        down = (p50 - p20).clip(lower=0.0)
        up = (p80 - p50).clip(lower=0.0)
        info = (n_eff / (n_eff + 280.0)).clip(lower=0.0, upper=1.0)
        outlier = np.abs(p50).clip(lower=0.0, upper=4.0)
        keep_base = (0.22 + (0.42 * info)) * (1.0 / (1.0 + (0.20 * outlier)))
        keep_base = np.clip(keep_base, 0.10, 0.70)
        # Compression amount is (1 - keep_base). Scale that amount by strength:
        # strength=1.0 -> prior behavior, strength=0.5 -> ~half as much compression.
        keep = 1.0 - ((1.0 - keep_base) * s)
        keep = np.clip(keep, 0.10, 1.0)

        new_down = down * keep
        new_up = up * keep
        p20n = (p50 - new_down).clip(lower=lo, upper=hi)
        p80n = (p50 + new_up).clip(lower=lo, upper=hi)
        p50n = p50.clip(lower=lo, upper=hi)
        p50n = np.maximum(p50n, p20n)
        p80n = np.maximum(p80n, p50n)
        out[c20] = p20n
        out[c50] = p50n
        out[c80] = p80n
        out[f"{metric}_proj_spread"] = p80n - p20n
    return out


def _apply_ops_sensitive_pa_projections(
    proj: pd.DataFrame,
    *,
    hist: pd.DataFrame,
    source_season: int,
    recency_weights: list[float],
    ops_pivot: float = 0.650,
) -> pd.DataFrame:
    out = proj.copy()
    if out.empty:
        return out

    req_hist = {
        "mlbid",
        "season",
        "plate_appearances_agg",
        "ops_est",
        "appeared_in_MLB_hist",
    }
    if not req_hist.issubset(hist.columns):
        out["PA_proj_p20"] = 600.0
        out["PA_proj_p50"] = 600.0
        out["PA_proj_p80"] = 600.0
        out["PA_proj_spread"] = 0.0
        return out

    h = hist[list(req_hist)].copy()
    h["mlbid"] = pd.to_numeric(h["mlbid"], errors="coerce")
    h["season"] = pd.to_numeric(h["season"], errors="coerce")
    h["plate_appearances_agg"] = pd.to_numeric(
        h["plate_appearances_agg"], errors="coerce"
    )
    h["ops_est"] = pd.to_numeric(h["ops_est"], errors="coerce")
    h["appeared_in_MLB_hist"] = (
        pd.to_numeric(h["appeared_in_MLB_hist"], errors="coerce").fillna(0).astype(int)
    )
    h = h[h["mlbid"].notna() & h["season"].notna()].copy()
    h["mlbid"] = h["mlbid"].astype("int64")
    h["season"] = h["season"].astype("int64")

    mlb_only = h[h["appeared_in_MLB_hist"] == 1].copy()
    pa_avg_by_season = (
        mlb_only.groupby("season", dropna=False)["plate_appearances_agg"]
        .mean()
        .to_dict()
    )
    ops_avg_by_season = (
        mlb_only.groupby("season", dropna=False)["ops_est"].mean().to_dict()
    )

    global_pa_avg = float(
        pd.to_numeric(mlb_only["plate_appearances_agg"], errors="coerce").mean()
    )
    global_ops_avg = float(pd.to_numeric(mlb_only["ops_est"], errors="coerce").mean())
    if not np.isfinite(global_pa_avg):
        global_pa_avg = 420.0
    if not np.isfinite(global_ops_avg):
        global_ops_avg = float(ops_pivot)

    pa_lookup = h.set_index(["mlbid", "season"])["plate_appearances_agg"].to_dict()
    ops_lookup = h.set_index(["mlbid", "season"])["ops_est"].to_dict()

    ids = (
        pd.to_numeric(out["mlbid"], errors="coerce").fillna(-1).astype("int64").tolist()
    )
    weights = [float(w) for w in recency_weights]
    if len(weights) < 3:
        weights = [5.0, 4.0, 3.0]
    weights = weights[:3]
    wsum = float(sum(weights))
    if wsum <= 0:
        weights = [5.0, 4.0, 3.0]
        wsum = 12.0

    pa20_list: list[float] = []
    pa50_list: list[float] = []
    pa80_list: list[float] = []
    pa_spread_list: list[float] = []
    pa_marcel_list: list[float] = []
    ops_perf_list: list[float] = []
    pa_adj_pct_list: list[float] = []
    pa_adj_abs_list: list[float] = []

    def _safe_col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype="float64")

    k50_series = _safe_col("strikeout_rate_mlb_eq_non_ar_delta_proj_p50")
    bb50_series = _safe_col("walk_rate_mlb_eq_non_ar_delta_proj_p50")
    hbp50_series = _safe_col("hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_p50")
    bbe50_series = _safe_col("batted_ball_rate_mlb_eq_non_ar_delta_proj_p50")
    hr_bbe50_series = _safe_col("home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p50")
    sf_bbe50_series = _safe_col("sac_fly_rate_bbe_mlb_eq_non_ar_delta_proj_p50")
    sh_bbe50_series = _safe_col("sac_hit_rate_bbe_mlb_eq_non_ar_delta_proj_p50")
    babip50_series = _safe_col("babip_recalc_rate_mlb_eq_non_ar_delta_proj_p50")

    obp_skill_list: list[float] = []
    k_skill_list: list[float] = []
    pa_risk_factor_list: list[float] = []

    for row_idx, pid in enumerate(ids):
        lag_pa_vals: list[float] = []
        lag_ops_vals: list[float] = []
        for lag in range(3):
            season_i = int(source_season - lag)
            pa_val = pd.to_numeric(
                pa_lookup.get((pid, season_i), np.nan), errors="coerce"
            )
            if not np.isfinite(pa_val):
                pa_val = float(
                    pd.to_numeric(pa_avg_by_season.get(season_i), errors="coerce")
                )
            if not np.isfinite(pa_val):
                pa_val = global_pa_avg

            ops_val = pd.to_numeric(
                ops_lookup.get((pid, season_i), np.nan), errors="coerce"
            )
            if not np.isfinite(ops_val):
                ops_val = float(
                    pd.to_numeric(ops_avg_by_season.get(season_i), errors="coerce")
                )
            if not np.isfinite(ops_val):
                ops_val = global_ops_avg

            lag_pa_vals.append(float(max(0.0, pa_val)))
            lag_ops_vals.append(float(np.clip(ops_val, 0.300, 1.500)))

        pa_marcel = float(np.dot(lag_pa_vals, weights) / wsum)
        ops_perf = float(np.dot(lag_ops_vals, weights) / wsum)
        diff = float(ops_perf - float(ops_pivot))
        pos = max(0.0, diff)
        neg = max(0.0, -diff)

        # PA flow: Marcel-like PA base first, then apply bounded performance adjustment.
        # This keeps playing-time history as the primary driver while still reflecting results.
        adj_curve = 0.24 * np.tanh(diff / 0.110)
        adj_tail = (
            (np.sign(diff) * (abs(diff) ** 1.40) * 1.20) if np.isfinite(diff) else 0.0
        )
        pa_adj_pct = float(np.clip(adj_curve + adj_tail, -0.35, 0.35))
        pa_adj_abs = float(np.clip(pa_marcel * pa_adj_pct, -220.0, 220.0))
        pa50 = float(np.clip(pa_marcel + pa_adj_abs, 30.0, 705.0))

        k50 = float(pd.to_numeric(k50_series.iloc[row_idx], errors="coerce"))
        bb50 = float(pd.to_numeric(bb50_series.iloc[row_idx], errors="coerce"))
        hbp50 = float(pd.to_numeric(hbp50_series.iloc[row_idx], errors="coerce"))
        bbe50 = float(pd.to_numeric(bbe50_series.iloc[row_idx], errors="coerce"))
        hr_bbe50 = float(pd.to_numeric(hr_bbe50_series.iloc[row_idx], errors="coerce"))
        sf_bbe50 = float(pd.to_numeric(sf_bbe50_series.iloc[row_idx], errors="coerce"))
        sh_bbe50 = float(pd.to_numeric(sh_bbe50_series.iloc[row_idx], errors="coerce"))
        babip50 = float(pd.to_numeric(babip50_series.iloc[row_idx], errors="coerce"))

        k50 = float(np.clip(k50, 0.0, 0.70)) if np.isfinite(k50) else np.nan
        bb50 = float(np.clip(bb50, 0.0, 0.40)) if np.isfinite(bb50) else np.nan
        hbp50 = float(np.clip(hbp50, 0.0, 0.12)) if np.isfinite(hbp50) else np.nan
        bbe50 = float(np.clip(bbe50, 0.0, 0.95)) if np.isfinite(bbe50) else np.nan
        hr_bbe50 = (
            float(np.clip(hr_bbe50, 0.0, 0.35)) if np.isfinite(hr_bbe50) else np.nan
        )
        sf_bbe50 = (
            float(np.clip(sf_bbe50, 0.0, 0.20)) if np.isfinite(sf_bbe50) else np.nan
        )
        sh_bbe50 = (
            float(np.clip(sh_bbe50, 0.0, 0.20)) if np.isfinite(sh_bbe50) else np.nan
        )
        babip50 = float(np.clip(babip50, 0.0, 0.60)) if np.isfinite(babip50) else np.nan

        obp_skill = np.nan
        if all(
            np.isfinite(v)
            for v in [k50, bb50, hbp50, bbe50, hr_bbe50, sf_bbe50, sh_bbe50, babip50]
        ):
            hr_pa = bbe50 * hr_bbe50
            sf_pa = bbe50 * sf_bbe50
            sh_pa = bbe50 * sh_bbe50
            ab_pa = 1.0 - (bb50 + hbp50 + sf_pa + sh_pa)
            bip_pa = ab_pa - k50 - hr_pa + sf_pa
            bip_pa = float(max(0.0, bip_pa))
            hit_pa = hr_pa + (babip50 * bip_pa)
            obp_skill = float(np.clip(hit_pa + bb50 + hbp50, 0.150, 0.550))

        if np.isfinite(obp_skill) and np.isfinite(k50):
            obp_term = float(_sigmoid((obp_skill - 0.305) / 0.020))
            k_term = float(_sigmoid((0.33 - k50) / 0.035))
            pa_risk_factor = float(
                np.clip(0.78 + (0.30 * obp_term * k_term), 0.72, 1.08)
            )
        else:
            pa_risk_factor = 1.0

        pa50 = float(np.clip(pa50 * pa_risk_factor, 30.0, 705.0))

        down = float(np.clip(0.24 + (0.75 * neg) - (0.30 * pos), 0.10, 0.55))
        up = float(np.clip(0.24 + (0.70 * pos) - (0.25 * neg), 0.10, 0.60))
        pa20 = float(np.clip(pa50 * (1.0 - down), 0.0, pa50))
        pa80 = float(np.clip(pa50 * (1.0 + up), pa50, 750.0))

        pa_marcel_list.append(pa_marcel)
        ops_perf_list.append(ops_perf)
        pa_adj_pct_list.append(pa_adj_pct)
        pa_adj_abs_list.append(pa_adj_abs)
        obp_skill_list.append(obp_skill)
        k_skill_list.append(k50)
        pa_risk_factor_list.append(pa_risk_factor)
        pa20_list.append(pa20)
        pa50_list.append(pa50)
        pa80_list.append(pa80)
        pa_spread_list.append(pa80 - pa20)

    out["PA_marcel_base"] = pa_marcel_list
    out["OPS_perf_anchor"] = ops_perf_list
    out["PA_perf_adj_pct"] = pa_adj_pct_list
    out["PA_perf_adj_abs"] = pa_adj_abs_list
    out["OBP_skill_est_p50"] = obp_skill_list
    out["K_skill_est_p50"] = k_skill_list
    out["PA_risk_factor"] = pa_risk_factor_list
    out["PA_proj_p20"] = pa20_list
    out["PA_proj_p50"] = pa50_list
    out["PA_proj_p80"] = pa80_list
    out["PA_proj_spread"] = pa_spread_list
    return out


def _sigmoid(x: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


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
        mlb = work[
            pd.to_numeric(work["appeared_in_MLB_hist"], errors="coerce").fillna(0) == 1
        ].copy()
        if len(mlb) >= 500:
            work = mlb

    for metric in metrics:
        d_lo, d_hi = SKILL_DOMAIN_BOUNDS.get(metric, (0.0, 1.0))
        s = (
            pd.to_numeric(work.get(metric), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(s) < 200:
            bounds[metric] = (float(d_lo), float(d_hi))
            continue
        lo = float(np.nanquantile(s, q_lo))
        hi = float(np.nanquantile(s, q_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            bounds[metric] = (float(d_lo), float(d_hi))
            continue
        span = max(hi - lo, 1e-6)
        lo = lo - (0.20 * span)
        hi = hi + (0.20 * span)
        lo = float(np.clip(lo, d_lo, d_hi))
        hi = float(np.clip(hi, lo + 1e-6, d_hi))
        bounds[metric] = (lo, hi)
    return bounds


def _finalize_rate_percentiles(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = proj.copy()
    for metric in metric_cols:
        c20 = f"{metric}_proj_p20"
        c50 = f"{metric}_proj_p50"
        c80 = f"{metric}_proj_p80"
        if not all(c in out.columns for c in [c20, c50, c80]):
            continue
        lo, hi = bounds.get(metric, (0.0, 1.0))
        p20 = pd.to_numeric(out[c20], errors="coerce").clip(lower=lo, upper=hi)
        p50 = pd.to_numeric(out[c50], errors="coerce").clip(lower=lo, upper=hi)
        p80 = pd.to_numeric(out[c80], errors="coerce").clip(lower=lo, upper=hi)
        p50 = np.maximum(p50, p20)
        p80 = np.maximum(p80, p50)
        out[c20] = p20
        out[c50] = p50
        out[c80] = p80
        out[f"{metric}_proj_spread"] = p80 - p20
    return out


def _apply_sigmoid_skill_shape(
    proj: pd.DataFrame,
    *,
    metric_cols: list[str],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = proj.copy()
    req_pa = {"PA_proj_p20", "PA_proj_p50", "PA_proj_p80"}
    if not req_pa.issubset(out.columns):
        return out

    pa20 = pd.to_numeric(out["PA_proj_p20"], errors="coerce").clip(lower=0.0)
    pa50 = pd.to_numeric(out["PA_proj_p50"], errors="coerce").clip(lower=1.0)
    pa80 = pd.to_numeric(out["PA_proj_p80"], errors="coerce").clip(lower=0.0)
    pa_dn = ((pa50 - pa20) / pa50).clip(lower=0.0, upper=1.5)
    pa_up = ((pa80 - pa50) / pa50).clip(lower=0.0, upper=1.5)

    # S-curve gates: middle percentiles stay tight, tail outcomes open gradually with PA tail width.
    keep_dn = 0.18 + (0.48 * _sigmoid((pa_dn - 0.22) / 0.06))
    keep_up = 0.18 + (0.48 * _sigmoid((pa_up - 0.22) / 0.06))
    keep_dn = np.clip(keep_dn, 0.18, 0.62)
    keep_up = np.clip(keep_up, 0.18, 0.62)
    # Absolute PA level gates: elite skill tails should mostly appear with larger PA outcomes.
    pa50_gate = _sigmoid((pa50 - 500.0) / 90.0)
    pa80_gate = _sigmoid((pa80 - 540.0) / 85.0)
    level_mult = np.clip(0.55 + (0.45 * pa50_gate), 0.55, 1.0)
    up_mult = np.clip(0.45 + (0.55 * pa80_gate), 0.45, 1.0)
    keep_dn = np.clip(keep_dn * level_mult, 0.12, 0.62)
    keep_up = np.clip(keep_up * level_mult * up_mult, 0.10, 0.62)

    for metric in metric_cols:
        c20 = f"{metric}_proj_p20"
        c50 = f"{metric}_proj_p50"
        c80 = f"{metric}_proj_p80"
        if not all(c in out.columns for c in [c20, c50, c80]):
            continue

        lo, hi = bounds.get(metric, (0.0, 1.0))
        p20 = pd.to_numeric(out[c20], errors="coerce")
        p50 = pd.to_numeric(out[c50], errors="coerce")
        p80 = pd.to_numeric(out[c80], errors="coerce")

        down = (p50 - p20).clip(lower=0.0)
        up = (p80 - p50).clip(lower=0.0)
        cap = float(MAX_SKILL_TAIL_SWING.get(metric, 0.04))
        span = max(float(hi - lo), 1e-6)
        cap = float(min(cap, 0.45 * span))

        new_down = np.minimum(down * keep_dn, cap)
        new_up = np.minimum(up * keep_up, cap)

        p20n = (p50 - new_down).clip(lower=lo, upper=hi)
        p80n = (p50 + new_up).clip(lower=lo, upper=hi)
        p50n = p50.clip(lower=lo, upper=hi)

        p50n = np.maximum(p50n, p20n)
        p80n = np.maximum(p80n, p50n)
        out[c20] = p20n
        out[c50] = p50n
        out[c80] = p80n
        out[f"{metric}_proj_spread"] = p80n - p20n
    return out


def _enforce_component_rate_consistency(proj: pd.DataFrame) -> pd.DataFrame:
    out = proj.copy()
    for pct in [20, 50, 80]:
        cols = {
            "s1b": f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "s2b": f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "s3b": f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "hr": f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "sf": f"sac_fly_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
            "sh": f"sac_hit_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}",
        }
        if not all(c in out.columns for c in cols.values()):
            continue
        s1b = pd.to_numeric(out[cols["s1b"]], errors="coerce")
        s2b = pd.to_numeric(out[cols["s2b"]], errors="coerce")
        s3b = pd.to_numeric(out[cols["s3b"]], errors="coerce")
        hr = pd.to_numeric(out[cols["hr"]], errors="coerce")
        sf = pd.to_numeric(out[cols["sf"]], errors="coerce")
        sh = pd.to_numeric(out[cols["sh"]], errors="coerce")

        total = (s1b + s2b + s3b + hr + sf + sh).replace([np.inf, -np.inf], np.nan)
        # Keep batted-ball outcome rates in a plausible aggregate band.
        cap_total = 0.58
        scale = pd.Series(1.0, index=out.index, dtype="float64")
        mask = total.notna() & (total > cap_total)
        scale.loc[mask] = cap_total / total.loc[mask]

        out[cols["s1b"]] = (s1b * scale).clip(lower=0.0, upper=1.0)
        out[cols["s2b"]] = (s2b * scale).clip(lower=0.0, upper=1.0)
        out[cols["s3b"]] = (s3b * scale).clip(lower=0.0, upper=1.0)
        out[cols["hr"]] = (hr * scale).clip(lower=0.0, upper=1.0)
        out[cols["sf"]] = (sf * scale).clip(lower=0.0, upper=1.0)
        out[cols["sh"]] = (sh * scale).clip(lower=0.0, upper=1.0)
    return out


def _build_k_overrides(constants_path: Path, k_scale: float = 1.0) -> dict[str, float]:
    if not constants_path.exists():
        return {}
    c = pd.read_parquet(constants_path)
    needed = {"stat", "bp_level_id", "K"}
    if not needed.issubset(c.columns):
        return {}
    c = c.copy()
    c["bp_level_id"] = pd.to_numeric(c["bp_level_id"], errors="coerce")
    c["K"] = pd.to_numeric(c["K"], errors="coerce")
    c = c[c["bp_level_id"] == 1]
    if c.empty:
        return {}

    med = c.groupby("stat", dropna=False)["K"].median(numeric_only=True).to_dict()
    k_overrides: dict[str, float] = {}
    for metric in RATE_METRICS:
        stat = metric.replace("_mlb_eq_non_ar_delta", "")
        kval = pd.to_numeric(med.get(stat), errors="coerce")
        if np.isfinite(kval) and float(kval) > 0.0:
            kval_scaled = float(kval) * float(k_scale)
            k_overrides[metric] = float(np.clip(float(kval_scaled), 20.0, 20000.0))
    return k_overrides


def _compute_age_curves(
    df: pd.DataFrame,
    *,
    id_col: str,
    season_col: str,
    age_col: str,
    exposure_col: str,
    metrics: list[str],
    min_pairs_per_age: int = 30,
    smooth_window: int = 3,
    anchor_age: int = 27,
) -> pd.DataFrame:
    work = df[[id_col, season_col, age_col, exposure_col, *metrics]].copy()
    work = work.sort_values([id_col, season_col])
    work[season_col] = pd.to_numeric(work[season_col], errors="coerce")
    age_cur = pd.to_numeric(work[age_col], errors="coerce")
    age_int = age_cur.round().astype("Int64")
    season_nxt = pd.to_numeric(
        work.groupby(id_col, dropna=False)[season_col].shift(-1), errors="coerce"
    )
    gap = season_nxt - work[season_col]
    valid_gap = gap > 0

    exp_cur = pd.to_numeric(work[exposure_col], errors="coerce")
    exp_nxt = pd.to_numeric(
        work.groupby(id_col, dropna=False)[exposure_col].shift(-1), errors="coerce"
    )
    weight = np.minimum(exp_cur.to_numpy(dtype=float), exp_nxt.to_numpy(dtype=float))

    out_frames: list[pd.DataFrame] = []
    for metric in metrics:
        cur = pd.to_numeric(work[metric], errors="coerce")
        nxt = pd.to_numeric(
            work.groupby(id_col, dropna=False)[metric].shift(-1), errors="coerce"
        )
        delta_per_year = (nxt - cur) / gap

        mask = (
            valid_gap
            & age_int.notna()
            & cur.notna()
            & nxt.notna()
            & np.isfinite(weight)
            & (weight > 0)
        )
        if int(mask.sum()) == 0:
            continue

        tmp = pd.DataFrame(
            {
                "age": age_int[mask].astype(int),
                "delta": pd.to_numeric(delta_per_year[mask], errors="coerce"),
                "weight": weight[mask.to_numpy()],
            }
        )
        tmp = tmp[
            tmp["delta"].notna()
            & np.isfinite(tmp["delta"])
            & np.isfinite(tmp["weight"])
            & (tmp["weight"] > 0)
        ]
        if tmp.empty:
            continue

        tmp["weighted_delta"] = tmp["delta"] * tmp["weight"]
        grp = (
            tmp.groupby("age", as_index=False)
            .agg(
                n_pairs=("delta", "size"),
                total_weight=("weight", "sum"),
                weighted_delta=("weighted_delta", "sum"),
            )
            .sort_values("age")
        )
        grp["delta_raw"] = grp["weighted_delta"] / grp["total_weight"]
        grp = grp.drop(columns=["weighted_delta"])
        grp = grp[grp["n_pairs"] >= int(min_pairs_per_age)]
        if grp.empty:
            continue

        grp["delta_smoothed"] = (
            grp["delta_raw"]
            .rolling(window=max(1, int(smooth_window)), center=True, min_periods=1)
            .mean()
        )

        min_age = int(grp["age"].min())
        max_age = int(grp["age"].max())
        all_ages = np.arange(min_age, max_age + 1, dtype=int)
        deltas = pd.Series(np.nan, index=all_ages, dtype=float)
        for row in grp.itertuples(index=False):
            deltas.loc[int(row.age)] = float(row.delta_smoothed)

        if anchor_age < min_age:
            anchor_used = min_age
        elif anchor_age > max_age:
            anchor_used = max_age
        else:
            anchor_used = anchor_age

        curve = pd.Series(np.nan, index=all_ages, dtype=float)
        curve.loc[anchor_used] = 0.0
        for age in range(anchor_used, max_age):
            d = deltas.loc[age]
            curv = curve.loc[age]
            if np.isfinite(d) and np.isfinite(curv):
                curve.loc[age + 1] = curv + d
        for age in range(anchor_used - 1, min_age - 1, -1):
            d = deltas.loc[age]
            nxt_val = curve.loc[age + 1]
            if np.isfinite(d) and np.isfinite(nxt_val):
                curve.loc[age] = nxt_val - d

        curve_df = pd.DataFrame({"age": all_ages, "curve_vs_anchor": curve.values})
        merged = grp.merge(curve_df, on="age", how="left")
        merged.insert(0, "metric", metric)
        merged["anchor_age"] = anchor_used
        out_frames.append(merged)

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "metric",
                "age",
                "n_pairs",
                "total_weight",
                "delta_raw",
                "delta_smoothed",
                "curve_vs_anchor",
                "anchor_age",
            ]
        )
    return pd.concat(out_frames, ignore_index=True)


def _weighted_or_mean(vals: pd.Series, w: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    mask = vals.notna() & np.isfinite(vals) & w.notna() & np.isfinite(w) & (w > 0)
    if mask.any():
        return vals  # caller handles aggregate context; keep function for symmetry.
    return out


def _add_p600_totals(proj: pd.DataFrame) -> pd.DataFrame:
    out = proj.copy()
    # Outcome-percentile mapping for "bad when higher" metrics.
    # For K-rate specifically: weaker season tails should use higher K outcomes.
    strikeout_pct_map = {20: 80, 50: 50, 80: 20}
    for pct in [20, 50, 80]:
        pa_col = f"PA_proj_p{pct}"
        if pa_col in out.columns:
            pa = pd.to_numeric(out[pa_col], errors="coerce").clip(lower=0.0)
        else:
            pa = pd.Series(600.0, index=out.index, dtype=float)
            out[pa_col] = pa

        so_src_pct = int(strikeout_pct_map.get(int(pct), int(pct)))
        bbe_rate = pd.to_numeric(
            out[f"batted_ball_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        so_rate = pd.to_numeric(
            out[f"strikeout_rate_mlb_eq_non_ar_delta_proj_p{so_src_pct}"],
            errors="coerce",
        )
        bb_rate = pd.to_numeric(
            out[f"walk_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        hbp_rate = pd.to_numeric(
            out[f"hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        one_rate = pd.to_numeric(
            out[f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        two_rate = pd.to_numeric(
            out[f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        three_rate = pd.to_numeric(
            out[f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        hr_rate = pd.to_numeric(
            out[f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        sf_rate = pd.to_numeric(
            out[f"sac_fly_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        sh_rate = pd.to_numeric(
            out[f"sac_hit_rate_bbe_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        sba_pa_rate = pd.to_numeric(
            out[f"stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta_proj_p{pct}"],
            errors="coerce",
        )
        sba_sbo_rate = pd.to_numeric(
            out[f"stolen_base_attempt_rate_sbo_mlb_eq_non_ar_delta_proj_p{pct}"],
            errors="coerce",
        )
        sb_succ_rate = pd.to_numeric(
            out[f"stolen_base_success_rate_mlb_eq_non_ar_delta_proj_p{pct}"],
            errors="coerce",
        )
        xbh_h_rate = pd.to_numeric(
            out[f"xbh_from_h_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        runs_rate = pd.to_numeric(
            out[f"runs_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        rbi_rate = pd.to_numeric(
            out[f"rbi_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )
        babip_rate = pd.to_numeric(
            out[f"babip_recalc_rate_mlb_eq_non_ar_delta_proj_p{pct}"], errors="coerce"
        )

        bbe = (pa * bbe_rate).clip(lower=0.0)
        so = (pa * so_rate).clip(lower=0.0)
        bb = (pa * bb_rate).clip(lower=0.0)
        hbp = (pa * hbp_rate).clip(lower=0.0)

        singles_raw = (bbe * one_rate).clip(lower=0.0)
        doubles_raw = (bbe * two_rate).clip(lower=0.0)
        triples_raw = (bbe * three_rate).clip(lower=0.0)
        hr = (bbe * hr_rate).clip(lower=0.0)
        sf = (bbe * sf_rate).clip(lower=0.0)
        sh = (bbe * sh_rate).clip(lower=0.0)
        # Keep BABIP chaining aligned with slashline identities used in the app:
        # AB = PA - (BB + HBP + SH + SF), then BIP = AB - SO - HR + SF.
        ab = (pa - (bb + hbp + sh + sf)).clip(lower=0.0)
        bip = (ab - so - hr + sf).clip(lower=0.0)
        babip_hits_target = (bip * babip_rate).clip(lower=0.0)

        # Enforce BABIP consistency: distribute BIP hits across 1B/2B/3B using the raw mix.
        bip_comp_raw = (singles_raw + doubles_raw + triples_raw).clip(lower=0.0)
        singles = singles_raw.copy()
        doubles = doubles_raw.copy()
        triples = triples_raw.copy()

        valid_babip = (
            babip_hits_target.notna()
            & bip_comp_raw.notna()
            & np.isfinite(babip_hits_target)
            & np.isfinite(bip_comp_raw)
        )
        has_mix = valid_babip & (bip_comp_raw > 0)
        no_mix = valid_babip & (bip_comp_raw <= 0)
        if has_mix.any():
            scale = (babip_hits_target.loc[has_mix] / bip_comp_raw.loc[has_mix]).clip(
                lower=0.0
            )
            singles.loc[has_mix] = singles_raw.loc[has_mix] * scale
            doubles.loc[has_mix] = doubles_raw.loc[has_mix] * scale
            triples.loc[has_mix] = triples_raw.loc[has_mix] * scale
        if no_mix.any():
            singles.loc[no_mix] = babip_hits_target.loc[no_mix]
            doubles.loc[no_mix] = 0.0
            triples.loc[no_mix] = 0.0

        hits = (singles + doubles + triples + hr).clip(lower=0.0)
        xbh_from_h = (hits * xbh_h_rate).clip(lower=0.0)
        xbh_components = (doubles + triples + hr).clip(lower=0.0)

        est_sbo = (singles + bb + hbp + sh).clip(lower=0.0)
        sba_from_pa = (pa * sba_pa_rate).clip(lower=0.0)
        sba_from_sbo = (est_sbo * sba_sbo_rate).clip(lower=0.0)
        sba = sba_from_pa.copy()
        both = sba_from_pa.notna() & sba_from_sbo.notna()
        sba.loc[both] = (sba_from_pa.loc[both] + sba_from_sbo.loc[both]) / 2.0
        only_sbo = sba_from_pa.isna() & sba_from_sbo.notna()
        sba.loc[only_sbo] = sba_from_sbo.loc[only_sbo]
        sb = (sba * sb_succ_rate).clip(lower=0.0)
        cs = (sba - sb).clip(lower=0.0)

        runs = (pa * runs_rate).clip(lower=0.0)
        rbi = (pa * rbi_rate).clip(lower=0.0)

        bip_hits = (hits - hr).clip(lower=0.0)
        babip_hits_from_rate = babip_hits_target

        out[f"BBE_proj_p{pct}"] = bbe
        out[f"SO_proj_p{pct}"] = so
        out[f"BB_proj_p{pct}"] = bb
        out[f"HBP_proj_p{pct}"] = hbp
        out[f"1B_proj_p{pct}"] = singles
        out[f"2B_proj_p{pct}"] = doubles
        out[f"3B_proj_p{pct}"] = triples
        out[f"HR_proj_p{pct}"] = hr
        out[f"SF_proj_p{pct}"] = sf
        out[f"SH_proj_p{pct}"] = sh
        out[f"H_proj_p{pct}"] = hits
        out[f"XBH_from_H_proj_p{pct}"] = xbh_from_h
        out[f"XBH_from_components_proj_p{pct}"] = xbh_components
        out[f"SBO_proj_p{pct}"] = est_sbo
        out[f"SBA_proj_p{pct}"] = sba
        out[f"SB_proj_p{pct}"] = sb
        out[f"CS_proj_p{pct}"] = cs
        out[f"Runs_proj_p{pct}"] = runs
        out[f"RBI_proj_p{pct}"] = rbi
        out[f"BIP_proj_p{pct}"] = bip
        out[f"BIP_H_proj_p{pct}"] = bip_hits
        out[f"BIP_H_from_BABIP_proj_p{pct}"] = babip_hits_from_rate

    # Spread columns requested around p20/p50/p80 framework.
    spread_targets = [
        "PA",
        "BBE",
        "SO",
        "BB",
        "HBP",
        "1B",
        "2B",
        "3B",
        "HR",
        "SF",
        "SH",
        "H",
        "XBH_from_H",
        "XBH_from_components",
        "SBO",
        "SBA",
        "SB",
        "CS",
        "Runs",
        "RBI",
        "BIP",
        "BIP_H",
        "BIP_H_from_BABIP",
    ]
    for s in spread_targets:
        c20 = f"{s}_proj_p20"
        c50 = f"{s}_proj_p50"
        c80 = f"{s}_proj_p80"
        if not all(c in out.columns for c in [c20, c50, c80]):
            continue
        p20 = pd.to_numeric(out[c20], errors="coerce").clip(lower=0.0)
        p50 = pd.to_numeric(out[c50], errors="coerce").clip(lower=0.0)
        p80 = pd.to_numeric(out[c80], errors="coerce").clip(lower=0.0)
        p50 = np.maximum(p50, p20)
        p80 = np.maximum(p80, p50)
        out[c20] = p20
        out[c50] = p50
        out[c80] = p80
        out[f"{s}_proj_spread"] = p80 - p20
    return out


def _build_source_season_context(
    bp_hitting_path: Path, source_season: int
) -> pd.DataFrame:
    out_cols = [
        "mlbid",
        f"levels_played_{source_season}",
        "appeared_in_MLB",
        "team_abbreviations",
    ]
    if not bp_hitting_path.exists():
        return pd.DataFrame(columns=out_cols)

    cols = [
        "mlbid",
        "season",
        "bp_level_id",
        "team_abbreviation",
        "plate_appearances_agg",
    ]
    bp = pd.read_parquet(bp_hitting_path)
    if not set(cols).issubset(bp.columns):
        return pd.DataFrame(columns=out_cols)

    work = bp[cols].copy()
    work["mlbid"] = pd.to_numeric(work["mlbid"], errors="coerce")
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["bp_level_id"] = pd.to_numeric(work["bp_level_id"], errors="coerce")
    work["plate_appearances_agg"] = pd.to_numeric(
        work["plate_appearances_agg"], errors="coerce"
    )
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
                for v in sorted(
                    set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist())
                )
            )
        )
        .reset_index(name=lvl_col)
    )
    appeared = (
        work.groupby("mlbid", dropna=False)["bp_level_id"]
        .apply(lambda s: "T" if (pd.to_numeric(s, errors="coerce") == 1).any() else "F")
        .reset_index(name="appeared_in_MLB")
    )

    team_work = work[work["team_abbreviation"] != ""].copy()
    if team_work.empty:
        team = pd.DataFrame(columns=["mlbid", "team_abbreviations"])
    else:
        team_tot = (
            team_work.groupby(["mlbid", "team_abbreviation"], as_index=False)[
                "plate_appearances_agg"
            ]
            .sum(numeric_only=True)
            .sort_values(
                ["mlbid", "plate_appearances_agg", "team_abbreviation"],
                ascending=[True, False, True],
            )
        )
        team = (
            team_tot.groupby("mlbid", dropna=False)["team_abbreviation"]
            .apply(lambda s: "|".join(s.astype(str).tolist()))
            .reset_index(name="team_abbreviations")
        )

    meta = levels.merge(appeared, on="mlbid", how="outer")
    meta = meta.merge(team, on="mlbid", how="left")
    return meta


def _load_metric_recency_weights(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metric recency weight file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Metric recency weight file must be a JSON object of metric -> [w0,w1,w2].")
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
                raise ValueError(f"Metric {metric} has non-positive/invalid weight: {w}")
            vals.append(wf)
        out[metric] = vals
    return out


def build_bp_rate_projections_2026(
    *,
    input_path: Path,
    constants_path: Path,
    bp_hitting_path: Path,
    historical_ar_path: Path,
    z_coherence_mode: str = "inverse",
    z_anchor_k: float = 260.0,
    hr_anchor_k: float = 240.0,
    z_tail_strength: float = 1.0,
    coherence_mode: str = "inverse",
    out_projection_path: Path,
    out_age_curve_path: Path,
    uncertainty_draws: int = 2000,
    seed: int = 7,
    metric_recency_weights: dict[str, list[float]] | None = None,
    default_k: float = 200.0,
    k_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(input_path)
    required = {
        "mlbid",
        "season",
        "plate_appearances_agg",
        "baseball_age",
        "player_display_text",
    }
    missing_req = [c for c in required if c not in raw.columns]
    if missing_req:
        raise ValueError(f"Missing required columns in {input_path}: {missing_req}")
    missing_metrics = [m for m in RATE_METRICS if m not in raw.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metrics in {input_path}: {missing_metrics}")

    hist_raw = raw.copy()
    hist_raw["mlbid"] = pd.to_numeric(hist_raw["mlbid"], errors="coerce")
    hist_raw["season"] = pd.to_numeric(hist_raw["season"], errors="coerce")
    hist_raw["plate_appearances_agg"] = pd.to_numeric(
        hist_raw["plate_appearances_agg"], errors="coerce"
    )
    hist_raw["baseball_age"] = pd.to_numeric(hist_raw["baseball_age"], errors="coerce")
    hist_raw = hist_raw[hist_raw["mlbid"].notna() & hist_raw["season"].notna()].copy()
    hist_raw["mlbid"] = hist_raw["mlbid"].astype("int64")
    hist_raw["season"] = hist_raw["season"].astype("int64")
    hist_raw["level_id_source"] = 1
    hist_raw = hist_raw.rename(columns={"player_display_text": "player_name"})
    hist_raw["player_name"] = (
        hist_raw["player_name"].fillna(hist_raw["mlbid"].astype(str)).astype(str)
    )
    hist_raw = _add_hist_ops_from_rates(hist_raw)
    appeared_hist = _build_mlb_appearance_by_season(bp_hitting_path)
    if not appeared_hist.empty:
        hist_raw = hist_raw.merge(appeared_hist, on=["mlbid", "season"], how="left")
    if "appeared_in_MLB_hist" not in hist_raw.columns:
        hist_raw["appeared_in_MLB_hist"] = 0
    hist_raw["appeared_in_MLB_hist"] = (
        pd.to_numeric(hist_raw["appeared_in_MLB_hist"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    z_params = _build_mlb_t_z_params(
        hist_raw,
        metric_cols=RATE_METRICS,
        flag_col="appeared_in_MLB_hist",
    )
    hist = _to_z_space(
        hist_raw,
        metric_cols=RATE_METRICS,
        params=z_params,
    )
    z_bounds = _derive_z_bounds(hist, metric_cols=RATE_METRICS)
    z_profile = _build_metric_correlation_profile(hist, metric_cols=RATE_METRICS)

    age_curves = _compute_age_curves(
        hist,
        id_col="mlbid",
        season_col="season",
        age_col="baseball_age",
        exposure_col="plate_appearances_agg",
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
        recency_weights=[5, 4, 3],
        recency_weights_by_metric=dict(metric_recency_weights or {}),
        default_k=float(default_k),
        uncertainty_draws=int(uncertainty_draws),
        seed=int(seed),
        cov_blend_global=0.6,
        local_k=200,
        local_min_k=40,
        uncertainty_c=120.0,
        uncertainty_d=20.0,
        projection_version="bp_eq_non_ar_post_inv_coh_zspace_2026",
        min_age_samples=20,
    )
    k_overrides = _build_k_overrides(constants_path, k_scale=float(k_scale))
    coherence_profile = (
        _build_level1_correlation_profile(historical_ar_path)
        if coherence_mode != "none"
        else None
    )

    source_season = int(pd.to_numeric(hist["season"], errors="coerce").max())
    src_meta = _build_source_season_context(
        bp_hitting_path, source_season=source_season
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
        exposure_col="plate_appearances_agg",
        global_cfg=gcfg,
        k_overrides=k_overrides,
        bounds=z_bounds,
        source_season=source_season,
        passthrough_cols=[],
    )
    point = _apply_z_anchor_recenter(
        point,
        metric_cols=RATE_METRICS,
        anchor_k=float(z_anchor_k),
    )

    transitions = build_transition_deltas(
        hist,
        id_col="mlbid",
        season_col="season",
        metric_cols=RATE_METRICS,
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
    so_metric = "strikeout_rate_mlb_eq_non_ar_delta"
    so_cols_z = [
        f"{so_metric}_proj_p20",
        f"{so_metric}_proj_p50",
        f"{so_metric}_proj_p80",
    ]
    so_snapshot_z = {
        c: pd.to_numeric(proj[c], errors="coerce").copy()
        for c in so_cols_z
        if c in proj.columns
    }
    proj = _apply_rate_correlation_congruity(
        proj,
        profile=z_profile,
        mode=z_coherence_mode,
        ridge=0.06,
        alpha_map=Z_COHERENCE_ALPHA,
        max_delta_map=Z_COHERENCE_MAX_DELTA,
    )
    for c, s in so_snapshot_z.items():
        proj[c] = s
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=z_bounds,
    )
    proj = _apply_z_tail_shape(
        proj,
        metric_cols=RATE_METRICS,
        z_bounds=z_bounds,
        strength=float(z_tail_strength),
    )
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=z_bounds,
    )
    proj = _from_z_space_projection(
        proj,
        metric_cols=RATE_METRICS,
        params=z_params,
    )
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=RATE_BOUNDS,
    )
    so_cols_raw = [
        f"{so_metric}_proj_p20",
        f"{so_metric}_proj_p50",
        f"{so_metric}_proj_p80",
    ]
    so_snapshot_raw = {
        c: pd.to_numeric(proj[c], errors="coerce").copy()
        for c in so_cols_raw
        if c in proj.columns
    }
    proj = _apply_rate_correlation_congruity(
        proj,
        profile=coherence_profile,
        mode=coherence_mode,
        ridge=0.06,
    )
    for c, s in so_snapshot_raw.items():
        proj[c] = s
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=RATE_BOUNDS,
    )
    proj = _apply_hr_population_anchor(
        proj,
        hist=hist_raw,
        metric="home_run_rate_bbe_mlb_eq_non_ar_delta",
        anchor_k=float(hr_anchor_k),
        mlb_flag_col="appeared_in_MLB_hist",
    )
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=RATE_BOUNDS,
    )

    proj = _apply_ops_sensitive_pa_projections(
        proj,
        hist=hist_raw,
        source_season=source_season,
        recency_weights=list(gcfg.recency_weights),
        ops_pivot=0.650,
    )
    skill_bounds = _derive_empirical_skill_bounds(hist_raw, metrics=RATE_METRICS)
    for metric in NO_SKILL_BOUND_CLIP_METRICS:
        if metric in RATE_BOUNDS:
            skill_bounds[metric] = RATE_BOUNDS[metric]
    proj = _apply_sigmoid_skill_shape(
        proj,
        metric_cols=RATE_METRICS,
        bounds=skill_bounds,
    )
    proj = _enforce_component_rate_consistency(proj)
    proj = _finalize_rate_percentiles(
        proj,
        metric_cols=RATE_METRICS,
        bounds=skill_bounds,
    )

    lvl_col = f"levels_played_{source_season}"
    if not src_meta.empty:
        proj = proj.merge(src_meta, on="mlbid", how="left")
    if lvl_col not in proj.columns:
        proj[lvl_col] = np.nan
    if "appeared_in_MLB" not in proj.columns:
        proj["appeared_in_MLB"] = "F"
    if "team_abbreviations" not in proj.columns:
        proj["team_abbreviations"] = np.nan
    proj["appeared_in_MLB"] = (
        proj["appeared_in_MLB"]
        .fillna("F")
        .astype(str)
        .str.upper()
        .replace({"TRUE": "T", "FALSE": "F"})
    )

    proj = _add_p600_totals(proj)

    keep_cols = [
        "mlbid",
        "player_name",
        "source_season",
        "target_season",
        "projection_version",
        "age_used",
        "age_source",
        "PA_marcel_base",
        "OPS_perf_anchor",
        "PA_perf_adj_pct",
        "PA_perf_adj_abs",
        "OBP_skill_est_p50",
        "K_skill_est_p50",
        "PA_risk_factor",
        "HR_anchor_mu_mlbT",
        "HR_anchor_k",
        lvl_col,
        "appeared_in_MLB",
        "team_abbreviations",
    ]
    for metric in RATE_METRICS:
        for pct in [20, 50, 80]:
            col = f"{metric}_proj_p{pct}"
            if col in proj.columns:
                keep_cols.append(col)
        s = f"{metric}_proj_spread"
        if s in proj.columns:
            keep_cols.append(s)
    keep_cols.extend(
        [
            c
            for c in proj.columns
            if (
                (c.startswith("PA_proj_p") or c.startswith("PA_proj_spread"))
                or c.startswith("BBE_proj_p")
                or c.startswith("SO_proj_p")
                or c.startswith("BB_proj_p")
                or c.startswith("HBP_proj_p")
                or c.startswith("1B_proj_p")
                or c.startswith("2B_proj_p")
                or c.startswith("3B_proj_p")
                or c.startswith("HR_proj_p")
                or c.startswith("SF_proj_p")
                or c.startswith("SH_proj_p")
                or c.startswith("H_proj_p")
                or c.startswith("XBH_from_H_proj_p")
                or c.startswith("XBH_from_components_proj_p")
                or c.startswith("SBO_proj_p")
                or c.startswith("SBA_proj_p")
                or c.startswith("SB_proj_p")
                or c.startswith("CS_proj_p")
                or c.startswith("Runs_proj_p")
                or c.startswith("RBI_proj_p")
                or c.startswith("BIP_proj_p")
                or c.startswith("BIP_H_proj_p")
                or c.startswith("BIP_H_from_BABIP_proj_p")
                or c.endswith("_proj_spread")
            )
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
            "Build hitter age curves and 2026 p20/p50/p80 rate projections from "
            "BP_single_source_mlb_eq_non_ar_delta.parquet in MLB-T-centered z-space, "
            "inverse-transform to raw rates, then apply inverse coherence."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("BP_single_source_mlb_eq_non_ar_delta.parquet"),
    )
    parser.add_argument(
        "--constants-path",
        type=Path,
        default=Path("BP_data_AR_2015_2025_constants.parquet"),
    )
    parser.add_argument(
        "--bp-hitting-path",
        type=Path,
        default=Path("projection_outputs/bp_hitting_api/bp_hitting_table.parquet"),
    )
    parser.add_argument(
        "--historical-ar-path",
        type=Path,
        default=Path("BP_data_AR_2015_2025.parquet"),
    )
    parser.add_argument(
        "--z-coherence-mode",
        choices=["direct", "inverse", "none"],
        default="inverse",
        help="Z-space rate congruity adjustment before inverse-transform.",
    )
    parser.add_argument(
        "--z-anchor-k",
        type=float,
        default=260.0,
        help="Additional z-space shrinkage toward 0 for p50 rates: n_eff/(n_eff+K). Set 0 to disable.",
    )
    parser.add_argument(
        "--z-tail-strength",
        type=float,
        default=0.0,
        help="Z-space tail compression strength (1=full, 0=disabled, 0.5=about half as much compression).",
    )
    parser.add_argument(
        "--hr-anchor-k",
        type=float,
        default=240.0,
        help="HR-rate-only pull toward MLB-T mean: n_eff/(n_eff+K). Set 0 to disable.",
    )
    parser.add_argument(
        "--coherence-mode",
        choices=["direct", "inverse", "none"],
        default="inverse",
        help="Raw-space rate congruity adjustment after inverse-transform.",
    )
    parser.add_argument(
        "--out-projections",
        type=Path,
        default=Path("BP_rate_projections_2026_non_ar_post_inv_coh.parquet"),
    )
    parser.add_argument(
        "--out-age-curves",
        type=Path,
        default=Path("BP_rate_age_curves_2015_2025_non_ar_post_inv_coh.parquet"),
    )
    parser.add_argument(
        "--metric-recency-weights-json",
        type=Path,
        default=None,
        help="Optional JSON mapping metric -> [w0,w1,w2] for per-metric Marcel recency weights.",
    )
    parser.add_argument("--uncertainty-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--default-k",
        type=float,
        default=200.0,
        help="Fallback regression K used when a metric has no constant override.",
    )
    parser.add_argument(
        "--k-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to metric-specific K overrides from constants.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metric_recency_weights = _load_metric_recency_weights(args.metric_recency_weights_json)
    proj, ages = build_bp_rate_projections_2026(
        input_path=args.input_path,
        constants_path=args.constants_path,
        bp_hitting_path=args.bp_hitting_path,
        historical_ar_path=args.historical_ar_path,
        z_coherence_mode=args.z_coherence_mode,
        z_anchor_k=float(args.z_anchor_k),
        hr_anchor_k=float(args.hr_anchor_k),
        z_tail_strength=float(args.z_tail_strength),
        coherence_mode=args.coherence_mode,
        out_projection_path=args.out_projections,
        out_age_curve_path=args.out_age_curves,
        uncertainty_draws=int(args.uncertainty_draws),
        seed=int(args.seed),
        metric_recency_weights=metric_recency_weights,
        default_k=float(args.default_k),
        k_scale=float(args.k_scale),
    )
    print(f"Wrote {len(proj):,} rows to {args.out_projections}")
    print(f"Wrote {len(ages):,} rows to {args.out_age_curves}")


if __name__ == "__main__":
    main()
