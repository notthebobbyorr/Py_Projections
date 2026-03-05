from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    fixed = (fixed + fixed.T) / 2.0
    return fixed


def build_transition_deltas(
    player_season_df: pd.DataFrame,
    id_col: str,
    season_col: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    if player_season_df.empty:
        return pd.DataFrame()

    frame = player_season_df[[id_col, season_col, *metric_cols]].copy()
    frame[season_col] = pd.to_numeric(frame[season_col], errors="coerce")
    frame = frame[frame[season_col].notna()].copy()
    frame[season_col] = frame[season_col].astype(int)
    frame = frame.sort_values([id_col, season_col])
    nxt = frame.groupby(id_col, dropna=False).shift(-1)
    next_season = pd.to_numeric(nxt[season_col], errors="coerce")
    is_consecutive = next_season.eq(frame[season_col] + 1)

    rows = frame[[id_col, season_col]].copy()
    rows = rows[is_consecutive].copy()
    if rows.empty:
        return pd.DataFrame()

    for metric in metric_cols:
        cur = pd.to_numeric(frame.loc[is_consecutive, metric], errors="coerce")
        fut = pd.to_numeric(nxt.loc[is_consecutive, metric], errors="coerce")
        rows[f"src_{metric}"] = cur.to_numpy(dtype=float)
        rows[f"delta_{metric}"] = (fut - cur).to_numpy(dtype=float)
    return rows.reset_index(drop=True)


def _safe_cov(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.eye(arr.shape[1], dtype=float) * 1e-4
    cov = np.cov(arr, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    return np.asarray(cov, dtype=float)


def apply_uncertainty_bands(
    projection_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    metric_cols: list[str],
    draws: int = 2000,
    seed: int = 7,
    global_weight: float = 0.6,
    local_k: int = 200,
    local_min_k: int = 40,
    uncertainty_c: float = 120.0,
    uncertainty_d: float = 20.0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    out = projection_df.copy()
    bounds = bounds or {}
    if out.empty:
        return out

    if transitions_df.empty:
        for metric in metric_cols:
            p50 = pd.to_numeric(out[f"{metric}_proj_p50"], errors="coerce")
            out[f"{metric}_proj_p20"] = p50
            out[f"{metric}_proj_p25"] = p50
            out[f"{metric}_proj_p75"] = p50
            out[f"{metric}_proj_p80"] = p50
            out[f"{metric}_proj_spread"] = 0.0
        out["volatility_index"] = 0.0
        return out

    src_cols = [f"src_{m}" for m in metric_cols]
    delta_cols = [f"delta_{m}" for m in metric_cols]
    t = transitions_df[src_cols + delta_cols].copy()
    t = t.replace([np.inf, -np.inf], np.nan).dropna()
    if t.empty:
        for metric in metric_cols:
            p50 = pd.to_numeric(out[f"{metric}_proj_p50"], errors="coerce")
            out[f"{metric}_proj_p20"] = p50
            out[f"{metric}_proj_p25"] = p50
            out[f"{metric}_proj_p75"] = p50
            out[f"{metric}_proj_p80"] = p50
            out[f"{metric}_proj_spread"] = 0.0
        out["volatility_index"] = 0.0
        return out

    src_mat = t[src_cols].to_numpy(dtype=float)
    delta_mat = t[delta_cols].to_numpy(dtype=float)
    src_std = src_mat.std(axis=0, ddof=1)
    src_std = np.where(src_std > 0, src_std, 1.0)
    global_cov = nearest_psd(_safe_cov(delta_mat))
    hist_metric_std = src_mat.std(axis=0, ddof=1)
    hist_metric_std = np.where(hist_metric_std > 0, hist_metric_std, 1.0)

    rng = np.random.default_rng(seed)
    p20 = {m: [] for m in metric_cols}
    p25 = {m: [] for m in metric_cols}
    p75 = {m: [] for m in metric_cols}
    p80 = {m: [] for m in metric_cols}
    vol = []

    for _, row in out.iterrows():
        mean_vec = np.array(
            [float(pd.to_numeric(row.get(f"{m}_proj_p50"), errors="coerce")) for m in metric_cols],
            dtype=float,
        )
        src_vec = np.array(
            [
                float(pd.to_numeric(row.get(f"source_{m}", row.get(f"{m}_proj_p50")), errors="coerce"))
                for m in metric_cols
            ],
            dtype=float,
        )
        if not np.isfinite(src_vec).all():
            src_vec = np.where(np.isfinite(src_vec), src_vec, mean_vec)

        dists = np.sqrt((((src_mat - src_vec) / src_std) ** 2).sum(axis=1))
        order = np.argsort(dists)
        local_n = min(local_k, len(order))
        if local_n >= local_min_k:
            local_cov = nearest_psd(_safe_cov(delta_mat[order[:local_n], :]))
            cov = global_weight * global_cov + (1.0 - global_weight) * local_cov
        else:
            cov = global_cov

        scales = []
        for metric in metric_cols:
            n_eff_col = f"n_eff_{metric}"
            n_eff = float(pd.to_numeric(row.get(n_eff_col), errors="coerce"))
            if not np.isfinite(n_eff):
                n_eff = 0.0
            scales.append(1.0 + (uncertainty_c / (n_eff + uncertainty_d)))
        scale_diag = np.diag(np.sqrt(np.asarray(scales, dtype=float)))
        cov = scale_diag @ cov @ scale_diag
        cov = nearest_psd(cov)

        sims = rng.multivariate_normal(mean_vec, cov, size=int(draws), check_valid="ignore")
        for i, metric in enumerate(metric_cols):
            if metric in bounds:
                low, high = bounds[metric]
                sims[:, i] = np.clip(sims[:, i], low, high)
        lower20 = np.quantile(sims, 0.20, axis=0)
        lower = np.quantile(sims, 0.25, axis=0)
        upper = np.quantile(sims, 0.75, axis=0)
        upper80 = np.quantile(sims, 0.80, axis=0)
        width = (upper - lower) / hist_metric_std
        vol.append(float(np.nanmean(width)))

        for i, metric in enumerate(metric_cols):
            p20[metric].append(float(lower20[i]))
            p25[metric].append(float(lower[i]))
            p75[metric].append(float(upper[i]))
            p80[metric].append(float(upper80[i]))

    for metric in metric_cols:
        out[f"{metric}_proj_p20"] = p20[metric]
        out[f"{metric}_proj_p25"] = p25[metric]
        out[f"{metric}_proj_p75"] = p75[metric]
        out[f"{metric}_proj_p80"] = p80[metric]
        out[f"{metric}_proj_spread"] = (
            pd.to_numeric(out[f"{metric}_proj_p75"], errors="coerce")
            - pd.to_numeric(out[f"{metric}_proj_p25"], errors="coerce")
        )
    out["volatility_index"] = vol
    return out
