from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_step(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> None:
    started = _utc_now_iso()
    print(f"[{name}]")
    print(f"  cwd: {cwd}")
    print(f"  cmd: {_cmd_str(cmd)}")
    if dry_run:
        records.append(
            {
                "name": name,
                "started_utc": started,
                "ended_utc": _utc_now_iso(),
                "status": "dry_run",
                "returncode": None,
                "cmd": cmd,
            }
        )
        return

    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    ended = _utc_now_iso()
    records.append(
        {
            "name": name,
            "started_utc": started,
            "ended_utc": ended,
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "cmd": cmd,
        }
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit={proc.returncode})")


@dataclass
class PipelineArgs:
    python_exe: Path
    repo_root: Path
    dry_run: bool
    skip_enrich_regress: bool
    skip_kpi_projection: bool
    skip_bp_stage1: bool
    skip_sandbox_and_two_stage: bool
    manifest_out: Path
    enrich_min_season: int
    enrich_max_season: int
    stability_config: Path
    stability_constants: Path
    hitters_agg: Path
    pitchers_agg: Path
    pitch_types_agg: Path
    lb2_refresh_dir: Path
    projection_config_lb2: Path
    bp_input_path: Path
    bp_constants_path: Path
    bp_hitting_path: Path
    bp_historical_ar_path: Path
    bp_metric_recency_weights_json: Path | None
    bp_z_coherence_mode: str
    bp_z_anchor_k: float
    bp_hr_anchor_k: float
    bp_z_tail_strength: float
    bp_coherence_mode: str
    bp_uncertainty_draws: int
    bp_seed: int
    bp_default_k: float
    bp_k_scale: float
    bp_out_projections: Path
    bp_out_age_curves: Path
    sandbox_naive_input: Path
    sandbox_old_naive_input: Path
    sandbox_old_advanced_input: Path
    sandbox_bp_hitting_path: Path
    sandbox_historical_rates_path: Path
    sandbox_constants_path: Path
    sandbox_historical_ar_path: Path
    sandbox_stability_pa_per_season: float
    sandbox_spread_boost_max: float
    sandbox_no_metric_expert_composite: bool
    sandbox_out_kpi: Path
    sandbox_out_traditional: Path
    sandbox_out_composite: Path
    two_stage_out_traditional: Path
    two_stage_model_bundle: Path
    two_stage_diagnostics_out: Path
    two_stage_train_start_season: int
    two_stage_train_end_season: int
    two_stage_cap_sd_mult: float
    two_stage_mode: str
    two_stage_zspace_min_mlb_pa: float
    two_stage_emit_p25p75: bool


def _parse_args() -> PipelineArgs:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible end-to-end runner for KPI -> Stage-1 BP -> "
            "sandbox Stage-1 traditional -> two-stage KPI-adjusted fusion."
        )
    )

    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-enrich-regress", action="store_true")
    parser.add_argument("--skip-kpi-projection", action="store_true")
    parser.add_argument("--skip-bp-stage1", action="store_true")
    parser.add_argument("--skip-sandbox-and-two-stage", action="store_true")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("projection_outputs/sandbox/two_stage_repro_manifest.json"),
    )

    parser.add_argument("--enrich-min-season", type=int, default=2015)
    parser.add_argument("--enrich-max-season", type=int, default=2025)
    parser.add_argument("--stability-config", type=Path, default=Path("stability_config.yml"))
    parser.add_argument("--stability-constants", type=Path, default=Path("stability_constants.csv"))
    parser.add_argument("--hitters-agg", type=Path, default=Path("damage_pos_2015_2025.parquet"))
    parser.add_argument("--pitchers-agg", type=Path, default=Path("pitcher_stuff_new.parquet"))
    parser.add_argument("--pitch-types-agg", type=Path, default=Path("new_pitch_types.parquet"))
    parser.add_argument("--lb2-refresh-dir", type=Path, default=Path("projection_outputs/lb2_refresh"))
    parser.add_argument(
        "--projection-config-lb2",
        type=Path,
        default=Path("projections_v1/projection_config_lb2.yml"),
    )

    parser.add_argument("--bp-input-path", type=Path, default=Path("BP_single_source_mlb_eq_non_ar_delta.parquet"))
    parser.add_argument("--bp-constants-path", type=Path, default=Path("BP_data_AR_2015_2025_constants.parquet"))
    parser.add_argument(
        "--bp-hitting-path",
        type=Path,
        default=Path("projection_outputs/bp_hitting_api/bp_hitting_table.parquet"),
    )
    parser.add_argument("--bp-historical-ar-path", type=Path, default=Path("BP_data_AR_2015_2025.parquet"))
    parser.add_argument("--bp-metric-recency-weights-json", type=Path, default=None)
    parser.add_argument("--bp-z-coherence-mode", choices=["direct", "inverse", "none"], default="inverse")
    parser.add_argument("--bp-z-anchor-k", type=float, default=0.0)
    parser.add_argument("--bp-hr-anchor-k", type=float, default=240.0)
    parser.add_argument("--bp-z-tail-strength", type=float, default=0.0)
    parser.add_argument("--bp-coherence-mode", choices=["direct", "inverse", "none"], default="inverse")
    parser.add_argument("--bp-uncertainty-draws", type=int, default=2000)
    parser.add_argument("--bp-seed", type=int, default=7)
    parser.add_argument("--bp-default-k", type=float, default=200.0)
    parser.add_argument("--bp-k-scale", type=float, default=1.0)
    parser.add_argument(
        "--bp-out-projections",
        type=Path,
        default=Path("BP_rate_projections_2026_non_ar_post_inv_coh_no_z_anchor.parquet"),
    )
    parser.add_argument(
        "--bp-out-age-curves",
        type=Path,
        default=Path("BP_rate_age_curves_2015_2025_non_ar_post_inv_coh_no_z_anchor.parquet"),
    )

    parser.add_argument(
        "--sandbox-naive-input",
        type=Path,
        default=Path("projection_outputs/sandbox/naive_marcel_age_projections_2026.parquet"),
    )
    parser.add_argument(
        "--sandbox-old-naive-input",
        type=Path,
        default=Path("projection_outputs/sandbox/naive_marcel_age_projections_2026.parquet"),
    )
    parser.add_argument(
        "--sandbox-old-advanced-input",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_projections_2026_pre_predictive.parquet"),
    )
    parser.add_argument(
        "--sandbox-bp-hitting-path",
        type=Path,
        default=Path("projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet"),
    )
    parser.add_argument(
        "--sandbox-historical-rates-path",
        type=Path,
        default=Path("BP_single_source_mlb_eq_non_ar_delta.parquet"),
    )
    parser.add_argument("--sandbox-constants-path", type=Path, default=Path("BP_data_AR_2015_2025_constants.parquet"))
    parser.add_argument("--sandbox-historical-ar-path", type=Path, default=Path("BP_data_AR_2015_2025.parquet"))
    parser.add_argument("--sandbox-stability-pa-per-season", type=float, default=500.0)
    parser.add_argument("--sandbox-spread-boost-max", type=float, default=0.65)
    parser.add_argument(
        "--sandbox-no-metric-expert-composite",
        dest="sandbox_no_metric_expert_composite",
        action="store_true",
        default=True,
        help="Disable metric-expert composite output in sandbox build (default).",
    )
    parser.add_argument(
        "--sandbox-build-metric-expert-composite",
        dest="sandbox_no_metric_expert_composite",
        action="store_false",
        help="Enable metric-expert composite output in sandbox build.",
    )
    parser.add_argument(
        "--sandbox-out-kpi",
        type=Path,
        default=Path("projection_outputs/sandbox/kpi_projections_2026.parquet"),
    )
    parser.add_argument(
        "--sandbox-out-traditional",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_projections_2026.parquet"),
    )
    parser.add_argument(
        "--sandbox-out-composite",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_composite_projections_2026.parquet"),
    )
    parser.add_argument(
        "--two-stage-out-traditional",
        type=Path,
        default=Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet"),
    )
    parser.add_argument(
        "--two-stage-model-bundle",
        type=Path,
        default=Path("projection_outputs/sandbox/two_stage_models_2026.joblib"),
    )
    parser.add_argument(
        "--two-stage-diagnostics-out",
        type=Path,
        default=Path("projection_outputs/sandbox/two_stage_diagnostics_2026.parquet"),
    )
    parser.add_argument("--two-stage-train-start-season", type=int, default=2018)
    parser.add_argument("--two-stage-train-end-season", type=int, default=2025)
    parser.add_argument("--two-stage-cap-sd-mult", type=float, default=1.5)
    parser.add_argument(
        "--two-stage-mode",
        choices=["raw", "zspace"],
        default="raw",
    )
    parser.add_argument(
        "--two-stage-zspace-min-mlb-pa",
        type=float,
        default=200.0,
    )
    parser.add_argument("--two-stage-emit-p25p75", action="store_true")

    ns = parser.parse_args()
    return PipelineArgs(
        python_exe=ns.python_exe,
        repo_root=ns.repo_root,
        dry_run=bool(ns.dry_run),
        skip_enrich_regress=bool(ns.skip_enrich_regress),
        skip_kpi_projection=bool(ns.skip_kpi_projection),
        skip_bp_stage1=bool(ns.skip_bp_stage1),
        skip_sandbox_and_two_stage=bool(ns.skip_sandbox_and_two_stage),
        manifest_out=ns.manifest_out,
        enrich_min_season=int(ns.enrich_min_season),
        enrich_max_season=int(ns.enrich_max_season),
        stability_config=ns.stability_config,
        stability_constants=ns.stability_constants,
        hitters_agg=ns.hitters_agg,
        pitchers_agg=ns.pitchers_agg,
        pitch_types_agg=ns.pitch_types_agg,
        lb2_refresh_dir=ns.lb2_refresh_dir,
        projection_config_lb2=ns.projection_config_lb2,
        bp_input_path=ns.bp_input_path,
        bp_constants_path=ns.bp_constants_path,
        bp_hitting_path=ns.bp_hitting_path,
        bp_historical_ar_path=ns.bp_historical_ar_path,
        bp_metric_recency_weights_json=ns.bp_metric_recency_weights_json,
        bp_z_coherence_mode=ns.bp_z_coherence_mode,
        bp_z_anchor_k=float(ns.bp_z_anchor_k),
        bp_hr_anchor_k=float(ns.bp_hr_anchor_k),
        bp_z_tail_strength=float(ns.bp_z_tail_strength),
        bp_coherence_mode=ns.bp_coherence_mode,
        bp_uncertainty_draws=int(ns.bp_uncertainty_draws),
        bp_seed=int(ns.bp_seed),
        bp_default_k=float(ns.bp_default_k),
        bp_k_scale=float(ns.bp_k_scale),
        bp_out_projections=ns.bp_out_projections,
        bp_out_age_curves=ns.bp_out_age_curves,
        sandbox_naive_input=ns.sandbox_naive_input,
        sandbox_old_naive_input=ns.sandbox_old_naive_input,
        sandbox_old_advanced_input=ns.sandbox_old_advanced_input,
        sandbox_bp_hitting_path=ns.sandbox_bp_hitting_path,
        sandbox_historical_rates_path=ns.sandbox_historical_rates_path,
        sandbox_constants_path=ns.sandbox_constants_path,
        sandbox_historical_ar_path=ns.sandbox_historical_ar_path,
        sandbox_stability_pa_per_season=float(ns.sandbox_stability_pa_per_season),
        sandbox_spread_boost_max=float(ns.sandbox_spread_boost_max),
        sandbox_no_metric_expert_composite=bool(ns.sandbox_no_metric_expert_composite),
        sandbox_out_kpi=ns.sandbox_out_kpi,
        sandbox_out_traditional=ns.sandbox_out_traditional,
        sandbox_out_composite=ns.sandbox_out_composite,
        two_stage_out_traditional=ns.two_stage_out_traditional,
        two_stage_model_bundle=ns.two_stage_model_bundle,
        two_stage_diagnostics_out=ns.two_stage_diagnostics_out,
        two_stage_train_start_season=int(ns.two_stage_train_start_season),
        two_stage_train_end_season=int(ns.two_stage_train_end_season),
        two_stage_cap_sd_mult=float(ns.two_stage_cap_sd_mult),
        two_stage_mode=str(ns.two_stage_mode),
        two_stage_zspace_min_mlb_pa=float(ns.two_stage_zspace_min_mlb_pa),
        two_stage_emit_p25p75=bool(ns.two_stage_emit_p25p75),
    )


def run_pipeline(cfg: PipelineArgs) -> None:
    records: list[dict[str, Any]] = []
    py = str(cfg.python_exe)
    root = cfg.repo_root.resolve()

    lb2_hitter_proj = cfg.lb2_refresh_dir / "hitter_projections.parquet"

    if not cfg.skip_enrich_regress:
        cmd = [
            py,
            "enrich_and_regress_from_agg.py",
            "--min-season",
            str(cfg.enrich_min_season),
            "--max-season",
            str(cfg.enrich_max_season),
            "--config",
            str(cfg.stability_config),
            "--constants",
            str(cfg.stability_constants),
            "--hitters",
            str(cfg.hitters_agg),
            "--pitchers",
            str(cfg.pitchers_agg),
            "--pitch-types",
            str(cfg.pitch_types_agg),
            "--enriched-out-dir",
            str(cfg.lb2_refresh_dir),
            "--regressed-out-dir",
            str(cfg.lb2_refresh_dir),
        ]
        _run_step(name="LB2 Enrich + Regress", cmd=cmd, cwd=root, dry_run=cfg.dry_run, records=records)

    if not cfg.skip_kpi_projection:
        cmd = [
            py,
            "-m",
            "projections_v1.run",
            "--config",
            str(cfg.projection_config_lb2),
        ]
        _run_step(name="KPI Projection Build (v1 LB2)", cmd=cmd, cwd=root, dry_run=cfg.dry_run, records=records)

    if not cfg.skip_bp_stage1:
        cmd = [
            py,
            "build_bp_rate_projections_2026_non_ar_post_inv_coh.py",
            "--input-path",
            str(cfg.bp_input_path),
            "--constants-path",
            str(cfg.bp_constants_path),
            "--bp-hitting-path",
            str(cfg.bp_hitting_path),
            "--historical-ar-path",
            str(cfg.bp_historical_ar_path),
            "--z-coherence-mode",
            cfg.bp_z_coherence_mode,
            "--z-anchor-k",
            str(cfg.bp_z_anchor_k),
            "--hr-anchor-k",
            str(cfg.bp_hr_anchor_k),
            "--z-tail-strength",
            str(cfg.bp_z_tail_strength),
            "--coherence-mode",
            cfg.bp_coherence_mode,
            "--uncertainty-draws",
            str(cfg.bp_uncertainty_draws),
            "--seed",
            str(cfg.bp_seed),
            "--default-k",
            str(cfg.bp_default_k),
            "--k-scale",
            str(cfg.bp_k_scale),
            "--out-projections",
            str(cfg.bp_out_projections),
            "--out-age-curves",
            str(cfg.bp_out_age_curves),
        ]
        if cfg.bp_metric_recency_weights_json is not None:
            cmd.extend(["--metric-recency-weights-json", str(cfg.bp_metric_recency_weights_json)])
        _run_step(name="Stage-1 BP Rate Projection Build", cmd=cmd, cwd=root, dry_run=cfg.dry_run, records=records)

    if not cfg.skip_sandbox_and_two_stage:
        cmd = [
            py,
            "build_projection_sandbox_sets.py",
            "--kpi-input",
            str(lb2_hitter_proj),
            "--traditional-input",
            str(cfg.bp_out_projections),
            "--use-prebuilt-traditional",
            "--naive-input",
            str(cfg.sandbox_naive_input),
            "--old-naive-input",
            str(cfg.sandbox_old_naive_input),
            "--old-advanced-input",
            str(cfg.sandbox_old_advanced_input),
            "--historical-rates-path",
            str(cfg.sandbox_historical_rates_path),
            "--constants-path",
            str(cfg.sandbox_constants_path),
            "--historical-ar-path",
            str(cfg.sandbox_historical_ar_path),
            "--bp-hitting-path",
            str(cfg.sandbox_bp_hitting_path),
            "--stability-pa-per-season",
            str(cfg.sandbox_stability_pa_per_season),
            "--spread-boost-max",
            str(cfg.sandbox_spread_boost_max),
            "--out-kpi",
            str(cfg.sandbox_out_kpi),
            "--out-traditional",
            str(cfg.sandbox_out_traditional),
            "--out-composite",
            str(cfg.sandbox_out_composite),
            "--build-two-stage-fused",
            "--out-two-stage-traditional",
            str(cfg.two_stage_out_traditional),
            "--two-stage-model-bundle",
            str(cfg.two_stage_model_bundle),
            "--two-stage-diagnostics-out",
            str(cfg.two_stage_diagnostics_out),
            "--two-stage-train-start-season",
            str(cfg.two_stage_train_start_season),
            "--two-stage-train-end-season",
            str(cfg.two_stage_train_end_season),
            "--two-stage-cap-sd-mult",
            str(cfg.two_stage_cap_sd_mult),
            "--two-stage-mode",
            str(cfg.two_stage_mode),
            "--two-stage-zspace-min-mlb-pa",
            str(cfg.two_stage_zspace_min_mlb_pa),
        ]
        if cfg.sandbox_no_metric_expert_composite:
            cmd.append("--no-metric-expert-composite")
        if cfg.two_stage_emit_p25p75:
            cmd.append("--two-stage-emit-p25p75")
        _run_step(
            name="Sandbox Stage-1 + Stage-2 Fusion Build",
            cmd=cmd,
            cwd=root,
            dry_run=cfg.dry_run,
            records=records,
        )

    manifest = {
        "started_utc": records[0]["started_utc"] if records else _utc_now_iso(),
        "ended_utc": _utc_now_iso(),
        "repo_root": str(root),
        "python_exe": str(cfg.python_exe),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in cfg.__dict__.items()
        },
        "steps": records,
        "expected_outputs": {
            "kpi_projection_hitter": str(lb2_hitter_proj),
            "bp_stage1_projection": str(cfg.bp_out_projections),
            "bp_stage1_age_curves": str(cfg.bp_out_age_curves),
            "sandbox_kpi": str(cfg.sandbox_out_kpi),
            "sandbox_traditional_stage1": str(cfg.sandbox_out_traditional),
            "sandbox_two_stage_fused": str(cfg.two_stage_out_traditional),
            "two_stage_model_bundle": str(cfg.two_stage_model_bundle),
            "two_stage_diagnostics": str(cfg.two_stage_diagnostics_out),
        },
    }
    out_path = cfg.manifest_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote run manifest to {out_path}")


def main() -> None:
    cfg = _parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
