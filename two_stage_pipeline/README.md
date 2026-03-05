# Two-Stage Projection Repro Module

This module codifies the full reproducible build path for the current two-stage hitter projections:

1. KPI projection build from the original KPI training datasets.
2. Stage-1 BP projection build from the original BP training dataset.
3. Sandbox stage-1 post-processing and stage-2 KPI residual fusion.

It is designed so both a human operator and a Codex agent can run the same sequence with the same entrypoint.

## Scope

Current entrypoint:

- Script: `two_stage_pipeline/run_repro_pipeline.py`

Current methodology source of truth remains in the existing production scripts:

- KPI building blocks:
  - `enrich_and_regress_from_agg.py`
  - `projections_v1/run.py` (with `projections_v1/projection_config_lb2.yml`)
- Stage-1 BP building block:
  - `build_bp_rate_projections_2026_non_ar_post_inv_coh.py`
- Sandbox stage-1 + stage-2 fusion:
  - `build_projection_sandbox_sets.py`

This module orchestrates those scripts in sequence without re-implementing their internals.

## End-to-End Flow

### Step A: LB2 Enrich + Regress (KPI upstream inputs)

Builds enriched and regressed datasets used by KPI projections.

Input datasets (defaults):

- `damage_pos_2015_2025.parquet`
- `pitcher_stuff_new.parquet`
- `new_pitch_types.parquet`
- `stability_config.yml`
- `stability_constants.csv`

Output directory (default):

- `projection_outputs/lb2_refresh`

### Step B: KPI Projection Output

Runs the KPI projection pipeline (v1 LB2 config).

Config:

- `projections_v1/projection_config_lb2.yml`

Primary output used downstream:

- `projection_outputs/lb2_refresh/hitter_projections.parquet`

### Step C: Stage-1 BP Projection Output

Builds the BP stage-1 rate projection file from BP historical training data.

Default inputs:

- `BP_single_source_mlb_eq_non_ar_delta.parquet`
- `BP_data_AR_2015_2025_constants.parquet`
- `BP_data_AR_2015_2025.parquet`
- `projection_outputs/bp_hitting_api/bp_hitting_table.parquet`

Primary outputs:

- `BP_rate_projections_2026_non_ar_post_inv_coh_no_z_anchor.parquet`
- `BP_rate_age_curves_2015_2025_non_ar_post_inv_coh_no_z_anchor.parquet`

### Step D: Sandbox Stage-1 + Stage-2 Fusion

Runs `build_projection_sandbox_sets.py` using:

- KPI input from Step B (`hitter_projections.parquet`)
- prebuilt stage-1 BP input from Step C
- two-stage fusion enabled

Primary outputs:

- `projection_outputs/sandbox/kpi_projections_2026.parquet`
- `projection_outputs/sandbox/traditional_projections_2026.parquet`
- `projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet`
- `projection_outputs/sandbox/two_stage_models_2026.joblib`
- `projection_outputs/sandbox/two_stage_diagnostics_2026.parquet`

## Repro Command

From repo root:

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py
```

## Useful Flags

Skip completed upstream steps:

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py `
  --skip-enrich-regress `
  --skip-kpi-projection `
  --skip-bp-stage1
```

Enable p25/p75 emission in stage-2:

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py `
  --two-stage-emit-p25p75
```

Run stage-2 in z-space mode (current prototype path in `build_projection_sandbox_sets.py`):

```powershell
.\.venv\Scripts\python.exe build_projection_sandbox_sets.py `
  --use-prebuilt-traditional `
  --no-metric-expert-composite `
  --build-two-stage-fused `
  --two-stage-mode zspace
```

In `zspace` mode, reference parameters are built from the MLB-appeared population (`appeared_in_MLB == T`) with:

- `mu = median`
- `sigma = std (population, ddof=0)`
- default filter: `MLB PA >= 200` for the reference season
- season basis: prior season (`source_season`) with fallback to nearest available season
- two-stage target set now includes `PA` (mapped to `plate_appearances_agg` actuals)

You can override the PA filter threshold:

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py `
  --two-stage-mode zspace `
  --two-stage-zspace-min-mlb-pa 200
```

The orchestration runner also supports this directly:

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py --two-stage-mode zspace
```

Dry run (print commands only):

```powershell
.\.venv\Scripts\python.exe two_stage_pipeline\run_repro_pipeline.py --dry-run
```

## Manifest

Each run writes a manifest JSON with:

- resolved arguments
- executed command list
- timestamps
- expected output paths

Default manifest path:

- `projection_outputs/sandbox/two_stage_repro_manifest.json`

## Notes

- This module is orchestration-only by design; it calls the existing production scripts directly.
- If upstream script defaults change, re-run with explicit flags or update this module defaults to stay locked to your preferred recipe.
