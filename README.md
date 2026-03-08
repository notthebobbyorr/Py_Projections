# Py_Projections (Standalone subset)

This folder is a trimmed copy of the projection app + artifacts focused on 3 projection sources:

1. Traditional Two-Stage (KPI Adjusted, p50)
2. Two Stage KPI Projections, Pitchers
3. KPI Projections (Individual Stat Skills)

## App

- `projection_streamlit.py` was copied from `Py_Damage` and restricted to only the 3 sources above.
- Run locally:

```powershell
streamlit run projection_streamlit.py
```

## Included projection artifacts

- `projection_outputs/sandbox/kpi_projections_2026.parquet`
- `projection_outputs/sandbox/pitcher_kpi_projections_2026.parquet`
- `projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet` (optional 3/1/1 variant)
- `projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet`
- `projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026.parquet`
- `projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026_recency_3_1_1.parquet` (optional 3/1/1 variant)
- `projection_outputs/sandbox/traditional_projections_2026.parquet` (used by Before vs After tab)

## Included process scripts

- `two_stage_pipeline/run_repro_pipeline.py`
- `build_projection_sandbox_sets.py`
- `build_bp_rate_projections_2026_non_ar_post_inv_coh.py`
- `build_bp_pitching_rate_projections_2026_non_ar_post_inv_coh.py`
- `enrich_and_regress_from_agg.py`
- `apply_regression_from_agg.py`
- `join_lb2_metadata_to_agg.py`
- `projections_v1/` (full package + `projection_config_lb2.yml`)

## Regeneration notes

- Hitter KPI + two-stage outputs are produced by the two-stage runner.
- Pitcher KPI and pitcher BP output use different file names upstream. After regenerating upstream outputs, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_selected_projection_files.ps1
```

This script syncs upstream outputs into the sandbox filenames used by `projection_streamlit.py`.

For 3/1/1 pitcher variants, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_pitcher_recency_3_1_1_variants.ps1
```

## GitHub note

- Review `.streamlit/secrets.toml` before pushing. It may contain secrets and should usually be excluded.

## Added damage interface (open access)

- `damage_streamlit.py` is now copied into this repo and patched to run without login/subscription gating.
- Required damage datasets were copied as `.parquet` files (same names expected by the app's CSV loader).

Launch both interfaces from one entrypoint:

```powershell
streamlit run projection_streamlit.py
```

Then use the sidebar `Application` selector to switch between `Projection Sandbox` and `Damage Interface`.
