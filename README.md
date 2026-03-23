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

## Changelog

### 2026-03-23 - Roster Manager: fixed 15-slot reserve tables with st.form; show_table Apply Filters gate (projection_streamlit.py)

**Reserve Manager: fixed 15-slot st.form tables**
- Hitter and pitcher reserves now always display exactly 15 fixed slots (ROSTER_RESERVE_SLOTS = 15) instead of dynamically added/removed rows.
- Add/Remove buttons removed; all 15 player+pct selectboxes are inside st.form - no page reruns until Apply Changes is clicked.
- On Apply: player_id and percentile are committed to state; PA/IP number inputs, position/role picker, and manual stats controls render live beneath the form for committed (non-None) slots only.
- _roster_default_state() pre-populates 15 empty rows; _roster_prepare_live_state() and _deserialize_roster_state() pad/trim saved rosters to exactly 15 slots for backward compatibility.
- Widget keys changed from row_id-based to index-based (_hres_pick_{idx}, _pres_pick_{idx}).

**Projection Sandbox show_table: Apply Filters gate**
- All filter/sort/metrics/percentile widgets inside the show_table controls expander are now wrapped in st.form; no page rerun occurs on individual widget interaction.
- Apply Filters submit button at the bottom of the controls panel commits all selections at once.



### 2026-03-22 — Projection Sandbox: st.form Apply Filters gate on show_table controls (`projection_streamlit.py`)

**Apply Filters button on all show_table control panels**
- All filter/sort/metric/percentile widgets inside each table's controls expander are now wrapped in `st.form`; no page rerun occurs on any individual widget change.
- An "Apply Filters" submit button (primary) at the bottom of the controls expander commits all selections at once and triggers a single rerun.
- Affected widgets per table: season, source level, team, opening day status, position, player, min PA/IP quantity, OPS/ERA/WHIP environment checkboxes, percentile selector, metrics selector, sort metric, sort direction.
- Applies to all `show_table` calls: Hitters tab, Pitchers tab, and any other tab using `show_table`.
- `_apply_column_filters` (post-table column-level numeric/string filters) remains live outside the form.

### 2026-03-21 — Roster Manager: st.form Apply Changes + inline position/role pickers (`projection_streamlit.py`)

**Apply Changes via st.form (no page refresh on dropdown interaction)**
- Hitter and pitcher starter player + pct selections are wrapped in `st.form`; Streamlit does not rerun the page on dropdown interaction inside a form.
- An "Apply Changes" submit button commits all starter picks and pct selections to state at once, then reruns.
- Reserve picks remain live (immediate state write) since Add/Remove buttons require reruns outside a form.

**Inline position and role pickers (slot-based)**
- Each committed hitter starter slot shows a "Counts toward" selectbox (inside the form) listing eligible positions + UT, defaulting to the slot's natural position (e.g., C1→C, MI→2B, CI→1B, UT→UT).
- Each committed pitcher starter slot shows a "Counts toward" selectbox (SP/RP) inside the form, only visible when the player has dual eligibility; role defaults to slot natural (SP*/RP).
- Reserve position/role pickers remain live (outside form), keyed by row_id.
- Eligibility Counter reads from slot-based keys (`_helig_slot_{slot}`, `_pelig_slot_{slot}`) for starters and row_id-based keys for reserves.
