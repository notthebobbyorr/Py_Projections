param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Join-Path (Get-Location) "projections_v1")

Write-Host "Building LB2 KPI projections with 3/1/1 recency weights..."
& $PythonExe -m projections_v1.run --config projections_v1/projection_config_lb2_recency_3_1_1.yml
if($LASTEXITCODE -ne 0){ throw "KPI projection build failed." }

Write-Host "Building BP pitcher projections with 3/1/1 recency weights..."
$bpInputPath = "BP_pitching_single_source_mlb_eq_non_ar_delta.parquet"
$bpConstantsPath = "BP_pitching_non_ar_constants_2015_2025.parquet"
$bpHistoricalRawPath = "BP_pitching_equivalents_2015_2025_non_ar_delta.parquet"
if(-not (Test-Path $bpInputPath)){ $bpInputPath = "..\\Py_Damage\\BP_pitching_single_source_mlb_eq_non_ar_delta.parquet" }
if(-not (Test-Path $bpConstantsPath)){ $bpConstantsPath = "..\\Py_Damage\\BP_pitching_non_ar_constants_2015_2025.parquet" }
if(-not (Test-Path $bpHistoricalRawPath)){ $bpHistoricalRawPath = "..\\Py_Damage\\BP_pitching_equivalents_2015_2025_non_ar_delta.parquet" }

& $PythonExe build_bp_pitching_rate_projections_2026_non_ar_post_inv_coh.py `
  --input-path $bpInputPath `
  --constants-path $bpConstantsPath `
  --historical-raw-path $bpHistoricalRawPath `
  --recency-weights 3 1 1 `
  --out-projections BP_pitching_rate_projections_2026_non_ar_post_inv_coh_recency_3_1_1.parquet `
  --out-age-curves BP_pitching_rate_age_curves_2015_2025_non_ar_post_inv_coh_recency_3_1_1.parquet
if($LASTEXITCODE -ne 0){ throw "BP pitcher projection build failed." }

Write-Host "Syncing 3/1/1 files into sandbox paths..."
$destDir = "projection_outputs/sandbox"
if(-not (Test-Path $destDir)){ New-Item -ItemType Directory -Force -Path $destDir | Out-Null }
Copy-Item -LiteralPath "projection_outputs/lb2_refresh_recency_3_1_1/pitcher_projections.parquet" `
  -Destination "projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet" -Force
Copy-Item -LiteralPath "BP_pitching_rate_projections_2026_non_ar_post_inv_coh_recency_3_1_1.parquet" `
  -Destination "projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026_recency_3_1_1.parquet" -Force

Write-Host "Done. New sandbox files:"
Write-Host " - projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet"
Write-Host " - projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026_recency_3_1_1.parquet"
