param(
  [switch]$IncludeRecency311
)

$ErrorActionPreference = "Stop"

$map = @(
  @{
    Source = "projection_outputs/lb2_refresh/pitcher_projections.parquet"
    Dest   = "projection_outputs/sandbox/pitcher_kpi_projections_2026.parquet"
  },
  @{
    Source = "BP_pitching_rate_projections_2026_non_ar_post_inv_coh.parquet"
    Dest   = "projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026.parquet"
  }
)

if($IncludeRecency311){
  $map += @(
    @{
      Source = "projection_outputs/lb2_refresh_recency_3_1_1/pitcher_projections.parquet"
      Dest   = "projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet"
    },
    @{
      Source = "BP_pitching_rate_projections_2026_non_ar_post_inv_coh_recency_3_1_1.parquet"
      Dest   = "projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026_recency_3_1_1.parquet"
    }
  )
}

foreach($item in $map){
  if(-not (Test-Path $item.Source)){
    Write-Error "Missing source file: $($item.Source)"
  }
  $destDir = Split-Path -Parent $item.Dest
  if($destDir){ New-Item -ItemType Directory -Force -Path $destDir | Out-Null }
  Copy-Item -LiteralPath $item.Source -Destination $item.Dest -Force
  Write-Host "Synced $($item.Source) -> $($item.Dest)"
}
