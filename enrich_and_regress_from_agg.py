from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from apply_regression_from_agg import (
    add_league_contact_baseline,
    apply_mean_from_agg,
    apply_rate_from_agg,
    load_config as load_stability_config,
    merge_frames,
    round_floats,
)
from join_lb2_metadata_to_agg import fetch_lb2_metadata, join_dataset


def _enriched_path(src: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{src.stem}{suffix}{src.suffix}"


def run_regression_on_enriched(
    *,
    dataset: str,
    path: Path,
    keys: list[str],
    stats_cfg: dict[str, Any],
    constants: pl.DataFrame,
    round_places: int,
    out_dir: Path,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing enriched input for {dataset}: {path}")
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    frames: list[pl.DataFrame] = []
    for stat in stats_cfg.get(dataset, []):
        kind = stat.get("kind")
        name = stat.get("name")
        if kind == "derived":
            continue
        if kind == "rate":
            frame = apply_rate_from_agg(df, constants, dataset, name, keys)
        else:
            frame = apply_mean_from_agg(df, constants, dataset, name, keys)
        if frame is not None:
            frames.append(frame)

    if dataset == "hitters" and "whiff_rate_num" in df.columns:
        frame = apply_rate_from_agg(df, constants, dataset, "whiff_rate", keys)
        if frame is not None:
            frames.append(frame)

    base = df.select([c for c in keys if c in df.columns]).unique()
    merged = merge_frames(base, frames, keys)

    if dataset == "hitters":
        merged = add_league_contact_baseline(merged, df)
        if (
            "selection_skill_raw" in merged.columns
            and "hittable_pitches_taken_raw" in merged.columns
        ):
            merged = merged.with_columns(
                (pl.col("selection_skill_raw") - pl.col("hittable_pitches_taken_raw")).alias(
                    "SEAGER_raw"
                )
            )
        if (
            "selection_skill_reg" in merged.columns
            and "hittable_pitches_taken_reg" in merged.columns
        ):
            merged = merged.with_columns(
                (pl.col("selection_skill_reg") - pl.col("hittable_pitches_taken_reg")).alias(
                    "SEAGER_reg"
                )
            )
        if (
            "pred_whiff_loc_mean_raw" in merged.columns
            and "whiff_rate_raw" in merged.columns
        ):
            if "lg_contact_baseline" in merged.columns:
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_raw")
                        - pl.col("whiff_rate_raw")
                        - pl.col("lg_contact_baseline")
                    ).alias("contact_vs_avg_raw")
                )
            else:
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_raw")
                        - pl.col("whiff_rate_raw")
                    ).alias("contact_vs_avg_raw")
                )
        if (
            "pred_whiff_loc_mean_raw" in merged.columns
            and "whiff_rate_reg" in merged.columns
        ):
            if (
                "pred_whiff_loc_mean_reg" in merged.columns
                and "lg_contact_baseline" in merged.columns
            ):
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_reg")
                        - pl.col("whiff_rate_reg")
                        - pl.col("lg_contact_baseline")
                    ).alias("contact_vs_avg_reg")
                )
            elif "pred_whiff_loc_mean_reg" in merged.columns:
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_reg")
                        - pl.col("whiff_rate_reg")
                    ).alias("contact_vs_avg_reg")
                )
        drop_cols = [c for c in merged.columns if c.startswith("whiff_rate_")]
        if drop_cols:
            merged = merged.drop(drop_cols)

    merged = round_floats(merged, round_places)
    out_path = out_dir / f"{dataset}_regressed.parquet"
    merged.write_parquet(out_path)
    print(f"Wrote {len(merged):,} rows to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-pass pipeline: LB2 metadata join + regression from aggregate files."
        )
    )
    parser.add_argument("--min-season", type=int, default=2015)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument("--schema", default="applications")
    parser.add_argument("--table", default="lb2_player_season_metadata")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=[
            "mlbid",
            "season",
            "baseball_age",
            "primary_position",
            "bpid",
            "player_display_text",
            "player_sort_order",
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("stability_config.yml"),
        help="Stability config used for regression.",
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=Path("stability_constants.csv"),
        help="Path to stability_constants.csv.",
    )
    parser.add_argument(
        "--hitters",
        type=Path,
        default=Path("damage_pos_2015_2025.parquet"),
    )
    parser.add_argument(
        "--pitchers",
        type=Path,
        default=Path("pitcher_stuff_new.parquet"),
    )
    parser.add_argument(
        "--pitch-types",
        type=Path,
        default=Path("new_pitch_types.parquet"),
    )
    parser.add_argument(
        "--enriched-out-dir",
        type=Path,
        default=Path("projection_outputs/lb2_refresh"),
        help="Directory for LB2-enriched aggregate files.",
    )
    parser.add_argument(
        "--regressed-out-dir",
        type=Path,
        default=Path("projection_outputs/lb2_refresh"),
        help="Directory for regressed outputs generated from enriched files.",
    )
    parser.add_argument(
        "--enriched-suffix",
        default="_with_lb2_meta",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Decimal places for regression outputs.",
    )
    args = parser.parse_args()

    args.enriched_out_dir.mkdir(parents=True, exist_ok=True)
    args.regressed_out_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_lb2_metadata(
        min_season=args.min_season,
        max_season=args.max_season,
        schema=args.schema,
        table=args.table,
        columns=args.columns,
    )
    print(
        f"Fetched metadata rows: {metadata.height:,} "
        f"for seasons {args.min_season}-{args.max_season}"
    )

    inputs = {
        "hitters": (args.hitters, "batter_mlbid"),
        "pitchers": (args.pitchers, "pitcher_mlbid"),
        "pitch_types": (args.pitch_types, "pitcher_mlbid"),
    }
    enriched_paths: dict[str, Path] = {}
    for dataset, (src_path, id_col) in inputs.items():
        out = _enriched_path(src_path, args.enriched_out_dir, args.enriched_suffix)
        rows, matched, age_nonnull = join_dataset(
            src_path=src_path,
            out_path=out,
            id_col=id_col,
            metadata=metadata,
        )
        enriched_paths[dataset] = out
        print(
            f"{dataset}: wrote {out} | rows={rows:,} "
            f"| matched_rows={matched:,} | baseball_age_nonnull={age_nonnull:,}"
        )

    stability_cfg = load_stability_config(args.config)
    stats_cfg = stability_cfg.get("stats", {})
    constants = pl.read_csv(args.constants)

    regression_keys = {
        "hitters": ["batter_mlbid", "hitter_name", "season", "level_id"],
        "pitchers": ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand"],
        "pitch_types": [
            "pitcher_mlbid",
            "name",
            "pitcher_hand",
            "season",
            "level_id",
            "pitch_tag",
        ],
    }
    for dataset, keys in regression_keys.items():
        run_regression_on_enriched(
            dataset=dataset,
            path=enriched_paths[dataset],
            keys=keys,
            stats_cfg=stats_cfg,
            constants=constants,
            round_places=args.round,
            out_dir=args.regressed_out_dir,
        )


if __name__ == "__main__":
    main()
