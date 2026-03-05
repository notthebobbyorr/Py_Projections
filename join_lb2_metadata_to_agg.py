from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
import psycopg2

from data_pull import load_db_config


DEFAULT_COLUMNS = [
    "mlbid",
    "season",
    "baseball_age",
    "primary_position",
    "bpid",
    "player_display_text",
    "player_sort_order",
]


def _validate_ident(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def fetch_lb2_metadata(
    min_season: int,
    max_season: int,
    schema: str = "applications",
    table: str = "lb2_player_season_metadata",
    columns: Iterable[str] = DEFAULT_COLUMNS,
) -> pl.DataFrame:
    schema = _validate_ident(schema, "schema")
    table = _validate_ident(table, "table")
    cols = [c for c in columns if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c)]
    if "mlbid" not in cols:
        cols = ["mlbid", *cols]
    if "season" not in cols:
        cols = [*cols, "season"]
    select_cols = ", ".join(cols)
    query = f"""
        SELECT {select_cols}
        FROM {schema}.{table}
        WHERE season >= %s AND season <= %s
    """
    cfg = load_db_config()
    with psycopg2.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
    ) as conn:
        pdf = pd.read_sql_query(query, conn, params=[min_season, max_season])

    df = pl.from_pandas(pdf)
    for col in ["mlbid", "season"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))

    # Guard against potential duplicates in future table versions.
    if set(["mlbid", "season"]).issubset(df.columns):
        exprs = []
        for c in df.columns:
            if c in {"mlbid", "season"}:
                continue
            exprs.append(pl.col(c).drop_nulls().first().alias(c))
        if exprs:
            df = df.group_by(["mlbid", "season"]).agg(exprs)
        else:
            df = df.unique(subset=["mlbid", "season"])
    return df


def read_any(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_any(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.write_parquet(path)
        return
    if path.suffix.lower() == ".csv":
        df.write_csv(path)
        return
    raise ValueError(f"Unsupported file type: {path}")


def _output_path(path: Path, out_dir: Path, overwrite: bool, suffix: str) -> Path:
    if overwrite:
        return path
    return out_dir / f"{path.stem}{suffix}{path.suffix}"


def _safe_int_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    return df.with_columns(pl.col(col).cast(pl.Int64, strict=False))


def _coalesce_age(existing_col: str, incoming_col: str) -> pl.Expr:
    existing = pl.col(existing_col).cast(pl.Float64, strict=False)
    incoming = pl.col(incoming_col).cast(pl.Float64, strict=False)
    return (
        pl.when(existing.is_null() | (existing <= 0))
        .then(incoming)
        .otherwise(existing)
        .alias(existing_col)
    )


def join_dataset(
    src_path: Path,
    out_path: Path,
    id_col: str,
    metadata: pl.DataFrame,
    add_prefix: str = "lb2_",
) -> tuple[int, int, int]:
    df = read_any(src_path)
    rows_before = df.height
    if rows_before == 0:
        write_any(df, out_path)
        return rows_before, 0, 0

    df = _safe_int_col(df, id_col)
    df = _safe_int_col(df, "season")
    md = metadata.rename({"mlbid": "__join_mlbid", "season": "__join_season"})

    keep_cols = ["__join_mlbid", "__join_season"]
    md_cols = [c for c in md.columns if c not in keep_cols]
    rename_map = {}
    for col in md_cols:
        if col == "baseball_age":
            rename_map[col] = "__lb2_baseball_age"
        else:
            rename_map[col] = f"{add_prefix}{col}"
    md = md.rename(rename_map)
    keep_cols += list(rename_map.values())
    md = md.select([c for c in keep_cols if c in md.columns])

    joined = df.join(
        md,
        left_on=[id_col, "season"],
        right_on=["__join_mlbid", "__join_season"],
        how="left",
    )

    if "__lb2_baseball_age" in joined.columns:
        if "baseball_age" in joined.columns:
            joined = joined.with_columns(_coalesce_age("baseball_age", "__lb2_baseball_age"))
        else:
            joined = joined.rename({"__lb2_baseball_age": "baseball_age"})
        joined = joined.drop("__lb2_baseball_age")

    write_any(joined, out_path)
    matched = joined.filter(pl.col(f"{add_prefix}bpid").is_not_null()).height if f"{add_prefix}bpid" in joined.columns else 0
    age_nonnull = joined.filter(pl.col("baseball_age").cast(pl.Float64, strict=False).is_not_null()).height if "baseball_age" in joined.columns else 0
    return rows_before, matched, age_nonnull


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join applications.lb2_player_season_metadata to aggregated files "
            "(same inputs used by apply_regression_from_agg.py) on mlbid+season."
        )
    )
    parser.add_argument("--min-season", type=int, default=2015)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument("--schema", default="applications")
    parser.add_argument("--table", default="lb2_player_season_metadata")
    parser.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    parser.add_argument("--hitters", type=Path, default=Path("damage_pos_2015_2025.parquet"))
    parser.add_argument("--pitchers", type=Path, default=Path("pitcher_stuff_new.parquet"))
    parser.add_argument("--pitch-types", type=Path, default=Path("new_pitch_types.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--suffix", default="_with_lb2_meta")
    args = parser.parse_args()

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

    tasks = [
        ("hitters", args.hitters, "batter_mlbid"),
        ("pitchers", args.pitchers, "pitcher_mlbid"),
        ("pitch_types", args.pitch_types, "pitcher_mlbid"),
    ]
    for name, src, id_col in tasks:
        out = _output_path(src, args.out_dir, args.overwrite, args.suffix)
        rows, matched, age_nonnull = join_dataset(src, out, id_col=id_col, metadata=metadata)
        print(
            f"{name}: wrote {out} | rows={rows:,} "
            f"| matched_rows={matched:,} | baseball_age_nonnull={age_nonnull:,}"
        )


if __name__ == "__main__":
    main()
