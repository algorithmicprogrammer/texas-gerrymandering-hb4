#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import sys

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

# Optional: used to clean raw PL-94 census TXT into a tidy Parquet
try:
    import duckdb
except Exception:
    duckdb = None


# ---------------- Utility helpers ----------------
def mkdir_p(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_compactness(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute Polsby-Popper compactness for each geometry."""
    gdf = gdf.copy()
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    gdf["compactness"] = 4 * 3.14159 * gdf["area"] / (gdf["perimeter"] ** 2)
    return gdf


def clean_pl94_raw_to_parquet(raw_path: Path, out_parquet_dir: Path) -> Path:
    """
    Reads the raw PL-94 demographics TXT/CSV with DuckDB, computes typed columns
    and share features, and writes a cleaned Parquet named 'tx_pl94_clean.parquet'.
    Returns the output path.

    Expected raw columns: SCTBKEY,total,vap,anglo,black,asian,hisp,
    anglovap,blackvap,asianvap,hispvap
    """
    if duckdb is None:
        raise SystemExit("DuckDB required for --pl94-raw. Install: pip install duckdb")

    mkdir_p(out_parquet_dir)
    out_path = out_parquet_dir / "tx_pl94_clean.parquet"

    con = duckdb.connect()
    con.execute(f"""
    CREATE OR REPLACE TABLE pl94_clean AS
    WITH base AS (
      SELECT
        CAST(SCTBKEY AS VARCHAR) AS geoid20,
        CAST(total   AS BIGINT)  AS total_pop,
        CAST(vap     AS BIGINT)  AS vap_total,

        -- Race/Ethnicity counts (TOTAL)
        CAST(anglo   AS BIGINT)  AS nh_white,
        CAST(black   AS BIGINT)  AS nh_black,
        CAST(asian   AS BIGINT)  AS nh_asian,
        CAST(hisp    AS BIGINT)  AS hispanic,

        -- Race/Ethnicity counts (VAP)
        CAST(anglovap AS BIGINT) AS nh_white_vap,
        CAST(blackvap AS BIGINT) AS nh_black_vap,
        CAST(asianvap AS BIGINT) AS nh_asian_vap,
        CAST(hispvap  AS BIGINT) AS hispanic_vap
      FROM read_csv_auto('{raw_path.as_posix()}', header=True)
    ),
    shares AS (
      SELECT
        *,
        -- Shares (TOTAL)
        (nh_white  ::DOUBLE / NULLIF(total_pop,0)) AS share_nh_white_total,
        (nh_black  ::DOUBLE / NULLIF(total_pop,0)) AS share_nh_black_total,
        (nh_asian  ::DOUBLE / NULLIF(total_pop,0)) AS share_nh_asian_total,
        (hispanic  ::DOUBLE / NULLIF(total_pop,0)) AS share_hispanic_total,

        -- Shares (VAP)
        (nh_white_vap  ::DOUBLE / NULLIF(vap_total,0)) AS share_nh_white_vap,
        (nh_black_vap  ::DOUBLE / NULLIF(vap_total,0)) AS share_nh_black_vap,
        (nh_asian_vap  ::DOUBLE / NULLIF(vap_total,0)) AS share_nh_asian_vap,
        (hispanic_vap  ::DOUBLE / NULLIF(vap_total,0)) AS share_hispanic_vap
      FROM base
    ),
    qa AS (
      SELECT
        *,
        GREATEST(total_pop - COALESCE(nh_white,0) - COALESCE(nh_black,0) - COALESCE(nh_asian,0) - COALESCE(hispanic,0), 0) AS other_pop,
        GREATEST(vap_total - COALESCE(nh_white_vap,0) - COALESCE(nh_black_vap,0) - COALESCE(nh_asian_vap,0) - COALESCE(hispanic_vap,0), 0) AS other_vap
      FROM shares
    )
    SELECT * FROM qa;
    """)
    con.execute(f"COPY pl94_clean TO '{out_path.as_posix()}' (FORMAT PARQUET);")
    con.close()
    return out_path


# ---------------- Stage 1: ETL ----------------
def run_etl(input_files, tabular_dir: Path, geospatial_dir: Path, sqlite_path: Path):
    mkdir_p(tabular_dir)
    mkdir_p(geospatial_dir)
    mkdir_p(sqlite_path.parent)

    # Copy tabular files
    for f in input_files:
        if f.suffix in (".csv", ".tsv", ".parquet"):
            shutil.copy(f, tabular_dir / f.name)

    # Copy geospatial files
    for f in input_files:
        if f.suffix in (".gpkg", ".shp"):
            shutil.copy(f, geospatial_dir / f.name)

    # Load into SQLite
    engine = create_engine(f"sqlite:///{sqlite_path}")
    for f in tabular_dir.glob("*.*"):
        if f.suffix == ".parquet":
            df = pd.read_parquet(f)
        elif f.suffix == ".csv":
            df = pd.read_csv(f)
        elif f.suffix == ".tsv":
            df = pd.read_csv(f, sep="\t")
        else:
            continue
        table_name = f.stem
        df.to_sql(table_name, engine, if_exists="replace", index=False)

# ---------------- Stage 2: Build Final ----------------
def build_final(tabular_dir: Path, geospatial_dir: Path) -> pd.DataFrame:
    # Example final dataset merge placeholder
    dfs = []
    for f in tabular_dir.glob("*.parquet"):
        dfs.append(pd.read_parquet(f))
    if not dfs:
        raise ValueError("No tabular parquet files found for final build")
    final = pd.concat(dfs, axis=1)
    return final


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts", type=Path, required=True, help="Path to districts shapefile/GeoPackage")
    ap.add_argument("--census", type=Path, required=True, help="Path to census blocks shapefile/GeoPackage")
    ap.add_argument("--vtds", type=Path, required=True, help="Path to VTDs shapefile/GeoPackage")
    ap.add_argument("--pl94", type=Path, required=False, help="Path to *cleaned* PL-94 attributes file (CSV/TSV/Parquet)")
    ap.add_argument("--pl94-raw", type=Path, required=False, help="Path to *raw* PL-94 demographics TXT/CSV (SCTBKEY,total,vap,anglo,...)")
    ap.add_argument("--elections", type=Path, required=True, help="Path to elections results file (CSV/Parquet)")
    ap.add_argument("--data-processed-tabular", type=Path, required=True)
    ap.add_argument("--data-processed-geospatial", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    args = ap.parse_args()

    # If raw PL-94 file provided, clean it into processed tabular folder.
    if args.pl94_raw:
        cleaned_pl94 = clean_pl94_raw_to_parquet(args.pl94_raw, args.data_processed_tabular)
    elif args.pl94:
        cleaned_pl94 = args.pl94
    else:
        raise SystemExit("Provide either --pl94 (cleaned) or --pl94-raw (raw TXT/CSV).")

    # Run ETL on all inputs, including the cleaned PL-94 file
    input_files = [args.districts, args.census, args.vtds, cleaned_pl94, args.elections]

    run_etl(input_files, args.data_processed_tabular, args.data_processed_geospatial, args.sqlite)
    final = build_final(args.data_processed_tabular, args.data_processed_geospatial)

    print("Final dataset built with shape:", final.shape)


if __name__ == "__main__":
    main()

