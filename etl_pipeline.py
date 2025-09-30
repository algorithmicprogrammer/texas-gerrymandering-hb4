# ETL + 38-district mart (compactness + race % + partisanship)

from __future__ import annotations
import argparse, math, re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# ---------------- Geo deps ----------------
try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit("GeoPandas required. Install: pip install geopandas shapely fiona pyproj") from e

# attempts to import Shapely's minimum_bounding_radius for Reock compactness
try:
    from shapely import minimum_bounding_radius
    HAS_MINCIRCLE = True
except Exception:
    HAS_MINCIRCLE = False

# ---------------- Config ------------------
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet")
SUPPORTED_GEO = (".gpkg", ".shp")
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO
AREA_CRS = "EPSG:3083"

# ----- helpers -----
def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_")
        for c in df.columns
    ]
    return df

def coerce_types_light(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({"": pd.NA, "na": pd.NA, "null": pd.NA, "none": pd.NA})
            )
    for c in out.columns:
        if re.search(r"(date|time|timestamp|dt)$", c):
            out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        s = out[c].dropna().astype(str).head(12)
        if len(s) and (s.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.6):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def dataset_key(path: Path) -> str:
    return re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", path.stem.lower())).strip("_")

def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in SUPPORTED_GEO:
        return gpd.read_file(path)
    raise ValueError(f"Unsupported ext: {ext}")


def is_geodf(df: pd.DataFrame) -> bool:
    return isinstance(df, gpd.GeoDataFrame)


def to_sqlite(df: pd.DataFrame, sqlite_path: Path, table: str) -> None:
    eng = create_engine(f"sqlite:///{sqlite_path}")
    with eng.begin() as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    mkdir_p(out_path.parent)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def write_geo_parquet(gdf: "gpd.GeoDataFrame", out_path: Path) -> None:
    mkdir_p(out_path.parent)
    gdf.to_parquet(out_path, index=False)


def ensure_crs(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def geo_to_sql_ready(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    out = gdf.copy()
    if "geometry" in out.columns:
        out["geometry_wkb"] = out.geometry.to_wkb(hex=True)
        b = out.geometry.bounds
        out["bbox_minx"] = b["minx"]
        out["bbox_miny"] = b["miny"]
        out["bbox_maxx"] = b["maxx"]
        out["bbox_maxy"] = b["maxy"]
        out = out.drop(columns=["geometry"])
    return out


def align_columns(dfs):
    cols = set()
    for d in dfs:
        cols.update(d.columns)
    cols = list(sorted(cols))
    return [d.reindex(columns=cols) for d in dfs]


def normalize_vtd_key(df: pd.DataFrame, prefer="cntyvtd") -> pd.DataFrame:
    for c in ["cntyvtd", "cntyvtdkey", "cnty_vtd", "vtdkey", "vtd_key", "county_vtd_key"]:
        if c in df.columns:
            return df if c == prefer else df.rename(columns={c: prefer})
    raise AssertionError("No recognizable VTD key (cntyvtd/cntyvtdkey/…)")


def compute_compactness(districts_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = districts_gdf.to_crs(AREA_CRS).copy()
    A, P = g.geometry.area, g.geometry.length
    polsby = (4 * math.pi * A) / (P ** 2)
    schwartz = ((4 * math.pi * A) ** 0.5) / P
    hull_ratio = A / g.geometry.convex_hull.area.replace({0: pd.NA})
    if HAS_MINCIRCLE:
        radii = g.geometry.map(minimum_bounding_radius)
        reock = A / (math.pi * (radii ** 2))
    else:
        reock = pd.Series([pd.NA] * len(g), index=g.index)
    return pd.DataFrame(
        {
            "polsby_popper": polsby,
            "schwartzberg": schwartz,
            "convex_hull_ratio": hull_ratio,
            "reock": reock,
        },
        index=g.index,
    )

# ---------------- Stage 1: ETL ----------------
# NOW accepts an explicit list of input files instead of scanning a directory

def run_etl(input_files: list[Path], out_parquet_dir: Path, out_geo_dir: Path, sqlite_path: Path):
    print("\n========== Stage 1: ETL ==========")
    mkdir_p(out_parquet_dir)
    mkdir_p(out_geo_dir)
    mkdir_p(sqlite_path.parent)

    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if path.suffix.lower() not in ALL_EXTS:
            raise ValueError(f"Unsupported input type for {path.name} (ext {path.suffix})")

        name = dataset_key(path)
        df = read_any(path).drop_duplicates(ignore_index=True)

        if is_geodf(df):
            df = ensure_crs(df)
            attrs = stdcols(df.drop(columns=["geometry"])) if "geometry" in df.columns else stdcols(df)
            attrs = coerce_types_light(attrs)
            gdf = gpd.GeoDataFrame(attrs, geometry=df.get("geometry"), crs=df.crs)
            write_geo_parquet(gdf, out_geo_dir / f"{name}.parquet")
            to_sqlite(geo_to_sql_ready(gdf), sqlite_path, f"stg_{name}")
        else:
            df = coerce_types_light(stdcols(df))
            write_parquet(df, out_parquet_dir / f"{name}.parquet")
            to_sqlite(df, sqlite_path, f"stg_{name}")

    print("========== ETL complete ==========\n")

# ------------- Stage 2: Build final  -------------

def build_final(out_parquet_dir: Path, out_geo_dir: Path) -> pd.DataFrame:
    print("========== Stage 2: Build final 38-row dataset ==========")

    # Read cleaned layers produced by Stage 1 (filenames are derived from stems of the five inputs)
    districts = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir / "districts_clean.parquet")))
    census_geo = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir / "texas_census_blocks_clean.parquet")))
    vtd_geo = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir / "vtds_geo_clean.parquet")))
    pl94 = stdcols(pd.read_parquet(out_parquet_dir / "tx_pl94_clean.parquet"))
    vtd_elec = stdcols(pd.read_parquet(out_parquet_dir / "clean_vtd_election_results.parquet"))

    # ----- explicit PL-94 fields -----
    PL94_TOTAL_VAP = "vap_total"
    PL94_RACE_VAP = {
        "pct_white": "nh_white_vap",
        "pct_black": "nh_black_vap",
        "pct_asian": "nh_asian_vap",
        "pct_hispanic": "hispanic_vap",
    }
    need = ["geoid20", PL94_TOTAL_VAP, *PL94_RACE_VAP.values()]
    missing = [c for c in need if c not in pl94.columns]
    if missing:
        raise ValueError(f"PL-94 expected columns missing: {missing}")

    if "geoid20" not in census_geo.columns:
        raise ValueError("Census geometry must contain 'geoid20'")

    census_merged = census_geo.merge(pl94[need], on="geoid20", how="left")

    # project to equal-area CRS
    census_proj, districts_proj = census_merged.to_crs(AREA_CRS), districts.to_crs(AREA_CRS)

    # Census -> Districts (within)
    c2d = gpd.sjoin(
        census_proj,
        districts_proj[["geometry"]].reset_index().rename(columns={"index": "district_idx"}),
        how="inner",
        predicate="within",
    )

    agg = c2d.groupby("district_idx")[[PL94_TOTAL_VAP] + list(PL94_RACE_VAP.values())].sum(min_count=1)
    den = agg[PL94_TOTAL_VAP].replace({0: pd.NA})
    race_pct = pd.DataFrame(index=agg.index)
    for out_name, src_col in PL94_RACE_VAP.items():
        race_pct[out_name] = (agg[src_col] / den).where(den > 0)

    # VTDs + Elections -> Districts
    vtd_geo = normalize_vtd_key(vtd_geo, "cntyvtd")
    vtd_elec = normalize_vtd_key(vtd_elec, "cntyvtd")
    vtd_with_votes = vtd_geo.merge(vtd_elec, on="cntyvtd", how="left").to_crs(AREA_CRS)

    v2d = gpd.sjoin(
        vtd_with_votes,
        districts_proj[["geometry"]].reset_index().rename(columns={"index": "district_idx"}),
        how="inner",
        predicate="intersects",
    )

    vote_cols = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"] if c in v2d.columns]
    votes_by_dist = (
        v2d.groupby("district_idx")[vote_cols].sum(min_count=1) if vote_cols else pd.DataFrame(index=districts_proj.index)
    )

    part = pd.DataFrame(index=votes_by_dist.index if len(votes_by_dist) else districts_proj.index)
    if {"dem_votes", "rep_votes"}.issubset(votes_by_dist.columns):
        tot = (
            votes_by_dist["total_votes"]
            if "total_votes" in votes_by_dist.columns
            else (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"] + votes_by_dist.get("third_party_votes", 0))
        )
        part["dem_share"] = (votes_by_dist["dem_votes"] / tot).where(tot > 0)
        part["rep_share"] = (votes_by_dist["rep_votes"] / tot).where(tot > 0)
    else:
        part["dem_share"] = pd.NA
        part["rep_share"] = pd.NA

    # Compactness on districts
    cmpx = compute_compactness(districts_proj)

    # Assemble final
    final = pd.DataFrame(index=districts_proj.index)

    id_col = None
    for cand in ["district", "district_id", "cd", "dist"]:
        if cand in districts.columns:
            id_col = cand
            break
    final["district_id"] = districts[id_col].values if id_col else (districts_proj.index + 1)

    final = final.join(cmpx).join(race_pct).join(part[["dem_share", "rep_share"]])
    final = final.sort_index()
    assert len(final) == 38, f"Expected 38 districts, found {len(final)}"
    return final

# ---------------- CLI ----------------
# Now accepts FIVE explicit input paths instead of a directory scan

def main():
    ap = argparse.ArgumentParser(description="ETL + final 38-district dataset")
    ap.add_argument("--districts", type=Path, required=True, help="Path to district polygons file (SHP/GPKG/GeoParquet)")
    ap.add_argument("--census", type=Path, required=True, help="Path to census blocks geo file (SHP/GPKG/GeoParquet)")
    ap.add_argument("--vtds", type=Path, required=True, help="Path to VTD geometries file (SHP/GPKG/GeoParquet)")
    ap.add_argument("--pl94", type=Path, required=True, help="Path to PL-94 attributes file (CSV/TSV/Parquet)")
    ap.add_argument("--elections", type=Path, required=True, help="Path to VTD election results file (CSV/TSV/Parquet)")

    ap.add_argument("--data-processed-tabular", type=Path, required=True, help="Output folder for cleaned Parquet")
    ap.add_argument("--data-processed-geospatial", type=Path, required=True, help="Output folder for cleaned GeoParquet")
    ap.add_argument("--sqlite", type=Path, required=True, help="SQLite warehouse path")
    args = ap.parse_args()

    input_files = [args.districts, args.census, args.vtds, args.pl94, args.elections]

    run_etl(input_files, args.data_processed_tabular, args.data_processed_geospatial, args.sqlite)
    final = build_final(args.data_processed_tabular, args.data_processed_geospatial)

    pq = args.data_processed_tabular / "districts_final.parquet"
    csv = args.data_processed_tabular / "districts_final.csv"
    write_parquet(final, pq)
    final.to_csv(csv, index=False)
    print("\nSaved:")
    print(" ", pq)
    print(" ", csv)

if __name__ == "__main__":
    main()
