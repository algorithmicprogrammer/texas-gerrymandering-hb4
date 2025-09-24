# ETL + 38-district mart (compactness + race % + partisanship)

# enables postponed valuations of type annotations
from __future__ import annotations
# argpase parses CLI flags (--input, etc)
import argparse, math, re
from pathlib import Path
import pandas as pd
#from PyQt5.QtCore import lowercasedigits
#creates a SQLite engine to stash "staging" tables
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
# allowed non-spatial file extensions
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet")
# allows geospatial file extensions
SUPPORTED_GEO = (".gpkg", ".shp")
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO
# CRS projection used for area/length/compactness
AREA_CRS = "EPSG:3083"  

# ----- helpers -----
# ensures a directory if it exists. Safe if it already exists.
def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# copies the dataframe
# for each column:
# lowercases
# trims whitespace
# replaces any non-word characters with _
# collapses multiple underscores to one, then strips leading/trailing underscores
# returns standardized columns dataframe
def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_")
        for c in df.columns
    ]
    return df

# "light-touch" typing
# copy input
# for object/string columns: map strings; map empty/"na"/"null"/"none" to pd.NA
# if column name ends with date/time/timestamp/dt: parse to datetime invalid -> NaT
#for non-datetime columns: peek at up to 12 non-NA values; if >60% look like numbers, coerce with errors ->NA. Return typed frame.
def coerce_types_light(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].astype(str).str.strip().replace({"": pd.NA, "na": pd.NA, "null": pd.NA, "none": pd.NA})
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

# make a safe predictable base name from a filename (mirrors stdcols logic) for output files/table names
def dataset_key(path: Path) -> str:
    return re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", path.stem.lower())).strip("_")

#reads a file into a (geo) dataframe based on extensions
def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv": return pd.read_csv(path)
    if ext == ".tsv": return pd.read_csv(path, sep="\t")
    if ext == ".parquet": return pd.read_parquet(path)
    if ext in SUPPORTED_GEO: return gpd.read_file(path)
    raise ValueError(f"Unsupported ext: {ext}")

#typeguard: "is this dataframe geospatial?"
def is_geodf(df: pd.DataFrame) -> bool:
    return isinstance(df, gpd.GeoDataFrame)

# opens a SQLite database (creates if needed) and writes the dataframe as a table (replacing any existing)
# uses a transaction (eng.begin())
def to_sqlite(df: pd.DataFrame, sqlite_path: Path, table: str) -> None:
    eng = create_engine(f"sqlite:///{sqlite_path}")
    with eng.begin() as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)

#ensures directory exists, writes parquet with pyarrow, no index
def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    mkdir_p(out_path.parent); df.to_parquet(out_path, engine="pyarrow", index=False)

#same idea for geo dataframes (Geoparquet)
# geopandas writes geometry + CRS metadata
def write_geo_parquet(gdf: "gpd.GeoDataFrame", out_path: Path) -> None:
    mkdir_p(out_path.parent); gdf.to_parquet(out_path, index=False)

# guarantees a CRS
# if missing, assume WGS84 (ESPG: 4326), then return
def ensure_crs(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)  
    return gdf

# converts a geodataframe to a pure tabular form for SQL
# copies
# if geometry is present: serialize to hex WKB (geometry_wkb), add bounding box columns,
# then drop geometry column (SQLite doesn't natively store shapely objects)
def geo_to_sql_ready(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    out = gdf.copy()
    if "geometry" in out.columns:
        out["geometry_wkb"] = out.geometry.to_wkb(hex=True)
        b = out.geometry.bounds
        out["bbox_minx"] = b["minx"]; out["bbox_miny"] = b["miny"]
        out["bbox_maxx"] = b["maxx"]; out["bbox_maxy"] = b["maxy"]
        out = out.drop(columns=["geometry"])
    return out

# utility to align multiple frames from the same column set (sorted)
# note: not used anywhere else in this file
def align_columns(dfs):
    cols = set()
    for d in dfs: cols.update(d.columns)
    cols = list(sorted(cols))
    return [d.reindex(columns=cols) for d in dfs]

#ensures both datasets use the same precinct join key
#if any alias is present, it's renamed to cntyvtd (or prefer)
# if none found, crash with a clear message
def normalize_vtd_key(df: pd.DataFrame, prefer="cntyvtd") -> pd.DataFrame:
    for c in ["cntyvtd","cntyvtdkey","cnty_vtd","vtdkey","vtd_key","county_vtd_key"]:
        if c in df.columns:
            return df if c == prefer else df.rename(columns={c: prefer})
    raise AssertionError("No recognizable VTD key (cntyvtd/cntyvtdkey/…)")

#reprojects to equal-area CRS and copies
# A=polygon areas, P=perimeters
# Polsby-Popper = 4 PI A/P^2 (1 is circle)
# Swartzberg: sqrt(4 PI A)/P (perimeter vs circle with same area)
# Convex-Hull ratio: A/area(convex_hull) (1 means convex). Replaces hull area 0 with NA to avoid divide-by-zero
# Reock: A/PI * r^2 where r is min enclosing circle radius, if function not available, returns NA series
# returns a dataframe aligned to district index
def compute_compactness(districts_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = districts_gdf.to_crs(AREA_CRS).copy()
    A, P = g.geometry.area, g.geometry.length
    polsby = (4 * math.pi * A) / (P**2)
    schwartz = ((4 * math.pi * A)**0.5) / P
    hull_ratio = A / g.geometry.convex_hull.area.replace({0: pd.NA})
    if HAS_MINCIRCLE:
        radii = g.geometry.map(minimum_bounding_radius)
        reock = A / (math.pi * (radii**2))
    else:
        reock = pd.Series([pd.NA] * len(g), index=g.index)
    return pd.DataFrame({
        "polsby_popper": polsby,
        "schwartzberg": schwartz,
        "convex_hull_ratio": hull_ratio,
        "reock": reock
    }, index=g.index)

# ---------------- Stage 1: ETL ----------------
# prints a stage header
#ensures output directories exist (clean Parquet, Geoparquet, and SQLLite's parent directory)
# collects all supported fies in input_dir. Logs how many, if none, return early
#prepares two unused lists (placeholders for future accumulation)
# loops over files alphabetically (case-insensitive):
# - build a safe name (Dataset_key)
# - reads the file and drops duplicate rows (resetting index)
# if geospatial:
# -ensure CRS exists
# -build an attribute table: drop geometry before stdcols, or just stdcols if no geometry, then coerce_types_light
# recombine attributes + geometry intoa clean geodataframe with the same CRS
# -write as geoparquet
# else (tabular):
# -standardize & lightly type columns
# -write parquet <name>.parquet
# write to SQLite as stg<name>
# print completion banner
def run_etl(input_dir: Path, out_parquet_dir: Path, out_geo_dir: Path, sqlite_path: Path):
    print("\n========== Stage 1: ETL ==========")
    mkdir_p(out_parquet_dir); mkdir_p(out_geo_dir); mkdir_p(sqlite_path.parent)
    files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in ALL_EXTS]
    print(f"[INFO] Found {len(files)} files in {input_dir}")
    if not files: return

    tabular_frames, geo_frames = [], []
    for path in sorted(files, key=lambda p: p.name.lower()):
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
#announces stage 2
# reads five inputs (produced in stage 1 or earlier):
# -district polygons, census block geometries, VTD geometries (geoparquet)
# - PL-94 (population) attributes and VTD election results (Parquet)
# immediately standardizes column names and ensures CRS for geodata

def build_final(out_parquet_dir: Path, out_geo_dir: Path) -> pd.DataFrame:
    print("========== Stage 2: Build final 38-row dataset ==========")

    # Read cleaned layers produced by Stage 1
    districts = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir/"districts_clean.parquet")))
    census_geo = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir/"texas_census_blocks_clean.parquet")))
    vtd_geo = ensure_crs(stdcols(gpd.read_parquet(out_geo_dir/"vtds_geo_clean.parquet")))
    pl94 = stdcols(pd.read_parquet(out_parquet_dir/"tx_pl94_clean.parquet"))
    vtd_elec = stdcols(pd.read_parquet(out_parquet_dir/"clean_vtd_election_results.parquet"))

    # ----- explicit PL-94 fields -----
# name the PL-94 columns tou expect
# build a need list and checks pl94 contains them; fails fast with a helpful error if not
# verifies census geometry has join key
    PL94_TOTAL_VAP = "vap_total"
    PL94_RACE_VAP = {
        "pct_white":    "nh_white_vap",
        "pct_black":    "nh_black_vap",
        "pct_asian":    "nh_asian_vap",
        "pct_hispanic": "hispanic_vap",
    }
    need = ["geoid20", PL94_TOTAL_VAP, *PL94_RACE_VAP.values()]
    missing = [c for c in need if c not in pl94.columns]
    if missing:
        raise ValueError(f"PL-94 expected columns missing: {missing}")

    # (A) Census attrs to census geo on geoid20 → spatial aggregate to districts
#verifies census geometry has join key
# left-joins PL-94 attributes onto census geometries by geoid20
    if "geoid20" not in census_geo.columns:
        raise ValueError("Census geometry must contain 'geoid20'")
    census_merged = census_geo.merge(pl94[need], on="geoid20", how="left")

    # robust geometries
    #commented hint: buffering(0) sometimes fixes invalid polygons; currently disabled
    # districts = districts.set_geometry(districts.geometry.buffer(0))

#projects both layers to the equal area CRS for accurate areal calculation
    census_proj, districts_proj = census_merged.to_crs(AREA_CRS), districts.to_crs(AREA_CRS)

#spatial join census -> districts: for each census polygon, find the district polygon it falls within
#keep only matches (inner)
#creates a district_idx from the district geodataframe's index (used for grouping)
    c2d = gpd.sjoin(
        census_proj,
        districts_proj[["geometry"]].reset_index().rename(columns={"index": "district_idx"}),
        how="inner",
        predicate="within",
    )

#sum vap totals and each race VAP over census units within each district
# creates a denominator (den) and turns zeros into NA to avoid divide-by-zero
# for each race column, computes share = race VAP/total VAP, but only where total>0
    agg = c2d.groupby("district_idx")[[PL94_TOTAL_VAP] + list(PL94_RACE_VAP.values())].sum(min_count=1)
    den = agg[PL94_TOTAL_VAP].replace({0: pd.NA})
    race_pct = pd.DataFrame(index=agg.index)
    for out_name, src_col in PL94_RACE_VAP.items():
        race_pct[out_name] = (agg[src_col] / den).where(den > 0)

    # (B) VTD geo + VTD election on cntyvtd → spatial aggregate to districts
# normalizes precinct join key names to cntyvtd on both frames
# merges election results into VTD geometries (left join: keeps all VTD polygons)
# projects to equal-area CRS
    vtd_geo  = normalize_vtd_key(vtd_geo,  "cntyvtd")
    vtd_elec = normalize_vtd_key(vtd_elec, "cntyvtd")
    vtd_with_votes = vtd_geo.merge(vtd_elec, on="cntyvtd", how="left").to_crs(AREA_CRS)

#spatial join VTDs -> districts using intersects
# VTDs may cross boundaries; this assigns them to any district they intersect - no areal weighting)
# produces district_indx for grouping
    v2d = gpd.sjoin(
        vtd_with_votes,
        districts_proj[["geometry"]].reset_index().rename(columns={"index":"district_idx"}),
        how="inner",
        predicate="intersects",
    )

# detects which columns exist after the merge
# if any, sum them by district, creates an empty frame aligned to districts
    vote_cols = [c for c in ["dem_votes","rep_votes","third_party_votes","total_votes"] if c in v2d.columns]
    votes_by_dist = v2d.groupby("district_idx")[vote_cols].sum(min_count=1) if vote_cols else pd.DataFrame(index=districts_proj.index)

#prep an index for the partisanship table
#if both D and R exist, compute total (either from total_votes or as a sum) and compute D/R shares (guarding against zero totals)
    # if needed columns are missing, fill shares with NA
    part = pd.DataFrame(index=votes_by_dist.index if len(votes_by_dist) else districts_proj.index)
    if {"dem_votes","rep_votes"}.issubset(votes_by_dist.columns):
        tot = votes_by_dist["total_votes"] if "total_votes" in votes_by_dist.columns else \
              (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"] + votes_by_dist.get("third_party_votes", 0))
        part["dem_share"] = (votes_by_dist["dem_votes"] / tot).where(tot > 0)
        part["rep_share"] = (votes_by_dist["rep_votes"] / tot).where(tot > 0)
    else:
        part["dem_share"] = pd.NA; part["rep_share"] = pd.NA

    # (C) Compactness on districts
#calculates polsby-popper, schwartzberg, hull ratio, and (if available) Reock for each district
    cmpx = compute_compactness(districts_proj)

    # (D) Assemble final
#start an empty dataframe aligned to district index
    final = pd.DataFrame(index=districts_proj.index)

#attempts to pick a human-readable district identifier if present, else uses -1 based as the ID
    id_col = None
    for cand in ["district","district_id","cd","dist"]:
        if cand in districts.columns:
            id_col = cand; break
    final["district_id"] = districts[id_col].values if id_col else (districts_proj.index + 1)

#joins compactness, race percentages, and partisanship onto the ID column
# sorts by district index
# sanity check: must have exactly 38 rows; otherwise raise AssertionError with actual count
# returns the assembled mart
    final = final.join(cmpx).join(race_pct).join(part[["dem_share","rep_share"]])
    final = final.sort_index()
    assert len(final) == 38, f"Expected 38 districts, found {len(final)}"
    return final

# ---------------- CLI ----------------
# defines CLI: requires 4 arguments (input folder, parquet output folder, geoparquet output folder, and SQLlite path)
# runs stage 1, then stage 2 to get final dataframe
#compares two output paths in the parquet output folder
#writes the final dataset to Parquet (with Pyarrow) and CSV
#prints the saved file locations
def main():
    ap = argparse.ArgumentParser(description="ETL + final 38-district dataset")
    ap.add_argument("--input", type=Path, required=True, default=Path("data/interim"), help="Folder with raw files (CSV/TSV/Parquet/GPKG/SHP)")
    ap.add_argument("--data-processed-tabular", type=Path, required=True, help="Output folder for cleaned Parquet")
    ap.add_argument("--data-processed-geospatial", type=Path, required=True, help="Output folder for cleaned GeoParquet")
    ap.add_argument("--sqlite", type=Path, required=True, default=Path("data/warehouse/warehouse.db"),  help="SQLite warehouse path")
    args = ap.parse_args()

    run_etl(args.input, args.data_processed_tabular, args.data_processed_geospatial, args.sqlite)
    final = build_final(args.data_processed_tabular, args.data_processed_geospatial)

    pq = args.data_processed_tabular / "districts_final.parquet"
    csv = args.data_processed_tabular / "districts_final.csv"
    write_parquet(final, pq); final.to_csv(csv, index=False)
    print("\nSaved:")
    print(" ", pq)
    print(" ", csv)

#standard python entrypoint guard: when you run the file directly, execute main()
if __name__ == "__main__":
    main()
