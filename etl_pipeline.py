#!/usr/bin/env python3
# etl_pipeline.py
# ETL for Texas gerrymandering project:
# - Cleans inputs (districts, census blocks, VTDs, PL-94, elections)
# - Stores geospatial layers in EPSG:4269 (as-issued from TIGER/Line)
# - Computes geometry/overlays in EPSG:3083 (Texas equal-area) with safety checks
# - Builds final 38-row district dataset (compactness + race % + partisanship)

from __future__ import annotations

import argparse
import math
import re
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
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".txt")
SUPPORTED_GEO = (".gpkg", ".shp", ".parquet")  # allow GeoParquet inputs
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO

# Equal-area CRS for Texas (for area/length/overlay math)
AREA_CRS = "EPSG:3083"


# ========================== Utilities ==========================
def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    """lowercase, trim, non-word -> '_', collapse '__', strip edges"""
    df = df.copy()
    df.columns = [
        re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def coerce_types_light(df: pd.DataFrame) -> pd.DataFrame:
    """Light-touch typing for messy CSV/TXT: normalize empties, try datetime, then numeric heuristics."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = (
                out[c].astype(str)
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
    """safe predictable base name from a filename (mirrors stdcols logic)"""
    return re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", path.stem.lower())).strip("_")


def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        # Could be tabular or geo; try GeoPandas and fall back
        try:
            g = gpd.read_parquet(path)
            return g
        except Exception:
            return pd.read_parquet(path)
    if ext == ".txt":
        # Try to auto-detect delimiter for PL-94 style TXT
        return pd.read_csv(path, sep=None, engine="python")
    if ext in (".gpkg", ".shp"):
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
    """Default to WGS84 if missing (but we prefer as-issued 4269 when known)."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def ensure_geoid20_str(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure geoid20 exists and is a 15-digit zero-padded string (strip .0 if present)."""
    if "geoid20" in df.columns:
        s = (
            df["geoid20"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
        df = df.copy()
        df["geoid20"] = s
    return df


def geo_to_sql_ready(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """Flatten geometry for SQLite staging: WKB + bbox."""
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


def normalize_vtd_key(df: pd.DataFrame, prefer="cntyvtd") -> pd.DataFrame:
    for c in ["cntyvtd", "cntyvtdkey", "cnty_vtd", "vtdkey", "vtd_key", "county_vtd_key"]:
        if c in df.columns:
            return df if c == prefer else df.rename(columns={c: prefer})
    raise AssertionError("No recognizable VTD key (cntyvtd/cntyvtdkey/…)")

# --- Robust CNTY+VTD key builders (canonical: CCC + DDDD + optional trailing letters) ---
def _std_vtd_code(vtd_raw: pd.Series) -> pd.Series:
    """
    Normalize precinct code to 4-digit + optional letter suffix.
    Keeps letters at the END (e.g., '1A' → '0001A', '12' → '0012').
    Removes spaces/dashes/punctuation.
    """
    s = vtd_raw.astype(str).str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    letters = s.str.extract(r"([A-Z]+)$", expand=False).fillna("")      # trailing letters, if any
    digits  = s.str.replace(r"[A-Z]+$", "", regex=True)                 # leading digits
    digits = digits.where(digits.str.len() > 0, pd.NA)
    digits = digits.mask(digits.isna(), None)
    digits = digits.apply(lambda x: None if x is None else x.zfill(4))  # 4-digit pad
    out = pd.Series(digits, index=s.index).fillna("") + letters
    return out.replace({"": pd.NA})

def _std_cnty_code_from_cnty(cnty_raw: pd.Series) -> pd.Series:
    """Normalize numeric CNTY (1..254) to 3-digit zero-padded string."""
    s = cnty_raw.astype(str).str.replace(r"[^0-9]", "", regex=True)
    s = s.str.lstrip("0").where(lambda x: x.str.len() > 0, "0")
    return s.str.zfill(3)

def _std_cnty_code_from_fips(fips_raw: pd.Series) -> pd.Series:
    """Normalize FIPS to 3-digit county code: last 3 of 5-digit (e.g., 48001→001), or pad 3 if shorter."""
    s = fips_raw.astype(str).str.replace(r"[^0-9]", "", regex=True)
    return s.apply(lambda x: x[-3:] if len(x) >= 3 else x.zfill(3))

def build_cntyvtd_from_parts(cnty_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = _std_cnty_code_from_cnty(cnty_series)
    vtd4  = _std_vtd_code(vtd_series)
    out = cnty3 + vtd4.fillna("")
    return out.replace({"": pd.NA})

def build_cntyvtd_from_fips_vtd(fips_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = _std_cnty_code_from_fips(fips_series)
    vtd4  = _std_vtd_code(vtd_series)
    out = cnty3 + vtd4.fillna("")
    return out.replace({"": pd.NA})

def normalize_cntyvtd_safely(key_series: pd.Series) -> pd.Series:
    """
    Canonicalize a prebuilt cntyvtd to CCC + DDDD + optional letters.
    If it doesn't parse, return NA so we can rebuild from parts when possible.
    """
    s = key_series.astype(str).str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    m = s.str.extract(r"^(?P<cnty>\d{1,3})(?P<vtd>\d{1,4})(?P<suf>[A-Z]*)$")
    bad = m.isna().any(axis=1)
    m.loc[~bad, "cnty"] = m.loc[~bad, "cnty"].str.zfill(3)
    m.loc[~bad, "vtd"]  = m.loc[~bad, "vtd"].str.zfill(4)
    out = (m["cnty"] + m["vtd"] + m["suf"]).where(~bad)
    return out

# --------- NEW: guardrail — prevent metric math in geographic CRS ---------
def assert_projected_planar(gdf: "gpd.GeoDataFrame", where: str = "") -> None:
    """
    Raise if CRS is missing or geographic (degrees). Call before area/length/overlays
    to ensure we switched to AREA_CRS.
    """
    if gdf.crs is None:
        raise ValueError(f"{where}: GeoDataFrame has no CRS; project to {AREA_CRS} before metric computations.")
    try:
        if gdf.crs.is_geographic:
            raise ValueError(f"{where}: CRS is geographic ({gdf.crs.to_string()}); reproject to {AREA_CRS} first.")
    except AttributeError:
        if any(u in str(gdf.crs).lower() for u in ["degree", "longlat", "epsg:4269", "epsg:4326"]):
            raise ValueError(f"{where}: CRS looks geographic; reproject to {AREA_CRS} first.")


# =================== Notebook logic: clean Texas 2020 blocks ===================
CENSUS_BLOCK_KEEP = [
    "STATEFP20", "COUNTYFP20", "TRACTCE20", "BLOCKCE20",
    "GEOID20", "NAME20", "ALAND20", "AWATER20",
    "INTPTLAT20", "INTPTLON20", "geometry",
]

def _cols_present_case_insensitive(gdf: gpd.GeoDataFrame, cols_upper: list[str]) -> bool:
    cols_up = {c.upper() for c in gdf.columns}
    need_up = set(cols_upper)
    return need_up.issubset(cols_up)

def _subset_by_upper(gdf: gpd.GeoDataFrame, cols_upper: list[str]) -> gpd.GeoDataFrame:
    """
    Select columns by a case-insensitive list (provided as 'upper' names).
    Guarantees that if 'geometry' exists (in any case), it is included.
    """
    name_map = {c.upper(): c for c in gdf.columns}
    want = []
    for u in cols_upper:
        key = u.upper()
        if key in name_map:
            want.append(name_map[key])

    # Ensure geometry survives (GeoPandas -> GeoDataFrame)
    if "geometry" in gdf.columns and "geometry" not in want:
        want.append("geometry")

    # Return a GeoDataFrame
    out = gdf[want]
    if not isinstance(out, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry="geometry" if "geometry" in out.columns else None, crs=gdf.crs)
    return out


def is_tx_2020_blocks(gdf: gpd.GeoDataFrame) -> bool:
    # Heuristic: presence of canonical 2020 block fields
    return _cols_present_case_insensitive(gdf, CENSUS_BLOCK_KEEP[:-1])  # ignore geometry in check

def clean_tx_2020_blocks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Replicates 03_clean_census_shpfile.ipynb:
      • keep key columns
      • enforce CRS NAD83 (EPSG:4269)
      • drop duplicate GEOID20
      • standardize geoid20 string
    """
    sub = _subset_by_upper(gdf, CENSUS_BLOCK_KEEP)  # now keeps geometry reliably
    # Ensure GeoDataFrame wrapper + set CRS to 4269 (as-issued)
    if not isinstance(sub, gpd.GeoDataFrame):
        sub = gpd.GeoDataFrame(sub, geometry="geometry" if "geometry" in sub.columns else None, crs=gdf.crs)
    try:
        sub = sub.set_crs("EPSG:4269", allow_override=True)
    except Exception:
        sub = sub if sub.crs else sub.set_crs("EPSG:4269")

    sub_std = stdcols(sub)
    if "geoid20" in sub_std.columns:
        sub_std = sub_std.drop_duplicates(subset=["geoid20"])
        sub_std = ensure_geoid20_str(sub_std)
    return gpd.GeoDataFrame(sub_std, geometry="geometry", crs="EPSG:4269")



# =================== Metrics & cleaners ===================
def compute_compactness(districts_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = districts_gdf.to_crs(AREA_CRS).copy()
    assert_projected_planar(g, "compute_compactness")
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


def unify_pl94_schema(pl94: pd.DataFrame) -> pd.DataFrame:
    """Normalize PL-94 column names to pipeline schema."""
    df = pl94.copy()

    geoid_candidates = [
        "geoid20", "geoid", "tabblock20", "tabblock2020", "block_geoid", "blk_geoid", "sctbkey", "ctbkey"
    ]
    geoid_col = next((c for c in geoid_candidates if c in df.columns), None)
    if geoid_col is None:
        raise ValueError("PL-94 file lacks a recognizable GEOID column (e.g., geoid20 or sctbkey).")
    if geoid_col != "geoid20":
        df = df.rename(columns={geoid_col: "geoid20"})
    df["geoid20"] = (
        df["geoid20"].astype(str).str.replace(".0$", "", regex=True).str.replace(r"[^\d]", "", regex=True).str.zfill(15)
    )

    # Map VAP fields if present
    rename_map = {}
    if "vap_total" not in df.columns and "vap" in df.columns:
        rename_map["vap"] = "vap_total"
    if "anglovap" in df.columns and "nh_white_vap" not in df.columns:
        rename_map["anglovap"] = "nh_white_vap"
    if "blackvap" in df.columns and "nh_black_vap" not in df.columns:
        rename_map["blackvap"] = "nh_black_vap"
    if "asianvap" in df.columns and "nh_asian_vap" not in df.columns:
        rename_map["asianvap"] = "nh_asian_vap"
    if "hispvap" in df.columns and "hispanic_vap" not in df.columns:
        rename_map["hispvap"] = "hispanic_vap"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ---------------- Elections cleaner ----------------
def is_tall_elections(df: pd.DataFrame) -> bool:
    req = {"office", "party", "votes"}
    has_vtd = {"cntyvtd", "cntyvtdkey", "cnty_vtd", "vtdkey", "vtd_key", "county_vtd_key"} & set(df.columns)
    return req.issubset(df.columns) and bool(has_vtd)


def clean_vtd_election_returns(df: pd.DataFrame, target_office: str = "President") -> pd.DataFrame:
    """
    Convert tall VTD returns (one row per candidate) into a single wide row per VTD with
    dem/rep/third/total vote columns, filtered to a specific contest (default: President).
    """
    t = df.copy()
    t = normalize_vtd_key(t, "cntyvtd")

    # Filter to target contest (case-insensitive match)
    if "office" in t.columns:
        mask = t["office"].astype(str).str.casefold() == str(target_office).casefold()
        sub = t.loc[mask].copy()
        if sub.empty:
            print(f"[WARN] No rows matched office='{target_office}'. Using all offices combined.")
            sub = t.copy()
    else:
        sub = t

    # Coerce votes to numeric
    sub["votes"] = pd.to_numeric(sub["votes"], errors="coerce").fillna(0).astype(int)
    sub["party"] = sub.get("party", "").astype(str).str.upper().str.strip()

    # Aggregate party totals per VTD
    agg = sub.groupby(["cntyvtd", "party"], as_index=False)["votes"].sum()

    # Pivot to party columns and compute aggregates
    parties = ["D", "R", "G", "L", "I", "W"]
    piv = agg.pivot_table(index="cntyvtd", columns="party", values="votes", fill_value=0, aggfunc="sum")
    for p in parties:
        if p not in piv.columns:
            piv[p] = 0

    out = pd.DataFrame({
        "cntyvtd": piv.index.astype(str),
        "dem_votes": piv["D"].astype(int),
        "rep_votes": piv["R"].astype(int),
    })
    out["third_party_votes"] = (piv[["G", "L", "I", "W"]].sum(axis=1)).astype(int)
    out["total_votes"] = (out["dem_votes"] + out["rep_votes"] + out["third_party_votes"]).astype(int)

    out["two_party_dem_share"] = (
        out["dem_votes"] / (out["dem_votes"] + out["rep_votes"]).replace({0: pd.NA})
    )

    cols = ["cntyvtd", "dem_votes", "rep_votes", "third_party_votes", "total_votes", "two_party_dem_share"]
    return out.loc[:, cols]


# ========================== Stage 1: ETL ==========================
def run_etl(
    input_files: list[Path],
    out_parquet_dir: Path,
    out_geo_dir: Path,
    sqlite_path: Path,
    elections_office: str = "President",
):
    print("\n========== Stage 1: ETL ==========")
    mkdir_p(out_parquet_dir)
    mkdir_p(out_geo_dir)
    mkdir_p(sqlite_path.parent)

    # By convention from CLI: last is --elections
    elections_path = input_files[-1]

    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if path.suffix.lower() not in ALL_EXTS:
            raise ValueError(f"Unsupported input type for {path.name} (ext {path.suffix})")

        name = dataset_key(path)
        print(f"[ETL] Reading {path} -> key={name}")
        df = read_any(path).drop_duplicates(ignore_index=True)

        # --- SPECIAL HANDLING: VTD election results cleaner ---
        if path == elections_path and not is_geodf(df):
            tdf = coerce_types_light(stdcols(df))
            if is_tall_elections(tdf):
                print(f"[ETL] Detected tall elections file. Cleaning office='{elections_office}' → wide VTD votes.")
                try:
                    tdf = clean_vtd_election_returns(tdf, target_office=elections_office)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed cleaning elections file '{path.name}': {e}"
                    ) from e
            write_parquet(tdf, out_parquet_dir / f"{name}.parquet")
            to_sqlite(tdf, sqlite_path, f"stg_{name}")
            continue

        # --- Normal handling for all other files ---
        if is_geodf(df):
            # Detect & clean census blocks
            if is_tx_2020_blocks(df):
                print("[ETL] Detected Texas 2020 Census blocks → applying notebook cleaning.")
                gdf = clean_tx_2020_blocks(df)
            else:
                # Default path: standardize non-geometry attrs but keep CRS as provided (or set WGS84 if missing)
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


# =================== Stage 2: Build final dataset ===================
def build_final(
    out_parquet_dir: Path,
    out_geo_dir: Path,
    districts_key: str,
    census_key: str,
    vtds_key: str,
    pl94_key: str,
    elections_key: str,
) -> pd.DataFrame:
    print("========== Stage 2: Build final 38-row dataset ==========")

    # Resolve cleaned file paths from dataset stems generated in Stage 1
    districts_fp = out_geo_dir / f"{districts_key}.parquet"
    census_fp = out_geo_dir / f"{census_key}.parquet"
    vtds_fp = out_geo_dir / f"{vtds_key}.parquet"
    pl94_fp = out_parquet_dir / f"{pl94_key}.parquet"
    elect_fp = out_parquet_dir / f"{elections_key}.parquet"

    missing_paths = [p for p in [districts_fp, census_fp, vtds_fp, pl94_fp, elect_fp] if not p.exists()]
    if missing_paths:
        msg = (
            "Stage 2 expected these cleaned files from Stage 1 but didn't find them:\n"
            + "\n".join(f" - {p}" for p in missing_paths)
            + "\n(Hint: cleaned filenames are derived from input stems via dataset_key; check stems/output folders.)"
        )
        raise FileNotFoundError(msg)

    print("[INFO] Using cleaned files:")
    print("  ", districts_fp)
    print("  ", census_fp)
    print("  ", vtds_fp)
    print("  ", pl94_fp)
    print("  ", elect_fp)

    # Read cleaned layers produced by Stage 1 (stored in EPSG:4269)
    districts = ensure_crs(stdcols(gpd.read_parquet(districts_fp)))
    census_geo = ensure_crs(stdcols(gpd.read_parquet(census_fp)))
    vtd_geo = ensure_crs(stdcols(gpd.read_parquet(vtds_fp)))
    pl94 = stdcols(pd.read_parquet(pl94_fp))
    pl94 = unify_pl94_schema(pl94)
    vtd_elec = stdcols(pd.read_parquet(elect_fp))

    # ----- Build a canonical cntyvtd_std on BOTH sides -----
    vtd_geo = vtd_geo.copy()
    vtd_elec = vtd_elec.copy()

    # Geo side: prefer CNTY + VTD if available; else parse existing cntyvtd
    if {"cnty", "vtd"} <= set(vtd_geo.columns):
        vtd_geo["cntyvtd_std"] = build_cntyvtd_from_parts(vtd_geo["cnty"], vtd_geo["vtd"])
    elif "cntyvtd" in vtd_geo.columns:
        vtd_geo["cntyvtd_std"] = normalize_cntyvtd_safely(vtd_geo["cntyvtd"])
    else:
        raise ValueError("VTD geometry lacks CNTY/VTD or CNTYVTD to build a key.")

    # Elections side: prefer existing cntyvtd from the cleaned wide table; lightly standardize only.
    if "cntyvtd" in vtd_elec.columns:
        vtd_elec["cntyvtd_std"] = (
            vtd_elec["cntyvtd"]
            .astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)  # drop spaces/dashes/punct ONLY
        )
    else:
        # Fallback only if cntyvtd truly missing (rare after Stage 1)
        if {"fips", "vtd"} <= set(vtd_elec.columns):
            vtd_elec["cntyvtd_std"] = build_cntyvtd_from_fips_vtd(vtd_elec["fips"], vtd_elec["vtd"])
        else:
            raise ValueError("Elections file lacks cntyvtd (and FIPS/VTD) to build a key.")

    # Drop rows with missing standardized keys (rare, but avoids NA-joins)
    vtd_geo  = vtd_geo.loc[vtd_geo["cntyvtd_std"].notna()].copy()
    vtd_elec = vtd_elec.loc[vtd_elec["cntyvtd_std"].notna()].copy()

    print(f"[INFO] VTD keys (standardized) — geo: {vtd_geo['cntyvtd_std'].nunique()} unique, "
          f"elections: {vtd_elec['cntyvtd_std'].nunique()} unique")


    # If elections file lacks a unified cntyvtd key, build it from county + precinct
    if "cntyvtd" not in vtd_elec.columns and {"cnty", "vtd"} <= set(vtd_elec.columns):
        # Common TLC pattern is county code + zero-padded precinct id.
        # If your source already has combined keys, this block is skipped.
        vtd_elec = vtd_elec.copy()
        vtd_elec["cnty"] = (
            vtd_elec["cnty"]
            .astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        vtd_elec["vtd"] = (
            vtd_elec["vtd"]
            .astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        # Zero-padding widths vary by source; 3+4 works for many TLC exports.
        # If join rate is still low, try removing zfill() or adjusting widths.
        vtd_elec["cntyvtd"] = vtd_elec["cnty"].str.zfill(3) + vtd_elec["vtd"].str.zfill(4)

    # 🔧 Ensure geoid20 is 15-digit string on BOTH sides before merge
    census_geo = ensure_geoid20_str(census_geo)
    pl94 = ensure_geoid20_str(pl94)

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
        msg = (
            "PL-94 expected columns missing: " + str(missing)
            + "\nColumns available: " + ", ".join(list(pl94.columns)[:40]) + ("..." if len(pl94.columns) > 40 else "")
            + "\n(Hint: rename your PL-94 columns before Stage 1 or adjust PL94_RACE_VAP in the pipeline.)"
        )
        raise ValueError(msg)

    # (A) Blocks + PL94 → area-weighted to districts (in AREA_CRS)
    if "geoid20" not in census_geo.columns:
        raise ValueError("Census geometry must contain 'geoid20'")
    census_merged = census_geo.merge(pl94[need], on="geoid20", how="left")

    census_proj, districts_proj = census_merged.to_crs(AREA_CRS), districts.to_crs(AREA_CRS)
    assert_projected_planar(census_proj, "blocks→districts")
    assert_projected_planar(districts_proj, "blocks→districts")

    d_sub = districts_proj[["geometry"]].reset_index(names="district_idx")
    blk = census_proj[["geoid20", "geometry"]].copy()
    blk_attrs = pl94[need].copy()

    # intersection overlay
    blk_inter = gpd.overlay(blk, d_sub, how="intersection", keep_geom_type=True)

    blk_inter = blk_inter.merge(blk_attrs, on="geoid20", how="left")
    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    blk_inter["blk_area"] = blk_inter["geoid20"].map(blk_area)
    blk_inter["inter_area"] = blk_inter.geometry.area
    blk_inter = blk_inter.loc[blk_inter["blk_area"] > 0].copy()
    blk_inter["w"] = (blk_inter["inter_area"] / blk_inter["blk_area"]).clip(0, 1)

    to_sum = [PL94_TOTAL_VAP] + list(PL94_RACE_VAP.values())
    for col in to_sum:
        blk_inter[col] = blk_inter[col].fillna(0) * blk_inter["w"]

    agg = (
        blk_inter.groupby("district_idx", observed=True)[to_sum]
        .sum(min_count=1)
        .reindex(districts_proj.reset_index(names="district_idx")["district_idx"], fill_value=0)
    )

    den = agg[PL94_TOTAL_VAP].replace({0: pd.NA})
    race_pct = pd.DataFrame(index=agg.index)
    for out_name, src_col in PL94_RACE_VAP.items():
        race_pct[out_name] = (agg[src_col] / den).where(den > 0)

    # --- (B) VTD geo + VTD election on standardized cntyvtd_std → area-weighted to districts ---
    # (We no longer need normalize_vtd_key/_norm_cntyvtd here.)
    vtd_with_votes = (
        vtd_geo.merge(
            vtd_elec.drop(columns=[c for c in ["cntyvtd", "cnty", "vtd", "fips"] if c in vtd_elec.columns]),
            on="cntyvtd_std",
            how="left",
            suffixes=("", "_elec"),
        )
        .to_crs(AREA_CRS)
    )
    assert_projected_planar(vtd_with_votes, "vtd→districts")

    # For the rest of your pipeline (which expects a column named 'cntyvtd'),
    # just alias cntyvtd_std → cntyvtd so you don't have to rewrite later code.
    vtd_with_votes["cntyvtd"] = vtd_with_votes["cntyvtd_std"]


    districts_proj = districts_proj.to_crs(AREA_CRS)
    assert_projected_planar(districts_proj, "vtd→districts")

    vote_cols_base = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]
                      if c in vtd_with_votes.columns]

    if not vote_cols_base:
        votes_by_dist = pd.DataFrame(index=districts_proj.index,
                                     columns=["dem_votes", "rep_votes", "third_party_votes", "total_votes"])
        part = pd.DataFrame(index=districts_proj.index, columns=["dem_share", "rep_share", "two_party_dem_share"])
    else:
        vtd_cols_keep = ["cntyvtd", "geometry"] + vote_cols_base
        vtd_compact = vtd_with_votes.loc[:, vtd_cols_keep].copy()

        possible = gpd.sjoin(
            vtd_compact[["cntyvtd", "geometry"]],
            districts_proj[["geometry"]].reset_index(names="district_idx"),
            how="inner",
            predicate="intersects",
        )
        pairs = possible[["cntyvtd", "district_idx"]].drop_duplicates()

        vtd_subset = vtd_compact[vtd_compact["cntyvtd"].isin(pairs["cntyvtd"])].copy()
        d_sub = districts_proj[["geometry"]].reset_index(names="district_idx")

        vtd_district_intersections = gpd.overlay(vtd_subset, d_sub, how="intersection", keep_geom_type=True)

        if vtd_district_intersections.empty:
            votes_by_dist = pd.DataFrame(index=districts_proj.index, columns=vote_cols_base).fillna(0)
            part = pd.DataFrame(index=districts_proj.index, columns=["dem_share", "rep_share", "two_party_dem_share"])
        else:
            vtd_areas = (
                vtd_with_votes[["cntyvtd", "geometry"]]
                .drop_duplicates("cntyvtd")
                .set_index("cntyvtd")
                .geometry.area
                .rename("vtd_area")
            )

            vtd_district_intersections["vtd_area"] = vtd_district_intersections["cntyvtd"].map(vtd_areas)

            vtd_district_intersections["inter_area"] = vtd_district_intersections.geometry.area
            eps = 1e-9
            vtd_district_intersections = vtd_district_intersections.loc[
                vtd_district_intersections["vtd_area"] > eps
            ].copy()

            vtd_district_intersections["weight"] = (
                vtd_district_intersections["inter_area"] / vtd_district_intersections["vtd_area"]
            ).clip(lower=0, upper=1)

            vote_cols = [c for c in vote_cols_base if c in vtd_district_intersections.columns]
            for col in vote_cols:
                vtd_district_intersections[col] = (
                    vtd_district_intersections[col].fillna(0) * vtd_district_intersections["weight"]
                )

            votes_by_dist = (
                vtd_district_intersections.groupby("district_idx", observed=True)[vote_cols]
                .sum(min_count=1)
                .reindex(districts_proj.reset_index(names="district_idx")["district_idx"], fill_value=0)
            )

            part = pd.DataFrame(index=votes_by_dist.index)
            if {"dem_votes", "rep_votes"}.issubset(votes_by_dist.columns):
                tot = (
                    votes_by_dist["total_votes"]
                    if "total_votes" in votes_by_dist.columns
                    else (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"] + votes_by_dist.get(
                        "third_party_votes", 0))
                )
                valid = tot > 0
                part["dem_share"] = (votes_by_dist["dem_votes"] / tot).where(valid)
                part["rep_share"] = (votes_by_dist["rep_votes"] / tot).where(valid)
                part["two_party_dem_share"] = (
                    votes_by_dist["dem_votes"] / (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"])
                ).where((votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"]) > 0)
            else:
                part["dem_share"] = pd.NA
                part["rep_share"] = pd.NA
                part["two_party_dem_share"] = pd.NA


    # --- Diagnostics for zero-vote districts ---
    if 'total_votes' in votes_by_dist.columns:
        tot = votes_by_dist['total_votes']
    else:
        tot = votes_by_dist.get('dem_votes', 0) + votes_by_dist.get('rep_votes', 0) + votes_by_dist.get('third_party_votes', 0)

    zero_vote_districts = list(votes_by_dist.index[tot.fillna(0) == 0])
    print(f"[DIAG] Districts with zero total votes (shares become NA): {zero_vote_districts}")

    # How many VTDs intersected each district?
    vtds_per_dist = possible[['district_idx','cntyvtd']].drop_duplicates().groupby('district_idx').size()
    print("[DIAG] Example intersect counts (first 10):")
    print(vtds_per_dist.head(10))

    # How many of those VTDs had matched vote rows (all vote cols non-null)?
    vote_cols = [c for c in ["dem_votes","rep_votes","third_party_votes","total_votes"] if c in vtd_with_votes.columns]
    had_votes = vtd_with_votes.dropna(subset=vote_cols, how='all')['cntyvtd'].unique()
    matched_per_dist = possible[possible['cntyvtd'].isin(had_votes)].groupby('district_idx')['cntyvtd'].nunique()
    print("[DIAG] Example matched-vote VTD counts (first 10):")
    print(matched_per_dist.head(10))

    # --- Extra diagnostics: join coverage & samples ---
    vote_cols = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"] if
                 c in vtd_with_votes.columns]

    total_vtds = len(vtd_with_votes)
    matched_mask = vtd_with_votes[vote_cols].notna().any(axis=1)
    matched_vtds = int(matched_mask.sum())
    print(f"[DIAG] VTDs with any matched votes: {matched_vtds}/{total_vtds} "
          f"({matched_vtds / total_vtds:.1%})")

    # Show a few unmatched VTD keys to spot a pattern
    unmatched_keys = vtd_with_votes.loc[~matched_mask, "cntyvtd"].dropna().astype(str).head(20).tolist()
    print("[DIAG] Sample unmatched VTD keys (cntyvtd):", unmatched_keys)

    # Per-problem-district matched counts (convert district_idx->1-based ID for readability)
    problem_idxs = [1, 5, 6, 19, 23, 29, 32, 37]
    per_dist = possible[['district_idx', 'cntyvtd']].drop_duplicates()
    per_dist['has_votes'] = per_dist['cntyvtd'].isin(vtd_with_votes.loc[matched_mask, 'cntyvtd'])
    summary = (per_dist.groupby('district_idx')['has_votes']
               .agg(['sum', 'count'])
               .loc[problem_idxs]
               .rename(columns={'sum': 'matched_vtds', 'count': 'intersecting_vtds'}))
    summary['district_id'] = summary.index + 1
    print("[DIAG] Problem districts — matched/total VTDs:")
    print(summary[['district_id', 'matched_vtds', 'intersecting_vtds']])

    # (C) Compactness on districts
    cmpx = compute_compactness(districts_proj)

    # (D) Assemble final
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


# =============================== CLI ===============================
def main():
    ap = argparse.ArgumentParser(description="ETL + final 38-district dataset")
    ap.add_argument("--districts", type=Path, required=True, help="Path to district polygons (SHP/GPKG/GeoParquet)")
    ap.add_argument("--census", type=Path, required=True, help="Path to census blocks (SHP/GPKG/GeoParquet)")
    ap.add_argument("--vtds", type=Path, required=True, help="Path to VTD geometries (SHP/GPKG/GeoParquet)")
    ap.add_argument("--pl94", type=Path, required=True, help="Path to PL-94 attributes (CSV/TSV/Parquet/TXT)")
    ap.add_argument("--elections", type=Path, required=True, help="Path to VTD election results (CSV/TSV/Parquet)")
    ap.add_argument("--elections-office", type=str, default="President",
                    help="Contest to aggregate for elections cleaning (e.g., 'President', 'U.S. Sen').")

    ap.add_argument("--data-processed-tabular", type=Path, required=True, help="Output folder for cleaned Parquet")
    ap.add_argument("--data-processed-geospatial", type=Path, required=True, help="Output folder for cleaned GeoParquet")
    ap.add_argument("--sqlite", type=Path, required=True, help="SQLite warehouse path")
    args = ap.parse_args()

    input_files = [args.districts, args.census, args.vtds, args.pl94, args.elections]

    run_etl(
        input_files,
        args.data_processed_tabular,
        args.data_processed_geospatial,
        args.sqlite,
        elections_office=args.elections_office,
    )

    # Derive dataset keys from stems for Stage 1
    dk, ck, vk, pk, ek = [dataset_key(p) for p in input_files]

    final = build_final(
        args.data_processed_tabular,
        args.data_processed_geospatial,
        dk, ck, vk, pk, ek,
    )

    pq = args.data_processed_tabular / "districts_final.parquet"
    csv = args.data_processed_tabular / "districts_final.csv"
    write_parquet(final, pq)
    final.to_csv(csv, index=False)
    print("\nSaved:")
    print(" ", pq)
    print(" ", csv)


if __name__ == "__main__":
    main()
