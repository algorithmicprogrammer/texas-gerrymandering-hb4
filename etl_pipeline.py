#!/usr/bin/env python3
# etl_pipeline.py
# Build a 38-row congressional dataset for Texas:
# - Stage 1: Clean and persist raw inputs (tabular + geospatial)
# - Stage 2: Join PL94 + elections to districts via area-weighting
# - Outputs: districts_final.{parquet,csv}

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ---------------- Geo deps ----------------
try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit("GeoPandas required. Install: pip install geopandas shapely fiona pyproj") from e

# Reock compactness (if available)
try:
    from shapely import minimum_bounding_radius
    HAS_MINCIRCLE = True
except Exception:
    HAS_MINCIRCLE = False

# ---------------- Config ------------------
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".txt")
SUPPORTED_GEO = (".gpkg", ".shp", ".parquet")
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO

AREA_CRS = "EPSG:3083"  # Texas equal-area for metrics/overlays


# ========================== Small utilities ==========================
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
    # normalize text-ish empties
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            col = out[c].astype("string")
            stripped = col.str.strip()
            # Preserve genuine missing values (pd.NA) instead of coercing them to the
            # string "<NA>" which happens with astype(str).
            out[c] = stripped.mask(
                stripped.isin({"", "na", "null", "none"}), pd.NA
            )
    # attempt datetime on obvious names
    for c in out.columns:
        if re.search(r"(date|time|timestamp|dt)$", c):
            out[c] = pd.to_datetime(out[c], errors="coerce")
    # light numeric inference
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
        # could be GeoParquet or regular Parquet
        try:
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)
    if ext == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if ext in (".gpkg", ".shp"):
        return gpd.read_file(path)
    raise ValueError(f"Unsupported input type: {ext}")


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


def ensure_geoid20_str(df: pd.DataFrame) -> pd.DataFrame:
    if "geoid20" in df.columns:
        s = (
            df["geoid20"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
        df = df.copy()
        df["geoid20"] = s
    return df


def geo_to_sql_ready(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    out = gdf.copy()
    if "geometry" in out.columns:
        out["geometry_wkb"] = out.geometry.to_wkb(hex=True)
        b = out.geometry.bounds
        out["bbox_minx"] = b["minx"]; out["bbox_miny"] = b["miny"]
        out["bbox_maxx"] = b["maxx"]; out["bbox_maxy"] = b["maxy"]
        out = out.drop(columns=["geometry"])
    return out


# ======================== VTD key builders ==========================
def normalize_vtd_key(df: pd.DataFrame, prefer="cntyvtd") -> pd.DataFrame:
    for c in ["cntyvtd", "cntyvtdkey", "cnty_vtd", "vtdkey", "vtd_key", "county_vtd_key"]:
        if c in df.columns:
            return df if c == prefer else df.rename(columns={c: prefer})
    raise AssertionError("No recognizable VTD key (cntyvtd/cntyvtdkey/…)")


def _std_vtd_code(vtd_raw: pd.Series) -> pd.Series:
    s = vtd_raw.astype("string").str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    letters = s.str.extract(r"([A-Z]+)$", expand=False).fillna("")
    digits = s.str.replace(r"[A-Z]+$", "", regex=True)
    digits = digits.where(digits.str.len() > 0, pd.NA).astype("string")
    letters = letters.astype("string")

    out = pd.Series(pd.NA, dtype="string", index=s.index)
    valid = digits.notna()
    if valid.any():
        padded = digits.loc[valid].str.zfill(5)
        out.loc[valid] = (padded + letters.loc[valid]).astype("string")
    return out


def _std_cnty_code_from_cnty(cnty_raw: pd.Series) -> pd.Series:
    s = cnty_raw.astype("string").str.replace(r"[^0-9]", "", regex=True)
    s = s.where(s.str.len() > 0, pd.NA)
    s = s.str[-3:].str.zfill(3)
    return s.where(~s.isin({"000"}), pd.NA)


def _std_cnty_code_from_fips(fips_raw: pd.Series) -> pd.Series:
    s = fips_raw.astype("string").str.replace(r"[^0-9]", "", regex=True)
    s = s.where(s.str.len() > 0, pd.NA)
    s = s.str[-3:].str.zfill(3)
    return s.where(~s.isin({"000"}), pd.NA)


def build_cntyvtd_from_parts(cnty_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = _std_cnty_code_from_cnty(cnty_series).astype("string")
    vtd5 = _std_vtd_code(vtd_series).astype("string")
    valid = cnty3.notna() & vtd5.notna()
    out = pd.Series(pd.NA, dtype="string", index=cnty3.index)
    out.loc[valid] = (cnty3.loc[valid] + vtd5.loc[valid]).astype("string")
    return out


def build_cntyvtd_from_fips_vtd(fips_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = _std_cnty_code_from_fips(fips_series).astype("string")
    vtd5 = _std_vtd_code(vtd_series).astype("string")
    valid = cnty3.notna() & vtd5.notna()
    out = pd.Series(pd.NA, dtype="string", index=cnty3.index)
    out.loc[valid] = (cnty3.loc[valid] + vtd5.loc[valid]).astype("string")
    return out


def normalize_cntyvtd_safely(key_series: pd.Series) -> pd.Series:
    s = (
        key_series.astype(str)
        .str.strip().str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    m = s.str.extract(r"^(?:48)?(?P<cnty>\d{3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
    bad = m.isna().any(axis=1)
    if bad.any():
        m2 = s[bad].str.extract(r"^(?P<cnty>\d{1,3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
        m.loc[bad, ["cnty", "vtd", "suf"]] = m2[["cnty", "vtd", "suf"]]
    bad = m.isna().any(axis=1)
    m.loc[~bad, "cnty"] = m.loc[~bad, "cnty"].astype(str).str.zfill(3)
    m.loc[~bad, "vtd"]  = m.loc[~bad, "vtd"].astype(str).str.zfill(5)   # ← pad to 5
    out = (m["cnty"] + m["vtd"] + m["suf"].fillna("")).where(~bad)
    return out


def _last4_elec_key(s: pd.Series) -> pd.Series:
    """
    From an elections-side key that may have 5 VTD digits (CCC + DDDDD [+ SUF]),
    build a 4-digit-compat key by taking the RIGHTMOST 4 VTD digits.
    Returns CNTY + last4 + SUF (suffix preserved).
    """
    s = s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    m = s.str.extract(r"^(?:48)?(?P<cnty>\d{3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
    ok = m["cnty"].notna() & m["vtd"].notna()
    out = pd.Series(pd.NA, index=s.index)
    last4 = m.loc[ok, "vtd"].astype(str).str[-4:].str.zfill(4)
    out.loc[ok] = m.loc[ok, "cnty"].astype(str).str.zfill(3) + last4 + m.loc[ok, "suf"].fillna("")
    return out

COUNTY_NAME_TO_FIPS = {
    # Add as needed — we only NEED Bexar to fix TX-20
    "BEXAR": "029",
    # (Optional) A few big ones:
    "HARRIS": "201", "DALLAS": "113", "TARRANT": "439", "TRAVIS": "453",
}


# =================== CRS Guardrail ===================
def assert_projected_planar(gdf: "gpd.GeoDataFrame", where: str = "") -> None:
    if gdf.crs is None:
        raise ValueError(f"{where}: GeoDataFrame has no CRS; project to {AREA_CRS} before metric computations.")
    try:
        if gdf.crs.is_geographic:
            raise ValueError(f"{where}: CRS is geographic ({gdf.crs.to_string()}); reproject to {AREA_CRS} first.")
    except AttributeError:
        if any(u in str(gdf.crs).lower() for u in ["degree", "longlat", "epsg:4269", "epsg:4326"]):
            raise ValueError(f"{where}: CRS looks geographic; reproject to {AREA_CRS} first.")


# ============ Clean Texas 2020 blocks (notebook parity) ============
CENSUS_BLOCK_KEEP = [
    "STATEFP20","COUNTYFP20","TRACTCE20","BLOCKCE20",
    "GEOID20","NAME20","ALAND20","AWATER20","INTPTLAT20","INTPTLON20","geometry",
]

def _cols_present_case_insensitive(gdf: gpd.GeoDataFrame, cols_upper: list[str]) -> bool:
    cols_up = {c.upper() for c in gdf.columns}
    return set(cols_upper).issubset(cols_up)

def _subset_by_upper(gdf: gpd.GeoDataFrame, cols_upper: list[str]) -> gpd.GeoDataFrame:
    name_map = {c.upper(): c for c in gdf.columns}
    want = [name_map[u] for u in cols_upper if u in name_map]
    if "geometry" in gdf.columns and "geometry" not in want:
        want.append("geometry")
    out = gdf[want]
    if not isinstance(out, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry="geometry" if "geometry" in out.columns else None, crs=gdf.crs)
    return out

def is_tx_2020_blocks(gdf: gpd.GeoDataFrame) -> bool:
    return _cols_present_case_insensitive(gdf, CENSUS_BLOCK_KEEP[:-1])

def clean_tx_2020_blocks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sub = _subset_by_upper(gdf, CENSUS_BLOCK_KEEP)
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


# =================== Metrics ===================
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
        {"polsby_popper": polsby, "schwartzberg": schwartz, "convex_hull_ratio": hull_ratio, "reock": reock},
        index=g.index,
    )


def unify_pl94_schema(pl94: pd.DataFrame) -> pd.DataFrame:
    df = pl94.copy()
    geoid_candidates = ["geoid20","geoid","tabblock20","tabblock2020","block_geoid","blk_geoid","sctbkey","ctbkey"]
    geoid_col = next((c for c in geoid_candidates if c in df.columns), None)
    if geoid_col is None:
        raise ValueError("PL-94 file lacks a recognizable GEOID column (e.g., geoid20).")
    if geoid_col != "geoid20":
        df = df.rename(columns={geoid_col: "geoid20"})
    df["geoid20"] = (
        df["geoid20"].astype(str).str.replace(".0$", "", regex=True).str.replace(r"[^\d]", "", regex=True).str.zfill(15)
    )

    # map common VAP columns
    rename_map = {}
    if "vap_total" not in df.columns and "vap" in df.columns: rename_map["vap"] = "vap_total"
    if "anglovap" in df.columns and "nh_white_vap" not in df.columns: rename_map["anglovap"] = "nh_white_vap"
    if "blackvap" in df.columns and "nh_black_vap" not in df.columns: rename_map["blackvap"] = "nh_black_vap"
    if "asianvap" in df.columns and "nh_asian_vap" not in df.columns: rename_map["asianvap"] = "nh_asian_vap"
    if "hispvap" in df.columns and "hispanic_vap" not in df.columns: rename_map["hispvap"] = "hispanic_vap"
    if rename_map: df = df.rename(columns=rename_map)
    return df


# =================== Elections cleaner ===================
def is_tall_elections(df: pd.DataFrame) -> bool:
    req = {"office", "party", "votes"}
    has_vtd = {"cntyvtd", "cntyvtdkey", "cnty_vtd", "vtdkey", "vtd_key", "county_vtd_key"} & set(df.columns)
    return req.issubset(df.columns) and bool(has_vtd)


def clean_vtd_election_returns(df: pd.DataFrame, target_office: str = "") -> pd.DataFrame:
    """
    Tall → wide by VTD: dem/rep/third/total and two_party_dem_share.
    - If target_office is falsy (""), aggregate ALL offices combined.
    - If target_office provided: try exact label, then US-Sen regex; if coverage is poor, fall back to ALL offices.
    """
    t = df.copy()
    t = normalize_vtd_key(t, "cntyvtd")
    t = t.assign(
        office=t.get("office", "").astype(str).str.strip(),
        party=t.get("party", "").astype(str).str.upper().str.strip(),
        votes=pd.to_numeric(t.get("votes", 0), errors="coerce").fillna(0).astype(int),
    )

    # Debug: show top office labels
    try:
        print("[DEBUG] Top office labels:\n",
              t["office"].astype(str).str.strip().value_counts().head(20).to_string())
    except Exception:
        pass

    # === PATCH 1: County→FIPS + robust elections VTD key build ===================
    import numpy as np
    import re

    def _norm_name(x):
        if pd.isna(x): return x
        return str(x).strip().title()

    # Minimal mapping; extend if needed (you already have Bexar here)
    COUNTY_NAME_TO_FIPS = {
        "Bexar": "029", "Harris": "201", "Dallas": "113", "Tarrant": "439", "Travis": "453",
        "Collin": "085", "Denton": "121", "Hidalgo": "215", "El Paso": "141", "Fort Bend": "157",
        "Williamson": "491", "Montgomery": "339", "Cameron": "061", "Nueces": "355",
        "Galveston": "167", "Brazoria": "039", "Lubbock": "303", "Bell": "027",
        "McLennan": "309", "Jefferson": "245"
    }

    # 1) Ensure 'cnty' exists (3-digit FIPS). Accept County name OR 5-digit FIPS columns.
    name_cols = [c for c in t.columns if c.lower() in {"county", "countyname", "county_name"}]
    five_cols = [c for c in t.columns if c.lower() in {"countyfips", "fips", "county_fips"}]

    if "cnty" not in t.columns:
        t["cnty"] = pd.NA

    if name_cols:
        nm = name_cols[0]
        t.loc[t["cnty"].isna(), "cnty"] = t[nm].map(lambda v: COUNTY_NAME_TO_FIPS.get(_norm_name(v), pd.NA))

    if five_cols:
        cf = five_cols[0]
        sel = t["cnty"].isna() & t[cf].notna()
        t.loc[sel, "cnty"] = (
            t.loc[sel, cf].astype(str).str.findall(r"\d").str.join("").str[-3:].str.zfill(3)
        )

    t["cnty"] = (
        t["cnty"].astype("string")
        .str.findall(r"\d").str.join("").str[-3:].str.zfill(3)
        .where(lambda s: ~s.isin({"", "000", "<NA>", "nan"}), pd.NA)
    )

    # 2) Build robust VTD digits (+ optional letter suffix) from flexible columns.
    def _first_present(df, cols):
        for c in cols:
            if c in df.columns: return c
        return None

    vtd_col = _first_present(t, [
        "VTD", "vtd", "Precinct", "precinct", "Pct", "pct", "PCT",
        "Precinct Number", "Pct Number", "VTDNumber", "VtdNumber", "pct_number", "precinct_number"
    ])

    if vtd_col is None and "cntyvtd" in t.columns:
        # derive digits from any existing combined key
        digits_src = t["cntyvtd"].astype(str)
    elif vtd_col:
        digits_src = t[vtd_col].astype(str)
    else:
        digits_src = pd.Series("", index=t.index, dtype="string")

    t["_e_vtd_digits"] = (
        digits_src.str.findall(r"\d").str.join("").str[-5:].str.zfill(5)
        .where(lambda s: ~s.isin({"", "00000", "<NA>", "nan"}), pd.NA)
        .astype("string")
    )

    # Try to capture split/letter suffix if present
    split_col = _first_present(t, ["Split", "split", "Suffix", "suffix", "Precinct Split", "PrecinctSplit"])
    if split_col:
        suf = t[split_col].astype(str).str.extract(r"([A-Za-z]+)$", expand=False).fillna("").str.upper()
    else:
        base_for_suffix = (t[vtd_col].astype(str) if vtd_col else pd.Series("", index=t.index))
        suf = base_for_suffix.str.extract(r"([A-Za-z]+)$", expand=False).fillna("").str.upper()

    # Make string-safe
    t["_e_vtd_digits"] = t["_e_vtd_digits"].astype("string")
    suf = suf.astype("string").fillna("")
    t["cnty"] = t["cnty"].astype("string")

    valid_parts = t["cnty"].notna() & t["_e_vtd_digits"].notna()
    combined = (t.loc[valid_parts, "cnty"] + t.loc[valid_parts, "_e_vtd_digits"])

    # Compose: <3-digit cnty><5-digit vtd><suffix?>
    t["cntyvtd"] = pd.Series(pd.NA, dtype="string", index=t.index)
    t.loc[valid_parts, "cntyvtd"] = (combined + suf.loc[valid_parts]).astype("string")

    # standardized (no suffix)
    t["cntyvtd_std"] = pd.Series(pd.NA, dtype="string", index=t.index)
    t.loc[valid_parts, "cntyvtd_std"] = combined.astype("string")

    # Helpful debug
    bexar = t.loc[t["cnty"] == "029"]
    print(f"[DEBUG] Elections rows with cnty='029' (Bexar) after build: {len(bexar)}")
    if len(bexar):
        print(f"[DEBUG] Sample Bexar cntyvtd: {bexar['cntyvtd'].head(10).tolist()}")

    # --------------------------------------------------------------------------
    # County name → FIPS override + key repair (helps BEXAR/029 etc.)
    county_cols = [c for c in t.columns if c.lower() in ("county", "county_name", "cty_name")]
    if county_cols:
        cname = (
            t[county_cols[0]].astype(str).str.upper()
            .str.replace(r"[^A-Z ]", "", regex=True).str.strip()
        )
        cnty_from_name = cname.map(COUNTY_NAME_TO_FIPS)

        # --- NEW: also recognize numeric county code columns (CountyId, CountyNumber, FIPS, etc.)
        cnty_code_cols = [
            c for c in t.columns
            if c.lower() in (
                "countyid","county_id","countynumber","county_number","countycode","county_code",
                "fips","countyfips","county_fips","cnty_fips","cntycode","cnty_code"
            )
        ]
        cnty_from_code = pd.Series(index=t.index, dtype="string")

        if cnty_code_cols:
            # Use the first matching numeric column
            raw_code = (
                t[cnty_code_cols[0]]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)  # keep digits
                .fillna("")
            )
            # If it's full FIPS (e.g., 48029), strip leading '48'
            raw_code = raw_code.str.replace(r"^48(?=\d{3}$)", "", regex=True)
            # Last 3 digits, left-pad
            cnty_from_code = raw_code.str[-3:].str.zfill(3).astype("string")

        # Prefer name mapping, else code mapping, else NA
        cnty_best = cnty_from_name.astype("string")
        use_code = cnty_best.isna() & cnty_from_code.notna() & cnty_from_code.str.match(r"^\d{3}$")
        cnty_best = cnty_best.mask(use_code, cnty_from_code)

        # Look for any precinct/VTD-ish column we can mine
        vtd_candidates = [
            "vtd", "vtdid", "vtd_id", "vtdkey", "vtd_key",
            "precinct", "precinct_id", "precinctid", "prec_id",
            "pct", "pctid", "pct_id", "pct_code", "pctname", "pct_name",
            "cnty_vtd", "cntyvtdkey"
        ]
        vtd_src = next((c for c in vtd_candidates if c in t.columns), None)

        if vtd_src:
            t["vtd_norm"] = (
                t[vtd_src].astype(str).str.upper()
                .str.replace(r"[^A-Z0-9]", "", regex=True).str.strip()
            )
        else:
            # fall back to using whatever is in cntyvtd for extracting digits
            t["vtd_norm"] = t["cntyvtd"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

        # Current parse of cntyvtd
        parsed = normalize_cntyvtd_safely(t["cntyvtd"])
        # Prefer 5-digit run if present, else 4
        digs = t["vtd_norm"].str.extract(r"(?P<d5>\d{5})|(?P<d4>\d{4})", expand=True)
        vtd_digits = digs["d5"].fillna(digs["d4"]).astype(str)
        suf = t["vtd_norm"].str.extract(r"([A-Z]+)$", expand=False).fillna("")

        needs_override = parsed.isna() & cnty_best.notna()
        have_real_digits = vtd_digits.str.match(r"^\d{4,5}$")
        ok_to_repair = needs_override & have_real_digits

        if ok_to_repair.any():
            d = vtd_digits.str.zfill(5)
            repaired = (cnty_best + d + suf).astype("string")   # ← use cnty_best here
            t["cntyvtd"] = t["cntyvtd"].astype("string")
            t.loc[ok_to_repair, "cntyvtd"] = repaired.loc[ok_to_repair]

    # --------------------------------------------------------------------------

    def _to_wide(sub: pd.DataFrame) -> pd.DataFrame:
        agg = sub.groupby(["cntyvtd", "party"], as_index=False)["votes"].sum()
        piv = agg.pivot_table(index="cntyvtd", columns="party", values="votes", fill_value=0, aggfunc="sum")
        for p in ["D","R","G","L","I","W"]:
            if p not in piv.columns: piv[p] = 0
        out = pd.DataFrame({
            "cntyvtd": piv.index.astype(str),
            "dem_votes": piv["D"].astype(int),
            "rep_votes": piv["R"].astype(int),
        })
        out["third_party_votes"] = piv[["G","L","I","W"]].sum(axis=1).astype(int)
        out["total_votes"] = (out["dem_votes"] + out["rep_votes"] + out["third_party_votes"]).astype(int)
        out["two_party_dem_share"] = (
            out["dem_votes"] / (out["dem_votes"] + out["rep_votes"]).replace({0: pd.NA})
        )
        return out

    # If no specific office, aggregate all
    if not target_office:
        return _to_wide(t)

    # Otherwise: robust office filter (exact label → US Senate regex → fallback all)
    t["office_norm"] = (
        t["office"].astype(str).str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    fed_sen_mask = (
        t["office_norm"].str.contains(r"(?:unitedstatessenate|ussen(?:ate)?|ussen)$", regex=True)
        & ~t["office_norm"].str.contains(r"statesenate", regex=True)
    )

    exact_mask = t["office"].astype(str).str.casefold() == str(target_office).casefold()
    sub = t.loc[exact_mask].copy()
    used_fed_regex = False
    if sub.empty:
        sub = t.loc[fed_sen_mask].copy()
        used_fed_regex = True
        if sub.empty:
            print(f"[WARN] No rows matched office='{target_office}' or US-Sen regex. Using ALL offices combined.")
            return _to_wide(t)

    wide = _to_wide(sub)

    # Coverage fallback to ALL offices (strict)
    distinct_vtds_all = t["cntyvtd"].nunique()
    covered_vtds = wide["cntyvtd"].nunique()
    if distinct_vtds_all and covered_vtds < 0.95 * distinct_vtds_all:
        src = f"office='{target_office}'" if not used_fed_regex else "US-Sen regex"
        print(f"[WARN] Low coverage for {src}: {covered_vtds}/{distinct_vtds_all} (~{covered_vtds/distinct_vtds_all:.1%}). "
              f"Falling back to ALL offices combined.")
        return _to_wide(t)

    return wide


# ========================== Stage 1: ETL ==========================
def run_etl(
    input_files: list[Path],
    out_parquet_dir: Path,
    out_geo_dir: Path,
    sqlite_path: Path,
    elections_office: str = "",   # default: ALL offices combined
):
    print("\n========== Stage 1: ETL ==========")
    mkdir_p(out_parquet_dir); mkdir_p(out_geo_dir); mkdir_p(sqlite_path.parent)

    elections_path = input_files[-1]
    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if path.suffix.lower() not in ALL_EXTS:
            raise ValueError(f"Unsupported input type for {path.name} (ext {path.suffix})")

        name = dataset_key(path)
        print(f"[ETL] Reading {path} -> key={name}")
        df = read_any(path).drop_duplicates(ignore_index=True)

        # Elections: tall → wide per VTD
        if path == elections_path and not is_geodf(df):
            tdf_tall = coerce_types_light(stdcols(df))
            if is_tall_elections(tdf_tall):
                print(f"[ETL] Detected tall elections file. Cleaning office='{elections_office}' → wide VTD votes.")
                try:
                    # (a) Requested office (or ALL if blank)
                    tdf = clean_vtd_election_returns(tdf_tall, target_office=elections_office)
                    # (b) Always also compute ALL-offices combined for Stage-2 fallback
                    tdf_all = clean_vtd_election_returns(tdf_tall, target_office="")
                except Exception as e:
                    raise RuntimeError(f"Failed cleaning elections file '{path.name}': {e}") from e
            else:
                # If it's already wide, just pass through and clone for _all
                tdf = tdf_tall
                tdf_all = tdf_tall.copy()

            # Save both versions
            write_parquet(tdf, out_parquet_dir / f"{name}.parquet")
            write_parquet(tdf_all, out_parquet_dir / f"{name}_all.parquet")

            # Warehouse
            to_sqlite(tdf, sqlite_path, f"stg_{name}")
            to_sqlite(tdf_all, sqlite_path, f"stg_{name}_all")
            continue

        # Geospatial / Tabular defaults
        if is_geodf(df):
            if is_tx_2020_blocks(df):
                print("[ETL] Detected Texas 2020 Census blocks → applying notebook cleaning.")
                gdf = clean_tx_2020_blocks(df)
            else:
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

    # Locate cleaned artifacts from Stage 1
    districts_fp = out_geo_dir / f"{districts_key}.parquet"
    census_fp    = out_geo_dir / f"{census_key}.parquet"
    vtds_fp      = out_geo_dir / f"{vtds_key}.parquet"
    pl94_fp      = out_parquet_dir / f"{pl94_key}.parquet"
    elect_fp     = out_parquet_dir / f"{elections_key}.parquet"

    elect_all_fp = out_parquet_dir / f"{elections_key}_all.parquet"
    if not elect_all_fp.exists():
        raise FileNotFoundError(f"Missing ALL-offices elections file: {elect_all_fp}\n"
                                f"(Re-run Stage 1 with the patch that writes *_all.parquet)")
    vtd_elec_all = stdcols(pd.read_parquet(elect_all_fp))

    missing = [p for p in [districts_fp, census_fp, vtds_fp, pl94_fp, elect_fp] if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 2 missing cleaned files:\n" + "\n".join(f" - {p}" for p in missing))

    print("[INFO] Using cleaned files:")
    print("  ", districts_fp)
    print("  ", census_fp)
    print("  ", vtds_fp)
    print("  ", pl94_fp)
    print("  ", elect_fp)

    # Load
    districts = ensure_crs(stdcols(gpd.read_parquet(districts_fp)))
    census_geo = ensure_crs(stdcols(gpd.read_parquet(census_fp)))
    vtd_geo = ensure_crs(stdcols(gpd.read_parquet(vtds_fp)))
    pl94 = stdcols(pd.read_parquet(pl94_fp)); pl94 = unify_pl94_schema(pl94)
    vtd_elec = stdcols(pd.read_parquet(elect_fp))

    # ----- Canonical cntyvtd_std on BOTH sides -----
    vtd_geo = vtd_geo.copy(); vtd_elec = vtd_elec.copy()

    # Geo side: prefer CNTY+VTD parts; else prebuilt cntyvtd cleaned
    if {"cnty", "vtd"} <= set(vtd_geo.columns):
        vtd_geo["cntyvtd_std"] = build_cntyvtd_from_parts(vtd_geo["cnty"], vtd_geo["vtd"])
    elif "cntyvtd" in vtd_geo.columns:
        vtd_geo["cntyvtd_std"] = (
            vtd_geo["cntyvtd"].astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
    else:
        raise ValueError("VTD geometry lacks CNTY/VTD or CNTYVTD to build a key.")

    # Elections side: parse w/ TX-aware normalizer; fallback to cleaned string
    if "cntyvtd" in vtd_elec.columns:
        parsed = normalize_cntyvtd_safely(vtd_elec["cntyvtd"])
        fallback = (
            vtd_elec["cntyvtd"].astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        vtd_elec["cntyvtd_std"] = parsed.fillna(fallback)
    elif {"fips", "vtd"} <= set(vtd_elec.columns):
        vtd_elec["cntyvtd_std"] = build_cntyvtd_from_fips_vtd(vtd_elec["fips"], vtd_elec["vtd"])
    else:
        raise ValueError("Elections file lacks cntyvtd (and FIPS/VTD) to build a key.")

    vtd_geo  = vtd_geo.loc[vtd_geo["cntyvtd_std"].notna()].copy()
    vtd_elec = vtd_elec.loc[vtd_elec["cntyvtd_std"].notna()].copy()

    print(f"[INFO] VTD keys (standardized) — geo: {vtd_geo['cntyvtd_std'].nunique()} unique, "
          f"elections: {vtd_elec['cntyvtd_std'].nunique()} unique")

    # Inspect VTD lengths in elections (helps spot 5-digit codes)
    lens = vtd_elec["cntyvtd_std"].dropna().astype(str).str.extract(r"^(?:48)?\d{3}(\d+)")[0].str.len().value_counts().sort_index()
    print("[DIAG] Elections VTD digit lengths (post-cnty):", lens.to_dict())

    # Build standardized keys for ALL-offices too
    if "cntyvtd" in vtd_elec_all.columns:
        parsed_all = normalize_cntyvtd_safely(vtd_elec_all["cntyvtd"])
        fallback_all = (
            vtd_elec_all["cntyvtd"].astype(str).str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        vtd_elec_all["cntyvtd_std"] = parsed_all.fillna(fallback_all)
    elif {"fips", "vtd"} <= set(vtd_elec_all.columns):
        vtd_elec_all["cntyvtd_std"] = build_cntyvtd_from_fips_vtd(vtd_elec_all["fips"], vtd_elec_all["vtd"])
    else:
        raise ValueError("ALL-offices elections file lacks cntyvtd (and FIPS/VTD) to build a key.")

    # Constrain ALL-offices to valid counties (same filter you use for the requested contest)
    def _county_from_key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.extract(r"^(?:48)?(\d{3})")[0]

    valid_cnties = set(_county_from_key(vtd_geo["cntyvtd_std"]).dropna().unique())
    # Constrain requested-contest elections to valid counties too
    vtd_elec = vtd_elec[_county_from_key(vtd_elec["cntyvtd_std"]).isin(valid_cnties)].copy()
    vtd_elec_all = vtd_elec_all[_county_from_key(vtd_elec_all["cntyvtd_std"]).isin(valid_cnties)].copy()

    # --- Elections key fallback: map 5-digit VTDs to last-4 so they match the 4-digit GEO keys ---
    geo_key_set = set(vtd_geo["cntyvtd_std"].astype(str))

    def _apply_last4_fallback(elec_df: pd.DataFrame) -> pd.DataFrame:
        elec_df = elec_df.copy()
        # Current standardized key
        k = elec_df["cntyvtd_std"].astype(str)
        # Which keys miss the geo side?
        miss = ~k.isin(geo_key_set)
        if miss.any():
            # Build the last-4 variant and use it ONLY where the original misses and the last-4 exists in geo
            k_last4 = _last4_elec_key(k)
            can_swap = miss & k_last4.notna() & k_last4.isin(geo_key_set)
            elec_df.loc[can_swap, "cntyvtd_std"] = k_last4.loc[can_swap]
        return elec_df

    vtd_elec = _apply_last4_fallback(vtd_elec)
    vtd_elec_all = _apply_last4_fallback(vtd_elec_all)

    # --- County coverage diagnostic (counting only elections keys that also exist in geo) ---
    geo_keys = vtd_geo["cntyvtd_std"].astype(str)
    elec_keys = vtd_elec["cntyvtd_std"].astype(str)
    geo_cnty = _county_from_key(geo_keys)
    elec_cnty = _county_from_key(elec_keys)
    elec_in_geo_mask = elec_keys.isin(set(geo_keys))
    elec_by_cnty_in_geo = elec_cnty[elec_in_geo_mask].value_counts().sort_index()
    coverage = (
        pd.DataFrame({
            "geo_vtds": geo_cnty.value_counts().sort_index(),
            "elec_vtds": elec_by_cnty_in_geo,
        }).fillna(0).astype(int)
    )
    coverage["pct_covered"] = 0.0
    mask_cov = coverage["geo_vtds"] > 0
    coverage.loc[mask_cov, "pct_covered"] = (
        (coverage.loc[mask_cov, "elec_vtds"] / coverage.loc[mask_cov, "geo_vtds"])
        .clip(upper=1).round(3)
    )

    print("[DIAG] County VTD coverage (first 20):")
    print(coverage.head(20))

    # Build a base key without trailing letter suffixes (optional '48' + 3-digit county + 1–5 digits)
    def _base_vtd_key(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        m = s.str.extract(r"^(?:48)?(?P<cnty>\d{3})(?P<vtd>\d{1,5})")
        m["cnty"] = m["cnty"].fillna("").astype(str).str.zfill(3)
        m["vtd"] = m["vtd"].fillna("").astype(str).str.zfill(5)  # ← pad to 5
        return (m["cnty"] + m["vtd"]).where(m["cnty"].ne("") & m["vtd"].ne(""))

    # Prepare base-key aggregate for ALL-offices
    vtd_elec_all["cntyvtd_base"] = _base_vtd_key(vtd_elec_all["cntyvtd_std"])
    vote_cols_all = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes", "two_party_dem_share"]
                     if c in vtd_elec_all.columns]
    sum_cols_all = [c for c in vote_cols_all if c != "two_party_dem_share"]
    elec_all_base = vtd_elec_all.dropna(subset=["cntyvtd_base"]).copy()
    if sum_cols_all:
        elec_all_base_agg = elec_all_base.groupby("cntyvtd_base", as_index=False)[sum_cols_all].sum(min_count=1)
        if {"dem_votes", "rep_votes"}.issubset(elec_all_base_agg.columns):
            twoden_all = elec_all_base_agg["dem_votes"] + elec_all_base_agg["rep_votes"]
            elec_all_base_agg["two_party_dem_share"] = (elec_all_base_agg["dem_votes"] / twoden_all).where(twoden_all > 0)
    else:
        elec_all_base_agg = elec_all_base[["cntyvtd_base"]].drop_duplicates().assign(
            dem_votes=pd.NA, rep_votes=pd.NA, third_party_votes=pd.NA, total_votes=pd.NA, two_party_dem_share=pd.NA
        )

    # --- Ensure GEOID alignment for PL94 merge ---
    census_geo = ensure_geoid20_str(census_geo)
    pl94 = ensure_geoid20_str(pl94)

    # ----- PL-94 fields -----
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

    # (A) Blocks + PL94 → area-weighted to districts
    if "geoid20" not in census_geo.columns:
        raise ValueError("Census geometry must contain 'geoid20'")
    census_merged = census_geo.merge(pl94[need], on="geoid20", how="left")

    census_proj, districts_proj = census_merged.to_crs(AREA_CRS), districts.to_crs(AREA_CRS)
    assert_projected_planar(census_proj, "blocks→districts")
    assert_projected_planar(districts_proj, "blocks→districts")

    d_sub = districts_proj[["geometry"]].reset_index(names="district_idx")
    blk = census_proj[["geoid20", "geometry"]].copy()
    blk_attrs = pl94[need].copy()

    blk_inter = gpd.overlay(blk, d_sub, how="intersection", keep_geom_type=True)
    blk_inter = blk_inter.merge(blk_attrs, on="geoid20", how="left")

    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    blk_inter["blk_area"] = blk_area.loc[blk_inter["geoid20"]].values
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

    # (B) VTD geo + election → area-weight to districts

    # Build base keys (drop letter suffixes) on both sides
    vtd_geo["cntyvtd_base"] = _base_vtd_key(vtd_geo["cntyvtd_std"])
    vtd_elec["cntyvtd_base"] = _base_vtd_key(vtd_elec["cntyvtd_std"])

    # Aggregate elections by base key (combine A/B/C variants if they exist)
    vote_cols_all2 = [c for c in ["dem_votes","rep_votes","third_party_votes","total_votes","two_party_dem_share"]
                      if c in vtd_elec.columns]
    sum_cols = [c for c in vote_cols_all2 if c != "two_party_dem_share"]
    elec_base = vtd_elec.dropna(subset=["cntyvtd_base"]).copy()
    if sum_cols:
        elec_base_agg = elec_base.groupby("cntyvtd_base", as_index=False)[sum_cols].sum(min_count=1)
        if {"dem_votes","rep_votes"}.issubset(elec_base_agg.columns):
            twoden = elec_base_agg["dem_votes"] + elec_base_agg["rep_votes"]
            elec_base_agg["two_party_dem_share"] = (elec_base_agg["dem_votes"] / twoden).where(twoden > 0)
    else:
        elec_base_agg = elec_base[["cntyvtd_base"]].drop_duplicates().assign(
            dem_votes=pd.NA, rep_votes=pd.NA, third_party_votes=pd.NA, total_votes=pd.NA, two_party_dem_share=pd.NA
        )

    # First pass: exact suffix-preserving join on cntyvtd_std
    vtd_with_votes = vtd_geo.merge(
        vtd_elec.drop(columns=[c for c in ["cntyvtd","cnty","vtd","fips"] if c in vtd_elec.columns]),
        on="cntyvtd_std", how="left", suffixes=("", "_elec")
    )

    # Second pass: fill any still-missing vote cells from base-key aggregates
    vote_cols_base = [c for c in ["dem_votes","rep_votes","third_party_votes","total_votes"] if c in vtd_with_votes.columns]
    missing_mask = vtd_with_votes[vote_cols_base].isna().all(axis=1) if vote_cols_base else pd.Series(False, index=vtd_with_votes.index)
    if missing_mask.any():
        filler = vtd_with_votes.loc[missing_mask, ["cntyvtd_base"]].merge(elec_base_agg, on="cntyvtd_base", how="left")
        for col in ["dem_votes","rep_votes","third_party_votes","total_votes","two_party_dem_share"]:
            if col in vtd_with_votes.columns and col in filler.columns:
                vtd_with_votes.loc[missing_mask, col] = vtd_with_votes.loc[missing_mask, col].fillna(filler[col])

    # Third pass: fill any remaining missing vote cells from ALL-offices aggregates
    if missing_mask.any():
        if "cntyvtd_base" not in vtd_with_votes.columns:
            vtd_with_votes["cntyvtd_base"] = _base_vtd_key(vtd_with_votes["cntyvtd_std"])
        fallback_fill = vtd_with_votes.loc[missing_mask, ["cntyvtd_base"]].merge(elec_all_base_agg, on="cntyvtd_base", how="left")
        for col in ["dem_votes","rep_votes","third_party_votes","total_votes","two_party_dem_share"]:
            if col in vtd_with_votes.columns and col in fallback_fill.columns:
                vtd_with_votes.loc[missing_mask, col] = vtd_with_votes.loc[missing_mask, col].fillna(fallback_fill[col])

    # CRS + alias
    vtd_with_votes = vtd_with_votes.to_crs(AREA_CRS)
    assert_projected_planar(vtd_with_votes, "vtd→districts")
    vtd_with_votes["cntyvtd"] = vtd_with_votes["cntyvtd_std"]  # alias for downstream

    districts_proj = districts_proj.to_crs(AREA_CRS)
    assert_projected_planar(districts_proj, "vtd→districts")

    if not vote_cols_base:
        votes_by_dist = pd.DataFrame(index=districts_proj.index,
                                     columns=["dem_votes","rep_votes","third_party_votes","total_votes"])
        part = pd.DataFrame(index=districts_proj.index,
                            columns=["dem_share","rep_share","two_party_dem_share"])
        possible = pd.DataFrame(columns=["district_idx","cntyvtd"])
    else:
        vtd_cols_keep = ["cntyvtd","geometry"] + vote_cols_base
        vtd_compact = vtd_with_votes.loc[:, vtd_cols_keep].copy()

        possible = gpd.sjoin(
            vtd_compact[["cntyvtd","geometry"]],
            districts_proj[["geometry"]].reset_index(names="district_idx"),
            how="inner", predicate="intersects",
        )
        pairs = possible[["cntyvtd","district_idx"]].drop_duplicates()

        vtd_subset = vtd_compact[vtd_compact["cntyvtd"].isin(pairs["cntyvtd"])].copy()
        d_sub = districts_proj[["geometry"]].reset_index(names="district_idx")

        vtd_district_intersections = gpd.overlay(vtd_subset, d_sub, how="intersection", keep_geom_type=True)

        if vtd_district_intersections.empty:
            votes_by_dist = pd.DataFrame(index=districts_proj.index, columns=vote_cols_base).fillna(0)
            part = pd.DataFrame(index=districts_proj.index,
                                columns=["dem_share","rep_share","two_party_dem_share"])
        else:
            vtd_areas = (
                vtd_with_votes[["cntyvtd","geometry"]]
                .drop_duplicates("cntyvtd").set_index("cntyvtd").geometry.area.rename("vtd_area")
            )
            vtd_district_intersections["vtd_area"] = vtd_district_intersections["cntyvtd"].map(vtd_areas)
            vtd_district_intersections["inter_area"] = vtd_district_intersections.geometry.area

            eps = 1e-9
            vtd_district_intersections = vtd_district_intersections.loc[
                vtd_district_intersections["vtd_area"] > eps
            ].copy()
            vtd_district_intersections["weight"] = (
                vtd_district_intersections["inter_area"] / vtd_district_intersections["vtd_area"]
            ).clip(0, 1)

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
            if {"dem_votes","rep_votes"}.issubset(votes_by_dist.columns):
                tot = (
                    votes_by_dist["total_votes"]
                    if "total_votes" in votes_by_dist.columns
                    else (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"] + votes_by_dist.get("third_party_votes", 0))
                )
                valid = tot > 0
                part["dem_share"] = (votes_by_dist["dem_votes"] / tot).where(valid)
                part["rep_share"] = (votes_by_dist["rep_votes"] / tot).where(valid)
                part["two_party_dem_share"] = (
                    votes_by_dist["dem_votes"] / (votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"])
                ).where((votes_by_dist["dem_votes"] + votes_by_dist["rep_votes"]) > 0)
            else:
                part["dem_share"] = pd.NA; part["rep_share"] = pd.NA; part["two_party_dem_share"] = pd.NA

    # --- Diagnostics ---
    if 'total_votes' in votes_by_dist.columns:
        tot = votes_by_dist['total_votes']
    else:
        tot = (votes_by_dist.get('dem_votes', 0)
               + votes_by_dist.get('rep_votes', 0)
               + votes_by_dist.get('third_party_votes', 0))
    zero_vote_districts = list(votes_by_dist.index[pd.Series(tot).fillna(0) == 0])
    print(f"[DIAG] Districts with zero total votes (shares become NA): {zero_vote_districts}")

    print("[DIAG] County VTD coverage (top by elections VTDs):")
    print(coverage.sort_values("elec_vtds", ascending=False).head(20))

    if not possible.empty:
        vtds_per_dist = possible[['district_idx','cntyvtd']].drop_duplicates().groupby('district_idx').size()
        print("[DIAG] Example intersect counts (first 10):"); print(vtds_per_dist.head(10))

        vote_cols = [c for c in ["dem_votes","rep_votes","third_party_votes","total_votes"] if c in vtd_with_votes.columns]
        had_votes = vtd_with_votes.dropna(subset=vote_cols, how='all')['cntyvtd'].unique()
        matched_per_dist = possible[possible['cntyvtd'].isin(had_votes)].groupby('district_idx')['cntyvtd'].nunique()
        print("[DIAG] Example matched-vote VTD counts (first 10):"); print(matched_per_dist.head(10))

        total_vtds = len(vtd_with_votes)
        matched_mask_any = vtd_with_votes[vote_cols].notna().any(axis=1) if vote_cols else pd.Series(False, index=vtd_with_votes.index)
        matched_vtds = int(matched_mask_any.sum()) if vote_cols else 0
        pct_str = f"{matched_vtds / total_vtds:.1%}" if total_vtds else "0.0%"
        print(f"[DIAG] VTDs with any matched votes: {matched_vtds}/{total_vtds} ({pct_str})")
        unmatched_keys = vtd_with_votes.loc[~matched_mask_any, "cntyvtd"].dropna().astype(str).head(20).tolist() if vote_cols else []
        print("[DIAG] Sample unmatched VTD keys (cntyvtd):", unmatched_keys)

        problem_idxs = zero_vote_districts
        per_dist = possible[['district_idx','cntyvtd']].drop_duplicates()
        per_dist['has_votes'] = per_dist['cntyvtd'].isin(vtd_with_votes.loc[matched_mask_any,'cntyvtd']) if vote_cols else False
        summary = (per_dist.groupby('district_idx')['has_votes']
                   .agg(['sum','count']).loc[problem_idxs]
                   .rename(columns={'sum':'matched_vtds','count':'intersecting_vtds'}))
        summary['district_id'] = summary.index + 1
        print("[DIAG] Problem districts — matched/total VTDs:"); print(summary[['district_id','matched_vtds','intersecting_vtds']])
    else:
        print("[DIAG] No VTD↔district intersections found (check CRS and geometries).")

    if zero_vote_districts:
        for d_idx in zero_vote_districts:
            print(f"\n[DEEPDIAG] Investigating district_idx={d_idx} (district_id={d_idx + 1})")
            dvtd = (possible.loc[possible['district_idx'] == d_idx, 'cntyvtd']
                    .drop_duplicates().astype(str))
            print(f"[DEEPDIAG] Intersecting VTDs: {len(dvtd)}")

            vote_cols_check = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]
                               if c in vtd_with_votes.columns]
            if vote_cols_check:
                has_any = vtd_with_votes[vote_cols_check].notna().any(axis=1)
                dvtd_has_votes = set(vtd_with_votes.loc[has_any, 'cntyvtd'].astype(str))
                matched = dvtd.isin(dvtd_has_votes).sum()
                print(f"[DEEPDIAG] Matched-vote VTDs in district: {matched}/{len(dvtd)}")

            missing_keys = dvtd[~dvtd.isin(vtd_with_votes.loc[has_any, 'cntyvtd'].astype(str))].head(25).tolist()
            print("[DEEPDIAG] Sample missing VTD keys:", missing_keys)

            def _county_from_key(s: pd.Series) -> pd.Series:
                return s.astype(str).str.extract(r"^(?:48)?(\d{3})")[0]

            d_cnties_geo = _county_from_key(pd.Series(dvtd)).value_counts().sort_index()
            print("[DEEPDIAG] Counties (GEO side) in district:", d_cnties_geo.to_dict())

            elec_keys_after = vtd_elec["cntyvtd_std"].astype(str)
            cnty_elec_after = _county_from_key(elec_keys_after).value_counts().sort_index()
            print("[DEEPDIAG] Elections rows per county (after fallback):",
                  {k: int(cnty_elec_after.get(k, 0)) for k in d_cnties_geo.index})

            if "029" in d_cnties_geo.index:
                in_geo_029 = set(k for k in dvtd if k.startswith("029"))
                in_elec_029 = set(elec_keys_after[elec_keys_after.str.startswith("029")])
                print(f"[DEEPDIAG] Bexar (029): GEO VTDs={len(in_geo_029)}; ELEC VTDs(after fallback)={len(in_elec_029)}")
                missing_029 = sorted(list(in_geo_029 - in_elec_029))[:15]
                print("[DEEPDIAG] Sample missing Bexar (029) keys:", missing_029)

    # (C) Compactness
    cmpx = compute_compactness(districts_proj)

    # (D) Assemble final
    final = pd.DataFrame(index=districts_proj.index)
    id_col = next((c for c in ["district","district_id","cd","dist"] if c in districts.columns), None)
    final["district_id"] = districts[id_col].values if id_col else (districts_proj.index + 1)
    final = final.join(cmpx).join(race_pct).join(part[["dem_share","rep_share"]]).sort_index()
    assert len(final) == 38, f"Expected 38 districts, found {len(final)}"

    # --- Fill missing vote shares with 0.0 ---
    if {"dem_share","rep_share"}.issubset(final.columns):
        final[["dem_share","rep_share"]] = final[["dem_share","rep_share"]].fillna(0.0)

    return final


# =============================== CLI ===============================
def main():
    ap = argparse.ArgumentParser(description="ETL + final 38-district dataset (Texas)")
    ap.add_argument("--districts", type=Path, required=True, help="District polygons (SHP/GPKG/GeoParquet)")
    ap.add_argument("--census", type=Path, required=True, help="2020 Census blocks (SHP/GPKG/GeoParquet)")
    ap.add_argument("--vtds", type=Path, required=True, help="VTD geometries (SHP/GPKG/GeoParquet)")
    ap.add_argument("--pl94", type=Path, required=True, help="PL-94 attributes (CSV/TSV/Parquet/TXT)")
    ap.add_argument("--elections", type=Path, required=True, help="VTD election returns (CSV/TSV/Parquet)")
    ap.add_argument("--elections-office", type=str, default="",
                    help="Contest to aggregate (e.g., 'U.S. Sen', 'President'). "
                         "Empty string aggregates ALL offices combined (default).")
    ap.add_argument("--data-processed-tabular", type=Path, required=True, help="Output folder for cleaned Parquet")
    ap.add_argument("--data-processed-geospatial", type=Path, required=True, help="Output folder for cleaned GeoParquet")
    ap.add_argument("--sqlite", type=Path, required=True, help="SQLite warehouse path")
    args = ap.parse_args()

    input_files = [args.districts, args.census, args.vtds, args.pl94, args.elections]

    # Stage 1
    run_etl(
        input_files,
        args.data_processed_tabular,
        args.data_processed_geospatial,
        args.sqlite,
        elections_office=args.elections_office,
    )

    # Resolve cleaned dataset keys (derived from input filenames)
    dk, ck, vk, pk, ek = [dataset_key(p) for p in input_files]

    # Stage 2
    final = build_final(
        args.data_processed_tabular,
        args.data_processed_geospatial,
        dk, ck, vk, pk, ek,
    )

    # Save final outputs
    pq = args.data_processed_tabular / "districts_final.parquet"
    csv = args.data_processed_tabular / "districts_final.csv"
    write_parquet(final, pq)
    final.to_csv(csv, index=False)
    print("\nSaved:\n ", pq, "\n ", csv)


if __name__ == "__main__":
    main()
