#!/usr/bin/env python3
# etl_pipeline.py
# Build a VTD-level dataset for Texas (one row per voting district):
# - Stage 1: Clean and persist raw inputs (tabular + geospatial)
# - Stage 2: Join PL94 + elections to VTDs via area-weighting
# - Outputs: vtds_final.{parquet,csv}

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import numpy as np

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
def mkdir_p(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_key(path: Path) -> str:
    # derive a stable key from the filename
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
        # heuristically: tab vs comma
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = "".join(f.readlines()[:10])
        sep = "\t" if sample.count("\t") > sample.count(",") else ","
        return pd.read_csv(path, sep=sep)
    if ext in (".shp", ".gpkg"):
        return gpd.read_file(path)

    raise ValueError(f"Unsupported input type for {path}")


def is_geodf(df: pd.DataFrame) -> bool:
    return isinstance(df, gpd.GeoDataFrame) or "geometry" in df.columns


def ensure_crs(gdf: "gpd.GeoDataFrame", target: str | None = None) -> "gpd.GeoDataFrame":
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise TypeError("ensure_crs: expected a GeoDataFrame or a DataFrame with 'geometry'")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    if gdf.crs is None and target is None:
        raise ValueError(
            "GeoDataFrame has no CRS. Either set gdf.crs or pass a 'target' CRS to reproject."
        )
    if target is not None:
        return gdf.to_crs(target)
    return gdf


def assert_projected_planar(gdf: "gpd.GeoDataFrame", where: str) -> None:
    if gdf.crs is None:
        raise ValueError(f"{where}: GeoDataFrame has no CRS; project to {AREA_CRS} before metric computations.")
    # crude heuristic: degrees vs meters
    if hasattr(gdf.crs, "is_geographic"):
        if gdf.crs.is_geographic:
            raise ValueError(f"{where}: CRS is geographic ({gdf.crs.to_string()}); reproject to {AREA_CRS} first.")
    else:
        s = gdf.crs.to_string().upper()
        if "EPSG:4326" in s or "+UNITS=DEG" in s or "LONGITUDE" in s or "LATITUDE" in s:
            raise ValueError(f"{where}: CRS looks geographic; reproject to {AREA_CRS} first.")


def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    # lower_snake case
    df = df.copy()
    df.columns = [
        re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def col_like(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def to_int_str(s: pd.Series, width: int | None = None) -> pd.Series:
    out = s.astype("string").str.replace(r"[^\d]", "", regex=True)
    out = out.where(out.str.len() > 0, pd.NA)
    if width is not None:
        out = out.str.zfill(width)
    return out


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
        out["geometry_wkb"] = out.geometry.to_wkb()
    return pd.DataFrame(out.drop(columns=["geometry"], errors="ignore"))


def coerce_types_light(df: pd.DataFrame) -> pd.DataFrame:
    # Just some light numeric coercion
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            try:
                out[c] = pd.to_numeric(out[c])
            except Exception:
                pass
    return out


# ==================== Elections helpers ====================
def is_tall_elections(df: pd.DataFrame) -> bool:
    # Heuristic: tall format has columns like: cntyvtd / county / vtd, office, candidate, party, votes
    cols = set(df.columns)
    has_key = bool({"cntyvtd", "vtd"}.intersection(cols)) or ({"county", "precinct"} <= cols)
    has_office = "office" in cols or "race" in cols
    has_votes = "votes" in cols or "vote" in cols
    return has_key and has_office and has_votes


def _normalize_party(party: str | float | int | None) -> str:
    if pd.isna(party):
        return "UNK"
    s = str(party).strip().upper()
    if s in ("DEM", "D", "DFL"):
        return "DEM"
    if s in ("REP", "R", "GOP"):
        return "REP"
    if s in ("LIB", "LBT"):
        return "LIB"
    if "GRN" in s or "GREEN" in s:
        return "GRN"
    return s


def normalize_cntyvtd_safely(s: pd.Series) -> pd.Series:
    """
    Try to parse Texas CNTYVTD codes robustly, returning a 3+5+suffix pattern where possible.
    """
    s = s.astype("string")
    # Strip junk
    cleaned = s.str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    # Often prefixed by '48' for Texas FIPS
    no48 = cleaned.str.replace(r"^48", "", regex=True)

    # Attempt to parse 3-digit county + 5-digit VTD + optional letters
    m = no48.str.extract(r"^(?P<cnty>\d{3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
    # digits-only fallback if that fails
    bad = m["cnty"].isna()
    if bad.any():
        m2 = s[bad].str.extract(r"^(?P<cnty>\d{1,3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
        m.loc[bad, ["cnty", "vtd", "suf"]] = m2[["cnty", "vtd", "suf"]]
    bad = m["cnty"].isna()
    if bad.any():
        # last-3 as county, rest as vtd
        tmp = s[bad].str.extract(r"^(\d+)([A-Z]*)$")
        digits = tmp[0].str.replace(r"[^\d]", "", regex=True)
        suf = tmp[1].fillna("")
        cnty = digits.str[-3:]
        vtd = digits.str[:-3]
        m.loc[bad, "cnty"] = cnty
        m.loc[bad, "vtd"] = vtd
        m.loc[bad, "suf"] = suf

    cnty = m["cnty"].astype("string").str.zfill(3)
    vtd = m["vtd"].astype("string")
    vtd = vtd.where(vtd.str.len() > 0, pd.NA)
    vtd = vtd.str.zfill(5)
    suf = m["suf"].astype("string")

    out = pd.Series(pd.NA, dtype="string", index=s.index)
    valid = cnty.notna() & vtd.notna()
    out.loc[valid] = (cnty.loc[valid] + vtd.loc[valid] + suf.loc[valid]).astype("string")
    return out


def _std_vtd_code(vtd_raw: pd.Series) -> pd.Series:
    s = vtd_raw.astype("string").str.strip().str.upper()
    digits = s.str.replace(r"[^0-9]", "", regex=True)
    letters = s.str.replace(r"[^A-Z]", "", regex=True)
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
    return s.astype("string")


def build_cntyvtd_from_parts(cnty_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = _std_cnty_code_from_cnty(cnty_series).astype("string")
    vtd5 = _std_vtd_code(vtd_series).astype("string")
    valid = cnty3.notna() & vtd5.notna()
    out = pd.Series(pd.NA, dtype="string", index=cnty3.index)
    out.loc[valid] = (cnty3.loc[valid] + vtd5.loc[valid]).astype("string")
    return out


def build_cntyvtd_from_fips_vtd(fips_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    # fips-series may contain 5-digit county FIPS with 48 prefix, so use last-3
    fips = fips_series.astype("string").str.replace(r"[^\d]", "", regex=True)
    cnty = fips.str[-3:].str.zfill(3)
    vtd = _std_vtd_code(vtd_series)
    out = pd.Series(pd.NA, dtype="string", index=fips_series.index)
    valid = cnty.notna() & vtd.notna()
    out.loc[valid] = (cnty.loc[valid] + vtd.loc[valid]).astype("string")
    return out


def clean_vtd_election_returns(df: pd.DataFrame, target_office: str = "") -> pd.DataFrame:
    """
    Clean tall-form VTD elections and return wide-by-VTD for a single office or ALL offices combined.
    """
    df = stdcols(df)
    office_col = col_like(df, "office", "race")
    cand_col = col_like(df, "candidate", "cand", "name")
    party_col = col_like(df, "party")
    votes_col = col_like(df, "votes", "vote")

    # VTD key: prefer explicit cntyvtd, else county + vtd
    if "cntyvtd" in df.columns:
        key = normalize_cntyvtd_safely(df["cntyvtd"])
        df["cntyvtd"] = key
    elif {"county", "precinct"}.issubset(df.columns):
        df["cntyvtd"] = build_cntyvtd_from_parts(df["county"], df["precinct"])
    else:
        raise ValueError("Elections file lacks cntyvtd or (county,precinct) to build a VTD key.")

    if office_col is None or votes_col is None:
        raise ValueError("Elections file lacks office or votes columns.")

    df["party_norm"] = df[party_col].map(_normalize_party)
    df["office_norm"] = df[office_col].astype("string").str.strip()

    if target_office:
        mask = df["office_norm"].str.contains(re.escape(target_office), case=False, na=False)
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError(f"No rows matched office '{target_office}' in elections file.")

    grp_cols = ["cntyvtd", "party_norm"]
    agg = df.groupby(grp_cols)[votes_col].sum(min_count=1).reset_index(name="votes")

    # pivot to wide
    wide = agg.pivot(index="cntyvtd", columns="party_norm", values="votes").reset_index()
    wide = wide.rename_axis(None, axis=1)

    # standard vote columns
    wide["dem_votes"] = wide.get("DEM")
    wide["rep_votes"] = wide.get("REP")

    # third-party = everything that is not DEM/REP/UNK
    party_cols = [c for c in wide.columns if c not in ("cntyvtd", "dem_votes", "rep_votes")]
    if party_cols:
        wide["third_party_votes"] = wide[party_cols].fillna(0).sum(axis=1)
    else:
        wide["third_party_votes"] = 0

    # total votes
    wide["total_votes"] = (
        wide[["dem_votes", "rep_votes", "third_party_votes"]]
        .fillna(0)
        .sum(axis=1)
    )

    # two-party dem share
    twoden = wide[["dem_votes", "rep_votes"]].fillna(0).sum(axis=1)
    wide["two_party_dem_share"] = (wide["dem_votes"] / twoden).where(twoden > 0)

    return wide


# ======================== PL94 helpers ========================
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

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ===================== Compactness metrics =====================
def compute_compactness(districts_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = districts_gdf.to_crs(AREA_CRS).copy()
    assert_projected_planar(g, "compute_compactness")
    A, P = g.geometry.area, g.geometry.length

    out = pd.DataFrame(index=g.index)
    out["area"] = A
    out["perimeter"] = P

    # Polsby–Popper
    out["polsby_popper"] = (4 * math.pi * A) / (P ** 2).where(P > 0)

    # Convex hull ratio
    hull = g.geometry.convex_hull
    hull_area = hull.area
    out["convex_hull_ratio"] = (A / hull_area).where(hull_area > 0)

    # Reock: bounding circle area ratio (if available)
    if HAS_MINCIRCLE:
        centers_radii = hull.apply(minimum_bounding_radius)
        radii = centers_radii.apply(lambda t: t[1] if isinstance(t, tuple) and len(t) == 2 else math.nan)
        circle_area = math.pi * (radii ** 2)
        out["reock"] = (A / circle_area).where(circle_area > 0)
    else:
        out["reock"] = pd.NA

    return out


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
                except Exception as exc:
                    raise RuntimeError(f"Error cleaning elections file {path} (office='{elections_office}')") from exc

                # Persist both
                parquet_path = out_parquet_dir / f"{name}.parquet"
                parquet_all_path = out_parquet_dir / f"{name}_all.parquet"
                mkdir_p(parquet_path.parent)
                tdf.to_parquet(parquet_path, index=False)
                tdf_all.to_parquet(parquet_all_path, index=False)

                print(f"[ETL] Wrote elections wide (office='{elections_office or 'ALL'}') → {parquet_path}")
                print(f"[ETL] Wrote elections wide (ALL offices) → {parquet_all_path}")
                df = tdf  # for SQLite, etc.

            # if not tall, fall through and treat as already-wide

        # Persist to Parquet
        if is_geodf(df):
            gdf = ensure_crs(df)
            out_geo = out_geo_dir / f"{name}.parquet"
            mkdir_p(out_geo.parent)
            gdf.to_parquet(out_geo, index=False)
            sql_df = geo_to_sql_ready(gdf)
        else:
            out_parquet = out_parquet_dir / f"{name}.parquet"
            mkdir_p(out_parquet.parent)
            df.to_parquet(out_parquet, index=False)
            sql_df = df

        # Also write to SQLite (tolerant to first-time tables)
        from sqlalchemy import create_engine, text
        eng = create_engine(f"sqlite:///{sqlite_path}")
        with eng.begin() as conn:
            # Drop if exists without reflecting metadata (avoids InvalidRequestError)
            conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{name}"')
            sql_df.to_sql(name, conn, if_exists="append", index=False)


    print("========== ETL complete ==========\n")


def to_sqlite(df: pd.DataFrame, sqlite_path: Path, table: str) -> None:
    from sqlalchemy import create_engine
    eng = create_engine(f"sqlite:///{sqlite_path}")
    with eng.begin() as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    mkdir_p(out_path.parent)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def write_geo_parquet(gdf: "gpd.GeoDataFrame", out_path: Path) -> None:
    mkdir_p(out_path.parent)
    gdf.to_parquet(out_path, index=False)


# =================== Stage 2: Build VTD-level dataset ===================
def build_final(
    out_parquet_dir: Path,
    out_geo_dir: Path,
    districts_key: str,
    census_key: str,
    vtds_key: str,
    pl94_key: str,
    elections_key: str,
) -> "gpd.GeoDataFrame":
    """
    Stage 2: Build a VTD-level dataset.

    One row per voting district (cntyvtd), with:
      - PL94 racial VAP (area-weighted from blocks)
      - elections (dem/rep/third/total + shares)
      - compactness metrics
      - parent congressional district_id (area-dominant)
    """
    print("========== Stage 2: Build VTD-level dataset ==========")

    # Locate cleaned artifacts from Stage 1
    districts_fp = out_geo_dir / f"{districts_key}.parquet"
    census_fp    = out_geo_dir / f"{census_key}.parquet"
    vtds_fp      = out_geo_dir / f"{vtds_key}.parquet"
    pl94_fp      = out_parquet_dir / f"{pl94_key}.parquet"
    elect_fp     = out_parquet_dir / f"{elections_key}.parquet"

    elect_all_fp = out_parquet_dir / f"{elections_key}_all.parquet"

    # Basic existence checks
    missing = [p for p in [districts_fp, census_fp, vtds_fp, pl94_fp, elect_fp] if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 2 missing cleaned files:\n" + "\n".join(f" - {p}" for p in missing))

    if not elect_all_fp.exists():
        print(f"[WARN] Missing ALL-offices elections file: {elect_all_fp}")
        vtd_elec_all = None
    else:
        vtd_elec_all = stdcols(pd.read_parquet(elect_all_fp))

    print("[INFO] Using cleaned files:")
    print("  ", districts_fp)
    print("  ", census_fp)
    print("  ", vtds_fp)
    print("  ", pl94_fp)
    print("  ", elect_fp)

    # ---------------- Load core datasets ----------------
    districts = ensure_crs(stdcols(gpd.read_parquet(districts_fp)))
    census_geo = ensure_crs(stdcols(gpd.read_parquet(census_fp)))
    vtd_geo = ensure_crs(stdcols(gpd.read_parquet(vtds_fp)))
    pl94 = stdcols(pd.read_parquet(pl94_fp))
    pl94 = unify_pl94_schema(pl94)
    vtd_elec = stdcols(pd.read_parquet(elect_fp))

    # ---------------- Canonical VTD keys ----------------
    vtd_geo = vtd_geo.copy()
    vtd_elec = vtd_elec.copy()

    # Geo side: CNTY + VTD if available, else cntyvtd
    if {"cnty", "vtd"} <= set(vtd_geo.columns):
        vtd_geo["cntyvtd_std"] = build_cntyvtd_from_parts(vtd_geo["cnty"], vtd_geo["vtd"])
    elif "cntyvtd" in vtd_geo.columns:
        vtd_geo["cntyvtd_std"] = (
            vtd_geo["cntyvtd"].astype(str)
            .str.strip().str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
    else:
        raise ValueError("VTD geometry lacks CNTY/VTD or CNTYVTD to build a key.")

    # Elections side: parse with TX-aware normalizer; fallback to cleaned string
    if "cntyvtd" in vtd_elec.columns:
        parsed = normalize_cntyvtd_safely(vtd_elec["cntyvtd"])
        fallback = (
            vtd_elec["cntyvtd"].astype(str)
            .str.strip().str.upper()
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

    # ---------------- PL94: blocks -> VTDs via area weighting ----------------
    census_geo = ensure_geoid20_str(census_geo)
    pl94 = ensure_geoid20_str(pl94)

    PL94_TOTAL_VAP = "vap_total"
    PL94_RACE_VAP = {
        "pct_white": "nh_white_vap",
        "pct_black": "nh_black_vap",
        "pct_asian": "nh_asian_vap",
        "pct_hispanic": "hispanic_vap",
    }
    need = ["geoid20", PL94_TOTAL_VAP, *PL94_RACE_VAP.values()]
    missing_pl = [c for c in need if c not in pl94.columns]
    if missing_pl:
        raise ValueError(f"PL-94 expected columns missing: {missing_pl}")

    if "geoid20" not in census_geo.columns:
        raise ValueError("Census geometry must contain 'geoid20'")

    census_merged = census_geo.merge(pl94[need], on="geoid20", how="left")

    census_proj = census_merged.to_crs(AREA_CRS)
    vtd_proj = vtd_geo.to_crs(AREA_CRS)
    assert_projected_planar(census_proj, "blocks→VTDs")
    assert_projected_planar(vtd_proj, "blocks→VTDs")

    # give VTDs a simple integer index for grouping
    vtd_proj = vtd_proj.reset_index(drop=False).rename(columns={"index": "vtd_idx"})
    vtd_proj["vtd_idx"] = vtd_proj.index

    blk = census_proj[["geoid20", "geometry"]].copy()
    blk_attrs = census_proj[["geoid20", PL94_TOTAL_VAP, *PL94_RACE_VAP.values()]].copy()

    vtd_sub = vtd_proj[["vtd_idx", "geometry"]].copy()

    print("[INFO] Overlay blocks -> VTDs (intersection)...")
    blk_inter = gpd.overlay(blk, vtd_sub, how="intersection", keep_geom_type=True)
    blk_inter = blk_inter.merge(blk_attrs, on="geoid20", how="left")

    # area weights at block level
    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    blk_inter["blk_area"] = blk_area.loc[blk_inter["geoid20"]].values
    blk_inter["inter_area"] = blk_inter.geometry.area
    blk_inter = blk_inter.loc[blk_inter["blk_area"] > 0].copy()
    blk_inter["w"] = (blk_inter["inter_area"] / blk_inter["blk_area"]).clip(0, 1)

    to_sum = [PL94_TOTAL_VAP] + list(PL94_RACE_VAP.values())
    for col in to_sum:
        blk_inter[col] = blk_inter[col].fillna(0) * blk_inter["w"]

    agg_vtd = (
        blk_inter.groupby("vtd_idx", observed=True)[to_sum]
        .sum(min_count=1)
        .reindex(vtd_proj["vtd_idx"], fill_value=0)
    )

    den = agg_vtd[PL94_TOTAL_VAP].replace({0: pd.NA})
    race_pct = pd.DataFrame(index=agg_vtd.index)
    for out_name, src_col in PL94_RACE_VAP.items():
        race_pct[out_name] = (agg_vtd[src_col] / den).where(den > 0)

    # ---------------- Elections per VTD ----------------
    # Aggregate to one record per cntyvtd_std (sum duplicates)
    vote_cols = [c for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"] if c in vtd_elec.columns]
    if vote_cols:
        elec_agg = (
            vtd_elec.groupby("cntyvtd_std", as_index=False)[vote_cols]
            .sum(min_count=1)
        )
    else:
        elec_agg = vtd_elec[["cntyvtd_std"]].drop_duplicates()
        elec_agg["dem_votes"] = pd.NA
        elec_agg["rep_votes"] = pd.NA
        elec_agg["third_party_votes"] = pd.NA
        elec_agg["total_votes"] = pd.NA

    vtd_with_votes = vtd_proj.merge(
        elec_agg,
        how="left",
        left_on="cntyvtd_std",
        right_on="cntyvtd_std",
    )

    # vote shares
    tot = vtd_with_votes["total_votes"]
    valid_tot = tot > 0
    vtd_with_votes["dem_share"] = (vtd_with_votes["dem_votes"] / tot).where(valid_tot)
    vtd_with_votes["rep_share"] = (vtd_with_votes["rep_votes"] / tot).where(valid_tot)

    two_den = (vtd_with_votes["dem_votes"] + vtd_with_votes["rep_votes"])
    vtd_with_votes["two_party_dem_share"] = (
        vtd_with_votes["dem_votes"] / two_den
    ).where(two_den > 0)

    # ---------------- Compactness at VTD level ----------------
    print("[INFO] Computing VTD compactness ...")
    cmpx_vtd = compute_compactness(vtd_proj)

    # ---------------- Parent congressional district (area-dominant) ----------------
    districts_proj = districts.to_crs(AREA_CRS)
    assert_projected_planar(districts_proj, "VTD→districts")

    d_sub = districts_proj[["geometry"]].reset_index(names="district_idx")
    vtd_for_overlay = vtd_proj[["vtd_idx", "geometry"]].copy()

    print("[INFO] Overlay VTDs -> districts (intersection)...")
    vtd_dist_inter = gpd.overlay(vtd_for_overlay, d_sub, how="intersection", keep_geom_type=True)

    if not vtd_dist_inter.empty:
        vtd_dist_inter["inter_area"] = vtd_dist_inter.geometry.area
        best = (
            vtd_dist_inter
            .sort_values("inter_area", ascending=False)
            .drop_duplicates("vtd_idx")[["vtd_idx", "district_idx"]]
        )

        id_col = next((c for c in ["district", "district_id", "cd", "dist"] if c in districts.columns), None)
        if id_col is not None:
            best = best.merge(
                districts_proj.reset_index(names="district_idx")[["district_idx", id_col]],
                on="district_idx",
                how="left",
            )
            best = best.rename(columns={id_col: "district_id"})
        else:
            best["district_id"] = best["district_idx"] + 1

        vtd_with_votes = vtd_with_votes.merge(best[["vtd_idx", "district_id"]], on="vtd_idx", how="left")
    else:
        print("[WARN] No VTD↔district intersections found; district_id will be NA.")
        vtd_with_votes["district_id"] = pd.NA

    # ---------------- Assemble final VTD-level GeoDataFrame ----------------
    final_vtd = vtd_with_votes.copy()
    final_vtd = final_vtd.join(cmpx_vtd, on="vtd_idx").join(race_pct, on="vtd_idx")

    # Avoid duplicate key columns if the original shapefile already had 'cntyvtd'
    if "cntyvtd" in final_vtd.columns and "cntyvtd_std" in final_vtd.columns:
        final_vtd = final_vtd.drop(columns=["cntyvtd"])

    final_vtd = final_vtd.rename(columns={"cntyvtd_std": "cntyvtd"})

    # Safety: ensure there are no duplicate column names left
    dup_mask = pd.Index(final_vtd.columns).duplicated(keep="first")
    if dup_mask.any():
        dups = list(pd.Index(final_vtd.columns)[dup_mask])
        raise ValueError(f"Unexpected duplicate columns even after cleanup: {dups}")


    # Optional: fill NA vote shares with 0.0
    for col in ["dem_share", "rep_share"]:
        if col in final_vtd.columns:
            final_vtd[col] = final_vtd[col].fillna(0.0)

    # Column ordering: key + high-level metrics first
    front = [
        c for c in [
            "cntyvtd",
            "district_id",
            "dem_votes",
            "rep_votes",
            "third_party_votes",
            "total_votes",
            "dem_share",
            "rep_share",
            "two_party_dem_share",
            "pct_white",
            "pct_black",
            "pct_asian",
            "pct_hispanic",
        ]
        if c in final_vtd.columns
    ]
    other = [c for c in final_vtd.columns if c not in front]
    final_vtd = final_vtd[front + other]

    print(f"[INFO] Final VTD rows: {len(final_vtd)}")
    return gpd.GeoDataFrame(final_vtd, geometry="geometry", crs=vtd_proj.crs)


# =============================== CLI ===============================
def main():
    ap = argparse.ArgumentParser(description="ETL + VTD-level dataset (Texas; one row per voting district)")
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
    pq = args.data_processed_tabular / "vtds_final.parquet"
    csv = args.data_processed_tabular / "vtds_final.csv"
    write_parquet(final, pq)
    final.to_csv(csv, index=False)
    print("\nSaved:\n ", pq, "\n ", csv)


if __name__ == "__main__":
    main()
