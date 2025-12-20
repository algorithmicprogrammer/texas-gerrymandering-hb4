#!/usr/bin/env python3
# pipelines/data/etl_pipeline.py

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit("GeoPandas required. Install: pip install geopandas shapely pyproj fiona pyarrow sqlalchemy") from e


# ============================== CONFIG ==============================
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".txt")
SUPPORTED_GEO = (".gpkg", ".shp", ".parquet")
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO

AREA_CRS = "EPSG:3083"  # Texas-centric equal-area projection for overlays + compactness

# Output final columns (exactly as requested)
FINAL_COLUMNS = [
    "cntyvtd",
    "dem_share",
    "rep_share",
    "pct_white",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "polsby_popper",
    "convex_hull_ratio",
    "schwartzberg",
    "reock",
]
KEEP_GEOMETRY = False  # set True if you want geometry included in the final output


# ============================== UTIL ==============================
def mkdir_p(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_key(path: Path) -> str:
    return re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", path.stem.lower())).strip("_")


def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_{2,}", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        try:
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)
    if ext == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = "".join(f.readlines()[:10])
        sep = "\t" if sample.count("\t") > sample.count(",") else ","
        return pd.read_csv(path, sep=sep)
    if ext in (".shp", ".gpkg"):
        return gpd.read_file(path)
    raise ValueError(f"Unsupported input type: {path}")


def is_geodf(df: pd.DataFrame) -> bool:
    return isinstance(df, gpd.GeoDataFrame) or "geometry" in df.columns


def ensure_crs(gdf: "gpd.GeoDataFrame", target: str | None = None) -> "gpd.GeoDataFrame":
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise TypeError("Expected a GeoDataFrame or a DataFrame with 'geometry'")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    if gdf.crs is None and target is None:
        raise ValueError("GeoDataFrame has no CRS; set it or pass a target CRS.")
    if target is not None:
        return gdf.to_crs(target)
    return gdf


def assert_projected_planar(gdf: "gpd.GeoDataFrame", where: str) -> None:
    if gdf.crs is None:
        raise ValueError(f"{where}: GeoDataFrame has no CRS; project to {AREA_CRS}.")
    if hasattr(gdf.crs, "is_geographic") and gdf.crs.is_geographic:
        raise ValueError(f"{where}: CRS is geographic; project to {AREA_CRS}.")


def ensure_geoid20_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "geoid20" in df.columns:
        df["geoid20"] = (
            df["geoid20"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
    return df


# ============================== ELECTIONS HELPERS ==============================
def is_tall_elections(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    has_key = "cntyvtd" in cols or ({"county", "precinct"} <= cols) or ({"fips", "vtd"} <= cols)
    has_office = "office" in cols or "race" in cols
    has_votes = "votes" in cols or "vote" in cols
    return has_key and has_office and has_votes


def _normalize_party(party: str | float | int | None) -> str:
    """
    Very robust party normalizer. This is what prevents DEM/REP being missing everywhere.
    """
    if pd.isna(party):
        return "UNK"
    s = str(party).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)  # remove punctuation/spaces

    if s in ("DEM", "D", "DFL") or "DEMOCRAT" in s:
        return "DEM"
    if s in ("REP", "R", "GOP") or "REPUBLICAN" in s:
        return "REP"
    if s.startswith("LIB") or "LIBERTARIAN" in s:
        return "LIB"
    if s.startswith("GRN") or "GREEN" in s:
        return "GRN"
    return s if s else "UNK"


def normalize_cntyvtd_safely(s: pd.Series) -> pd.Series:
    """
    Normalize CNTYVTD: attempts TX-like patterns; returns string like CCCVVVVV[SUF]
    """
    s = s.astype("string")
    cleaned = s.str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    no48 = cleaned.str.replace(r"^48", "", regex=True)

    m = no48.str.extract(r"^(?P<cnty>\d{3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
    bad = m["cnty"].isna()
    if bad.any():
        m2 = s[bad].str.extract(r"^(?P<cnty>\d{1,3})(?P<vtd>\d{1,5})(?P<suf>[A-Z]*)$")
        m.loc[bad, ["cnty", "vtd", "suf"]] = m2[["cnty", "vtd", "suf"]]

    cnty = m["cnty"].astype("string").str.zfill(3)
    vtd = m["vtd"].astype("string")
    vtd = vtd.where(vtd.str.len() > 0, pd.NA).str.zfill(5)
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
    out.loc[valid] = (digits.loc[valid].str.zfill(5) + letters.loc[valid]).astype("string")
    return out


def build_cntyvtd_from_parts(cnty_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    cnty3 = cnty_series.astype("string").str.replace(r"[^\d]", "", regex=True)
    cnty3 = cnty3.where(cnty3.str.len() > 0, pd.NA).str[-3:].str.zfill(3)
    vtd5 = _std_vtd_code(vtd_series).astype("string")
    out = pd.Series(pd.NA, dtype="string", index=cnty3.index)
    valid = cnty3.notna() & vtd5.notna()
    out.loc[valid] = (cnty3.loc[valid] + vtd5.loc[valid]).astype("string")
    return out


def build_cntyvtd_from_fips_vtd(fips_series: pd.Series, vtd_series: pd.Series) -> pd.Series:
    fips = fips_series.astype("string").str.replace(r"[^\d]", "", regex=True)
    cnty = fips.str[-3:].str.zfill(3)
    vtd = _std_vtd_code(vtd_series)
    out = pd.Series(pd.NA, dtype="string", index=fips_series.index)
    valid = cnty.notna() & vtd.notna()
    out.loc[valid] = (cnty.loc[valid] + vtd.loc[valid]).astype("string")
    return out


def clean_vtd_election_returns(df: pd.DataFrame, target_office: str = "") -> pd.DataFrame:
    """
    Tall elections -> wide by VTD with columns:
      cntyvtd, dem_votes, rep_votes, third_party_votes, total_votes, dem_share, rep_share, two_party_dem_share
    """
    df = stdcols(df)

    office_col = "office" if "office" in df.columns else ("race" if "race" in df.columns else None)
    votes_col = "votes" if "votes" in df.columns else ("vote" if "vote" in df.columns else None)
    party_col = "party" if "party" in df.columns else None

    if office_col is None or votes_col is None:
        raise ValueError("Elections file must contain office/race and votes/vote columns.")

    # Build cntyvtd
    if "cntyvtd" in df.columns:
        df["cntyvtd"] = normalize_cntyvtd_safely(df["cntyvtd"])
    elif {"county", "precinct"} <= set(df.columns):
        df["cntyvtd"] = build_cntyvtd_from_parts(df["county"], df["precinct"])
    elif {"fips", "vtd"} <= set(df.columns):
        df["cntyvtd"] = build_cntyvtd_from_fips_vtd(df["fips"], df["vtd"])
    else:
        raise ValueError("Elections file lacks cntyvtd or sufficient fields to construct it.")

    df = df.loc[df["cntyvtd"].notna()].copy()

    df["office_norm"] = df[office_col].astype("string").str.strip()
    if target_office:
        mask = df["office_norm"].str.contains(re.escape(target_office), case=False, na=False)
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError(f"No rows matched elections office filter: {target_office}")

    # Party normalization
    if party_col is None:
        # If there's no party column, we can't create DEM/REP counts reliably.
        # Fail loudly so you don't silently get all-missing votes.
        raise ValueError("Elections file has no 'party' column; cannot compute dem/rep votes.")
    df["party_norm"] = df[party_col].map(_normalize_party)

    # Votes numeric
    df[votes_col] = pd.to_numeric(df[votes_col], errors="coerce").fillna(0)

    # Aggregate votes by (cntyvtd, party_norm)
    agg = df.groupby(["cntyvtd", "party_norm"], as_index=False)[votes_col].sum()
    agg = agg.rename(columns={votes_col: "votes"})

    # Pivot to wide
    wide = agg.pivot(index="cntyvtd", columns="party_norm", values="votes").reset_index()
    wide = wide.rename_axis(None, axis=1)

    # Always present and numeric
    wide["dem_votes"] = pd.to_numeric(wide.get("DEM", 0), errors="coerce").fillna(0).astype("int64")
    wide["rep_votes"] = pd.to_numeric(wide.get("REP", 0), errors="coerce").fillna(0).astype("int64")

    # Third party = sum of all party columns except DEM/REP
    party_cols = [c for c in wide.columns if c not in ("cntyvtd", "dem_votes", "rep_votes")]
    party_cols = [c for c in party_cols if c not in ("DEM", "REP")]  # avoid double-counting if still present
    if party_cols:
        wide["third_party_votes"] = (
            wide[party_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype("int64")
        )
    else:
        wide["third_party_votes"] = 0

    wide["total_votes"] = (wide["dem_votes"] + wide["rep_votes"] + wide["third_party_votes"]).astype("int64")

    # Shares (safe)
    tot = wide["total_votes"]
    wide["dem_share"] = (wide["dem_votes"] / tot).where(tot > 0, 0.0)
    wide["rep_share"] = (wide["rep_votes"] / tot).where(tot > 0, 0.0)

    two_den = wide["dem_votes"] + wide["rep_votes"]
    wide["two_party_dem_share"] = (wide["dem_votes"] / two_den).where(two_den > 0, 0.0)

    return wide


# ============================== PL94 HELPERS ==============================
def unify_pl94_schema(pl94: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal schema unifier. We require geoid20 and some race/pop columns.
    Your file is Blocks_Pop.txt, so we treat it as "population-like" rather than VAP.
    """
    df = stdcols(pl94)

    geoid_candidates = [
        "geoid20", "geoid", "tabblock20", "tabblock2020",
        "block_geoid", "blk_geoid", "sctbkey", "ctbkey",
    ]
    geoid_col = next((c for c in geoid_candidates if c in df.columns), None)
    if geoid_col is None:
        raise ValueError("PL file lacks a GEOID column (need geoid20/geoid/tabblock20/etc).")
    if geoid_col != "geoid20":
        df = df.rename(columns={geoid_col: "geoid20"})

    df["geoid20"] = (
        df["geoid20"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^\d]", "", regex=True)
        .str.zfill(15)
    )

    return df

def pick_pop_columns(pl: pd.DataFrame) -> tuple[str, dict[str, str]]:
    """
    Choose denominator + race-count columns from Blocks_Pop-style files.

    Preference order:
      1) VAP-based shares (vap, anglovap, blackvap, asianvap, hispvap) if present
      2) Total-pop shares (total, anglo, black, asian, hisp) if present

    Returns:
      total_col, race_map where race_map keys are output pct_* column names
    """
    cols = set(pl.columns)

    # --- Prefer VAP if available ---
    if {"vap", "anglovap", "blackvap", "asianvap", "hispvap"} <= cols:
        total_col = "vap"
        race_map = {
            "pct_white": "anglovap",   # non-Hispanic white VAP in your file
            "pct_black": "blackvap",
            "pct_asian": "asianvap",
            "pct_hispanic": "hispvap",
        }
        return total_col, race_map

    # --- Otherwise use total population ---
    if {"total", "anglo", "black", "asian", "hisp"} <= cols:
        total_col = "total"
        race_map = {
            "pct_white": "anglo",      # non-Hispanic white pop in your file
            "pct_black": "black",
            "pct_asian": "asian",
            "pct_hispanic": "hisp",
        }
        return total_col, race_map

    # --- Last-resort fuzzy fallbacks (optional) ---
    # Allow pop20 as denominator if total is missing
    if "pop20" in cols and {"anglo", "black", "asian", "hisp"} <= cols:
        total_col = "pop20"
        race_map = {
            "pct_white": "anglo",
            "pct_black": "black",
            "pct_asian": "asian",
            "pct_hispanic": "hisp",
        }
        return total_col, race_map

    raise ValueError(
        "Could not detect required population/VAP columns in Blocks_Pop file.\n"
        "Expected either VAP set: vap + (anglovap, blackvap, asianvap, hispvap)\n"
        "or total-pop set: total + (anglo, black, asian, hisp)\n"
        f"Available columns (sample): {list(pl.columns)[:80]}"
    )




# ============================== REOCK (ROBUST) ==============================
def _mec_circle_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Minimal enclosing circle (randomized incremental).
    Returns (center_xy, radius). points: (n,2)
    """
    pts = points.astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.array([np.nan, np.nan]), np.nan
    if len(pts) == 1:
        return pts[0], 0.0

    pts = pts.copy()
    np.random.shuffle(pts)

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def circle_two(a, b):
        c = (a + b) / 2.0
        r = dist(a, c)
        return c, r

    def circle_three(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return np.array([np.nan, np.nan]), np.nan
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        center = np.array([ux, uy])
        r = dist(center, a)
        return center, r

    def contains(center, r, p):
        return dist(center, p) <= r + 1e-9

    center, r = circle_two(pts[0], pts[1])
    for i in range(len(pts)):
        p = pts[i]
        if contains(center, r, p):
            continue
        center, r = p, 0.0
        for j in range(i):
            q = pts[j]
            if contains(center, r, q):
                continue
            center, r = circle_two(p, q)
            for k in range(j):
                t = pts[k]
                if contains(center, r, t):
                    continue
                center, r = circle_three(p, q, t)
                if not np.isfinite(r):
                    c1, r1 = circle_two(p, q)
                    c2, r2 = circle_two(p, t)
                    c3, r3 = circle_two(q, t)
                    center, r = min([(c1, r1), (c2, r2), (c3, r3)], key=lambda x: x[1])
    return center, float(r)


def _reock_for_geom(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    try:
        geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    except Exception:
        geoms = [geom]

    coords = []
    for g in geoms:
        if g.is_empty:
            continue
        if getattr(g, "exterior", None) is not None and g.exterior is not None:
            coords.append(np.asarray(g.exterior.coords))
    if not coords:
        return np.nan

    pts = np.vstack(coords)[:, :2]
    _, r = _mec_circle_from_points(pts)
    if not np.isfinite(r) or r <= 0:
        return np.nan
    circle_area = math.pi * (r ** 2)
    a = geom.area
    return float(a / circle_area) if circle_area > 0 else np.nan


# ============================== COMPACTNESS ==============================
def compute_compactness(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = gdf.to_crs(AREA_CRS).copy()
    assert_projected_planar(g, "compute_compactness")

    # Fix invalid polys that can break measurements
    g["geometry"] = g.geometry.buffer(0)

    A = g.geometry.area
    P = g.geometry.length

    out = pd.DataFrame(index=g.index)

    out["polsby_popper"] = ((4 * math.pi * A) / (P ** 2)).where(P > 0)

    hull = g.geometry.convex_hull
    hull_area = hull.area
    out["convex_hull_ratio"] = (A / hull_area).where(hull_area > 0)

    # Schwartzberg: perimeter / circumference of equal-area circle
    out["schwartzberg"] = (P / (2 * np.sqrt(math.pi * A))).where(A > 0)

    # Reock: area / area of minimum enclosing circle
    out["reock"] = g.geometry.apply(_reock_for_geom)

    return out


# ============================== STAGE 1: ETL ==============================
def geo_to_sql_ready(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    out = gdf.copy()
    if "geometry" in out.columns:
        out["geometry_wkb"] = out.geometry.to_wkb()
    return pd.DataFrame(out.drop(columns=["geometry"], errors="ignore"))


def run_etl(
    input_files: list[Path],
    out_parquet_dir: Path,
    out_geo_dir: Path,
    sqlite_path: Path,
    elections_office: str = "",
):
    print("\n========== Stage 1: ETL ==========")
    mkdir_p(out_parquet_dir)
    mkdir_p(out_geo_dir)
    mkdir_p(sqlite_path.parent)

    elections_path = input_files[-1]

    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if path.suffix.lower() not in ALL_EXTS:
            raise ValueError(f"Unsupported input type: {path}")

        name = dataset_key(path)
        print(f"[ETL] Reading {path} -> key={name}")
        df = read_any(path).drop_duplicates(ignore_index=True)
        df = stdcols(df)

        # Elections: tall -> wide
        if path == elections_path and not is_geodf(df):
            if is_tall_elections(df):
                print(f"[ETL] Detected tall elections file. Cleaning office='{elections_office}' → wide VTD votes.")
                tdf = clean_vtd_election_returns(df, target_office=elections_office)
                tdf_all = clean_vtd_election_returns(df, target_office="")

                parquet_path = out_parquet_dir / f"{name}.parquet"
                parquet_all_path = out_parquet_dir / f"{name}_all.parquet"
                mkdir_p(parquet_path.parent)
                tdf.to_parquet(parquet_path, index=False)
                tdf_all.to_parquet(parquet_all_path, index=False)

                print(f"[ETL] Wrote elections wide (office='{elections_office or 'ALL'}') → {parquet_path}")
                print(f"[ETL] Wrote elections wide (ALL offices) → {parquet_all_path}")

                df = tdf  # for sqlite

        # Persist cleaned parquet/geoparquet
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

        # SQLite write (robust; avoids reflection error)
        from sqlalchemy import create_engine
        eng = create_engine(f"sqlite:///{sqlite_path}")
        with eng.begin() as conn:
            conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{name}"')
            sql_df.to_sql(name, conn, if_exists="append", index=False)

    print("========== ETL complete ==========\n")


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    mkdir_p(out_path.parent)
    df.to_parquet(out_path, engine="pyarrow", index=False)


# ============================== STAGE 2: BUILD VTD FINAL ==============================
def build_final(
    out_parquet_dir: Path,
    out_geo_dir: Path,
    districts_key: str,
    census_key: str,
    vtds_key: str,
    pl_key: str,
    elections_key: str,
) -> pd.DataFrame:
    print("========== Stage 2: Build VTD-level dataset ==========")

    districts_fp = out_geo_dir / f"{districts_key}.parquet"
    census_fp = out_geo_dir / f"{census_key}.parquet"
    vtds_fp = out_geo_dir / f"{vtds_key}.parquet"
    pl_fp = out_parquet_dir / f"{pl_key}.parquet"
    elect_fp = out_parquet_dir / f"{elections_key}.parquet"

    missing = [p for p in [districts_fp, census_fp, vtds_fp, pl_fp, elect_fp] if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 2 missing cleaned files:\n" + "\n".join(f" - {p}" for p in missing))

    print("[INFO] Using cleaned files:")
    print("  ", districts_fp)
    print("  ", census_fp)
    print("  ", vtds_fp)
    print("  ", pl_fp)
    print("  ", elect_fp)

    districts = ensure_crs(stdcols(gpd.read_parquet(districts_fp)))
    blocks = ensure_crs(stdcols(gpd.read_parquet(census_fp)))
    vtd = ensure_crs(stdcols(gpd.read_parquet(vtds_fp)))

    pl = unify_pl94_schema(pd.read_parquet(pl_fp))
    elec = stdcols(pd.read_parquet(elect_fp))

    # ---- Build robust VTD key from geometry ----
    # Prefer cntykey+vtdkey (most reliable with your VTDs_24PG file)
    vtd = vtd.copy()
    if {"cntykey", "vtdkey"} <= set(vtd.columns):
        vtd["cntyvtd_std"] = (
            vtd["cntykey"].astype("string").str.zfill(3)
            + vtd["vtdkey"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.zfill(5)
        )
    elif {"cnty", "vtd"} <= set(vtd.columns):
        vtd["cntyvtd_std"] = build_cntyvtd_from_parts(vtd["cnty"], vtd["vtd"])
    elif "cntyvtd" in vtd.columns:
        vtd["cntyvtd_std"] = normalize_cntyvtd_safely(vtd["cntyvtd"])
    else:
        raise ValueError("Cannot construct CNTYVTD key for VTD geometries.")

    # Elections key
    elec = elec.copy()
    if "cntyvtd" not in elec.columns:
        raise ValueError("Wide elections parquet missing cntyvtd column (should be produced by Stage 1).")
    elec["cntyvtd_std"] = normalize_cntyvtd_safely(elec["cntyvtd"]).fillna(
        elec["cntyvtd"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    vtd = vtd.loc[vtd["cntyvtd_std"].notna()].copy()
    elec = elec.loc[elec["cntyvtd_std"].notna()].copy()

    print(f"[INFO] VTD keys — geo: {vtd['cntyvtd_std'].nunique()} unique, elections: {elec['cntyvtd_std'].nunique()} unique")

    # ---- Blocks + PL -> VTD via area weighting ----
    blocks = ensure_geoid20_str(blocks)
    pl = ensure_geoid20_str(pl)

    if "geoid20" not in blocks.columns:
        raise ValueError("Blocks geometry must have geoid20 column after standardization.")
    if "geoid20" not in pl.columns:
        raise ValueError("Blocks_Pop file must have geoid20 after unification.")

    blocks_pl = blocks.merge(pl, on="geoid20", how="left")

    total_col, race_map = pick_pop_columns(blocks_pl)

    # Coerce to numeric
    blocks_pl[total_col] = pd.to_numeric(blocks_pl[total_col], errors="coerce").fillna(0)
    for rc in race_map.values():
        blocks_pl[rc] = pd.to_numeric(blocks_pl[rc], errors="coerce").fillna(0)

    blocks_proj = blocks_pl.to_crs(AREA_CRS)
    vtd_proj = vtd.to_crs(AREA_CRS)
    assert_projected_planar(blocks_proj, "blocks->VTD")
    assert_projected_planar(vtd_proj, "blocks->VTD")

    # Fix invalid polys
    blocks_proj["geometry"] = blocks_proj.geometry.buffer(0)
    vtd_proj["geometry"] = vtd_proj.geometry.buffer(0)

    # Index VTDs for grouping
    vtd_proj = vtd_proj.reset_index(drop=True)
    vtd_proj["vtd_idx"] = vtd_proj.index

    blk = blocks_proj[["geoid20", "geometry"]].copy()
    blk_attrs = blocks_proj[["geoid20", total_col, *race_map.values()]].copy()
    vtd_sub = vtd_proj[["vtd_idx", "geometry"]].copy()

    print("[INFO] Overlay blocks -> VTDs (intersection)...")
    inter = gpd.overlay(blk, vtd_sub, how="intersection", keep_geom_type=True)
    inter = inter.merge(blk_attrs, on="geoid20", how="left")

    # Area weighting
    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    inter["blk_area"] = blk_area.loc[inter["geoid20"]].values
    inter["inter_area"] = inter.geometry.area
    inter = inter.loc[inter["blk_area"] > 0].copy()
    inter["w"] = (inter["inter_area"] / inter["blk_area"]).clip(0, 1)

    # Weighted sums
    sum_cols = [total_col] + list(race_map.values())
    for c in sum_cols:
        inter[c] = inter[c].fillna(0) * inter["w"]

    agg = inter.groupby("vtd_idx", observed=True)[sum_cols].sum().reindex(vtd_proj["vtd_idx"], fill_value=0)

    # Race shares
    den = agg[total_col].replace({0: np.nan})
    race_pct = pd.DataFrame(index=agg.index)
    for out_name, src_col in race_map.items():
        race_pct[out_name] = (agg[src_col] / den).where(den > 0)

    # ---- Elections join (ensure not-missing shares) ----
    vote_cols = ["dem_votes", "rep_votes", "third_party_votes", "total_votes", "dem_share", "rep_share", "two_party_dem_share"]
    for c in vote_cols:
        if c not in elec.columns:
            raise ValueError(f"Elections wide table missing expected column: {c}")

    elec_agg = elec.groupby("cntyvtd_std", as_index=False)[["dem_votes", "rep_votes", "third_party_votes", "total_votes"]].sum()

    vtd_with_votes = vtd_proj.merge(elec_agg, how="left", left_on="cntyvtd_std", right_on="cntyvtd_std")

    # Define no-match as zero votes, and safe shares
    for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]:
        vtd_with_votes[c] = pd.to_numeric(vtd_with_votes[c], errors="coerce").fillna(0)

    tot = vtd_with_votes["total_votes"]
    vtd_with_votes["dem_share"] = (vtd_with_votes["dem_votes"] / tot).where(tot > 0, 0.0)
    vtd_with_votes["rep_share"] = (vtd_with_votes["rep_votes"] / tot).where(tot > 0, 0.0)

    # ---- Compactness (VTD polygons) ----
    print("[INFO] Computing VTD compactness ...")
    cmpx = compute_compactness(vtd_proj)

    # ---- Assemble final ----
    final = vtd_with_votes.join(cmpx, on="vtd_idx").join(race_pct, on="vtd_idx")

    # Ensure single cntyvtd (avoid duplicates if original file already had one)
    if "cntyvtd" in final.columns and "cntyvtd_std" in final.columns:
        final = final.drop(columns=["cntyvtd"], errors="ignore")
    final = final.rename(columns={"cntyvtd_std": "cntyvtd"})

    # Select only requested columns
    keep = FINAL_COLUMNS + (["geometry"] if KEEP_GEOMETRY else [])
    final = final[[c for c in keep if c in final.columns]].copy()

    # If any required columns are missing, fail loudly
    missing_cols = [c for c in FINAL_COLUMNS if c not in final.columns]
    if missing_cols:
        raise ValueError(f"Final dataset missing required columns: {missing_cols}")

    print(f"[INFO] Final VTD rows: {len(final)}")
    return final


# ============================== CLI ==============================
def main():
    ap = argparse.ArgumentParser(description="ETL + VTD-level dataset (one row per VTD)")
    ap.add_argument("--districts", type=Path, required=True, help="District polygons (SHP/GPKG/GeoParquet)")
    ap.add_argument("--census", type=Path, required=True, help="2020 Census blocks polygons (SHP/GPKG/GeoParquet)")
    ap.add_argument("--vtds", type=Path, required=True, help="VTD polygons (SHP/GPKG/GeoParquet)")
    ap.add_argument("--pl94", type=Path, required=True, help="Blocks_Pop-style attributes with GEOID20 (CSV/TSV/Parquet/TXT)")
    ap.add_argument("--elections", type=Path, required=True, help="Tall elections file (CSV/TSV/Parquet)")
    ap.add_argument("--elections-office", type=str, default="U.S. Sen", help="Office substring to filter (default: U.S. Sen)")
    ap.add_argument("--data-processed-tabular", type=Path, required=True)
    ap.add_argument("--data-processed-geospatial", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    args = ap.parse_args()

    input_files = [args.districts, args.census, args.vtds, args.pl94, args.elections]

    # Stage 1
    run_etl(
        input_files=input_files,
        out_parquet_dir=args.data_processed_tabular,
        out_geo_dir=args.data_processed_geospatial,
        sqlite_path=args.sqlite,
        elections_office=args.elections_office,
    )

    # Keys from filenames
    dk, ck, vk, pk, ek = [dataset_key(p) for p in input_files]

    # Stage 2
    final = build_final(
        out_parquet_dir=args.data_processed_tabular,
        out_geo_dir=args.data_processed_geospatial,
        districts_key=dk,
        census_key=ck,
        vtds_key=vk,
        pl_key=pk,
        elections_key=ek,
    )

    # Save
    out_pq = args.data_processed_tabular / "vtds_final.parquet"
    out_csv = args.data_processed_tabular / "vtds_final.csv"
    write_parquet(final, out_pq)
    final.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(" ", out_pq)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
