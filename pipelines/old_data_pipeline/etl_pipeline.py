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
    raise SystemExit(
        "GeoPandas required. Install: pip install geopandas shapely pyproj fiona pyarrow sqlalchemy"
    ) from e


# ============================== CONFIG ==============================
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".txt")
SUPPORTED_GEO = (".gpkg", ".shp", ".parquet")
ALL_EXTS = SUPPORTED_TABULAR + SUPPORTED_GEO

AREA_CRS = "EPSG:3083"  # Texas equal-area for overlays/area-based compactness

FINAL_COLUMNS = [
    "cntyvtd",
    "district_id",
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
KEEP_GEOMETRY = False  # set True if you also want geometry in vtds_final.parquet


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


def infer_vtd_width_from_series(s: pd.Series, default: int = 6) -> int:
    x = s.astype("string").str.upper().str.replace(r"[^0-9]", "", regex=True)
    lens = x.str.len().dropna()
    if lens.empty:
        return default
    if (lens >= 6).any():
        return 6
    if (lens >= 5).any():
        return 5
    return default


def normalize_cntyvtd_flexible(raw: pd.Series, vtd_width: int) -> pd.Series:
    s = raw.astype("string").str.strip().str.upper()
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)
    s2 = s.str.replace(r"^48", "", regex=True)

    m = s2.str.extract(r"^(?P<cnty>\d{3})(?P<rest>[A-Z0-9]+)$")
    cnty = m["cnty"].astype("string")
    rest = m["rest"].astype("string")

    vtd_digits = rest.str.replace(r"[^0-9]", "", regex=True)
    suf = rest.str.replace(r"[^A-Z]", "", regex=True)

    vtd_digits = vtd_digits.where(vtd_digits.str.len() > 0, pd.NA).str.zfill(vtd_width)
    out = pd.Series(pd.NA, dtype="string", index=s.index)

    valid = cnty.notna() & cnty.str.fullmatch(r"\d{3}", na=False) & vtd_digits.notna()
    out.loc[valid] = (cnty.loc[valid] + vtd_digits.loc[valid] + suf.loc[valid].fillna("")).astype("string")
    return out


def digits_only_cntyvtd(std_key: pd.Series, vtd_width: int) -> pd.Series:
    s = std_key.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    cnty = s.str.slice(0, 3)
    rest = s.str.slice(3)
    vtd_digits = rest.str.replace(r"[^0-9]", "", regex=True).str.zfill(vtd_width)
    out = pd.Series(pd.NA, dtype="string", index=s.index)
    valid = cnty.str.fullmatch(r"\d{3}", na=False) & vtd_digits.str.fullmatch(r"\d{" + str(vtd_width) + r"}", na=False)
    out.loc[valid] = (cnty.loc[valid] + vtd_digits.loc[valid]).astype("string")
    return out


# ============================== ELECTIONS HELPERS ==============================
def is_tall_elections(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    has_key = "cntyvtd" in cols or ({"county", "precinct"} <= cols) or ({"fips", "vtd"} <= cols)
    has_office = "office" in cols or "race" in cols
    has_votes = "votes" in cols or "vote" in cols
    return has_key and has_office and has_votes


def _normalize_party(party) -> str:
    if pd.isna(party):
        return "UNK"
    s = str(party).strip().upper()
    letters = re.sub(r"[^A-Z]", "", s)
    if letters in ("DEM", "D", "DFL") or "DEMOCRAT" in letters:
        return "DEM"
    if letters in ("REP", "R", "GOP") or "REPUBLICAN" in letters:
        return "REP"
    if letters.startswith("LIB") or "LIBERTARIAN" in letters:
        return "LIB"
    if letters.startswith("GRN") or "GREEN" in letters:
        return "GRN"
    if letters.startswith("IND") or "INDEPENDENT" in letters:
        return "IND"
    return letters if letters else "UNK"


def clean_vtd_election_returns(df: pd.DataFrame, target_office: str = "") -> pd.DataFrame:
    df = stdcols(df)

    office_col = "office" if "office" in df.columns else ("race" if "race" in df.columns else None)
    votes_col = "votes" if "votes" in df.columns else ("vote" if "vote" in df.columns else None)
    party_col = "party" if "party" in df.columns else None

    if office_col is None or votes_col is None:
        raise ValueError("Elections file must contain office/race and votes/vote columns.")
    if party_col is None:
        raise ValueError("Elections file has no 'party' column; cannot compute dem/rep votes.")

    if "cntyvtd" in df.columns:
        df["cntyvtd"] = df["cntyvtd"].astype("string")
    elif {"county", "precinct"} <= set(df.columns):
        df["cntyvtd"] = (df["county"].astype("string") + df["precinct"].astype("string")).astype("string")
    elif {"fips", "vtd"} <= set(df.columns):
        df["cntyvtd"] = (df["fips"].astype("string") + df["vtd"].astype("string")).astype("string")
    else:
        raise ValueError("Elections file lacks cntyvtd or sufficient fields to construct it.")

    df = df.loc[df["cntyvtd"].notna()].copy()

    df["office_norm"] = df[office_col].astype("string").str.strip()
    if target_office:
        mask = df["office_norm"].str.contains(re.escape(target_office), case=False, na=False)
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError(f"No rows matched elections office filter: {target_office}")

    df["party_norm"] = df[party_col].map(_normalize_party)

    df[votes_col] = df[votes_col].astype("string").str.replace(r"[^\d\-\.]", "", regex=True)
    df[votes_col] = pd.to_numeric(df[votes_col], errors="coerce").fillna(0)

    if float(df[votes_col].sum()) == 0.0:
        party_sample = df[party_col].astype("string").dropna().head(15).tolist()
        raise ValueError(
            f"Votes column '{votes_col}' parsed to all zeros after cleaning. "
            f"Party sample: {party_sample}. "
            "This usually means you're not using the correct votes column or values contain non-numeric encodings."
        )

    agg = df.groupby(["cntyvtd", "party_norm"], as_index=False)[votes_col].sum()
    agg = agg.rename(columns={votes_col: "votes"})

    wide = agg.pivot(index="cntyvtd", columns="party_norm", values="votes").reset_index()
    wide = wide.rename_axis(None, axis=1)

    wide["dem_votes"] = pd.to_numeric(wide.get("DEM", 0), errors="coerce").fillna(0).astype("int64")
    wide["rep_votes"] = pd.to_numeric(wide.get("REP", 0), errors="coerce").fillna(0).astype("int64")

    party_cols = [c for c in wide.columns if c not in ("cntyvtd", "dem_votes", "rep_votes")]
    party_cols = [c for c in party_cols if c not in ("DEM", "REP")]
    if party_cols:
        wide["third_party_votes"] = wide[party_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype("int64")
    else:
        wide["third_party_votes"] = 0

    wide["total_votes"] = (wide["dem_votes"] + wide["rep_votes"] + wide["third_party_votes"]).astype("int64")

    tot = wide["total_votes"]
    wide["dem_share"] = (wide["dem_votes"] / tot).where(tot > 0, 0.0)
    wide["rep_share"] = (wide["rep_votes"] / tot).where(tot > 0, 0.0)

    two_den = wide["dem_votes"] + wide["rep_votes"]
    wide["two_party_dem_share"] = (wide["dem_votes"] / two_den).where(two_den > 0, 0.0)

    return wide


# ============================== BLOCKS/PL HELPERS ==============================
def unify_pl94_schema(pl94: pd.DataFrame) -> pd.DataFrame:
    df = stdcols(pl94)
    if "geoid20" in df.columns:
        df["geoid20"] = (
            df["geoid20"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
        return df

    geoid_candidates = [
        "geoid", "tabblock20", "tabblock2020",
        "block_geoid", "blk_geoid", "sctbkey", "ctbkey",
    ]
    geoid_col = next((c for c in geoid_candidates if c in df.columns), None)
    if geoid_col is not None:
        df["geoid20"] = (
            df[geoid_col].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
    return df


def pick_pop_columns(pl: pd.DataFrame) -> tuple[str, dict[str, str]]:
    cols = set(pl.columns)

    if {"vap", "anglovap", "blackvap", "asianvap", "hispvap"} <= cols:
        return "vap", {
            "pct_white": "anglovap",
            "pct_black": "blackvap",
            "pct_asian": "asianvap",
            "pct_hispanic": "hispvap",
        }

    if {"total", "anglo", "black", "asian", "hisp"} <= cols:
        return "total", {
            "pct_white": "anglo",
            "pct_black": "black",
            "pct_asian": "asian",
            "pct_hispanic": "hisp",
        }

    if "pop20" in cols and {"anglo", "black", "asian", "hisp"} <= cols:
        return "pop20", {
            "pct_white": "anglo",
            "pct_black": "black",
            "pct_asian": "asian",
            "pct_hispanic": "hisp",
        }

    raise ValueError(
        "Could not detect required population/VAP columns in Blocks_Pop file.\n"
        "Expected either VAP set: vap + (anglovap, blackvap, asianvap, hispvap)\n"
        "or total-pop set: total + (anglo, black, asian, hisp)\n"
        f"Available columns (sample): {list(pl.columns)[:80]}"
    )


def pick_district_id_col(districts: pd.DataFrame) -> str | None:
    candidates = [
        "district_id", "district", "cd", "congress", "congdist", "cong_dist",
        "district_n", "dist",
        "cd116fp", "cd117fp", "cd118fp", "cd119fp",
        "cdfp",
    ]
    cols = set(districts.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# ============================== REOCK + COMPACTNESS ==============================
def _mec_circle_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
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
        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d
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


def compute_compactness(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    g = gdf.to_crs(AREA_CRS).copy()
    assert_projected_planar(g, "compute_compactness")

    g["geometry"] = g.geometry.buffer(0)
    A = g.geometry.area
    P = g.geometry.length

    out = pd.DataFrame(index=g.index)
    out["polsby_popper"] = ((4 * math.pi * A) / (P ** 2)).where(P > 0)

    hull = g.geometry.convex_hull
    hull_area = hull.area
    out["convex_hull_ratio"] = (A / hull_area).where(hull_area > 0)

    out["schwartzberg"] = (P / (2 * np.sqrt(math.pi * A))).where(A > 0)
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

    # ---- FIX: avoid duplicate 'cntyvtd' later by renaming the raw shapefile field immediately ----
    if "cntyvtd" in vtd.columns:
        vtd = vtd.rename(columns={"cntyvtd": "cntyvtd_raw"})

    pl = unify_pl94_schema(pd.read_parquet(pl_fp))
    elec = stdcols(pd.read_parquet(elect_fp))

    # ---------- VTD keys ----------
    vtd = vtd.copy()
    vtd_width = 6
    if "vtdkey" in vtd.columns:
        vtd_width = infer_vtd_width_from_series(vtd["vtdkey"], default=6)

    if "cntyvtd_raw" in vtd.columns:
        vtd["cntyvtd_std"] = normalize_cntyvtd_flexible(vtd["cntyvtd_raw"], vtd_width=vtd_width)
    elif {"cntykey", "vtdkey"} <= set(vtd.columns):
        cnty = vtd["cntykey"].astype("string").str.replace(r"[^\d]", "", regex=True).str.zfill(3)
        vtdk = vtd["vtdkey"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        digits = vtdk.str.replace(r"[^0-9]", "", regex=True).str.zfill(vtd_width)
        suf = vtdk.str.replace(r"[^A-Z]", "", regex=True)
        vtd["cntyvtd_std"] = (cnty + digits + suf.fillna("")).astype("string")
    elif {"cnty", "vtd"} <= set(vtd.columns):
        raw = (vtd["cnty"].astype("string") + vtd["vtd"].astype("string")).astype("string")
        vtd["cntyvtd_std"] = normalize_cntyvtd_flexible(raw, vtd_width=vtd_width)
    else:
        raise ValueError("Cannot construct CNTYVTD key for VTD geometries.")

    vtd["cntyvtd_digits"] = digits_only_cntyvtd(vtd["cntyvtd_std"], vtd_width=vtd_width)

    # ---------- Elections keys ----------
    if "cntyvtd" not in elec.columns:
        raise ValueError("Wide elections parquet missing cntyvtd column.")

    elec = elec.copy()
    elec["k_std_5"] = normalize_cntyvtd_flexible(elec["cntyvtd"], vtd_width=5)
    elec["k_std_6"] = normalize_cntyvtd_flexible(elec["cntyvtd"], vtd_width=6)
    elec["k_dig_5"] = digits_only_cntyvtd(elec["k_std_5"], vtd_width=5)
    elec["k_dig_6"] = digits_only_cntyvtd(elec["k_std_6"], vtd_width=6)

    vtd = vtd.loc[vtd["cntyvtd_std"].notna()].copy()
    elec = elec.loc[elec["cntyvtd"].notna()].copy()

    print(f"[INFO] VTD keys — geo: {vtd['cntyvtd_std'].nunique()} unique, elections: {elec['cntyvtd'].nunique()} raw unique")

    # ---------- Blocks + PL merge (robust) ----------
    blocks = ensure_geoid20_str(blocks)
    pl = ensure_geoid20_str(pl)

    total_col, race_map = pick_pop_columns(pl)

    candidate_join_keys = ["geoid20", "ctbkey", "blkkey"]
    available_pairs = [k for k in candidate_join_keys if (k in blocks.columns and k in pl.columns)]
    if not available_pairs:
        raise ValueError(
            "No common join key between blocks shapefile and Blocks_Pop table. "
            f"Need at least one of {candidate_join_keys} present in BOTH."
        )

    pl_cols_needed = list(dict.fromkeys(available_pairs + [total_col, *race_map.values()]))
    pl_small = pl[pl_cols_needed].copy()

    for k in ["ctbkey", "blkkey"]:
        if k in blocks.columns:
            blocks[k] = blocks[k].astype("string").str.strip()
        if k in pl_small.columns:
            pl_small[k] = pl_small[k].astype("string").str.strip()

    if "geoid20" in blocks.columns:
        blocks["geoid20"] = (
            blocks["geoid20"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )
    if "geoid20" in pl_small.columns:
        pl_small["geoid20"] = (
            pl_small["geoid20"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^\d]", "", regex=True)
            .str.zfill(15)
        )

    def _merge_and_rate(left: gpd.GeoDataFrame, right: pd.DataFrame, key: str) -> tuple[gpd.GeoDataFrame, float]:
        out = left.merge(right, on=key, how="left")
        rate = out[total_col].notna().mean() if total_col in out.columns else 0.0
        return out, float(rate)

    best_df = None
    best_key = None
    best_rate = -1.0
    for key in ["geoid20", "ctbkey", "blkkey"]:
        if key not in available_pairs:
            continue
        merged, rate = _merge_and_rate(blocks, pl_small, key)
        if rate > best_rate:
            best_df, best_key, best_rate = merged, key, rate

    if best_df is None:
        raise ValueError("Internal error: could not merge blocks with Blocks_Pop using any candidate key.")
    if best_rate < 0.90:
        raise ValueError(
            f"Low join match rate when merging blocks with Blocks_Pop: {best_rate:.2%}. "
            f"Tried keys: {available_pairs}. Best key was '{best_key}'. "
            "This will make pct_* null."
        )

    blocks_pl = best_df
    print(f"[INFO] Blocks_Pop join key selected: {best_key} (match rate {best_rate:.2%})")

    blocks_pl[total_col] = pd.to_numeric(blocks_pl[total_col], errors="coerce").fillna(0)
    for rc in race_map.values():
        blocks_pl[rc] = pd.to_numeric(blocks_pl[rc], errors="coerce").fillna(0)

    # ---------- Project + clean geometry ----------
    blocks_proj = blocks_pl.to_crs(AREA_CRS)
    vtd_proj = vtd.to_crs(AREA_CRS)  # carries key cols forward
    assert_projected_planar(blocks_proj, "blocks->VTD")
    assert_projected_planar(vtd_proj, "blocks->VTD")

    blocks_proj["geometry"] = blocks_proj.geometry.buffer(0)
    vtd_proj["geometry"] = vtd_proj.geometry.buffer(0)

    vtd_proj = vtd_proj.reset_index(drop=True)
    vtd_proj["vtd_idx"] = vtd_proj.index

    # ============================================================
    # Assign district_id to each VTD (dominant area overlap)
    # ============================================================
    districts_proj = districts.to_crs(AREA_CRS).copy()
    districts_proj["geometry"] = districts_proj.geometry.buffer(0)

    d = districts_proj.reset_index(drop=True).copy()
    d["district_idx"] = d.index

    inter_vtd_dist = gpd.overlay(
        vtd_proj[["vtd_idx", "geometry"]],
        d[["district_idx", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    if inter_vtd_dist.empty:
        vtd_proj["district_id"] = pd.NA
    else:
        inter_vtd_dist["inter_area"] = inter_vtd_dist.geometry.area
        best = (
            inter_vtd_dist
            .sort_values("inter_area", ascending=False)
            .drop_duplicates("vtd_idx")[["vtd_idx", "district_idx"]]
        )
        id_col = pick_district_id_col(districts_proj)
        if id_col is not None:
            best = best.merge(d[["district_idx", id_col]], on="district_idx", how="left").rename(columns={id_col: "district_id"})
        else:
            best["district_id"] = best["district_idx"] + 1
        vtd_proj = vtd_proj.merge(best[["vtd_idx", "district_id"]], on="vtd_idx", how="left")

    # ============================================================
    # Blocks -> VTD overlay and area-weighting
    # ============================================================
    print("[INFO] Overlay blocks -> VTDs (intersection)...")
    blk = blocks_proj[["geoid20", "geometry"]].copy()
    blk_attrs = blocks_proj[["geoid20", total_col, *race_map.values()]].copy()
    vtd_sub = vtd_proj[["vtd_idx", "geometry"]].copy()

    inter = gpd.overlay(blk, vtd_sub, how="intersection", keep_geom_type=False)
    if inter.empty:
        raise ValueError("blocks→VTD overlay returned 0 rows (CRS/geometry mismatch).")
    inter = inter.merge(blk_attrs, on="geoid20", how="left")

    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    inter["blk_area"] = blk_area.reindex(inter["geoid20"]).values
    inter["inter_area"] = inter.geometry.area

    inter = inter.loc[inter["blk_area"] > 0].copy()
    inter["w"] = (inter["inter_area"] / inter["blk_area"]).clip(0, 1)

    sum_cols = [total_col] + list(race_map.values())
    for c in sum_cols:
        inter[c] = pd.to_numeric(inter[c], errors="coerce").fillna(0) * inter["w"]

    agg = (
        inter.groupby("vtd_idx", observed=True)[sum_cols]
        .sum()
        .reindex(vtd_proj["vtd_idx"], fill_value=0)
    )

    den = agg[total_col].replace({0: np.nan})
    race_pct = pd.DataFrame(index=agg.index)
    for out_name, src_col in race_map.items():
        race_pct[out_name] = (agg[src_col] / den).where(den > 0)

    # ============================================================
    # Elections join: auto-pick best key variant
    # ============================================================
    required_vote_cols = ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]
    for c in required_vote_cols:
        if c not in elec.columns:
            raise ValueError(f"Elections wide table missing expected column: {c}")

    if float(pd.to_numeric(elec["total_votes"], errors="coerce").fillna(0).sum()) == 0.0:
        raise ValueError("Your elections-wide table has total_votes sum=0 BEFORE any join (Stage 1 produced zeros).")

    vtd_proj["k_std_5"] = normalize_cntyvtd_flexible(vtd_proj["cntyvtd_std"], vtd_width=5)
    vtd_proj["k_std_6"] = normalize_cntyvtd_flexible(vtd_proj["cntyvtd_std"], vtd_width=6)
    vtd_proj["k_dig_5"] = digits_only_cntyvtd(vtd_proj["k_std_5"], vtd_width=5)
    vtd_proj["k_dig_6"] = digits_only_cntyvtd(vtd_proj["k_std_6"], vtd_width=6)

    elec_aggs = {}
    for kcol in ["k_std_5", "k_std_6", "k_dig_5", "k_dig_6"]:
        tmp = elec.loc[elec[kcol].notna()].groupby(kcol, as_index=False)[required_vote_cols].sum()
        elec_aggs[kcol] = tmp

    candidates = []
    for kcol in ["k_std_6", "k_dig_6", "k_std_5", "k_dig_5"]:
        geo_keys = set(vtd_proj[kcol].dropna().unique().tolist())
        ele_keys = set(elec_aggs[kcol][kcol].dropna().unique().tolist())
        overlap = len(geo_keys & ele_keys)
        candidates.append((kcol, overlap))

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_k = candidates[0][0]
    if candidates[0][1] == 0:
        raise ValueError(
            "After building multiple CNTYVTD key variants, overlap is still 0 for all joins.\n"
            f"Overlap by variant: {candidates}\n"
            "This means elections precinct IDs are not the same universe/format as the VTD shapefile."
        )

    joined = vtd_proj.merge(
        elec_aggs[best_k],
        how="left",
        left_on=best_k,
        right_on=best_k,
    )

    for c in required_vote_cols:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0)

    if float(joined["total_votes"].sum()) == 0.0:
        raise ValueError(
            f"Elections join selected variant '{best_k}' (overlap {candidates[0][1]}) "
            "but total_votes still sums to 0 after join."
        )

    print(f"[INFO] Elections join selected key variant: {best_k} (overlap {candidates[0][1]})")

    tot = joined["total_votes"]
    joined["dem_share"] = (joined["dem_votes"] / tot).where(tot > 0, 0.0)
    joined["rep_share"] = (joined["rep_votes"] / tot).where(tot > 0, 0.0)

    # ---- Compactness ----
    print("[INFO] Computing VTD compactness ...")
    cmpx = compute_compactness(vtd_proj)

    # ---- Assemble final ----
    final = joined.join(cmpx, on="vtd_idx").join(race_pct, on="vtd_idx")

    # rename standardized key to your required output column
    final = final.rename(columns={"cntyvtd_std": "cntyvtd"})

    keep = FINAL_COLUMNS + (["geometry"] if KEEP_GEOMETRY else [])
    keep = [c for c in keep if c in final.columns]
    final = final[keep].copy()

    # ---- Safety net: remove duplicate column names (pyarrow forbids them) ----
    if final.columns.duplicated().any():
        final = final.loc[:, ~final.columns.duplicated()].copy()

    missing_cols = [c for c in FINAL_COLUMNS if c not in final.columns]
    if missing_cols:
        raise ValueError(f"Final dataset missing required columns: {missing_cols}")

    print(f"[INFO] Final VTD rows: {len(final)}")
    return final


# ============================== CLI ==============================
def main():
    ap = argparse.ArgumentParser(description="ETL + VTD-level dataset (one row per VTD)")
    ap.add_argument("--districts", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--vtds", type=Path, required=True)
    ap.add_argument("--pl94", type=Path, required=True)
    ap.add_argument("--elections", type=Path, required=True)
    ap.add_argument("--elections-office", type=str, default="U.S. Sen")
    ap.add_argument("--data-processed-tabular", type=Path, required=True)
    ap.add_argument("--data-processed-geospatial", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    args = ap.parse_args()

    input_files = [args.districts, args.census, args.vtds, args.pl94, args.elections]

    run_etl(
        input_files=input_files,
        out_parquet_dir=args.data_processed_tabular,
        out_geo_dir=args.data_processed_geospatial,
        sqlite_path=args.sqlite,
        elections_office=args.elections_office,
    )

    dk, ck, vk, pk, ek = [dataset_key(p) for p in input_files]

    final = build_final(
        out_parquet_dir=args.data_processed_tabular,
        out_geo_dir=args.data_processed_geospatial,
        districts_key=dk,
        census_key=ck,
        vtds_key=vk,
        pl_key=pk,
        elections_key=ek,
    )

    out_pq = args.data_processed_tabular / "vtds_final.parquet"
    out_csv = args.data_processed_tabular / "vtds_final.csv"
    write_parquet(final, out_pq)
    final.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(" ", out_pq)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
