"""
ingest.py
---------
Load and lightly parse every raw data source into clean in-memory objects.

No demographic construction or spatial joining happens here — only parsing
and filtering to the geographic level we need (Census blocks, SUMLEV=750).
"""

import logging
import re
import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PL 94-171 fixed-width positions for the geo header file (txgeo2020.pl).
# Positions are 1-indexed per Census documentation; we convert to 0-indexed
# slices.  Source: Legacy Format Summary File Header Records spec.
# ---------------------------------------------------------------------------
GEO_HEADER_FIELDS = {
    # (start_0idx, end_0idx_exclusive)
    "FILEID":   (0,  6),
    "STUSAB":   (6,  8),
    "SUMLEV":   (8,  11),    # summary level; filter to "750" for blocks
    "LOGRECNO": (18, 25),    # 7-char join key to data segment
    "STATE":    (27, 29),    # 2-digit state FIPS
    "COUNTY":   (29, 32),    # 3-digit county FIPS
    "TRACT":    (54, 60),    # 6-digit tract code
    "BLOCK":    (61, 65),    # 4-char block code (only at SUMLEV 750)
    "GEOID":    (25, 85),    # full 60-char GEOID (trimmed after parsing)
}


def _parse_geo_header(path: str, block_sumlev: str = "750") -> pd.DataFrame:
    """
    Parse the fixed-width PL 94-171 geo header file.

    Returns a DataFrame with one row per Census block (SUMLEV == block_sumlev)
    containing LOGRECNO, GEOID20 (bare 15-digit block FIPS), and tract GEOID.
    """
    records = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if len(line) < 85:
                continue
            sumlev = line[8:11].strip()
            if sumlev != block_sumlev:
                continue

            raw_geoid = line[25:85].strip()
            # Full block GEOID has prefix like "7500000US484530001001000"
            # Strip the prefix to get the bare 15-digit block FIPS code
            bare_geoid = re.sub(r"^\d{3}0+US", "", raw_geoid)

            state  = line[27:29].strip()
            county = line[29:32].strip()
            tract  = line[54:60].strip()
            tract_geoid = state + county + tract   # 11-digit tract GEOID

            records.append({
                "LOGRECNO": line[18:25].strip().zfill(7),
                "GEOID20":    bare_geoid,
                "tract_geoid": tract_geoid,
            })

    df = pd.DataFrame(records)
    log.info(f"  Geo header: {len(df):,} Census block rows parsed")
    return df


def _parse_pl_data_segment(path: str, logrecnos: set) -> pd.DataFrame:
    """
    Parse the pipe-delimited PL 94-171 data segment (tx000022020.pl).

    Only rows whose LOGRECNO appears in `logrecnos` are kept (block rows only).

    Column layout (0-indexed):
      0-4   : header fields (FILEID, STUSAB, CHARITER, CIFSN, LOGRECNO)
      5-75  : P3  — VAP by race, 18+ (71 cells, P0030001–P0030071)
      76-148: P4  — VAP by race for non-Hispanic 18+ (73 cells, P0040001–P0040073)
      149-151: H1 — housing occupancy (3 cells)
    """
    p3_names = [f"P{i:07d}" for i in range(30_001, 30_072)]   # P0030001-P0030071
    p4_names = [f"P{i:07d}" for i in range(40_001, 40_074)]   # P0040001-P0040073
    h1_names = [f"H{i:07d}" for i in range(10_001, 10_004)]

    col_names = (
        ["FILEID", "STUSAB", "CHARITER", "CIFSN", "LOGRECNO"]
        + p3_names
        + p4_names
        + h1_names
    )

    chunks = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 152:
                continue
            logrecno = parts[4].strip().zfill(7)
            if logrecno not in logrecnos:
                continue
            chunks.append(parts[:152])

    df = pd.DataFrame(chunks, columns=col_names)
    # Re-apply zfill after strip so LOGRECNO matches the geo header keys
    df["LOGRECNO"] = df["LOGRECNO"].str.strip().str.zfill(7)

    # Cast numeric columns
    numeric_cols = p3_names + p4_names + h1_names
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    log.info(f"  PL data segment: {len(df):,} block rows loaded")
    return df


def _load_acs_cvap(path: str) -> pd.DataFrame:
    """
    Load the ACS CVAP special tabulation tract-level CSV.

    Keeps only Texas rows (state FIPS = 48) and pivots lnnumber categories
    to wide format so every tract has one row with columns for each racial
    group's cvap_est.

    geoid format: "1400000USsscccccctttttt" → we strip the prefix to get
    the 11-digit tract GEOID.
    """
    df = pd.read_csv(path, dtype={"geoid": str, "lnnumber": str})
    df.columns = df.columns.str.strip().str.lower()

    # Filter Texas (state FIPS 48)
    # geoid example: "1400000US48001950100"
    df = df[df["geoid"].str.contains("US48", na=False)].copy()

    # Bare 11-digit tract GEOID
    df["tract_geoid"] = df["geoid"].str.extract(r"US(\d{11})")[0]
    df = df.dropna(subset=["tract_geoid"])

    # Pivot: one row per tract, columns are lnnumber values
    df["lnnumber"] = df["lnnumber"].str.strip()
    pivot = df.pivot_table(
        index="tract_geoid",
        columns="lnnumber",
        values="cvap_est",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot.columns = ["tract_geoid"] + [f"cvap_ln{c}" for c in pivot.columns[1:]]

    log.info(f"  ACS CVAP: {len(pivot):,} Texas tract rows loaded")
    return pivot


def load_all_sources(paths: dict, block_sumlev: str = "750") -> dict:
    """
    Load and return all raw data sources.

    Returns a dict with keys:
      pl_blocks  : merged geo-header + PL data (one row = one Census block)
      acs_cvap   : pivoted ACS CVAP tract data
      blocks_shp : TIGER/LINE Census block GeoDataFrame (GEOID20, geometry, POP20)
      vtd_shp    : VTD shapefile GeoDataFrame
      planc      : PLANC2333 congressional district GeoDataFrame
      gen_returns, dem_primary, rep_primary : raw election CSVs
    """
    log.info("Loading geo header …")
    geo = _parse_geo_header(str(paths["pl_geo"]), block_sumlev=block_sumlev)

    log.info("Loading PL data segment …")
    pl = _parse_pl_data_segment(str(paths["pl_data"]), set(geo["LOGRECNO"]))

    # Merge on LOGRECNO
    pl_blocks = geo.merge(pl.drop(columns=["FILEID", "STUSAB", "CHARITER", "CIFSN"]),
                          on="LOGRECNO", how="inner")
    log.info(f"  Merged PL blocks: {len(pl_blocks):,} rows")

    log.info("Loading ACS CVAP …")
    acs_cvap = _load_acs_cvap(str(paths["acs_cvap"]))

    log.info("Loading TIGER/LINE Census block shapefile …")
    blocks_shp = gpd.read_file(str(paths["blocks_shp"]))
    blocks_shp = blocks_shp[["GEOID20", "POP20", "geometry"]]
    blocks_shp["GEOID20"] = blocks_shp["GEOID20"].str.strip()
    log.info(f"  Census block geometries: {len(blocks_shp):,} rows")

    log.info("Loading VTD shapefile …")
    vtd_shp = gpd.read_file(str(paths["vtd_shp"]))
    log.info(f"  VTDs: {len(vtd_shp):,} rows")

    log.info("Loading PLANC2333 congressional district shapefile …")
    planc = gpd.read_file(str(paths["planc"]))
    log.info(f"  Congressional districts: {len(planc):,} rows")

    log.info("Loading election return CSVs …")
    gen_returns = pd.read_csv(str(paths["gen_returns"]), dtype={"VTD": str, "FIPS": str})
    dem_primary = pd.read_csv(str(paths["dem_primary"]), dtype={"VTD": str, "FIPS": str})
    rep_primary = pd.read_csv(str(paths["rep_primary"]), dtype={"VTD": str, "FIPS": str})

    return {
        "pl_blocks":   pl_blocks,
        "acs_cvap":    acs_cvap,
        "blocks_shp":  blocks_shp,
        "vtd_shp":     vtd_shp,
        "planc":       planc,
        "gen_returns": gen_returns,
        "dem_primary": dem_primary,
        "rep_primary": rep_primary,
    }
