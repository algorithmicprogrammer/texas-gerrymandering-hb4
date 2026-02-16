# data/demographics.py
from __future__ import annotations

import pandas as pd


def unify_pl94_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to reduce schema friction between data sources.
    Lowercasing column names and stripping leading/trailing whitespace.

    Args:
        df: Pandas DataFrame

    Returns:
        out: cleaned DataFrame
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def ensure_geoid20_str(df: pd.DataFrame, col: str = "geoid20") -> pd.DataFrame:
    """
    Guarantees that the DataFrame has a GEOID column as a 15-character zero-padded string.

    Args:
        df: Pandas DataFrame
        col: optional column name to enforce (defaults to "geoid20")

    Returns:
        out: DataFrame with constructed geoid20.

    Raises:
        ValueError: If col, an alias, and component columns don't exist.
    """

    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    if col in out.columns:
        out[col] = out[col].astype(str).str.strip().str.zfill(15)
        return out

    alias_map = {
        "geoid20": "geoid20",
        "geoid": "geoid20",
        "tabblock20": "geoid20",
        "block_geoid": "geoid20",
        "block_geoid20": "geoid20",
        "geoid_20": "geoid20",
    }

    for a, target in alias_map.items():
        if a in out.columns and target == col:
            out = out.rename(columns={a: col})
            out[col] = out[col].astype(str).str.strip().str.zfill(15)
            return out

    # Components to construct 2020 block GEOID
    required = ["state", "fips", "trt", "blk"]
    if all(c in out.columns for c in required):
        st = pd.to_numeric(out["state"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
        co = out["fips"].astype(str).str.strip().str.zfill(3)
        tr = out["trt"].astype(str).str.strip().str.zfill(6)
        bl = out["blk"].astype(str).str.strip().str.zfill(4)
        out[col] = (st + co + tr + bl).astype(str).str.strip().str.zfill(15)
        return out

    raise ValueError(
        f"Missing {col}. Could not find an alias (GEOID/GEOID20/TABBLOCK20) "
        "and could not construct from State+FIPS+TRT+BLK."
    )


def pick_pop_columns(df: pd.DataFrame):
    """
    Identify VAP columns in the merged blocks table.

    Returns:
        total_vap: column name for total VAP
        race_map: mapping of canonical VAP group names -> source columns
        mode: string ("vap")

    Raises:
        ValueError: If total VAP is missing.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    total_vap = cols.get("vap")
    anglo_vap = cols.get("anglovap")
    black_vap = cols.get("blackvap")
    hisp_vap = cols.get("hispvap")
    asian_vap = cols.get("asianvap")

    if total_vap is None:
        raise ValueError("Could not find total VAP column 'vap' in merged blocks table.")

    missing = [
        name
        for name, col in [
            ("anglovap", anglo_vap),
            ("blackvap", black_vap),
            ("hispvap", hisp_vap),
            ("asianvap", asian_vap),
        ]
        if col is None
    ]

    if missing:
        return total_vap, {}, "vap"

    race_map = {
        "vap_nh_white": anglo_vap,
        "vap_nh_black": black_vap,
        "vap_hisp": hisp_vap,
        "vap_nh_asian": asian_vap,
        "vap_nh_native": None,
    }
    race_map = {k: v for k, v in race_map.items() if v is not None}
    return total_vap, race_map, "vap"


def pick_total_pop_column(df: pd.DataFrame) -> str:
    """
    Infer a block-level TOTAL population column from a merged blocks table.

    Raises:
        ValueError if no recognizable total population column is present.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    candidates = [
        "p1_001n",
        "p1_001",
        "p001001",
        "p0010001",
        "total_pop",
        "totpop",
        "population",
        "pop_total",
        "totalpopulation",
        "total",
    ]

    for key in candidates:
        if key in cols:
            return cols[key]

    raise ValueError(
        "Could not infer TOTAL population column in merged blocks table. "
        "Expected something like P1_001N / P001001 / total_pop / totpop."
    )


def pick_total_race_columns(df: pd.DataFrame):
    """
    Identify TOTAL population by race/ethnicity columns in the merged blocks table.

    We want to support the common 4-bucket table:
      Latino, Black, White, Other (where White/Black are typically NH-only)

    Many PL94-derived pipelines export simplified columns such as:
      anglo, black, hisp, asian  (and total pop)

    Returns:
        race_map_total: mapping canonical total-pop group names -> source columns

    Canonical outputs we try to produce:
      total_hisp
      total_nh_black
      total_nh_white
      total_nh_asian
      total_nh_native (optional)
      total_nh_pi (optional)
    """
    cols = {c.strip().lower(): c for c in df.columns}

    # Common simplified totals in some PL94 merges
    # (These are *not* official PL94 names; they are common in preprocessed block text files.)
    anglo = cols.get("anglo") or cols.get("white") or cols.get("whitetot") or cols.get("totwhite") or cols.get("white_total")
    black = cols.get("black") or cols.get("blacktot") or cols.get("totblack") or cols.get("black_total")
    hisp = cols.get("hisp") or cols.get("hispanic") or cols.get("latino") or cols.get("hisp_total")
    asian = cols.get("asian") or cols.get("asiantot") or cols.get("totasian") or cols.get("asian_total")
    native = cols.get("native") or cols.get("aian") or cols.get("amerind") or cols.get("natam") or cols.get("native_total")
    pi = cols.get("pi") or cols.get("nhpi") or cols.get("pacific") or cols.get("nhopi") or cols.get("pi_total")

    race_map_total = {}

    # Treat anglo as NH white proxy if present
    if anglo is not None:
        race_map_total["total_nh_white"] = anglo
    if black is not None:
        race_map_total["total_nh_black"] = black
    if hisp is not None:
        race_map_total["total_hisp"] = hisp
    if asian is not None:
        race_map_total["total_nh_asian"] = asian
    if native is not None:
        race_map_total["total_nh_native"] = native
    if pi is not None:
        race_map_total["total_nh_pi"] = pi

    return race_map_total



