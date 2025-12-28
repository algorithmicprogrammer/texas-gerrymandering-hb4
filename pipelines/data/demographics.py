from __future__ import annotations

import pandas as pd


def unify_pl94_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light harmonization: lowercase/strip column names.
    (Your pipeline uses stdcols() elsewhere; this is an extra safety net.)
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def ensure_geoid20_str(df: pd.DataFrame, col: str = "geoid20") -> pd.DataFrame:
    """
    Ensure the dataframe has a block GEOID column `col` as a 15-character string.

    Works for BOTH:
      - TIGER blocks shapefile attribute tables (often already have GEOID20)
      - Your PL94-like Blocks_Pop table (constructable from State + FIPS + TRT + BLK)

    Construction rule for your Blocks_Pop.txt:
      geoid20 = zfill(State,2) + zfill(FIPS,3) + zfill(TRT,6) + zfill(BLK,4)
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    # If requested col exists, just normalize
    if col in out.columns:
        out[col] = out[col].astype(str).str.strip().str.zfill(15)
        return out

    # Common aliases in TIGER-like tables
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

    # Construct from your PL table's components (after unify/stdcols -> lowercase)
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
    Return (total_col, race_map, mode) where:
      - total_col is the column holding total VAP
      - race_map maps canonical output names -> source columns
      - mode is 'vap' (what we want)
    This matches what cli.py expects.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    # Your Blocks_Pop schema has these (lowercased by stdcols/unify)
    total_vap = cols.get("vap")
    anglo_vap = cols.get("anglovap")
    black_vap = cols.get("blackvap")
    hisp_vap = cols.get("hispvap")
    asian_vap = cols.get("asianvap")

    if total_vap is None:
        raise ValueError("Could not find total VAP column 'vap' in merged blocks table.")

    missing = [name for name, col in [
        ("anglovap", anglo_vap),
        ("blackvap", black_vap),
        ("hispvap", hisp_vap),
        ("asianvap", asian_vap),
    ] if col is None]

    # If race breakdown missing, still allow pipeline to proceed with total only
    if missing:
        return total_vap, {}, "vap"

    race_map = {
        # Canonical outputs -> source cols
        "vap_nh_white": anglo_vap,   # 'anglo' used as NH-white proxy in this TX dataset
        "vap_nh_black": black_vap,
        "vap_hisp": hisp_vap,
        "vap_nh_asian": asian_vap,
        # not explicitly present in your schema
        "vap_nh_native": None,
    }

    # Remove None values for downstream list(race_map.values())
    race_map = {k: v for k, v in race_map.items() if v is not None}

    return total_vap, race_map, "vap"

