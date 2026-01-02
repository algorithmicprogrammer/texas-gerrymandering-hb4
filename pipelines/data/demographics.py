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

    # Avoids mutating the input DataFrame.
    out = df.copy()
    # Normalizing column names immediately so subsequent checks work reliably.
    out.columns = [c.strip().lower() for c in out.columns]

    # If GEOID column already exists, forcing it into standardized format.
    if col in out.columns:
        out[col] = out[col].astype(str).str.strip().str.zfill(15)
        return out

    # Defines common alternative column names and maps them to the canonical meaning "geoid20".
    alias_map = {
        "geoid20": "geoid20",
        "geoid": "geoid20",
        "tabblock20": "geoid20",
        "block_geoid": "geoid20",
        "block_geoid20": "geoid20",
        "geoid_20": "geoid20",
    }

    # Iterates over each alias and its target meaning.
    for a, target in alias_map.items():
        # Checks if the alias exists in the DataFrame and if it corresponds with the enforced column name.
        if a in out.columns and target == col:
            # Renames the alias column to the requested canonical name.
            out = out.rename(columns={a: col})
            # Normalizes columns by type checking for string, trimming whitespaces, and zero-padding to 15 characters.
            out[col] = out[col].astype(str).str.strip().str.zfill(15)
            # Returns DataFrame with constructed geoid20.
            return out

    # The components need to construct a 2020 block GEOID.
    required = ["state", "fips", "trt", "blk"]
    # Only proceed if all required component columns exist.
    if all(c in out.columns for c in required):
        # Build a 2-digit state FIPS.
        st = pd.to_numeric(out["state"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
        # County FIPS: set to string type, trim whitespaces, and zero-pad to three characters.
        co = out["fips"].astype(str).str.strip().str.zfill(3)
        # Tract: set to string type, trim whitespaces, and zero-pad to 3 characters.
        tr = out["trt"].astype(str).str.strip().str.zfill(6)
        # Block: set to string type, trim whitespaces, and zero-pad to 4 characters.
        bl = out["blk"].astype(str).str.strip().str.zfill(4)
        # Concatenates the four pieces into one GEOID string.
        out[col] = (st + co + tr + bl).astype(str).str.strip().str.zfill(15)
        # Returns DataFrame with constructed geoid20.
        return out

    # If col, alias, or component columns do not exist, error loudly.
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

