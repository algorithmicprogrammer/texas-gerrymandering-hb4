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
    Args:
        df: Pandas DataFrame of a merged block-level table

    Returns:
        total_vap: which column is total voting age population
        race_map: which columns represent voting age population by race/ethnicity
        mode: returns them in a format expected by the CLI

    Raises:
        ValueError: If the total voting age population is none.
    """

    # Builds a mapping from normalized names to original column names.
    cols = {c.strip().lower(): c for c in df.columns}

    # Attempts to find total voting-age population.
    total_vap = cols.get("vap")
    # Attempts to find White voting-age population.
    anglo_vap = cols.get("anglovap")
    # Attempts to find Black voting-age population.
    black_vap = cols.get("blackvap")
    # Attempts to find Hispanic voting-age population.
    hisp_vap = cols.get("hispvap")
    # Attempts to find Asian voting-age population.
    asian_vap = cols.get("asianvap")

    # Total voting age population is required for the pipeline's demographic calculations, so it hard-fails if total_vap missing.
    if total_vap is None:
        raise ValueError("Could not find total VAP column 'vap' in merged blocks table.")

    # Creates a list of race breakdown fields that weren't found by iterating over race-specific VAP columns.
    missing = [name for name, col in [
        ("anglovap", anglo_vap),
        ("blackvap", black_vap),
        ("hispvap", hisp_vap),
        ("asianvap", asian_vap),
    ] if col is None]

    # If race breakdown is missing, then return the total only and an empty race_map.
    if missing:
        return total_vap, {}, "vap"

    # Creates a mapping from canonical output names to source columns.
    race_map = {
        # "Anglo" is treated as a proxy for non-Hispanic white.
        "vap_nh_white": anglo_vap,
        "vap_nh_black": black_vap,
        "vap_hisp": hisp_vap,
        "vap_nh_asian": asian_vap,
        # vap_nh_native is not present in our schema.
        "vap_nh_native": None,
    }

    # Removes any entries whose source column is none.
    race_map = {k: v for k, v in race_map.items() if v is not None}

    # Returns a tuple with total voting age population column name, the mapping of outputs to input columns, and the mode string "vap".
    return total_vap, race_map, "vap"

