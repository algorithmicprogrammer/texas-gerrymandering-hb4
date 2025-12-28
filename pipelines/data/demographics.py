from __future__ import annotations
import re
import pandas as pd

def ensure_geoid20_str(df: pd.DataFrame, col: str = "geoid20") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"Missing {col} in blocks/census data.")
    df[col] = df[col].astype("string").str.strip()
    return df

def unify_pl94_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to harmonize PL94-like columns to a standard set.
    This is intentionally flexible; you can point it at your own PL/ACS extracts.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_pop_columns(df: pd.DataFrame):
    """
    Heuristic: Prefer VAP columns if present, otherwise fall back to total pop columns.

    Returns:
      total_col: str
      race_map: dict[out_col -> src_col] where out_col matches canonical names
    """
    cols = set(df.columns)
    # Try VAP first
    vap_total_candidates = [c for c in cols if re.search(r"\bvap\b.*total|vap_total|totvap", c)]
    if vap_total_candidates:
        total_col = sorted(vap_total_candidates)[0]
        # These mappings depend on your extract naming; adjust if needed.
        # We'll search for common patterns.
        def find(patterns):
            for pat in patterns:
                for c in cols:
                    if re.search(pat, c):
                        return c
            return None

        m = {
            "vap_nh_white": find([r"nh_?white.*vap", r"vap.*nh_?white", r"white_nh_vap"]),
            "vap_nh_black": find([r"nh_?black.*vap", r"vap.*nh_?black", r"black_nh_vap"]),
            "vap_hisp": find([r"hisp.*vap", r"vap.*hisp"]),
            "vap_nh_asian": find([r"nh_?asian.*vap", r"vap.*nh_?asian", r"asian_nh_vap"]),
            "vap_nh_native": find([r"nh_?(ai(an)?|native).*vap", r"vap.*nh_?(ai(an)?|native)"]),
        }
        # Only keep found
        race_map = {k:v for k,v in m.items() if v is not None}
        return total_col, race_map, "vap"
    # Fallback: total population style
    total_candidates = [c for c in cols if c in {"total","totpop","pop_total","total_pop","tot"}]
    total_col = sorted(total_candidates)[0] if total_candidates else None
    if total_col is None:
        raise ValueError("Could not infer total population column in PL/blocks table.")
    # Very rough fallback mapping; you will likely want to adjust based on your PL schema.
    race_map = {}
    return total_col, race_map, "pop"
