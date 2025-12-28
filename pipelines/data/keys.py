from __future__ import annotations
import pandas as pd

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
    s = s.str.replace(r"[^0-9]", "", regex=True)
    county = s.str.slice(0, 3)
    vtd = s.str.slice(3, 3 + vtd_width).str.zfill(vtd_width)
    return county + vtd

def digits_only_cntyvtd(s: pd.Series, vtd_width: int) -> pd.Series:
    x = s.astype("string").str.replace(r"[^0-9]", "", regex=True)
    county = x.str.slice(0, 3)
    vtd = x.str.slice(3, 3 + vtd_width).str.zfill(vtd_width)
    return county + vtd
