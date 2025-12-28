from __future__ import annotations
import pandas as pd

def pick_district_id_col(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for cand in ["district","district_id","dist","dist_id","districtnum","district_n","cd","cd116","cd118","cd"]:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None
