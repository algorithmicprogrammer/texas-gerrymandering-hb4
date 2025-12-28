from __future__ import annotations
import pandas as pd
import numpy as np

def _normalize_party(p: str) -> str:
    if not isinstance(p, str):
        return ""
    p = p.strip().upper()
    if p in {"DEM", "D", "DEMOCRATIC", "DEMOCRAT"}:
        return "DEM"
    if p in {"REP", "R", "REPUBLICAN"}:
        return "REP"
    return "OTHER"

def is_tall_elections(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"party", "votes"}.issubset(cols)

def clean_vtd_election_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      * tall format with columns: cntyvtd, party, votes (plus optional election_id)
      * wide format already: cntyvtd, dem_votes, rep_votes, third_party_votes, total_votes

    Returns wide standardized columns.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "cntyvtd" not in df.columns:
        raise ValueError("Election returns must include cntyvtd key column.")
    df["cntyvtd"] = df["cntyvtd"].astype("string").str.strip()

    if is_tall_elections(df):
        df["party"] = df["party"].map(_normalize_party)
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
        pivot = df.pivot_table(index=["cntyvtd"], columns="party", values="votes", aggfunc="sum", fill_value=0)
        dem = pivot.get("DEM", 0)
        rep = pivot.get("REP", 0)
        other = pivot.drop(columns=[c for c in ["DEM","REP"] if c in pivot.columns], errors="ignore").sum(axis=1) if len(pivot.columns) else 0
        out = pd.DataFrame({
            "cntyvtd": pivot.index.astype("string"),
            "dem_votes": dem.to_numpy(),
            "rep_votes": rep.to_numpy(),
            "third_party_votes": other.to_numpy(),
        })
        out["total_votes"] = out["dem_votes"] + out["rep_votes"] + out["third_party_votes"]
        return out.reset_index(drop=True)

    # wide
    required = {"dem_votes","rep_votes","total_votes"}
    if not required.issubset(df.columns):
        raise ValueError(f"Wide election returns missing required columns: {sorted(required - set(df.columns))}")
    for c in ["dem_votes","rep_votes","third_party_votes","total_votes"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df[["cntyvtd","dem_votes","rep_votes","third_party_votes","total_votes"]].copy()
