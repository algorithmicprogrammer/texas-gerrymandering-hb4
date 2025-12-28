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


def _maybe_filter_office(df: pd.DataFrame, office_filter: str | None) -> pd.DataFrame:
    """Filter tall election returns to a single contest if an office column exists.

    Many Texas election files are *candidate-level* with an `Office` column. If you
    don't filter, you'll accidentally sum across multiple contests.
    """
    if office_filter is None:
        # If office exists and there are multiple contests, fail loudly.
        if "office" in df.columns:
            uniq = df["office"].dropna().astype(str).str.strip().unique()
            uniq = [u for u in uniq if u]
            if len(uniq) > 1:
                raise ValueError(
                    "Election returns contain multiple Office values, but no office_filter was provided. "
                    f"Provide --elections-office-filter. Found examples: {uniq[:10]}"
                )
        return df

    if "office" not in df.columns:
        # Nothing to filter on.
        return df

    off = df["office"].astype(str).str.strip().str.casefold()
    target = str(office_filter).strip().casefold()
    out = df.loc[off == target].copy()
    if out.empty:
        examples = df["office"].dropna().astype(str).str.strip().unique().tolist()
        raise ValueError(
            f"No rows matched office_filter={office_filter!r}. "
            f"Office examples in file: {examples[:15]}"
        )
    return out

def clean_vtd_election_returns(df: pd.DataFrame, office_filter: str | None = None) -> pd.DataFrame:
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

    # If Office exists, filter to a single contest (or error if ambiguous)
    df = _maybe_filter_office(df, office_filter)

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
