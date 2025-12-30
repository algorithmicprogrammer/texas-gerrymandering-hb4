from __future__ import annotations

import pandas as pd
import numpy as np


def _normalize_party(p: str) -> str:
    """Map raw party labels to DEM/REP/OTHER robustly."""
    if not isinstance(p, str):
        return "OTHER"
    p = p.strip().upper()

    if p in {"D", "DEM", "DEMOCRAT", "DEMOCRATIC", "DFL"}:
        return "DEM"
    if p in {"R", "REP", "REPUBLICAN", "GOP"}:
        return "REP"
    return "OTHER"


def is_tall_elections(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"party", "votes"}.issubset(cols)


def _maybe_filter_office(df: pd.DataFrame, office_filter: str | None) -> pd.DataFrame:
    """Filter tall election returns to a single contest if an office column exists."""
    if office_filter is None:
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


def clean_vtd_election_returns(
    df: pd.DataFrame,
    office_filter: str | None = None,
    prefer_key: str = "vtdkey",
) -> pd.DataFrame:
    """
    Standardize election returns to a wide per-VTD table.

    Accepts either:
      * Tall format with columns:
          - vtdkeyvalue OR cntyvtd  (key)
          - party
          - votes
          - (optional) office
      * Wide format already:
          - vtdkey OR cntyvtd
          - dem_votes, rep_votes, third_party_votes (optional), total_votes

    Returns wide standardized columns:
      - vtdkey (Int64)   [preferred join key]
      - cntyvtd (string) [optional, only if that was the only key available]
      - dem_votes, rep_votes, third_party_votes, total_votes
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # If Office exists, filter to a single contest (or error if ambiguous)
    df = _maybe_filter_office(df, office_filter)

    # Determine which key column we have
    has_vtdkeyvalue = "vtdkeyvalue" in df.columns
    has_cntyvtd = "cntyvtd" in df.columns

    if prefer_key == "vtdkey" and has_vtdkeyvalue:
        key_mode = "vtdkey"
        df["vtdkey"] = pd.to_numeric(df["vtdkeyvalue"], errors="coerce").astype("Int64")
    elif has_cntyvtd:
        key_mode = "cntyvtd"
        df["cntyvtd"] = df["cntyvtd"].astype("string").str.strip()
    elif has_vtdkeyvalue:
        key_mode = "vtdkey"
        df["vtdkey"] = pd.to_numeric(df["vtdkeyvalue"], errors="coerce").astype("Int64")
    else:
        raise ValueError("Election returns must include either vtdkeyvalue or cntyvtd.")

    # ----------------------------
    # Tall -> wide
    # ----------------------------
    if is_tall_elections(df):
        df["party"] = df["party"].map(_normalize_party)
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int64")

        if key_mode == "vtdkey":
            idx = ["vtdkey"]
        else:
            idx = ["cntyvtd"]

        pivot = df.pivot_table(index=idx, columns="party", values="votes", aggfunc="sum", fill_value=0)

        dem = pivot["DEM"] if "DEM" in pivot.columns else 0
        rep = pivot["REP"] if "REP" in pivot.columns else 0

        if isinstance(dem, int):
            dem_arr = np.zeros(len(pivot), dtype="int64")
        else:
            dem_arr = dem.to_numpy(dtype="int64")

        if isinstance(rep, int):
            rep_arr = np.zeros(len(pivot), dtype="int64")
        else:
            rep_arr = rep.to_numpy(dtype="int64")

        if len(pivot.columns) == 0:
            other_arr = np.zeros(len(pivot), dtype="int64")
        else:
            other_cols = [c for c in pivot.columns if c not in {"DEM", "REP"}]
            other_arr = pivot[other_cols].sum(axis=1).to_numpy(dtype="int64") if other_cols else np.zeros(len(pivot), dtype="int64")

        out = pd.DataFrame({
            idx[0]: pivot.index,
            "dem_votes": dem_arr,
            "rep_votes": rep_arr,
            "third_party_votes": other_arr,
        })
        out["total_votes"] = out["dem_votes"] + out["rep_votes"] + out["third_party_votes"]
        return out.reset_index(drop=True)

    # ----------------------------
    # Wide format
    # ----------------------------
    required = {"dem_votes", "rep_votes", "total_votes"}
    if not required.issubset(df.columns):
        raise ValueError(f"Wide election returns missing required columns: {sorted(required - set(df.columns))}")

    for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    keep = []
    if "vtdkey" in df.columns:
        df["vtdkey"] = pd.to_numeric(df["vtdkey"], errors="coerce").astype("Int64")
        keep.append("vtdkey")
    if "cntyvtd" in df.columns:
        df["cntyvtd"] = df["cntyvtd"].astype("string").str.strip()
        keep.append("cntyvtd")

    if not keep:
        raise ValueError("Wide election returns must include vtdkey or cntyvtd.")

    return df[keep + ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]].copy()

