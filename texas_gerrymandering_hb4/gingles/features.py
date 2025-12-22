import numpy as np
import pandas as pd
from typing import List, Optional

def to_two_party_share(df: pd.DataFrame, dem_col: str, rep_col: str, out_col: str) -> pd.DataFrame:
    dem = df[dem_col].astype(float)
    rep = df[rep_col].astype(float)
    s = dem + rep
    df[out_col] = np.where(s > 0, dem / s, np.nan)
    return df

def normalize_race_props(df: pd.DataFrame, race_cols: List[str], tol: float = 1e-6) -> pd.DataFrame:
    s = df[race_cols].sum(axis=1)
    df[race_cols] = np.where(
        (s.values[:, None] > tol),
        df[race_cols].values / s.values[:, None],
        df[race_cols].values
    )
    return df

def gini(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def build_vtd_units(
    df_raw: pd.DataFrame,
    vtd_id_col: str,
    cd_col: str,
    pop_col: str,
    dem_col: str,
    rep_col: str,
    dem_share_2p_col: str,
    race_cols: List[str],
) -> pd.DataFrame:
    df = df_raw.copy()

    for c in [vtd_id_col, cd_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Pop weights: strongly recommended, but fallback to 1
    if pop_col not in df.columns:
        df[pop_col] = 1.0

    # Two-party share
    if dem_share_2p_col not in df.columns:
        if dem_col not in df.columns or rep_col not in df.columns:
            raise ValueError(f"Need either {dem_share_2p_col} OR both {dem_col} and {rep_col}.")
        df = to_two_party_share(df, dem_col=dem_col, rep_col=rep_col, out_col=dem_share_2p_col)

    # Race props
    missing = [c for c in race_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing race proportion columns: {missing}")

    df = normalize_race_props(df, race_cols=race_cols)

    # Keep compactness cols if present
    compactness_cols = [c for c in df.columns if c.startswith("compactness_")]

    keep = [vtd_id_col, cd_col, pop_col, dem_share_2p_col] + race_cols + compactness_cols
    return df[keep].copy()

def build_elections_long(vtd_units: pd.DataFrame, vtd_id_col: str, dem_share_2p_col: str, election_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        "election_id": election_id,
        vtd_id_col: vtd_units[vtd_id_col].values,
        "dem_share_2p": vtd_units[dem_share_2p_col].values,
        "rep_share_2p": 1.0 - vtd_units[dem_share_2p_col].values,
    })

def aggregate_to_cd(
    vtd_units: pd.DataFrame,
    vtd_id_col: str,
    cd_col: str,
    pop_col: str,
    race_cols: List[str],
    minority: str = "black",
    coalition: bool = False,
) -> pd.DataFrame:
    df = vtd_units.copy()

    # minority share at unit level
    if coalition:
        df["p_minority"] = df["p_black"] + df["p_latino"]
    else:
        key = f"p_{minority}"
        if key not in df.columns:
            raise ValueError(f"Minority column not found: {key}")
        df["p_minority"] = df[key]

    def wavg(g: pd.DataFrame, col: str) -> float:
        w = g[pop_col].values.astype(float)
        x = g[col].values.astype(float)
        return float(np.average(x, weights=w)) if np.sum(w) > 0 else np.nan

    cd = df.groupby(cd_col, as_index=False).apply(
        lambda g: pd.Series({
            "pop_total": g[pop_col].sum(),
            **{c: wavg(g, c) for c in race_cols},
            "minority_share": wavg(g, "p_minority"),
        })
    ).reset_index(drop=True).rename(columns={cd_col: "cd"})

    disp = df.groupby(cd_col).apply(lambda g: pd.Series({
        "minority_dispersion_var": float(np.nanvar(g["p_minority"].values)),
        "minority_dispersion_gini": float(gini(g["p_minority"].values)),
        "minority_top10_mean": (
            float(np.nanmean(np.sort(g["p_minority"].values)[int(0.9*len(g)):]))
            if len(g) >= 10 else np.nan
        ),
    })).reset_index().rename(columns={cd_col: "cd"})

    cd = cd.merge(disp, on="cd", how="left")

    # compactness: dedupe per CD if repeated on each unit row
    compactness_cols = [c for c in df.columns if c.startswith("compactness_")]
    if compactness_cols:
        comp = df[[cd_col] + compactness_cols].drop_duplicates(cd_col).rename(columns={cd_col: "cd"})
        cd = cd.merge(comp, on="cd", how="left")

    return cd
