"""
ei_export.py
------------
Export the EI-ready CSV from the final dataset.

The CSV is a flat, geometry-free version of tx_precincts_final.parquet
containing only the columns that run_ei.R needs:

  - CNTYVTD                       (join key)
  - CVAP                          (total eligible voters, used to weight EI)
  - hisp_prop, black_prop,        (group fractions — must sum to ~1 per row)
    white_prop, other_prop
  - All candidate vote count cols (int)
  - All TOTVOTE_* cols            (int)

Notes
-----
- Rows where CVAP == 0 are kept; run_ei.R filters them out itself so that
  the CNTYVTD index stays consistent with the full precinct file.
- other_prop is derived here as 1 - (hisp + black + white + asian + amin)
  clipped to [0, 1].  This matches the eiPack convention that group
  fractions must sum to 1 within each precinct.
- UncommittedR_24P is excluded from the CSV.  It has no racial identity
  and is omitted from run_ei.R's PRESR_24P election definition.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger as log

# ---------------------------------------------------------------------------
# Candidate vote columns to include in the EI CSV (no geometry, no UncommittedR)
# ---------------------------------------------------------------------------
GENERAL_VOTE_COLS = [
    "TrumpR_24G", "HarrisD_24G",
    "CruzR_24G",  "AllredD_24G",
    "CraddickR_24G", "CulbertD_24G",
]

DEM_PRIMARY_VOTE_COLS = [
    "BidenD_24P", "CornejoD_24P", "LockeD_24P",  "LozadaD_24P",
    "PerezD_24P", "PhillipsD_24P", "UygurD_24P", "WilliamsonD_24P",
    "AllredD_24P", "GomezD_24P",  "GonzalezD_24P", "GutierrezD_24P",
    "HassanD_24P", "KeoughD_24P", "PrillimanD_24P", "ShermanD_24P",
    "TchenkoD_24P",
    "BurchD_24P", "CulbertD_24P",
]

REP_PRIMARY_VOTE_COLS = [
    # UncommittedR_24P deliberately excluded
    "BinkleyR_24P", "HaleyR_24P",     "StuckenbergR_24P", "TrumpR_24P",
    "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P",
    "CruzR_24P", "GibsonR_24P", "LopezR_24P",
    "ClarkR_24P", "CraddickR_24P", "HowellR_24P", "MatlockR_24P", "ReyesR_24P",
]

TOTVOTE_COLS = [
    "TOTVOTE_PRES_24G",
    "TOTVOTE_SEN_24G",
    "TOTVOTE_RRC_24G",
    "TOTVOTE_PRESD_24P",
    "TOTVOTE_PRESR_24P",   # Note: does NOT include UncommittedR (see below)
    "TOTVOTE_SEND_24P",
    "TOTVOTE_SENR_24P",
    "TOTVOTE_RRCD_24P",
    "TOTVOTE_RRCR_24P",
]

ALL_VOTE_COLS = GENERAL_VOTE_COLS + DEM_PRIMARY_VOTE_COLS + REP_PRIMARY_VOTE_COLS


def export_for_ei(
    final: gpd.GeoDataFrame,
    out_dir: Path,
) -> Path:
    """
    Export the EI-ready CSV to ``out_dir / tx_precincts_for_ei.csv``.

    Parameters
    ----------
    final   : the fully validated, schema-enforced GeoDataFrame from pipeline
    out_dir : directory to write the CSV (same as PROCESSED_DATA_DIR)

    Returns
    -------
    Path to the written CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tx_precincts_for_ei.csv"

    df = pd.DataFrame(final)   # drop geometry

    # ── Build other_prop ────────────────────────────────────────────────────
    # other_prop = 1 - hisp - black - white - asian - amin, clipped to [0,1]
    # This ensures the four group fractions passed to eiPack sum to 1.
    df["other_prop"] = (
        1.0
        - df["hisp_prop"].fillna(0)
        - df["black_prop"].fillna(0)
        - df["white_prop"].fillna(0)
        - df["asian_prop"].fillna(0)
        - df["amin_prop"].fillna(0)
    ).clip(lower=0.0, upper=1.0)

    # ── Recompute TOTVOTE_PRESR_24P without UncommittedR ────────────────────
    # The schema's TOTVOTE_PRESR_24P includes UncommittedR_24P.  For EI we
    # need a total that matches exactly the candidates passed to ei.MD.bayes.
    df["TOTVOTE_PRESR_24P"] = df[[
        "BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
        "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P",
    ]].sum(axis=1).astype("int64")

    # ── Select and order columns ─────────────────────────────────────────────
    ei_cols = (
        ["CNTYVTD", "CVAP",
         "hisp_prop", "black_prop", "white_prop", "other_prop"]
        + ALL_VOTE_COLS
        + TOTVOTE_COLS
    )

    # Sanity-check all expected columns are present
    missing = [c for c in ei_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"export_for_ei: columns missing from final dataset: {missing}"
        )

    ei_df = df[ei_cols].copy()

    # ── Validation ───────────────────────────────────────────────────────────
    n_rows       = len(ei_df)
    n_zero_cvap  = (ei_df["CVAP"] == 0).sum()
    prop_cols    = ["hisp_prop", "black_prop", "white_prop", "other_prop"]
    prop_sum     = ei_df[prop_cols].sum(axis=1)
    n_bad_sum    = ((prop_sum - 1.0).abs() > 0.01).sum()

    log.info(f"  EI CSV: {n_rows:,} precincts")
    log.info(f"  EI CSV: {n_zero_cvap:,} precincts with CVAP=0 (kept; R filters them)")
    if n_bad_sum > 0:
        log.warning(
            f"  EI CSV: {n_bad_sum} precincts where group proportions "
            f"don't sum to 1 ± 0.01 — check CVAP discounting"
        )
    else:
        log.info("  EI CSV: group proportions sum check passed (all within 0.01 of 1)")

    ei_df.to_csv(out_path, index=False)
    log.info(f"  EI CSV written → {out_path}")
    return out_path
