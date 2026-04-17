"""
ei_export.py
------------
Export the EI-ready CSV (and updated parquet) from the final dataset.

The CSV is a flat, geometry-free version of tx_precincts_final.parquet
containing only the columns that run_ei.py needs:

  - CNTYVTD                                      (join key)
  - CVAP                                         (total eligible voters, used to weight EI)
  - hisp_prop, black_prop, white_prop,           (group fractions, passed through as-is
    asian_prop, amin_prop                         from the parquet schema)
  - All candidate vote count cols (int)
  - All TOTVOTE_* cols            (int)
  - Abstain cols (one per election)              (precomputed as max(CVAP - TOTVOTE_*, 0))

Abstain columns
---------------
Each abstain column is named ``<election_key>_abstain`` and is computed as::

    abstain = max(CVAP - TOTVOTE_<election>, 0)

This matches the R script convention (X18R_Governor_abstain) and is the
defensible approach: the formula is explicit, auditable in the CSV before
any model runs, and consistent with a single CVAP denominator throughout.

When turnout exceeds the ACS CVAP estimate the column floors at 0 rather
than going negative.  run_ei.py uses these columns directly and no longer
needs to construct abstain at runtime.

The abstain columns are also written back into
``tx_precincts_final.parquet`` so that downstream consumers share one
canonical version of these derived values.

Notes
-----
- Rows where CVAP == 0 are kept; run_ei.py filters them out itself so that
  the CNTYVTD index stays consistent with the full precinct file.
- Proportion columns (hisp_prop, black_prop, white_prop, asian_prop, amin_prop)
  are passed through directly from the parquet schema without collapsing or
  deriving other_prop.
- UncommittedR_24P is excluded from the CSV.  It has no racial identity
  and is omitted from the PRESR_24P election definition.
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

# ---------------------------------------------------------------------------
# Abstain columns: one per EI election, named <election_key>_abstain.
# Each is computed as max(CVAP - TOTVOTE_<election>, 0).
# The TOTVOTE column used here must exactly match the candidates passed to
# RowByColumnEI — i.e. TOTVOTE_PRESR_24P excludes UncommittedR (recomputed
# in export_for_ei below).
# ---------------------------------------------------------------------------
ABSTAIN_SPEC = {
    # abstain_col_name         : totvote_col_to_subtract_from_CVAP
    "24G_President_abstain"    : "TOTVOTE_PRES_24G",
    "24P_PresD_abstain"        : "TOTVOTE_PRESD_24P",
    "24P_PresR_abstain"        : "TOTVOTE_PRESR_24P",   # recomputed without UncommittedR
    "24G_US_Sen_abstain"       : "TOTVOTE_SEN_24G",
    "24P_SenD_abstain"         : "TOTVOTE_SEND_24P",
    "24P_SenR_abstain"         : "TOTVOTE_SENR_24P",
    "24G_RR_Comm_1_abstain"    : "TOTVOTE_RRC_24G",
    "24P_RRCD_abstain"         : "TOTVOTE_RRCD_24P",
    "24P_RRCR_abstain"         : "TOTVOTE_RRCR_24P",
}

ABSTAIN_COLS = list(ABSTAIN_SPEC.keys())


def export_for_ei(
    final: gpd.GeoDataFrame,
    out_dir: Path,
    parquet_path: Path | None = None,
) -> Path:
    """
    Export the EI-ready CSV to ``out_dir / tx_cvap_for_ei.csv`` and write
    abstain columns back into the parquet at ``parquet_path``.

    Abstain columns
    ~~~~~~~~~~~~~~~
    For each election the abstain count is precomputed here as::

        abstain = max(CVAP - TOTVOTE_<election>, 0)

    This matches the convention in the original R script (where abstain was
    a pre-baked CSV column) and is more defensible than computing it at
    runtime in run_ei.py: the formula is explicit, auditable in the CSV
    before any MCMC runs, and always uses CVAP as the single denominator.

    When observed turnout exceeds the ACS CVAP estimate the column floors
    at 0.  run_ei.py should use these columns directly and no longer needs
    to construct abstain from scratch.

    Parameters
    ----------
    final        : the fully validated, schema-enforced GeoDataFrame from pipeline
    out_dir      : directory to write the CSV (same as PROCESSED_DATA_DIR)
    parquet_path : path to tx_precincts_final.parquet; if provided, abstain
                   columns are written back so the parquet stays canonical.
                   Defaults to ``out_dir / tx_precincts_final.parquet``.

    Returns
    -------
    Path to the written CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tx_cvap_for_ei.csv"

    if parquet_path is None:
        parquet_path = out_dir / "tx_precincts_final.parquet"

    df = pd.DataFrame(final)   # drop geometry

    # ── Recompute TOTVOTE_PRESR_24P without UncommittedR ────────────────────
    # The schema's TOTVOTE_PRESR_24P includes UncommittedR_24P.  For EI we
    # need a total that matches exactly the candidates passed to RowByColumnEI.
    df["TOTVOTE_PRESR_24P"] = df[[
        "BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
        "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P",
    ]].sum(axis=1).astype("int64")

    # ── Precompute abstain columns ───────────────────────────────────────────
    # abstain = max(CVAP - total_votes_cast_in_election, 0)
    # Floored at 0 to handle precincts where turnout exceeds ACS CVAP estimate.
    cvap = df["CVAP"].values.astype(float)
    n_floored_total = 0

    for abstain_col, totvote_col in ABSTAIN_SPEC.items():
        totvotes = df[totvote_col].values.astype(float)
        raw      = cvap - totvotes
        floored  = np.maximum(raw, 0.0)
        n_floored = int((raw < 0).sum())
        if n_floored > 0:
            log.warning(
                f"  abstain/{abstain_col}: {n_floored} precincts where turnout > CVAP "
                f"(floored to 0); max overage = {int(-raw[raw < 0].min())} votes"
            )
            n_floored_total += n_floored
        df[abstain_col] = floored.astype("int64")

    if n_floored_total == 0:
        log.info("  Abstain columns: no precincts with turnout > CVAP")

    # ── Select and order columns for CSV ─────────────────────────────────────
    ei_cols = (
        ["CNTYVTD", "CVAP",
         "hisp_prop", "black_prop", "white_prop", "asian_prop", "amin_prop"]
        + ALL_VOTE_COLS
        + TOTVOTE_COLS
        + ABSTAIN_COLS
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
    prop_cols    = ["hisp_prop", "black_prop", "white_prop", "asian_prop", "amin_prop"]
    nonzero_cvap = ei_df["CVAP"] > 0
    prop_sum     = ei_df.loc[nonzero_cvap, prop_cols].sum(axis=1)
    n_bad_sum    = ((prop_sum - 1.0).abs() > 0.01).sum()

    log.info(f"  EI CSV: {n_rows:,} precincts")
    log.info(f"  EI CSV: {n_zero_cvap:,} precincts with CVAP=0 (kept; run_ei.py filters them)")
    if n_bad_sum > 0:
        log.warning(
            f"  EI CSV: {n_bad_sum} non-zero-CVAP precincts where group proportions "
            f"don't sum to 1 ± 0.01 — check CVAP discounting"
        )
    else:
        log.info("  EI CSV: group proportions sum check passed (all within 0.01 of 1)")

    ei_df.to_csv(out_path, index=False)
    log.info(f"  EI CSV written → {out_path}")

    # ── Write abstain columns back into the parquet ──────────────────────────
    # Read the existing parquet, add/overwrite the abstain columns, re-write.
    # This keeps tx_precincts_final.parquet as the single canonical source for
    # all derived columns so downstream consumers don't have to recompute them.
    if parquet_path.exists():
        pq_df = gpd.read_parquet(parquet_path)
        for abstain_col in ABSTAIN_COLS:
            pq_df[abstain_col] = df.set_index("CNTYVTD")[abstain_col].reindex(
                pq_df["CNTYVTD"]
            ).values
        pq_df.to_parquet(parquet_path, index=False)
        log.info(
            f"  Parquet updated with {len(ABSTAIN_COLS)} abstain columns → {parquet_path}"
        )
    else:
        log.warning(
            f"  Parquet not found at {parquet_path}; skipping parquet update. "
            f"Abstain columns are still present in the CSV."
        )

    return out_path
