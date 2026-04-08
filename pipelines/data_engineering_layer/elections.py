"""
elections.py
------------
Load, normalize, and pivot 2024 Texas election returns from three CSVs:
  - 2024 General Election Returns
  - 2024 Democratic Primary Election Returns
  - 2024 Republican Primary Election Returns

Offices retained (per slide): President, U.S. Senate ("U.S. Sen"), Railroad
Commissioner ("RR Comm 1" / "Railroad Commissioner").

Join key: cntyvtd = FIPS (zero-padded to 3 chars) + VTD (stripped).
This matches the CNTYVTD key rebuilt in spatial.py.

Output schema: one row per VTD with candidate vote columns named per the
Final Dataset Schema in the slides.
"""

import logging
import re
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Office filter: values we keep from the raw Office column
# ---------------------------------------------------------------------------
OFFICE_FILTER = {"President", "U.S. Sen", "RR Comm 1"}

# ---------------------------------------------------------------------------
# Candidate → final column name mapping
# General election (suffix _24G), Primary (suffix _24P + R/D party code)
#
# Format: (office_key, name_pattern, party, output_col)
#   name_pattern is a case-insensitive regex matched against the Name column.
# ---------------------------------------------------------------------------
GENERAL_CANDIDATE_MAP = [
    # Presidential
    ("President",   r"Trump",    "R", "TrumpR_24G"),
    ("President",   r"Harris",   "D", "HarrisD_24G"),
    # U.S. Senate
    ("U.S. Sen",    r"Cruz",     "R", "CruzR_24G"),
    ("U.S. Sen",    r"Allred",   "D", "AllredD_24G"),
    # Railroad Commissioner
    ("RR Comm 1",   r"Craddick", "R", "CraddickR_24G"),
    ("RR Comm 1",   r"Culbert",  "D", "CulbertD_24G"),
]

DEM_PRIMARY_MAP = [
    # Presidential
    ("President", r"Biden",      "D", "BidenD_24P"),
    ("President", r"Cornejo",    "D", "CornejoD_24P"),
    ("President", r"Locke",      "D", "LockeD_24P"),
    ("President", r"Lozada",     "D", "LozadaD_24P"),
    ("President", r"Perez",      "D", "PerezD_24P"),
    ("President", r"Phillips",   "D", "PhillipsD_24P"),
    ("President", r"Uygur",      "D", "UygurD_24P"),
    ("President", r"Williamson", "D", "WilliamsonD_24P"),
    # U.S. Senate
    ("U.S. Sen",  r"Allred",     "D", "AllredD_24P"),
    ("U.S. Sen",  r"Gomez",      "D", "GomezD_24P"),
    ("U.S. Sen",  r"Gonzalez",   "D", "GonzalezD_24P"),
    ("U.S. Sen",  r"Gutierrez",  "D", "GutierrezD_24P"),
    ("U.S. Sen",  r"Hassan",     "D", "HassanD_24P"),
    ("U.S. Sen",  r"Keough",     "D", "KeoughD_24P"),
    ("U.S. Sen",  r"Prilliman",  "D", "PrillimanD_24P"),
    ("U.S. Sen",  r"Sherman",    "D", "ShermanD_24P"),
    ("U.S. Sen",  r"Tchenko",    "D", "TchenkoD_24P"),
    # Railroad Commissioner
    ("RR Comm 1", r"Burch",      "D", "BurchD_24P"),
    ("RR Comm 1", r"Culbert",    "D", "CulbertD_24P"),
    ("RR Comm 1", r"Jones",      "D", "JonesD_24P"),
    ("RR Comm 1", r"Sarosdy",    "D", "SarosdyD_24P"),
    ("RR Comm 1", r"Goldstein",  "D", "GoldsteinD_24P"),
]

REP_PRIMARY_MAP = [
    # Presidential
    ("President", r"Binkley",      "R", "BinkleyR_24P"),
    ("President", r"Haley",        "R", "HaleyR_24P"),
    ("President", r"Stuckenberg",  "R", "StuckenbergR_24P"),
    ("President", r"Trump",        "R", "TrumpR_24P"),
    ("President", r"Christie",     "R", "ChristieR_24P"),
    ("President", r"Ramaswamy",    "R", "RamaswamyR_24P"),
    ("President", r"Hutchinson",   "R", "HutchinsonR_24P"),
    ("President", r"DeSantis",     "R", "DeSantisR_24P"),
    ("President", r"Uncommitted",  "R", "UncommittedR_24P"),
    # U.S. Senate
    ("U.S. Sen",  r"Cruz",         "R", "CruzR_24P"),
    ("U.S. Sen",  r"Gibson",       "R", "GibsonR_24P"),
    ("U.S. Sen",  r"Lopez",        "R", "LopezR_24P"),
    # Railroad Commissioner
    ("RR Comm 1", r"Clark",        "R", "ClarkR_24P"),
    ("RR Comm 1", r"Craddick",     "R", "CraddickR_24P"),
    ("RR Comm 1", r"Howell",       "R", "HowellR_24P"),
    ("RR Comm 1", r"Matlock",      "R", "MatlockR_24P"),
    ("RR Comm 1", r"Reyes",        "R", "ReyesR_24P"),
]

# ---------------------------------------------------------------------------
# Total vote columns: office + stage + party
# ---------------------------------------------------------------------------
TOTALS_SPEC = {
    # (output_col, candidate_cols_to_sum)
    "TOTVOTE_PRES_24G":  ["TrumpR_24G", "HarrisD_24G"],
    "TOTVOTE_SEN_24G":   ["CruzR_24G", "AllredD_24G"],
    "TOTVOTE_RRC_24G":   ["CraddickR_24G", "CulbertD_24G"],
    "TOTVOTE_PRESR_24P": ["BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P",
                          "TrumpR_24P", "ChristieR_24P", "RamaswamyR_24P",
                          "HutchinsonR_24P", "DeSantisR_24P", "UncommittedR_24P"],
    "TOTVOTE_PRESD_24P": ["BidenD_24P", "CornejoD_24P", "LockeD_24P",
                          "LozadaD_24P", "PerezD_24P", "PhillipsD_24P",
                          "UygurD_24P", "WilliamsonD_24P"],
    "TOTVOTE_SENR_24P":  ["CruzR_24P", "GibsonR_24P", "LopezR_24P"],
    "TOTVOTE_SEND_24P":  ["AllredD_24P", "GomezD_24P", "GonzalezD_24P",
                          "GutierrezD_24P", "HassanD_24P", "KeoughD_24P",
                          "PrillimanD_24P", "ShermanD_24P", "TchenkoD_24P"],
    "TOTVOTE_RRCR_24P":  ["ClarkR_24P", "CraddickR_24P", "HowellR_24P",
                          "MatlockR_24P", "ReyesR_24P"],
    "TOTVOTE_RRCD_24P":  ["BurchD_24P", "CulbertD_24P", "JonesD_24P",
                          "SarosdyD_24P", "GoldsteinD_24P"],
}


def _normalize_cntyvtd(df: pd.DataFrame) -> pd.DataFrame:
    """Build the cntyvtd join key from FIPS + VTD, matching the VTD shapefile."""
    df = df.copy()
    df["FIPS"] = df["FIPS"].astype(str).str.strip().str.zfill(3)
    df["VTD"]  = df["VTD"].astype(str).str.strip()
    df["CNTYVTD"] = df["FIPS"] + df["VTD"]
    return df


def _pivot_election(
    df: pd.DataFrame,
    candidate_map: list,
    label: str,
) -> pd.DataFrame:
    """
    Pivot a long-form election CSV to wide format (one row per CNTYVTD,
    one column per candidate) using the candidate_map definitions.

    Handles:
    - Office filter (keeps only relevant offices)
    - Duplicate precinct entries (summed)
    - Missing precincts (filled with 0)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normalize column names case-insensitively
    col_map = {c.lower(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in col_map.items()})

    required = {"office", "name", "party", "votes", "fips", "vtd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: Missing columns {missing}. Found: {list(df.columns)}")

    df = _normalize_cntyvtd(df)

    # Filter to relevant offices
    df = df[df["office"].isin(OFFICE_FILTER)].copy()

    wide_frames = []
    for (office, name_pat, party, out_col) in candidate_map:
        mask = (
            df["office"].str.strip().eq(office)
            & df["name"].str.contains(name_pat, case=False, na=False)
            & df["party"].str.strip().str.upper().eq(party)
        )
        subset = df[mask].groupby("CNTYVTD")["votes"].sum().rename(out_col)
        wide_frames.append(subset)

    if not wide_frames:
        log.warning(f"  {label}: No candidates matched — check office/name filters")
        return pd.DataFrame(columns=["CNTYVTD"])

    wide = pd.concat(wide_frames, axis=1).reset_index()

    n_unmatched = df["CNTYVTD"].nunique() - len(wide)
    if n_unmatched > 0:
        log.warning(f"  {label}: {n_unmatched} unmatched precincts (will be 0-filled)")

    log.info(f"  {label}: {len(wide):,} precincts × {len(wide.columns)-1} candidate columns")
    return wide


def load_election_data(
    gen_df: pd.DataFrame,
    dem_df: pd.DataFrame,
    rep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load and merge all election returns into a single wide DataFrame keyed
    on CNTYVTD, with one column per candidate per election.

    Returns
    -------
    DataFrame with CNTYVTD + all candidate vote columns + total vote columns.
    """
    log.info("  Pivoting 2024 general election returns …")
    gen_wide = _pivot_election(gen_df, GENERAL_CANDIDATE_MAP, "General 2024")

    log.info("  Pivoting 2024 Democratic primary returns …")
    dem_wide = _pivot_election(dem_df, DEM_PRIMARY_MAP, "Dem Primary 2024")

    log.info("  Pivoting 2024 Republican primary returns …")
    rep_wide = _pivot_election(rep_df, REP_PRIMARY_MAP, "Rep Primary 2024")

    # Merge all three on CNTYVTD
    merged = gen_wide.merge(dem_wide, on="CNTYVTD", how="outer")
    merged = merged.merge(rep_wide, on="CNTYVTD", how="outer")

    # All candidate columns — fill NaN with 0 and cast to int
    cand_cols = [c for c in merged.columns if c != "CNTYVTD"]
    merged[cand_cols] = merged[cand_cols].fillna(0).astype("int64")

    # Compute total vote columns
    for out_col, src_cols in TOTALS_SPEC.items():
        present = [c for c in src_cols if c in merged.columns]
        merged[out_col] = merged[present].sum(axis=1).astype("int64")

    log.info(
        f"  Elections merged: {len(merged):,} precincts  "
        f"{len(cand_cols)} candidate columns"
    )
    return merged
