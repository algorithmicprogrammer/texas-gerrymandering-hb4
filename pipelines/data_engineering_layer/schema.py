"""
schema.py
---------
Enforce the final dataset schema defined in the slides (sections 10-11).

Responsibilities
----------------
1. compute_derived_columns : add racial proportion columns
2. enforce_schema          : cast columns to correct types, order columns,
                             fill any remaining nulls with sentinels
"""

import logging
import geopandas as gpd
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Final schema: (column_name, dtype, fill_value)
# Order matches the slides.
# ---------------------------------------------------------------------------
SCHEMA = [
    # --- Identity / Geography ---
    ("CNTYVTD",             "str",     ""),
    ("geometry",            "geometry", None),
    ("CD",                  "Int64",   pd.NA),
    # --- Population ---
    ("TOTALPOP",            "float64", 0.0),    # note: filled from POP20 in census blocks
    # --- CVAP ---
    ("CVAP",                "float64", 0.0),
    ("HCVAP",               "float64", 0.0),
    ("WCVAP",               "float64", 0.0),
    ("BCVAP",               "float64", 0.0),
    ("ACVAP",               "float64", 0.0),
    ("AMINCVAP",            "float64", 0.0),
    # --- VAP ---
    ("BVAP",                "float64", 0.0),
    ("HVAP",                "float64", 0.0),
    ("WVAP",                "float64", 0.0),
    ("AVAP",                "float64", 0.0),
    ("AMINVAP",             "float64", 0.0),
    ("OVAP",                "float64", 0.0),
    ("VAP",                 "float64", 0.0),
    # --- 2024 General Election ---
    ("TrumpR_24G",          "int64",   0),
    ("HarrisD_24G",         "int64",   0),
    ("CruzR_24G",           "int64",   0),
    ("AllredD_24G",         "int64",   0),
    ("CraddickR_24G",       "int64",   0),
    ("CulbertD_24G",        "int64",   0),
    # --- 2024 Republican Presidential Primary ---
    ("BinkleyR_24P",        "int64",   0),
    ("HaleyR_24P",          "int64",   0),
    ("StuckenbergR_24P",    "int64",   0),
    ("TrumpR_24P",          "int64",   0),
    ("ChristieR_24P",       "int64",   0),
    ("RamaswamyR_24P",      "int64",   0),
    ("HutchinsonR_24P",     "int64",   0),
    ("DeSantisR_24P",       "int64",   0),
    ("UncommittedR_24P",    "int64",   0),
    # --- 2024 Democratic Presidential Primary ---
    ("BidenD_24P",          "int64",   0),
    ("CornejoD_24P",        "int64",   0),
    ("LockeD_24P",          "int64",   0),
    ("LozadaD_24P",         "int64",   0),
    ("PerezD_24P",          "int64",   0),
    ("PhillipsD_24P",       "int64",   0),
    ("UygurD_24P",          "int64",   0),
    ("WilliamsonD_24P",     "int64",   0),
    # --- 2024 Senate Primary ---
    ("CruzR_24P",           "int64",   0),
    ("GibsonR_24P",         "int64",   0),
    ("LopezR_24P",          "int64",   0),
    ("AllredD_24P",         "int64",   0),
    ("GomezD_24P",          "int64",   0),
    ("GonzalezD_24P",       "int64",   0),
    ("GutierrezD_24P",      "int64",   0),
    ("HassanD_24P",         "int64",   0),
    ("KeoughD_24P",         "int64",   0),
    ("PrillimanD_24P",      "int64",   0),
    ("ShermanD_24P",        "int64",   0),
    ("TchenkoD_24P",        "int64",   0),
    # --- 2024 Railroad Commissioner Primary ---
    ("ClarkR_24P",          "int64",   0),
    ("CraddickR_24P",       "int64",   0),
    ("HowellR_24P",         "int64",   0),
    ("MatlockR_24P",        "int64",   0),
    ("ReyesR_24P",          "int64",   0),
    ("BurchD_24P",          "int64",   0),
    ("CulbertD_24P",        "int64",   0),
    # --- Total vote columns ---
    ("TOTVOTE_PRES_24G",    "int64",   0),
    ("TOTVOTE_SEN_24G",     "int64",   0),
    ("TOTVOTE_RRC_24G",     "int64",   0),
    ("TOTVOTE_PRESR_24P",   "int64",   0),
    ("TOTVOTE_PRESD_24P",   "int64",   0),
    ("TOTVOTE_SENR_24P",    "int64",   0),
    ("TOTVOTE_SEND_24P",    "int64",   0),
    ("TOTVOTE_RRCR_24P",    "int64",   0),
    ("TOTVOTE_RRCD_24P",    "int64",   0),
    # --- Derived proportions ---
    ("white_prop",          "float64", np.nan),
    ("black_prop",          "float64", np.nan),
    ("hisp_prop",           "float64", np.nan),
    ("asian_prop",          "float64", np.nan),
    ("amin_prop",           "float64", np.nan),
]


def compute_derived_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add racial proportion columns based on CVAP.

    Proportions are defined as group CVAP / total CVAP.
    Where CVAP == 0 the proportion is set to NaN (not 0) to distinguish
    "no eligible voters" from "undetermined".
    """
    df = gdf.copy()
    cvap = df["CVAP"].replace(0, np.nan)

    df["white_prop"] = df["WCVAP"]    / cvap
    df["black_prop"] = df["BCVAP"]    / cvap
    df["hisp_prop"]  = df["HCVAP"]    / cvap
    df["asian_prop"] = df["ACVAP"]    / cvap
    df["amin_prop"]  = df["AMINCVAP"] / cvap

    return df


def enforce_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Enforce final column order, types, and fill values per the schema above.

    Columns not in the schema are dropped (with a warning).
    Columns in the schema but missing from the data are added with fill values.
    """
    df = gdf.copy()

    # Warn about extra columns
    schema_cols = {s[0] for s in SCHEMA}
    extra = set(df.columns) - schema_cols
    if extra:
        log.warning(f"  Dropping {len(extra)} non-schema columns: {sorted(extra)}")
        df = df.drop(columns=list(extra))

    # Add missing columns with fill values
    for col, dtype, fill in SCHEMA:
        if col == "geometry":
            continue
        if col not in df.columns:
            log.warning(f"  Schema column '{col}' missing — filling with {fill!r}")
            df[col] = fill

    # Cast types
    for col, dtype, fill in SCHEMA:
        if col == "geometry":
            continue
        if dtype == "str":
            df[col] = df[col].astype(str)
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Reorder columns: schema order, geometry last for GeoDataFrame convention
    ordered_cols = [s[0] for s in SCHEMA if s[0] != "geometry" and s[0] in df.columns]
    result = gpd.GeoDataFrame(df[ordered_cols], geometry=df["geometry"], crs=df.crs)

    log.info(f"  Schema enforced: {len(result):,} rows × {len(result.columns)} columns")
    return result
