"""
post_ei_processing.py
-----------------------
Reshapes EI outputs from run_ei.py into the formats expected by
run_functions.py / TX_elections_model.py.

Run this after run_ei.py finishes:
    python post_ei_processing.py

Inputs  (from ei_outputs/):
    statewide_rxc_EI_preferences.csv
    prec_count_quants.csv
    mean_prec_vote_counts.csv          (optional — only needed for district mode)

Outputs (written to recom_inputs/):
    statewide_rxc_EI_preferences.csv   — renamed/remapped columns
    prec_count_quants.csv              — pivoted to wide composite-column format
    mean_prec_vote_counts.csv          — pivoted to wide composite-column format
    ingroup_weight.csv                 — derived W2 weights

Edit GROUP_MAP and WEIGHT_VALUES below to match your setup.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your run_ei.py GROUP_LABELS and desired weights
# ---------------------------------------------------------------------------

# Maps your EI group labels -> the CVAP column names the chain filters on.
# Only groups the chain cares about need to be listed; others are ignored.
GROUP_MAP = {
    "Hispanic": "HCVAP",
    "Black":    "BCVAP",
    # Add others if your run_functions.py version uses them:
    # "White":  "WCVAP",
    # "Asian":  "ACVAP",
    # "AMIN":   "AMINCVAP",
}

# W2 ingroup weights written to ingroup_weight.csv.
# "Relevant Minority" = preferred candidate IS from the group
# "Other"             = preferred candidate is NOT from any minority group
# "Partial "          = one of {Black, Hispanic} but not both  (note trailing space)
WEIGHT_VALUES = {
    "Relevant Minority": 1.0,
    "Other":             0.5,
    "Partial ":          0.75,   # trailing space intentional — matches run_functions.py key
}

# Octile probability -> integer label used in the wide column names.
# run_functions.py uses 0,125,250,375,500,625,750,875,1000 (per-mille, i.e. *1000).
OCTILE_PERMILLE = {
    0.125: 125,
    0.250: 250,
    0.375: 375,
    0.500: 500,
    0.625: 625,
    0.750: 750,
    0.875: 875,
    1.000: 1000,
}

IN_DIR  = "ei_outputs"
OUT_DIR = "recom_inputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. statewide_rxc_EI_preferences.csv
# ---------------------------------------------------------------------------
# run_functions.py queries: EI_statewide["Election"], ["Demog"], ["Candidate"], ["prob"]
# your EI produces:         election, group, candidate_of_choice, frac_draws_preferred

print("Reshaping statewide_rxc_EI_preferences.csv ...")

sw = pd.read_csv(os.path.join(IN_DIR, "statewide_rxc_EI_preferences.csv"))

# Keep only groups the chain needs
sw = sw[sw["group"].isin(GROUP_MAP.keys())].copy()

sw_out = pd.DataFrame({
    "Election":  sw["election"],
    "Demog":     sw["group"].map(GROUP_MAP),
    "Candidate": sw["candidate_of_choice"],
    "prob":      sw["frac_draws_preferred"],
})

sw_out.to_csv(os.path.join(OUT_DIR, "statewide_rxc_EI_preferences.csv"), index=False)
print(f"  Written: {len(sw_out)} rows")

# ---------------------------------------------------------------------------
# 2. prec_count_quants.csv  (long -> wide)
# ---------------------------------------------------------------------------
# run_functions.py column naming pattern:
#   {CVAP_KEY}.{election}_{candidate}.{permille}
# e.g.  BCVAP.24P_SenD_AllredD_24P.125
#
# The leading ".0" column (the intercept / baseline) is the minimum quantile
# anchor; run_functions.py accesses base + '.0' as the floor value.

print("Pivoting prec_count_quants.csv ...")

qdf = pd.read_csv(
    os.path.join(IN_DIR, "prec_count_quants.csv"),
    dtype={"CNTYVTD": str},
)

# Keep only groups that appear in GROUP_MAP
qdf = qdf[qdf["group"].isin(GROUP_MAP.keys())].copy()
qdf["cvap_key"] = qdf["group"].map(GROUP_MAP)

# Build composite base key: "{CVAP_KEY}.{election}_{candidate}"
qdf["base_col"] = qdf["cvap_key"] + "." + qdf["election"] + "_" + qdf["candidate"]

# Map quantile probability -> permille integer
# Round to 3 dp to avoid float comparison noise
qdf["permille"] = qdf["quantile"].round(3).map(
    {round(k, 3): v for k, v in OCTILE_PERMILLE.items()}
)

missing_octiles = qdf["permille"].isna().sum()
if missing_octiles > 0:
    print(f"  WARNING: {missing_octiles} rows had unrecognised quantile values — dropped.")
    qdf = qdf.dropna(subset=["permille"])

qdf["permille"] = qdf["permille"].astype(int)

# Pivot: one row per CNTYVTD, columns = base_col + "." + permille
qdf["full_col"] = qdf["base_col"] + "." + qdf["permille"].astype(str)

wide_q = qdf.pivot_table(
    index="CNTYVTD",
    columns="full_col",
    values="value",
    aggfunc="first",
).reset_index()
wide_q.columns.name = None

# Add the ".0" baseline column (the octile-0 floor = 0 by definition for count data)
# run_functions.py does:  dist_prec_quant[base + '.' + '0']  as the starting floor
base_cols = sorted(set(qdf["base_col"].unique()))
for bc in base_cols:
    col0 = bc + ".0"
    if col0 not in wide_q.columns:
        wide_q[col0] = 0.0

wide_q = wide_q.sort_index(axis=1)   # sort columns alphabetically for readability

wide_q.to_csv(os.path.join(OUT_DIR, "prec_count_quants.csv"), index=False)
print(f"  Written: {len(wide_q)} rows x {len(wide_q.columns)} columns")

# ---------------------------------------------------------------------------
# 3. mean_prec_vote_counts.csv  (long -> wide)
# ---------------------------------------------------------------------------
# compute_district_weights uses this via the same cand_pref_* helpers.
# Wide format: one row per CNTYVTD, columns = {CVAP_KEY}.{election}_{candidate}_counts

print("Pivoting mean_prec_vote_counts.csv ...")

mdf = pd.read_csv(
    os.path.join(IN_DIR, "mean_prec_vote_counts.csv"),
    dtype={"CNTYVTD": str},
)

mdf = mdf[mdf["group"].isin(GROUP_MAP.keys())].copy()
mdf["cvap_key"] = mdf["group"].map(GROUP_MAP)

# Column name: "{CVAP_KEY}.{election}_{candidate}_counts"
mdf["col_name"] = (
    mdf["cvap_key"] + "." + mdf["election"] + "_" + mdf["candidate"] + "_counts"
)

wide_m = mdf.pivot_table(
    index="CNTYVTD",
    columns="col_name",
    values="mean_count",
    aggfunc="first",
).reset_index()
wide_m.columns.name = None

wide_m.to_csv(os.path.join(OUT_DIR, "mean_prec_vote_counts.csv"), index=False)
print(f"  Written: {len(wide_m)} rows x {len(wide_m.columns)} columns")

# ---------------------------------------------------------------------------
# 4. ingroup_weight.csv
# ---------------------------------------------------------------------------
# compute_W2() reads this as:
#   min_cand_weights_dict = {key: min_cand_weights.to_dict()[key][0] for key ...}
# so it expects a single-row CSV where each column is a weight category.

print("Writing ingroup_weight.csv ...")

iw = pd.DataFrame([WEIGHT_VALUES])
iw.to_csv(os.path.join(OUT_DIR, "ingroup_weight.csv"), index=False)
print(f"  Written with keys: {list(iw.columns)}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
=== Done ===
Files written to {OUT_DIR}/:
  statewide_rxc_EI_preferences.csv   columns: Election, Demog, Candidate, prob
  prec_count_quants.csv              wide format, {len(wide_q.columns)-1} data columns
  mean_prec_vote_counts.csv          wide format, {len(wide_m.columns)-1} data columns
  ingroup_weight.csv                 W2 weights

Next steps:
  1. Copy these files (and your existing TX_logit_params.csv) into your
     TX_elections_model.py working directory.
  2. Update TX_elections_model.py:
       - EI_statewide = pd.read_csv("recom_inputs/statewide_rxc_EI_preferences.csv")
       - prec_ei_df   = pd.read_csv("recom_inputs/prec_count_quants.csv", dtype={{'CNTYVTD':'str'}})
       - mean_prec_counts = pd.read_csv("recom_inputs/mean_prec_vote_counts.csv", dtype={{'CNTYVTD':'str'}})
       - min_cand_weights = pd.read_csv("recom_inputs/ingroup_weight.csv")
  3. Update elections_track in TX_elections_model.py to reference your 2024
     election column names (24G_President, 24G_US_Sen, etc.).
  4. Run: python TX_elections_model.py
""")
