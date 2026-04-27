# -*- coding: utf-8 -*-
"""
Diagnostic for the all-zero enacted-map score.

Runs the same data-loading and partition-build path as
besag_clifford_vra_opportunity.py, then calls compute_final_dist with
return_raw=True so we see the pre-logit, pre-Venn probabilities. Also
inspects logit_params and BCVAP/HCVAP totals per district so we can tell
whether the zeros come from:

  (1) all accrue arrays being False (-> raw probs all 0),
  (2) calibration mapping uniform input to near-zero output,
  (3) the gating coefficient (BCVAP / CVAP) being zero per district
      (suggesting a label-keyed join failure from the 0-indexed remap),
  (4) something else.

Run from repo root:
    ./venv/bin/python diagnose_zero_score.py
"""
import sys
import os
sys.path.insert(0, os.path.expanduser(
    "~/Documents/texas-gerrymandering-hb4"))
sys.path.insert(0, os.path.expanduser(
    "~/Documents/texas-gerrymandering-hb4/pipelines/ensemble_generation_layer"))

import numpy as np
import pandas as pd
import geopandas as gpd
from gerrychain import Graph, GeographicPartition, Election, updaters
from gerrychain.updaters import Tally, cut_edges

from texas_gerrymandering_hb4.config import (
    TX_ELECTIONS, CANDIDATE_RACE_PARTY, PREC_COUNT_QUANTS_INPUT,
    INGROUP_WEIGHT_CSV_FILE, DROPPED_ELECS,
    STATEWIDE_RXC_EI_PREFERENCES_INPUT, RECENCY_WEIGHTS,
    MEAN_PREC_VOTE_COUNTS_INPUT, PRECINCT_DATASET_PARQUET, TX_LOGIT_PARAMS_CSV,
)
from run_functions import (
    cand_pref_all_draws_outcomes, precompute_state_weights,
    compute_district_weights, compute_final_dist,
)

NUM_DISTRICTS = 38
GEO_ID = "CNTYVTD"
TOT_POP = "TOTALPOP"
CVAP = "CVAP"
WCVAP = "WCVAP"
HCVAP = "HCVAP"
BCVAP = "BCVAP"
C_X = "C_X"
C_Y = "C_Y"
EFFECTIVENESS_CUTOFF = 0.6

print("Loading data...")
state_gdf = gpd.read_parquet(PRECINCT_DATASET_PARQUET)
state_gdf["CD"] = state_gdf["CD"].astype("int")
graph = Graph.from_geodataframe(state_gdf, ignore_errors=True)

elec_data = pd.read_csv(TX_ELECTIONS)
dropped_elecs = pd.read_csv(DROPPED_ELECS)["Election Set"]
recency_weights = pd.read_csv(RECENCY_WEIGHTS)
min_cand_weights = pd.read_csv(INGROUP_WEIGHT_CSV_FILE)
cand_race_table = pd.read_csv(CANDIDATE_RACE_PARTY)
EI_statewide = pd.read_csv(STATEWIDE_RXC_EI_PREFERENCES_INPUT)
prec_ei_df = pd.read_csv(PREC_COUNT_QUANTS_INPUT, dtype={"CNTYVTD": "str"})
mean_prec_counts = pd.read_csv(MEAN_PREC_VOTE_COUNTS_INPUT,
                               dtype={"CNTYVTD": "str"})
logit_params = pd.read_csv(TX_LOGIT_PARAMS_CSV)

# --- 0. logit_params inspection ---
print("\n=== 0. logit_params content ===")
print(logit_params.to_string())

# --- minimal pipeline setup mirroring the main script ---
elec_data_trunc = elec_data[
    ~elec_data.Election.isin(list(dropped_elecs))
].reset_index(drop=True)
elections = list(elec_data_trunc["Election"])
elec_sets = list(set(elec_data_trunc["Election Set"]))
elec_set_dict = {}
for elec_set in elec_sets:
    sub = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(sub["Type"], sub["Election"]))
elec_match_dict = dict(zip(elec_data_trunc["Election"],
                           elec_data_trunc["Election Set"]))
primary_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == "Primary"].Election)
runoff_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == "Runoff"].Election)
general_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == "General"].Election)

cand_race_dict = {
    "_".join(full_key.split("_")[:2]): race
    for full_key, race in cand_race_table.set_index("Candidates")["Race"].items()
}
min_cand_weights_dict = {
    k: min_cand_weights.to_dict()[k][0]
    for k in min_cand_weights.to_dict().keys()
}

elec_years = [
    elec_data_trunc.loc[elec_data_trunc["Election Set"] == es, "Year"].iloc[0]
    for es in elec_sets
]
recency_W1 = np.tile(np.array(
    [recency_weights[str(y)].iloc[0] for y in elec_years])[:, None],
    (1, NUM_DISTRICTS))

ELECTION_TO_CANDIDATES = {
    "24G_President":  ["TrumpR_24G", "HarrisD_24G"],
    "24P_PresR":      ["BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
                       "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P",
                       "DeSantisR_24P", "UncommittedR_24P"],
    "24P_PresD":      ["BidenD_24P", "CornejoD_24P", "LockeD_24P", "LozadaD_24P",
                       "PerezD_24P", "PhillipsD_24P", "UygurD_24P", "WilliamsonD_24P"],
    "24G_US_Sen":     ["CruzR_24G", "AllredD_24G"],
    "24P_SenR":       ["CruzR_24P", "GibsonR_24P", "LopezR_24P"],
    "24P_SenD":       ["AllredD_24P", "GomezD_24P", "GonzalezD_24P", "GutierrezD_24P",
                       "HassanD_24P", "KeoughD_24P", "PrillimanD_24P", "ShermanD_24P",
                       "TchenkoD_24P"],
    "24G_RR_Comm_1":  ["CraddickR_24G", "CulbertD_24G"],
    "24P_RRCR":       ["ClarkR_24P", "CraddickR_24P", "HowellR_24P",
                       "MatlockR_24P", "ReyesR_24P"],
    "24P_RRCD":       ["BurchD_24P", "CulbertD_24P"],
}
TX_columns = sorted({c for cs in ELECTION_TO_CANDIDATES.values() for c in cs})

candidates = {}
for elec in elections:
    cands = ELECTION_TO_CANDIDATES[elec]
    if elec in general_elecs:
        cands = cands[:2]
    candidates[elec] = dict(zip(range(len(cands)), cands))

demogs = ["BCVAP", "HCVAP"]
bases = {col.split(".")[0] + "." + col.split(".")[1]
         for col in prec_ei_df.columns
         if col[:5] in demogs and "abstain" not in col
         and not any(x in col for x in general_elecs)}
base_dict = {b: (b.split(".")[0].split("_")[0],
                 "_".join(b.split(".")[1].split("_")[:-2]))
             for b in bases}
outcomes = {val: [] for val in base_dict.values()}
for b in bases:
    outcomes[base_dict[b]].append(b)

precs = list(state_gdf[GEO_ID])
prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

# Build statewide weights. precompute_state_weights returns 10 values:
# the (black, hisp, neither) weight arrays for both 'statewide' and 'equal'
# modes, plus the four state-level pref-cands DataFrames.
(black_state_weight,  hisp_state_weight,  neither_state_weight,
 black_equal_weight,  hisp_equal_weight,  neither_equal_weight,
 black_pref_cands_prim_state,   hisp_pref_cands_prim_state,
 black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state) = \
    precompute_state_weights(
        NUM_DISTRICTS, elec_sets, elec_set_dict, recency_W1, EI_statewide,
        primary_elecs, runoff_elecs, elec_match_dict,
        min_cand_weights_dict, cand_race_dict
    )

# Build the enacted partition, applying the 0-indexed remap.
raw_labels = sorted({int(graph.nodes[n]["CD"]) for n in graph.nodes()})
label_to_idx = {label: idx for idx, label in enumerate(raw_labels)}
assignment = {n: label_to_idx[int(graph.nodes[n]["CD"])]
              for n in graph.nodes()}

my_updaters = {
    "population": Tally(TOT_POP, alias="population"),
    "CVAP":       Tally(CVAP,    alias="CVAP"),
    "WCVAP":      Tally(WCVAP,   alias="WCVAP"),
    "HCVAP":      Tally(HCVAP,   alias="HCVAP"),
    "BCVAP":      Tally(BCVAP,   alias="BCVAP"),
    "cut_edges":  cut_edges,
}
my_updaters.update({
    e.name: e for e in [Election(j, candidates[j]) for j in elections]})
partition = GeographicPartition(graph=graph, assignment=assignment,
                                updaters=my_updaters)

# --- 1. demographics by 0-indexed district ---
print("\n=== 1. Per-district demographics under 0-indexed labels ===")
print("(internal_idx_k corresponds to real CD k+1)")
print(f"{'idx':>5} {'CD':>5} {'TOTPOP':>10} {'CVAP':>10} {'BCVAP':>10} "
      f"{'HCVAP':>10} {'B/CVAP':>8} {'H/CVAP':>8}")
for d in sorted(partition.parts):
    print(f"{d:>5} {raw_labels[d]:>5} "
          f"{partition['population'][d]:>10} {partition['CVAP'][d]:>10} "
          f"{partition['BCVAP'][d]:>10} {partition['HCVAP'][d]:>10} "
          f"{partition['BCVAP'][d]/partition['CVAP'][d]:>8.3f} "
          f"{partition['HCVAP'][d]/partition['CVAP'][d]:>8.3f}")

# --- 2. dist_elec_results for one election (24G_US_Sen, where Cruz vs Allred) ---
order = list(partition.parts)
dist_elec_results = {}
for elec in elections:
    cands = candidates[elec]
    outcome_list = [dict(zip(order, partition[elec].percents(cand)))
                    for cand in cands.keys()]
    dist_elec_results[elec] = {
        d: {cands[i]: outcome_list[i][d] for i in cands.keys()}
        for d in range(NUM_DISTRICTS)
    }

print("\n=== 2. dist_elec_results['24G_US_Sen'] (first 5 districts) ===")
for d in range(5):
    print(f"  d={d} (CD {raw_labels[d]}): {dist_elec_results['24G_US_Sen'][d]}")

# --- 3. district winners table ---
import operator
dist_changes = list(range(NUM_DISTRICTS))
map_winners = pd.DataFrame(columns=dist_changes)
map_winners["Election"] = elections
map_winners["Election Set"] = elec_data_trunc["Election Set"]
map_winners["Election Type"] = elec_data_trunc["Type"]
for i in dist_changes:
    map_winners[i] = [
        max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0]
        for elec in elections
    ]
print("\n=== 3. map_winners (first 6 districts) ===")
print(map_winners[["Election", "Election Type"] + list(range(6))].to_string())

# --- 4. RAW probabilities (before logit, before Venn) for statewide mode ---
final_state_raw = compute_final_dist(
    map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
    hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
    neither_state_weight, black_state_weight, hisp_state_weight,
    dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
    candidates, elec_sets, elec_set_dict, "statewide", partition,
    logit_params, logit=False, return_raw=True,
)
print("\n=== 4. RAW (pre-logit, pre-Venn) statewide probs (hisp, black, neither) ===")
print(f"{'idx':>5} {'CD':>5} {'hisp':>8} {'black':>8} {'neither':>8}")
for d in sorted(final_state_raw):
    h, b, n = final_state_raw[d]
    print(f"{d:>5} {raw_labels[d]:>5} {h:>8.4f} {b:>8.4f} {n:>8.4f}")

# --- 5. POST-logit, POST-Venn probs ---
final_state_logit = compute_final_dist(
    map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
    hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
    neither_state_weight, black_state_weight, hisp_state_weight,
    dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
    candidates, elec_sets, elec_set_dict, "statewide", partition,
    logit_params, logit=True,
)
print("\n=== 5. POST-logit, POST-Venn statewide probs (hisp, black, neither, overlap) ===")
print(f"{'idx':>5} {'CD':>5} {'hisp':>8} {'black':>8} {'neither':>8} {'overlap':>8}")
for d in sorted(final_state_logit):
    h, b, n, o = final_state_logit[d]
    print(f"{d:>5} {raw_labels[d]:>5} {h:>8.4f} {b:>8.4f} {n:>8.4f} {o:>8.4f}")

print("\nDone.")