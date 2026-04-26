# -*- coding: utf-8 -*-
"""
make_TX_logit_params.py
=======================

Create TX_logit_params.csv for besag_clifford_vra_opportunity_combined.py.

Output columns expected by run_functions.compute_final_dist(..., logit=True):
    model_type, subgroup, coef, intercept

What this script fits
---------------------
For each model_type in {statewide, equal, district} and each subgroup in
{Black, Latino, Neither}, it builds district-by-election-set training rows:

    x = pre-logit opportunity score used by the VRA ensemble model
        = weighted historical success rate * group-concentration adjustment

    y = 1 if the subgroup-preferred candidate/proxy wins under the model's
        election-set rules, else 0

Then it fits:

    Pr(y = 1 | x) = sigmoid(intercept + coef * x)

The resulting coefficients calibrate the raw opportunity score to a probability.

Usage
-----
Run from the same project root where your TX files live:

    python make_TX_logit_params.py

Optional:

    python make_TX_logit_params.py --assignment CD --output TX_logit_params.csv
    python make_TX_logit_params.py --assignment Seed_Demo --no-district-mode

Notes
-----
- This script intentionally mirrors the non-logit part of run_functions.compute_final_dist.
- It does not require sklearn; it uses scipy.optimize.
- If a fit has only one class or fails to converge, it falls back to a conservative
  increasing logistic curve and prints a warning.
"""

import argparse
import operator
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import minimize

from gerrychain import Graph, GeographicPartition, updaters, Election
from gerrychain.updaters import cut_edges

from texas_gerrymandering_hb4.config import (
    TX_ELECTIONS,
    CANDIDATE_RACE_PARTY,
    PREC_COUNT_QUANTS_INPUT,
    INGROUP_WEIGHT_CSV_FILE,
    DROPPED_ELECS,
    STATEWIDE_RXC_EI_PREFERENCES_INPUT,
    RECENCY_WEIGHTS,
    PRECINCT_DATASET_PARQUET,
)

from run_functions import (
    compute_W2,
    prob_conf_conversion,
    cand_pref_outcome_sum,
    cand_pref_all_draws_outcomes,
    precompute_state_weights,
    compute_district_weights,
)


# ---------------------------------------------------------------------
# Project-specific constants. Change here if your column names differ.
# ---------------------------------------------------------------------
NUM_DISTRICTS = 38
PLOT_PATH = PRECINCT_DATASET_PARQUET

TOT_POP = "TOTPOP_x"
CVAP = "1_2018"
HCVAP = "13_2018"
BCVAP = "5_2018"
GEO_ID = "CNTYVTD"
C_X = "C_X"
C_Y = "C_Y"

DEFAULT_FALLBACK_COEF = 8.0
DEFAULT_FALLBACK_CENTER = 0.5


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_1d(x, y, subgroup, model_type):
    """Fit y ~ sigmoid(intercept + coef*x), returning coef/intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 10 or len(np.unique(y)) < 2:
        warnings.warn(
            f"Fallback for {model_type}/{subgroup}: not enough variation "
            f"(n={len(x)}, classes={sorted(set(y.tolist()))})."
        )
        return DEFAULT_FALLBACK_COEF, -DEFAULT_FALLBACK_COEF * DEFAULT_FALLBACK_CENTER

    def nll(beta):
        intercept, coef = beta
        p = sigmoid(intercept + coef * x)
        eps = 1e-9
        # mild ridge penalty prevents extreme coefficients under separation
        ridge = 1e-4 * (intercept**2 + coef**2)
        return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + ridge

    # Initialize near an increasing curve centered at observed positive rate.
    pos_rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    init_intercept = np.log(pos_rate / (1 - pos_rate)) - 4.0 * np.mean(x)
    init = np.array([init_intercept, 4.0])
    res = minimize(nll, init, method="BFGS")

    if not res.success or not np.all(np.isfinite(res.x)):
        warnings.warn(f"Fallback for {model_type}/{subgroup}: optimizer failed: {res.message}")
        return DEFAULT_FALLBACK_COEF, -DEFAULT_FALLBACK_COEF * DEFAULT_FALLBACK_CENTER

    intercept, coef = res.x

    # Opportunity should be monotone increasing in x. If noisy data flips it,
    # use a conservative increasing calibration rather than a decreasing one.
    if coef <= 0:
        warnings.warn(
            f"Fallback for {model_type}/{subgroup}: fitted coef <= 0 ({coef:.4f})."
        )
        return DEFAULT_FALLBACK_COEF, -DEFAULT_FALLBACK_COEF * DEFAULT_FALLBACK_CENTER

    return float(coef), float(intercept)


def load_project_data():
    print("Loading project data...")
    elec_data = pd.read_csv(TX_ELECTIONS)
    TX_columns = list(pd.read_csv("TX_columns.csv")["Columns"])
    dropped_elecs = pd.read_csv(DROPPED_ELECS)["Dropped Elections"]
    recency_weights = pd.read_csv(RECENCY_WEIGHTS)
    min_cand_weights = pd.read_csv(INGROUP_WEIGHT_CSV_FILE)
    cand_race_table = pd.read_csv(CANDIDATE_RACE_PARTY)
    EI_statewide = pd.read_csv(STATEWIDE_RXC_EI_PREFERENCES_INPUT)
    prec_ei_df = pd.read_csv(PREC_COUNT_QUANTS_INPUT, dtype={GEO_ID: "str"})

    state_gdf = gpd.read_parquet(PLOT_PATH)
    state_gdf["CD"] = state_gdf["CD"].astype(int)
    if "Seed_Demo" in state_gdf.columns:
        state_gdf["Seed_Demo"] = state_gdf["Seed_Demo"].astype(int)
    state_gdf.columns = state_gdf.columns.str.replace("-", "_", regex=False)

    # Rename election columns to match TX_columns.csv, same as the main pipeline.
    state_gdf_cols = list(state_gdf.columns)
    cand1_index = state_gdf_cols.index("RomneyR_12")
    cand2_index = state_gdf_cols.index("ObamaD_12P")
    state_gdf_cols[cand1_index : cand2_index + 1] = TX_columns
    state_gdf.columns = state_gdf_cols

    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)
    centroids_geom = state_gdf.centroid
    for node in graph.nodes():
        graph.nodes[node][C_X] = centroids_geom.x[node]
        graph.nodes[node][C_Y] = centroids_geom.y[node]

    elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
    elec_data_trunc = elec_data[elecs_bool].reset_index(drop=True)
    elec_sets = list(set(elec_data_trunc["Election Set"]))
    elections = list(elec_data_trunc["Election"])
    general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "General"].Election)
    primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "Primary"].Election)
    runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "Runoff"].Election)
    elec_cand_list = TX_columns

    elec_set_dict = {}
    for elec_set in elec_sets:
        sub = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
        elec_set_dict[elec_set] = dict(zip(sub.Type, sub.Election))

    elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

    candidates = {}
    for elec in elections:
        cands = (
            [y for y in elec_cand_list if elec in y and "R_" not in y.split("1")[0]]
            if "R_" in elec[:4] or "P_" in elec[:4]
            else [y for y in elec_cand_list if elec in y]
        )
        if elec in general_elecs:
            cands = cands[:2]
        candidates[elec] = dict(zip(range(len(cands)), cands))

    cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
    min_cand_weights_dict = {k: min_cand_weights.to_dict()[k][0] for k in min_cand_weights.to_dict().keys()}

    elec_years = [
        elec_data_trunc.loc[elec_data_trunc["Election Set"] == es, "Year"].values[0].astype(str)
        for es in elec_sets
    ]
    recency_scores = [recency_weights[yr][0] for yr in elec_years]
    recency_W1 = np.tile(recency_scores, (NUM_DISTRICTS, 1)).transpose()

    state_weights = precompute_state_weights(
        NUM_DISTRICTS,
        elec_sets,
        elec_set_dict,
        recency_W1,
        EI_statewide,
        primary_elecs,
        runoff_elecs,
        elec_match_dict,
        min_cand_weights_dict,
        cand_race_dict,
    )

    demogs = ["BCVAP", "HCVAP"]
    bases = {
        col.split(".")[0] + "." + col.split(".")[1]
        for col in prec_ei_df.columns
        if col[:5] in demogs
        and "abstain" not in col
        and not any(x in col for x in general_elecs)
    }
    base_dict = {b: (b.split(".")[0].split("_")[0], "_".join(b.split(".")[1].split("_")[1:-1])) for b in bases}
    outcomes = {val: [] for val in base_dict.values()}
    for b in bases:
        outcomes[base_dict[b]].append(b)
    precs = list(state_gdf[GEO_ID])
    prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

    return locals()


def build_partition(data, assignment_col):
    state_gdf = data["state_gdf"]
    graph = data["graph"]
    candidates = data["candidates"]
    elections = data["elections"]

    my_updaters = {
        "population": updaters.Tally(TOT_POP, alias="population"),
        "CVAP": updaters.Tally(CVAP, alias="CVAP"),
        "HCVAP": updaters.Tally(HCVAP, alias="HCVAP"),
        "BCVAP": updaters.Tally(BCVAP, alias="BCVAP"),
        "cut_edges": cut_edges,
    }
    my_updaters.update({e.name: e for e in [Election(j, candidates[j]) for j in elections]})

    if assignment_col not in state_gdf.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in shapefile.")

    return GeographicPartition(graph=graph, assignment=assignment_col, updaters=my_updaters)


def map_winners_for_partition(partition, data):
    elections = data["elections"]
    candidates = data["candidates"]
    elec_data_trunc = data["elec_data_trunc"]
    order = [x for x in partition.parts]

    dist_elec_results = {}
    for elec in elections:
        cands = candidates[elec]
        outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]
        dist_elec_results[elec] = {
            d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in range(NUM_DISTRICTS)
        }

    map_winners = pd.DataFrame(columns=range(NUM_DISTRICTS))
    map_winners["Election"] = elections
    map_winners["Election Set"] = elec_data_trunc["Election Set"]
    map_winners["Election Type"] = elec_data_trunc["Type"]
    for i in range(NUM_DISTRICTS):
        map_winners[i] = [
            max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0]
            for elec in elections
        ]
    return map_winners, dist_elec_results


def training_rows_for_mode(partition, data, mode):
    """Return long dataframe with x/y rows for Black, Latino, Neither."""
    map_winners, dist_elec_results = map_winners_for_partition(partition, data)

    elec_sets = data["elec_sets"]
    elec_set_dict = data["elec_set_dict"]
    primary_elecs = data["primary_elecs"]
    runoff_elecs = data["runoff_elecs"]
    cand_race_table = data["cand_race_table"]
    candidates = data["candidates"]
    recency_W1 = data["recency_W1"]
    state_gdf = data["state_gdf"]
    prec_draws_outcomes = data["prec_draws_outcomes"]
    bases = data["bases"]
    outcomes = data["outcomes"]
    elec_match_dict = data["elec_match_dict"]
    cand_race_dict = data["cand_race_dict"]
    min_cand_weights_dict = data["min_cand_weights_dict"]

    (
        black_weight_state,
        hisp_weight_state,
        neither_weight_state,
        black_weight_equal,
        hisp_weight_equal,
        neither_weight_equal,
        black_pref_cands_prim_state,
        hisp_pref_cands_prim_state,
        black_pref_cands_runoffs_state,
        hisp_pref_cands_runoffs_state,
    ) = data["state_weights"]

    dist_changes = list(range(NUM_DISTRICTS))

    if mode == "statewide":
        black_weight_array = black_weight_state
        hisp_weight_array = hisp_weight_state
        neither_weight_array = neither_weight_state
        black_pref_cands_df = black_pref_cands_prim_state
        hisp_pref_cands_df = hisp_pref_cands_prim_state
        black_pref_cands_runoffs = black_pref_cands_runoffs_state
        hisp_pref_cands_runoffs = hisp_pref_cands_runoffs_state
    elif mode == "equal":
        black_weight_array = black_weight_equal
        hisp_weight_array = hisp_weight_equal
        neither_weight_array = neither_weight_equal
        black_pref_cands_df = black_pref_cands_prim_state
        hisp_pref_cands_df = hisp_pref_cands_prim_state
        black_pref_cands_runoffs = black_pref_cands_runoffs_state
        hisp_pref_cands_runoffs = hisp_pref_cands_runoffs_state
    elif mode == "district":
        (
            black_weight_array,
            hisp_weight_array,
            neither_weight_array,
            black_pref_cands_df,
            black_pref_cands_runoffs,
            hisp_pref_cands_df,
            hisp_pref_cands_runoffs,
        ) = compute_district_weights(
            dist_changes,
            elec_sets,
            elec_set_dict,
            state_gdf,
            partition,
            prec_draws_outcomes,
            GEO_ID,
            primary_elecs,
            runoff_elecs,
            elec_match_dict,
            bases,
            outcomes,
            recency_W1,
            cand_race_dict,
            min_cand_weights_dict,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    general_winners = map_winners[map_winners["Election Type"] == "General"].reset_index(drop=True)
    primary_winners = map_winners[map_winners["Election Type"] == "Primary"].reset_index(drop=True)
    runoff_winners = map_winners[map_winners["Election Type"] == "Runoff"].reset_index(drop=True)

    primary_races = [elec_set_dict[elec_set]["Primary"] for elec_set in elec_sets]
    runoff_races = [None if "Runoff" not in elec_set_dict[elec_set].keys() else elec_set_dict[elec_set]["Runoff"] for elec_set in elec_sets]
    cand_party_dict = cand_race_table.set_index("Candidates").to_dict()["Party"]

    rows = []

    for dist in dist_changes:
        black_pref_cands = list(black_pref_cands_df[dist])
        hisp_pref_cands = list(hisp_pref_cands_df[dist])

        primary_dict = primary_winners.set_index("Election Set").to_dict()[dist]
        general_dict = general_winners.set_index("Election Set").to_dict()[dist]
        runoffs_dict = runoff_winners.set_index("Election Set").to_dict()[dist] if len(runoff_winners) else {}

        primary_winner_list = [primary_dict[es] for es in elec_sets]
        general_winner_list = [general_dict[es] for es in elec_sets]
        runoff_winner_list = ["N/A" if es not in list(runoff_winners["Election Set"]) else runoffs_dict[es] for es in elec_sets]

        primary_race_share_dict = {primary_race: dist_elec_results[primary_race][dist] for primary_race in primary_races}
        primary_ranking = {
            primary_race: {
                key: rank
                for rank, key in enumerate(
                    sorted(primary_race_share_dict[primary_race], key=primary_race_share_dict[primary_race].get, reverse=True),
                    1,
                )
            }
            for primary_race in primary_race_share_dict.keys()
        }

        black_pref_prim_rank = [primary_ranking[pr][bpc] for pr, bpc in zip(primary_races, black_pref_cands)]
        hisp_pref_prim_rank = [primary_ranking[pr][hpc] for pr, hpc in zip(primary_races, hisp_pref_cands)]
        party_general_winner = [cand_party_dict[gw] for gw in general_winner_list]

        runoff_black_pref = [
            "N/A" if rw == "N/A" else bpc
            for rw, bpc in zip(runoff_winner_list, list(black_pref_cands_runoffs[dist]))
        ]
        runoff_hisp_pref = [
            "N/A" if rw == "N/A" else hpc
            for rw, hpc in zip(runoff_winner_list, list(hisp_pref_cands_runoffs[dist]))
        ]

        black_y = [
            (prim_win == bpc and party_win == "D")
            if run_race is None
            else ((bpp_rank < 3 and run_win == runbp and party_win == "D") or (primary_race_share_dict[prim_race][bpc] > 0.5 and party_win == "D"))
            for run_race, prim_win, bpc, party_win, bpp_rank, run_win, runbp, prim_race in zip(
                runoff_races,
                primary_winner_list,
                black_pref_cands,
                party_general_winner,
                black_pref_prim_rank,
                runoff_winner_list,
                runoff_black_pref,
                primary_races,
            )
        ]
        hisp_y = [
            (prim_win == hpc and party_win == "D")
            if run_race is None
            else ((hpp_rank < 3 and run_win == runhp and party_win == "D") or (primary_race_share_dict[prim_race][hpc] > 0.5 and party_win == "D"))
            for run_race, prim_win, hpc, party_win, hpp_rank, run_win, runhp, prim_race in zip(
                runoff_races,
                primary_winner_list,
                hisp_pref_cands,
                party_general_winner,
                hisp_pref_prim_rank,
                runoff_winner_list,
                runoff_hisp_pref,
                primary_races,
            )
        ]
        neither_y = [(not b) and (not h) for b, h in zip(black_y, hisp_y)]

        black_gc = min(1, (partition["BCVAP"][dist] / partition["CVAP"][dist]) * 2) if partition["CVAP"][dist] else 0
        hisp_gc = min(1, (partition["HCVAP"][dist] / partition["CVAP"][dist]) * 2) if partition["CVAP"][dist] else 0
        neither_gc = 1.0

        for j, elec_set in enumerate(elec_sets):
            rows.append({"model_type": mode, "subgroup": "Black", "district": dist, "election_set": elec_set, "x": float(black_weight_array[j, dist] * black_gc), "y": int(black_y[j])})
            rows.append({"model_type": mode, "subgroup": "Latino", "district": dist, "election_set": elec_set, "x": float(hisp_weight_array[j, dist] * hisp_gc), "y": int(hisp_y[j])})
            rows.append({"model_type": mode, "subgroup": "Neither", "district": dist, "election_set": elec_set, "x": float(neither_weight_array[j, dist] * neither_gc), "y": int(neither_y[j])})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", default="CD", help="Shapefile column used as training map assignment, e.g. CD or Seed_Demo")
    parser.add_argument("--output", default="TX_logit_params.csv")
    parser.add_argument("--training-output", default="outputs/TX_logit_training_rows.csv")
    parser.add_argument("--no-district-mode", action="store_true", help="Skip district-specific preference mode if it is slow")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    data = load_project_data()
    partition = build_partition(data, args.assignment)

    modes = ["statewide", "equal"] if args.no_district_mode else ["statewide", "equal", "district"]
    train_parts = []
    for mode in modes:
        print(f"Building training rows for mode={mode}...")
        train_parts.append(training_rows_for_mode(partition, data, mode))
    train = pd.concat(train_parts, ignore_index=True)
    train.to_csv(args.training_output, index=False)
    print(f"Saved training rows to {args.training_output}")

    rows = []
    for mode in modes:
        for subgroup in ["Black", "Latino", "Neither"]:
            sub = train[(train["model_type"] == mode) & (train["subgroup"] == subgroup)]
            coef, intercept = fit_logit_1d(sub["x"].values, sub["y"].values, subgroup, mode)
            rows.append({"model_type": mode, "subgroup": subgroup, "coef": coef, "intercept": intercept})
            print(f"{mode:9s} {subgroup:7s}: coef={coef: .6f}, intercept={intercept: .6f}, n={len(sub)}, y_mean={sub['y'].mean():.3f}")

    # If district mode is skipped, copy statewide params so compute_final_dist still has rows.
    if args.no_district_mode:
        for subgroup in ["Black", "Latino", "Neither"]:
            src = [r for r in rows if r["model_type"] == "statewide" and r["subgroup"] == subgroup][0]
            rows.append({"model_type": "district", "subgroup": subgroup, "coef": src["coef"], "intercept": src["intercept"]})

    params = pd.DataFrame(rows, columns=["model_type", "subgroup", "coef", "intercept"])
    params.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(params)


if __name__ == "__main__":
    main()
