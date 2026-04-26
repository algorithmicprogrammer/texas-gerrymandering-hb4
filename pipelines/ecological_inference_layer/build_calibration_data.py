#!/usr/bin/env python3
"""Build the calibration CSV consumed by generate_tx_logit_params.py.

This script scores the enacted Texas plan with the ensemble model's
machinery, harvests the per-district raw VRA probabilities (pre-logit,
pre-venn-overlap), and pairs them with observed opportunity labels from a
benchmark general election. The result is the (model_type, subgroup,
raw_prob, is_opportunity) table that fits the per-(mode, group) logistic
calibration in the spirit of MGGG's VRA-Ensembles paper.

Usage:
    # Single benchmark (run with a placeholder --benchmark-elec first to see
    # the list of available general elections):
    python build_calibration_data.py \
        --benchmark-elec <one-of-general_elecs> \
        --black-cand AllredD_24G \
        --latino-cand AllredD_24G \
        --output calibration.csv

    # Multiple benchmarks pooled in one run (raw probs computed once, labels
    # recomputed per benchmark). Provide a CSV with columns:
    #   benchmark_elec,black_cand,latino_cand
    python build_calibration_data.py \
        --benchmarks-csv benchmarks.csv \
        --output calibration.csv

Then:
    python generate_tx_logit_params.py \
        --input calibration.csv \
        --output recom_inputs/TX_logit_params.csv

Notes
-----
- "raw_prob" comes from compute_final_dist(..., logit=False, return_raw=True),
  which yields the unsquashed black_vra_prob / hisp_vra_prob / neither_vra_prob.
  Raw probs depend only on the enacted plan, not on the benchmark, so when
  multiple benchmarks are passed they are computed once and reused.
- "is_opportunity" is computed per benchmark general election. The Black/Latino
  label is 1 iff that group's named candidate won the district; the Neither
  label is 1 iff neither minority's named candidate won.
- The output CSV always carries a benchmark_elec column for traceability;
  generate_tx_logit_params.py ignores extra columns.
- This driver replicates the loader in besag_clifford_vra_opportunity.py rather
  than importing it, because that module reads TX_logit_params.csv at import
  time and we are bootstrapping it here.
"""
from __future__ import annotations

import argparse
import importlib.util
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from gerrychain import (
    Graph, GeographicPartition, Election, updaters,
)
from gerrychain.updaters import cut_edges

from texas_gerrymandering_hb4.config import (
    TX_ELECTIONS, CANDIDATE_RACE_PARTY, PREC_COUNT_QUANTS_INPUT,
    INGROUP_WEIGHT_CSV_FILE, DROPPED_ELECS,
    STATEWIDE_RXC_EI_PREFERENCES_INPUT, RECENCY_WEIGHTS,
    MEAN_PREC_VOTE_COUNTS_INPUT, PRECINCT_DATASET_PARQUET,
)

# run_functions.py lives in pipelines/ensemble_generation_layer alongside its
# script-style siblings (no __init__.py, not a package), so load it directly
# from its absolute file path rather than relying on sys.path manipulation.
_RUN_FUNCTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "ensemble_generation_layer"
    / "run_functions.py"
)
_spec = importlib.util.spec_from_file_location("run_functions", _RUN_FUNCTIONS_PATH)
_run_functions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_functions)
compute_final_dist = _run_functions.compute_final_dist
cand_pref_all_draws_outcomes = _run_functions.cand_pref_all_draws_outcomes
precompute_state_weights = _run_functions.precompute_state_weights
compute_district_weights = _run_functions.compute_district_weights

NUM_DISTRICTS = 38
TOT_POP, CVAP, WCVAP, HCVAP, BCVAP = "TOTALPOP", "CVAP", "WCVAP", "HCVAP", "BCVAP"
GEO_ID, C_X, C_Y = "CNTYVTD", "C_X", "C_Y"

EXPECTED_MODES = ("statewide", "equal", "district")


def load_pipeline_state():
    """Mirror the top-level loads from besag_clifford_vra_opportunity.py.

    Returns a namespace-like dict of every object compute_final_dist needs.
    """
    elec_data = pd.read_csv(TX_ELECTIONS)
    dropped_elecs = pd.read_csv(DROPPED_ELECS)["Dropped Elections"]
    recency_weights = pd.read_csv(RECENCY_WEIGHTS)
    min_cand_weights = pd.read_csv(INGROUP_WEIGHT_CSV_FILE)
    cand_race_table = pd.read_csv(CANDIDATE_RACE_PARTY)
    EI_statewide = pd.read_csv(STATEWIDE_RXC_EI_PREFERENCES_INPUT)
    prec_ei_df = pd.read_csv(PREC_COUNT_QUANTS_INPUT, dtype={"CNTYVTD": "str"})

    state_gdf = gpd.read_parquet(PRECINCT_DATASET_PARQUET)
    state_gdf["CD"] = state_gdf["CD"].astype("int")

    # Same TX_columns hardcoded list used in besag_clifford_vra_opportunity.py.
    TX_columns = [
        "TrumpR_24G", "HarrisD_24G", "CruzR_24G", "AllredD_24G",
        "CraddickR_24G", "CulbertD_24G",
        "BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
        "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P",
        "UncommittedR_24P",
        "BidenD_24P", "CornejoD_24P", "LockeD_24P", "LozadaD_24P",
        "PerezD_24P", "PhillipsD_24P", "UygurD_24P", "WilliamsonD_24P",
        "CruzR_24P", "GibsonR_24P", "LopezR_24P",
        "AllredD_24P", "GomezD_24P", "GonzalezD_24P", "GutierrezD_24P",
        "HassanD_24P", "KeoughD_24P", "PrillimanD_24P", "ShermanD_24P",
        "TchenkoD_24P",
        "ClarkR_24P", "CraddickR_24P", "HowellR_24P", "MatlockR_24P", "ReyesR_24P",
        "BurchD_24P", "CulbertD_24P",
    ]

    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)
    centroids = state_gdf.centroid
    for node in graph.nodes():
        graph.nodes[node]["C_X"] = centroids.x[node]
        graph.nodes[node]["C_Y"] = centroids.y[node]

    elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
    elec_data_trunc = elec_data[elecs_bool].reset_index(drop=True)
    elec_sets = list(set(elec_data_trunc["Election Set"]))
    elections = list(elec_data_trunc["Election"])
    primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "Primary"].Election)
    runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "Runoff"].Election)
    general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == "General"].Election)

    elec_set_dict = {}
    for elec_set in elec_sets:
        sub = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
        elec_set_dict[elec_set] = dict(zip(sub.Type, sub.Election))
    elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

    candidates = {}
    for elec in elections:
        cands = ([y for y in TX_columns if elec in y and "R_" not in y.split("1")[0]]
                 if "R_" in elec[:4] or "P_" in elec[:4]
                 else [y for y in TX_columns if elec in y])
        if elec in general_elecs:
            cands = cands[:2]
        candidates[elec] = dict(zip(range(len(cands)), cands))

    cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
    min_cand_weights_dict = {k: min_cand_weights.to_dict()[k][0]
                              for k in min_cand_weights.to_dict().keys()}

    elec_years = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == es, "Year"]
                  .values[0].astype(str) for es in elec_sets]
    recency_scores = [recency_weights[yr][0] for yr in elec_years]
    recency_W1 = np.tile(recency_scores, (NUM_DISTRICTS, 1)).transpose()

    (black_weight_state, hisp_weight_state, neither_weight_state,
     black_weight_equal, hisp_weight_equal, neither_weight_equal,
     black_pref_cands_prim_state, hisp_pref_cands_prim_state,
     black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state) = \
        precompute_state_weights(
            NUM_DISTRICTS, elec_sets, elec_set_dict, recency_W1, EI_statewide,
            primary_elecs, runoff_elecs, elec_match_dict,
            min_cand_weights_dict, cand_race_dict,
        )

    demogs = ["BCVAP", "HCVAP"]
    bases = {col.split(".")[0] + "." + col.split(".")[1]
             for col in prec_ei_df.columns
             if col[:5] in demogs and "abstain" not in col
             and not any(x in col for x in general_elecs)}
    base_dict = {b: (b.split(".")[0].split("_")[0],
                     "_".join(b.split(".")[1].split("_")[1:-1]))
                 for b in bases}
    outcomes = {val: [] for val in base_dict.values()}
    for b in bases:
        outcomes[base_dict[b]].append(b)

    precs = list(state_gdf[GEO_ID])
    prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

    return dict(
        state_gdf=state_gdf, graph=graph, elec_data_trunc=elec_data_trunc,
        elec_sets=elec_sets, elections=elections, primary_elecs=primary_elecs,
        runoff_elecs=runoff_elecs, general_elecs=general_elecs,
        elec_set_dict=elec_set_dict, elec_match_dict=elec_match_dict,
        candidates=candidates, cand_race_table=cand_race_table,
        cand_race_dict=cand_race_dict, min_cand_weights_dict=min_cand_weights_dict,
        recency_W1=recency_W1,
        bases=bases, outcomes=outcomes, prec_draws_outcomes=prec_draws_outcomes,
        black_weight_state=black_weight_state, hisp_weight_state=hisp_weight_state,
        neither_weight_state=neither_weight_state,
        black_weight_equal=black_weight_equal, hisp_weight_equal=hisp_weight_equal,
        neither_weight_equal=neither_weight_equal,
        black_pref_cands_prim_state=black_pref_cands_prim_state,
        hisp_pref_cands_prim_state=hisp_pref_cands_prim_state,
        black_pref_cands_runoffs_state=black_pref_cands_runoffs_state,
        hisp_pref_cands_runoffs_state=hisp_pref_cands_runoffs_state,
    )


def build_enacted_partition(s):
    """Build the enacted GeographicPartition with per-election updaters."""
    my_updaters = {
        "population": updaters.Tally(TOT_POP, alias="population"),
        "CVAP": updaters.Tally(CVAP, alias="CVAP"),
        "WCVAP": updaters.Tally(WCVAP, alias="WCVAP"),
        "HCVAP": updaters.Tally(HCVAP, alias="HCVAP"),
        "BCVAP": updaters.Tally(BCVAP, alias="BCVAP"),
        "Sum_CX": updaters.Tally(C_X, alias="Sum_CX"),
        "Sum_CY": updaters.Tally(C_Y, alias="Sum_CY"),
        "cut_edges": cut_edges,
    }
    my_updaters.update({
        e.name: e for e in [Election(j, s["candidates"][j]) for j in s["elections"]]
    })
    return GeographicPartition(graph=s["graph"], assignment="CD", updaters=my_updaters)


def validate_benchmark(s, benchmark_elec, black_cand, latino_cand):
    """Raise ValueError if the (benchmark_elec, black_cand, latino_cand) triple
    doesn't match a general election in elec_data_trunc."""
    if benchmark_elec not in s["general_elecs"]:
        raise ValueError(
            f"benchmark_elec={benchmark_elec} is not a general election. "
            f"Pick one of: {s['general_elecs']}"
        )
    bench_cands = s["candidates"][benchmark_elec]
    if black_cand not in bench_cands.values():
        raise ValueError(
            f"black_cand={black_cand} is not in {benchmark_elec} candidates {list(bench_cands.values())}"
        )
    if latino_cand not in bench_cands.values():
        raise ValueError(
            f"latino_cand={latino_cand} is not in {benchmark_elec} candidates {list(bench_cands.values())}"
        )


def per_district_winner(state_gdf, partition, elec_cand_columns, target_candidate):
    """Return {district: bool} for whether `target_candidate` got the most votes in
    that district. `elec_cand_columns` is the list of candidate vote columns in
    the same general election (so we know who the rivals are).
    """
    if target_candidate not in elec_cand_columns:
        raise ValueError(
            f"{target_candidate} not in benchmark candidate columns {elec_cand_columns}"
        )
    df = pd.DataFrame(state_gdf).copy()
    df["_dist"] = df.index.map(dict(partition.assignment))
    totals = df.groupby("_dist")[list(elec_cand_columns)].sum()
    winners = totals.idxmax(axis=1).to_dict()
    return {d: (winners[d] == target_candidate) for d in winners}


def collect_raw_probs(partition, s):
    """Run compute_final_dist for all three modes with return_raw=True."""
    dist_changes = list(range(NUM_DISTRICTS))
    order = [x for x in partition.parts]

    dist_elec_results = {}
    for elec in s["elections"]:
        cands = s["candidates"][elec]
        outcome_list = [dict(zip(order, partition[elec].percents(c)))
                        for c in cands.keys()]
        dist_elec_results[elec] = {
            d: {cands[i]: outcome_list[i][d] for i in cands.keys()}
            for d in range(NUM_DISTRICTS)
        }

    map_winners = pd.DataFrame(columns=dist_changes)
    map_winners["Election"] = s["elections"]
    map_winners["Election Set"] = s["elec_data_trunc"]["Election Set"]
    map_winners["Election Type"] = s["elec_data_trunc"]["Type"]
    for d in dist_changes:
        map_winners[d] = [
            max(dist_elec_results[elec][d].items(), key=operator.itemgetter(1))[0]
            for elec in s["elections"]
        ]

    common = dict(
        map_winners=map_winners,
        dist_elec_results=dist_elec_results,
        dist_changes=dist_changes,
        cand_race_table=s["cand_race_table"],
        num_districts=NUM_DISTRICTS,
        candidates=s["candidates"],
        elec_sets=s["elec_sets"],
        elec_set_dict=s["elec_set_dict"],
        partition=partition,
        logit_params=pd.DataFrame(),  # unused when logit=False
        logit=False,
        return_raw=True,
    )

    state = compute_final_dist(
        black_pref_cands_df=s["black_pref_cands_prim_state"],
        black_pref_cands_runoffs=s["black_pref_cands_runoffs_state"],
        hisp_pref_cands_df=s["hisp_pref_cands_prim_state"],
        hisp_pref_cands_runoffs=s["hisp_pref_cands_runoffs_state"],
        neither_weight_array=s["neither_weight_state"],
        black_weight_array=s["black_weight_state"],
        hisp_weight_array=s["hisp_weight_state"],
        mode="statewide", **common,
    )
    equal = compute_final_dist(
        black_pref_cands_df=s["black_pref_cands_prim_state"],
        black_pref_cands_runoffs=s["black_pref_cands_runoffs_state"],
        hisp_pref_cands_df=s["hisp_pref_cands_prim_state"],
        hisp_pref_cands_runoffs=s["hisp_pref_cands_runoffs_state"],
        neither_weight_array=s["neither_weight_equal"],
        black_weight_array=s["black_weight_equal"],
        hisp_weight_array=s["hisp_weight_equal"],
        mode="equal", **common,
    )

    (black_w_d, hisp_w_d, neither_w_d,
     black_pref_d, black_pref_run_d, hisp_pref_d, hisp_pref_run_d) = \
        compute_district_weights(
            dist_changes, s["elec_sets"], s["elec_set_dict"], s["state_gdf"],
            partition, s["prec_draws_outcomes"], GEO_ID, s["primary_elecs"],
            s["runoff_elecs"], s["elec_match_dict"], s["bases"], s["outcomes"],
            s["recency_W1"], s["cand_race_dict"], s["min_cand_weights_dict"],
        )
    district = compute_final_dist(
        black_pref_cands_df=black_pref_d, black_pref_cands_runoffs=black_pref_run_d,
        hisp_pref_cands_df=hisp_pref_d, hisp_pref_cands_runoffs=hisp_pref_run_d,
        neither_weight_array=neither_w_d, black_weight_array=black_w_d,
        hisp_weight_array=hisp_w_d, mode="district", **common,
    )
    return {"statewide": state, "equal": equal, "district": district}


def assemble_calibration_rows(raw_probs, black_wins, latino_wins):
    """Cross probs with labels and return long-form (model_type, subgroup, ...) DataFrame."""
    rows = []
    for mode, prob_dict in raw_probs.items():
        for d, (hisp_p, black_p, neither_p) in prob_dict.items():
            rows.append({"model_type": mode, "subgroup": "Black",
                         "district": d, "raw_prob": float(black_p),
                         "is_opportunity": int(bool(black_wins[d]))})
            rows.append({"model_type": mode, "subgroup": "Latino",
                         "district": d, "raw_prob": float(hisp_p),
                         "is_opportunity": int(bool(latino_wins[d]))})
            # Neither = neither minority's preferred candidate won.
            neither_label = int(not (black_wins[d] or latino_wins[d]))
            rows.append({"model_type": mode, "subgroup": "Neither",
                         "district": d, "raw_prob": float(neither_p),
                         "is_opportunity": neither_label})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--benchmark-elec", default=None,
                   help="Single benchmark general-election name from TX_elections.csv. "
                        "Mutually exclusive with --benchmarks-csv.")
    p.add_argument("--black-cand", default="AllredD_24G",
                   help="Black-preferred candidate column name in the single benchmark election.")
    p.add_argument("--latino-cand", default="AllredD_24G",
                   help="Latino-preferred candidate column name in the single benchmark election.")
    p.add_argument("--benchmarks-csv", type=Path, default=None,
                   help="CSV with columns benchmark_elec,black_cand,latino_cand to pool "
                        "multiple benchmarks in one run. Raw probs are computed once.")
    p.add_argument("--output", type=Path, default=Path("calibration.csv"),
                   help="Output CSV (consumed by generate_tx_logit_params.py --input).")
    return p.parse_args()


def resolve_benchmarks(args):
    """Return a list of (benchmark_elec, black_cand, latino_cand) tuples."""
    if args.benchmarks_csv is not None and args.benchmark_elec is not None:
        raise SystemExit("Pass either --benchmark-elec or --benchmarks-csv, not both.")
    if args.benchmarks_csv is not None:
        df = pd.read_csv(args.benchmarks_csv)
        required = {"benchmark_elec", "black_cand", "latino_cand"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(
                f"--benchmarks-csv missing columns: {sorted(missing)} "
                f"(required: {sorted(required)})"
            )
        return [(str(r.benchmark_elec), str(r.black_cand), str(r.latino_cand))
                for r in df.itertuples(index=False)]
    if args.benchmark_elec is not None:
        return [(args.benchmark_elec, args.black_cand, args.latino_cand)]
    raise SystemExit("Pass either --benchmark-elec or --benchmarks-csv.")


def main():
    args = parse_args()
    benchmarks = resolve_benchmarks(args)

    print("Loading pipeline state...")
    s = load_pipeline_state()
    print(f"Available general elections: {s['general_elecs']}")

    for elec, bcand, lcand in benchmarks:
        validate_benchmark(s, elec, bcand, lcand)

    partition = build_enacted_partition(s)

    print("Computing raw VRA probabilities (statewide / equal / district)...")
    raw_probs = collect_raw_probs(partition, s)

    parts = []
    for elec, bcand, lcand in benchmarks:
        print(f"Computing is_opportunity labels from {elec} (black={bcand}, latino={lcand})...")
        bench_cands = list(s["candidates"][elec].values())
        black_wins = per_district_winner(s["state_gdf"], partition, bench_cands, bcand)
        latino_wins = per_district_winner(s["state_gdf"], partition, bench_cands, lcand)
        df = assemble_calibration_rows(raw_probs, black_wins, latino_wins)
        df["benchmark_elec"] = elec
        parts.append(df)

    print("Assembling calibration table...")
    out = pd.concat(parts, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(out.groupby(["model_type", "subgroup"])
          .agg(n=("raw_prob", "size"),
               positives=("is_opportunity", "sum"),
               raw_mean=("raw_prob", "mean")))


if __name__ == "__main__":
    main()
