# -*- coding: utf-8 -*-
"""
besag_clifford_vra.py
=====================
Exact significance testing for VRA effectiveness using the Besag-Clifford (1989)
parallel (hub-and-spoke) method.

Background
----------
The existing TX_elections_model.py runs a single forward Markov chain initialized
at the enacted map. Using those samples to rank the enacted map's score yields a
p-value that is NOT guaranteed to be valid: early chain states are correlated with
the start, violating the exchangeability requirement for a valid Monte Carlo test.

Besag & Clifford (1989) fix this with the parallel method:
  1. Run the chain FORWARD from the enacted map for L steps  →  hub X*
     (Because ReCom with always_accept is reversible, forward = backward in distribution,
      so X* is a draw from the null that is L steps "away" from the enacted map.)
  2. From X*, independently spawn M chains each run forward L steps  →  spokes X̃_1,...,X̃_M
  3. Rank the enacted map's statistic among {X̃_1,...,X̃_M}.
     This ranking is EXACTLY valid regardless of mixing time (B-C Prop 3.3).

The p-value for "the enacted map is unusually low in minority effectiveness" is:
  p = (# spokes with score <= enacted score + 1) / (M + 1)

Usage
-----
  python besag_clifford_vra.py

Tunable parameters are in the USER PARAMETERS section below.
All other setup mirrors TX_elections_model.py exactly.
"""

import random as stdlib_random
import time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from functools import partial
import operator
import multiprocessing as mp
from copy import deepcopy

from gerrychain.random import random as gc_random
from gerrychain import (
    Graph, MarkovChain, GeographicPartition,
    accept, constraints, updaters, Election
)
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import recursive_tree_part

from run_functions import (
    compute_final_dist, compute_W2, prob_conf_conversion,
    cand_pref_outcome_sum, cand_pref_all_draws_outcomes,
    precompute_state_weights, compute_district_weights
)

# ============================================================
# USER PARAMETERS
# ============================================================

L_HUB   = 500    # steps to reach the hub from the enacted map
L_SPOKE = 500    # steps per spoke from the hub
M_SPOKES = 200   # number of independent spokes (samples)
                 # Total ReCom steps = L_HUB + M_SPOKES * L_SPOKE
                 # e.g. 500 + 200*500 = 100,500

N_WORKERS = max(1, mp.cpu_count() - 1)   # parallel workers for spokes
                                          # set to 1 to disable multiprocessing

START_MAP    = 'CD'     # 'CD', 'Seed_Demo', or 'new_seed'
RUN_NAME     = 'TX_BC_parallel'
MODEL_MODE   = 'statewide'   # 'statewide', 'equal', or 'district'
EFFECTIVENESS_CUTOFF = 0.6

# ============================================================
# FIXED PARAMETERS  (mirror TX_elections_model.py)
# ============================================================

NUM_DISTRICTS  = 36
POP_TOL        = 0.01
ENACTED_BLACK  = 4
ENACTED_HISP   = 8
ENACTED_DISTINCT = 11

PLOT_PATH = 'TX_VTDs/TX_VTDs.shp'
DIR       = ''

TOT_POP      = 'TOTPOP_x'
WHITE_POP    = 'NH_WHITE'
CVAP         = "1_2018"
WCVAP        = "7_2018"
HCVAP        = "13_2018"
BCVAP        = "5_2018"
GEO_ID       = 'CNTYVTD'
COUNTY_SPLIT_ID = "CNTY_x"
C_X = "C_X"
C_Y = "C_Y"

if not os.path.exists(DIR + 'outputs'):
    os.mkdir(DIR + 'outputs')

# ============================================================
# DATA LOADING  (identical to TX_elections_model.py)
# ============================================================

print("Loading data...")

elec_data        = pd.read_csv("TX_elections.csv")
TX_columns       = list(pd.read_csv("TX_columns.csv")["Columns"])
dropped_elecs    = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
recency_weights  = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("ingroup_weight.csv")
cand_race_table  = pd.read_csv("Candidate_Race_Party.csv")
EI_statewide     = pd.read_csv("statewide_rxc_EI_preferences.csv")
prec_ei_df       = pd.read_csv("prec_count_quants.csv",       dtype={'CNTYVTD': 'str'})
mean_prec_counts = pd.read_csv("mean_prec_vote_counts.csv",   dtype={'CNTYVTD': 'str'})
logit_params     = pd.read_csv('TX_logit_params.csv')

# ---- shapefile ----
state_gdf = gpd.read_file(PLOT_PATH)
state_gdf["CD"]        = state_gdf["CD"].astype('int')
state_gdf["Seed_Demo"] = state_gdf["Seed_Demo"].astype('int')
state_gdf.columns      = state_gdf.columns.str.replace("-", "_")

state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index + 1] = TX_columns
state_gdf.columns = state_gdf_cols

state_df = pd.DataFrame(state_gdf).drop(['geometry'], axis=1)

# ---- graph ----
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids_geom = state_gdf.centroid
for node in graph.nodes():
    graph.nodes[node]["C_X"] = centroids_geom.x[node]
    graph.nodes[node]["C_Y"] = centroids_geom.y[node]

# ---- elections data structures ----
elecs_bool       = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc  = elec_data[elecs_bool].reset_index(drop=True)
elec_sets        = list(set(elec_data_trunc["Election Set"]))
elections        = list(elec_data_trunc["Election"])
general_elecs    = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs    = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs     = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)
elec_cand_list   = TX_columns

elec_set_dict = {}
for elec_set in elec_sets:
    sub = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(sub.Type, sub.Election))

elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

candidates = {}
for elec in elections:
    cands = ([y for y in elec_cand_list if elec in y and "R_" not in y.split('1')[0]]
             if "R_" in elec[:4] or "P_" in elec[:4]
             else [y for y in elec_cand_list if elec in y])
    if elec in general_elecs:
        cands = cands[:2]
    candidates[elec] = dict(zip(range(len(cands)), cands))

cand_race_dict       = cand_race_table.set_index("Candidates").to_dict()["Race"]
min_cand_weights_dict = {k: min_cand_weights.to_dict()[k][0]
                         for k in min_cand_weights.to_dict().keys()}

# ---- recency W1 ----
elec_years     = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == es, 'Year'].values[0].astype(str)
                  for es in elec_sets]
recency_scores = [recency_weights[yr][0] for yr in elec_years]
recency_W1     = np.tile(recency_scores, (NUM_DISTRICTS, 1)).transpose()

# ---- statewide precompute ----
(black_weight_state, hisp_weight_state, neither_weight_state,
 black_weight_equal, hisp_weight_equal, neither_weight_equal,
 black_pref_cands_prim_state, hisp_pref_cands_prim_state,
 black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state) = \
    precompute_state_weights(
        NUM_DISTRICTS, elec_sets, elec_set_dict, recency_W1, EI_statewide,
        primary_elecs, runoff_elecs, elec_match_dict,
        min_cand_weights_dict, cand_race_dict
    )

# ---- district-mode precompute (precinct draws) ----
demogs = ['BCVAP', 'HCVAP']
bases  = {col.split('.')[0] + '.' + col.split('.')[1]
          for col in prec_ei_df.columns
          if col[:5] in demogs and 'abstain' not in col
          and not any(x in col for x in general_elecs)}
base_dict = {b: (b.split('.')[0].split('_')[0],
                 '_'.join(b.split('.')[1].split('_')[1:-1]))
             for b in bases}
outcomes = {val: [] for val in base_dict.values()}
for b in bases:
    outcomes[base_dict[b]].append(b)

precs               = list(state_gdf[GEO_ID])
prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

print("Data loaded.")

# ============================================================
# UPDATERS  (identical to TX_elections_model.py)
# ============================================================

def final_elec_model(partition):
    """Elections model updater — identical logic to TX_elections_model.py."""
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = (
            set([dict1[k] for k in dict1 if dict1[k] != dict2[k]]) |
            set([dict2[k] for k in dict2 if dict1[k] != dict2[k]])
        )
        dist_changes = sorted(differences)
    else:
        dist_changes = range(NUM_DISTRICTS)

    order = [x for x in partition.parts]

    dist_elec_results = {}
    for elec in elections:
        cands = candidates[elec]
        outcome_list = [dict(zip(order, partition[elec].percents(cand)))
                        for cand in cands.keys()]
        dist_elec_results[elec] = {
            d: {cands[i]: outcome_list[i][d] for i in cands.keys()}
            for d in range(NUM_DISTRICTS)
        }

    map_winners = pd.DataFrame(columns=dist_changes)
    map_winners["Election"]      = elections
    map_winners["Election Set"]  = elec_data_trunc["Election Set"]
    map_winners["Election Type"] = elec_data_trunc["Type"]
    for i in dist_changes:
        map_winners[i] = [
            max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0]
            for elec in elections
        ]

    # statewide & equal modes
    final_state_prob_dict = compute_final_dist(
        map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
        hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
        neither_weight_state, black_weight_state, hisp_weight_state,
        dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
        candidates, elec_sets, elec_set_dict, "statewide", partition,
        logit_params, logit=True
    )
    final_equal_prob_dict = compute_final_dist(
        map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
        hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
        neither_weight_equal, black_weight_equal, hisp_weight_equal,
        dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
        candidates, elec_sets, elec_set_dict, "equal", partition,
        logit_params, logit=True
    )

    # district mode
    (black_weight_dist, hisp_weight_dist, neither_weight_dist,
     black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,
     hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist) = \
        compute_district_weights(
            dist_changes, elec_sets, elec_set_dict, state_gdf, partition,
            prec_draws_outcomes, GEO_ID, primary_elecs, runoff_elecs,
            elec_match_dict, bases, outcomes, recency_W1,
            cand_race_dict, min_cand_weights_dict
        )
    final_dist_prob_dict = compute_final_dist(
        map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,
        hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist,
        neither_weight_dist, black_weight_dist, hisp_weight_dist,
        dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
        candidates, elec_sets, elec_set_dict, 'district', partition,
        logit_params, logit=True
    )

    if partition.parent is None:
        final_state_prob = {k: final_state_prob_dict[k] for k in sorted(final_state_prob_dict)}
        final_equal_prob = {k: final_equal_prob_dict[k] for k in sorted(final_equal_prob_dict)}
        final_dist_prob  = {k: final_dist_prob_dict[k]  for k in sorted(final_dist_prob_dict)}
    else:
        final_state_prob = partition.parent["final_elec_model"][0].copy()
        final_equal_prob = partition.parent["final_elec_model"][1].copy()
        final_dist_prob  = partition.parent["final_elec_model"][2].copy()
        for i in dist_changes:
            final_state_prob[i] = final_state_prob_dict[i]
            final_equal_prob[i] = final_equal_prob_dict[i]
            final_dist_prob[i]  = final_dist_prob_dict[i]

    return final_state_prob, final_equal_prob, final_dist_prob


def effective_districts(dictionary):
    """Count effective districts above the effectiveness cutoff."""
    if "N/A" in dictionary.values():
        return "N/A", "N/A", "N/A"
    hisp_effective  = [i + l for i, j, k, l in dictionary.values()]
    black_effective = [j + l for i, j, k, l in dictionary.values()]
    hisp_idx  = [i for i, n in enumerate(hisp_effective)  if n >= EFFECTIVENESS_CUTOFF]
    black_idx = [i for i, n in enumerate(black_effective) if n >= EFFECTIVENESS_CUTOFF]
    return len(hisp_idx), len(black_idx), len(set(hisp_idx + black_idx))


def _build_updaters():
    """Build the full updater dict including all election updaters."""
    my_updaters = {
        "population":        updaters.Tally(TOT_POP, alias="population"),
        "CVAP":              updaters.Tally(CVAP,    alias="CVAP"),
        "WCVAP":             updaters.Tally(WCVAP,   alias="WCVAP"),
        "HCVAP":             updaters.Tally(HCVAP,   alias="HCVAP"),
        "BCVAP":             updaters.Tally(BCVAP,   alias="BCVAP"),
        "Sum_CX":            updaters.Tally(C_X,     alias="Sum_CX"),
        "Sum_CY":            updaters.Tally(C_Y,     alias="Sum_CY"),
        "cut_edges":         cut_edges,
        "final_elec_model":  final_elec_model,
    }
    # benchmark elections
    benchmark = [
        Election("PRES16", {"Democratic": 'ClintonD_16G_President',
                             "Republican": 'TrumpR_16G_President'},   alias="PRES16"),
        Election("PRES12", {"Democratic": 'ObamaD_12G_President',
                             "Republican": 'RomneyR_12G_President'},  alias="PRES12"),
        Election("SEN18",  {"Democratic": "ORourkeD_18G_US_Sen",
                             "Republican": 'CruzR_18G_US_Sen'},       alias="SEN18"),
        Election("GOV18",  {"Democratic": "ValdezD_18G_Governor",
                             "Republican": 'AbbottR_18G_Governor'},   alias="GOV18"),
    ]
    my_updaters.update({e.name: e for e in benchmark})
    # all model elections
    my_updaters.update({e.name: e for e in [Election(j, candidates[j]) for j in elections]})
    return my_updaters


def _build_proposal_and_constraint(initial_partition):
    total_pop    = state_gdf[TOT_POP].sum()
    ideal_pop    = total_pop / NUM_DISTRICTS
    proposal = partial(recom, pop_col=TOT_POP, pop_target=ideal_pop,
                       epsilon=POP_TOL, node_repeats=3)
    pop_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, POP_TOL
    )
    return proposal, pop_constraint


def _vra_score_from_partition(partition):
    """
    Extract the scalar VRA summary statistics from a partition using MODEL_MODE.
    Returns (hisp_effective, black_effective, distinct_effective).
    """
    final_state_prob, final_equal_prob, final_dist_prob = partition["final_elec_model"]
    prob_dict = (final_state_prob  if MODEL_MODE == 'statewide' else
                 final_equal_prob  if MODEL_MODE == 'equal'     else
                 final_dist_prob)
    return effective_districts(prob_dict)


def _run_chain_get_endpoint(start_assignment, n_steps, seed):
    """
    Run a ReCom chain for n_steps from start_assignment.
    Returns the final partition (endpoint).
    This function is designed to be safe to call in a subprocess.
    """
    gc_random.seed(seed)

    my_updaters  = _build_updaters()
    init_part    = GeographicPartition(graph=graph,
                                       assignment=start_assignment,
                                       updaters=my_updaters)
    proposal, pop_constraint = _build_proposal_and_constraint(init_part)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint],
        accept=accept.always_accept,
        initial_state=init_part,
        total_steps=n_steps,
    )

    endpoint = init_part
    for step in chain:
        endpoint = step

    return dict(endpoint.assignment)   # return plain dict (serializable)


# ============================================================
# STEP 1 — Score the enacted map
# ============================================================

def score_enacted_map():
    """
    Build the enacted partition and compute its VRA scores.
    Returns (hisp, black, distinct, enacted_partition).
    """
    print("\n=== Scoring enacted map ===")
    total_pop = state_gdf[TOT_POP].sum()
    ideal_pop = total_pop / NUM_DISTRICTS

    if START_MAP == 'new_seed':
        assignment = recursive_tree_part(graph, range(NUM_DISTRICTS),
                                         ideal_pop, TOT_POP, POP_TOL, 3)
    else:
        assignment = START_MAP

    my_updaters = _build_updaters()
    enacted_partition = GeographicPartition(graph=graph,
                                            assignment=assignment,
                                            updaters=my_updaters)
    hisp, black, distinct = _vra_score_from_partition(enacted_partition)
    print(f"  Enacted map scores ({MODEL_MODE} mode):")
    print(f"    Latino-effective districts : {hisp}")
    print(f"    Black-effective districts  : {black}")
    print(f"    Distinct-effective         : {distinct}")
    return hisp, black, distinct, enacted_partition


# ============================================================
# STEP 2 — Run the hub chain
# ============================================================

def run_hub_chain(enacted_partition):
    """
    Run the chain forward L_HUB steps from the enacted map to reach the hub X*.
    Because ReCom with always_accept is reversible, this is equivalent to running
    the time-reversed chain, satisfying the Besag-Clifford parallel method requirement.

    Returns the hub assignment dict (plain dict, serializable).
    """
    print(f"\n=== Running hub chain ({L_HUB} steps) ===")
    t0 = time.time()

    hub_seed = stdlib_random.randint(0, 10**9)
    gc_random.seed(hub_seed)

    proposal, pop_constraint = _build_proposal_and_constraint(enacted_partition)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint],
        accept=accept.always_accept,
        initial_state=enacted_partition,
        total_steps=L_HUB,
    )

    hub_partition = enacted_partition
    for step in chain:
        hub_partition = step

    hub_assignment = dict(hub_partition.assignment)
    elapsed = time.time() - t0
    print(f"  Hub reached in {elapsed:.1f}s.")
    return hub_assignment


# ============================================================
# STEP 3 — Run M independent spoke chains from the hub
# ============================================================

def _spoke_worker(args):
    """
    Worker function for a single spoke.
    args = (spoke_index, hub_assignment, seed)
    Returns (spoke_index, hisp, black, distinct, assignment_dict)
    """
    spoke_idx, hub_assignment, seed = args
    endpoint_assignment = _run_chain_get_endpoint(hub_assignment, L_SPOKE, seed)

    # rebuild partition to score it
    my_updaters = _build_updaters()
    endpoint_partition = GeographicPartition(graph=graph,
                                             assignment=endpoint_assignment,
                                             updaters=my_updaters)
    hisp, black, distinct = _vra_score_from_partition(endpoint_partition)
    return spoke_idx, hisp, black, distinct, endpoint_assignment


def run_spokes(hub_assignment):
    """
    Spawn M_SPOKES independent chains from the hub, each of length L_SPOKE.
    Returns a DataFrame with one row per spoke.
    """
    print(f"\n=== Running {M_SPOKES} spokes (L={L_SPOKE} steps each, {N_WORKERS} workers) ===")
    t0 = time.time()

    seeds = [stdlib_random.randint(0, 10**9) for _ in range(M_SPOKES)]
    args  = [(i, hub_assignment, seeds[i]) for i in range(M_SPOKES)]

    results = []

    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, (idx, h, b, d, _) in enumerate(pool.imap_unordered(_spoke_worker, args)):
                results.append({'spoke': idx, 'hisp': h, 'black': b, 'distinct': d})
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  {i+1}/{M_SPOKES} spokes done  ({elapsed:.0f}s elapsed)")
    else:
        # single-process fallback (easier to debug)
        for i, arg in enumerate(args):
            idx, h, b, d, _ = _spoke_worker(arg)
            results.append({'spoke': idx, 'hisp': h, 'black': b, 'distinct': d})
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{M_SPOKES} spokes done  ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"  All spokes complete in {elapsed:.1f}s.")
    return pd.DataFrame(results)


# ============================================================
# STEP 4 — Compute exact Besag-Clifford p-values
# ============================================================

def compute_bc_pvalues(enacted_hisp, enacted_black, enacted_distinct, spoke_df):
    """
    Rank the enacted map among the spoke endpoints.

    The Besag-Clifford p-value (one-sided, lower tail — testing whether the
    enacted map has FEWER effective districts than expected under the null) is:

        p = (# spokes with score <= enacted score + 1) / (M + 1)

    A small p-value means the enacted map has unusually FEW effective minority
    districts relative to the null distribution of maps — evidence of vote dilution.

    We also compute the upper-tail p-value for completeness.
    """
    M = len(spoke_df)

    def pval(enacted_val, col):
        # lower tail: enacted is unusually low
        n_leq  = (spoke_df[col] <= enacted_val).sum()
        n_geq  = (spoke_df[col] >= enacted_val).sum()
        p_low  = (n_leq + 1) / (M + 1)
        p_high = (n_geq + 1) / (M + 1)
        rank   = n_leq + 1   # rank among (M+1) total including enacted
        return p_low, p_high, rank

    p_hisp_low,     p_hisp_high,     rank_hisp     = pval(enacted_hisp,     'hisp')
    p_black_low,    p_black_high,    rank_black    = pval(enacted_black,    'black')
    p_distinct_low, p_distinct_high, rank_distinct = pval(enacted_distinct, 'distinct')

    results = {
        'mode': MODEL_MODE,
        'L_hub': L_HUB, 'L_spoke': L_SPOKE, 'M_spokes': M,

        'enacted_hisp':     enacted_hisp,
        'enacted_black':    enacted_black,
        'enacted_distinct': enacted_distinct,

        'mean_hisp':     spoke_df['hisp'].mean(),
        'mean_black':    spoke_df['black'].mean(),
        'mean_distinct': spoke_df['distinct'].mean(),

        'median_hisp':     spoke_df['hisp'].median(),
        'median_black':    spoke_df['black'].median(),
        'median_distinct': spoke_df['distinct'].median(),

        # lower-tail: enacted has fewer effective dists than typical (vote dilution direction)
        'p_hisp_lower':     p_hisp_low,
        'p_black_lower':    p_black_low,
        'p_distinct_lower': p_distinct_low,

        # upper-tail: enacted has more effective dists than typical
        'p_hisp_upper':     p_hisp_high,
        'p_black_upper':    p_black_high,
        'p_distinct_upper': p_distinct_high,

        'rank_hisp':     rank_hisp,
        'rank_black':    rank_black,
        'rank_distinct': rank_distinct,
    }
    return results


def print_results(results, spoke_df):
    M = results['M_spokes']
    print("\n" + "="*60)
    print("BESAG-CLIFFORD EXACT SIGNIFICANCE TEST RESULTS")
    print("="*60)
    print(f"Model mode          : {results['mode']}")
    print(f"Hub length (L)      : {results['L_hub']}")
    print(f"Spoke length (L)    : {results['L_spoke']}")
    print(f"Num spokes (M)      : {M}")
    print(f"Effectiveness cutoff: {EFFECTIVENESS_CUTOFF}")
    print()

    for group, key in [("Latino", "hisp"), ("Black", "black"), ("Distinct", "distinct")]:
        enacted = results[f'enacted_{key}']
        mean    = results[f'mean_{key}']
        median  = results[f'median_{key}']
        p_low   = results[f'p_{key}_lower']
        p_high  = results[f'p_{key}_upper']
        rank    = results[f'rank_{key}']
        print(f"  {group} effective districts:")
        print(f"    Enacted map       : {enacted}")
        print(f"    Spoke mean/median : {mean:.2f} / {median:.1f}")
        print(f"    Rank (of {M+1})      : {rank}")
        print(f"    p-value (lower ←) : {p_low:.4f}  {'*** SIGNIFICANT' if p_low < 0.05 else ''}")
        print(f"    p-value (upper →) : {p_high:.4f}")
        print()

    print("Interpretation:")
    print("  Lower-tail p < 0.05 → enacted map has UNUSUALLY FEW effective districts")
    print("  (potential vote dilution under the VRA)")
    print("  Upper-tail p < 0.05 → enacted map has UNUSUALLY MANY effective districts")
    print()
    print("Note: these p-values are EXACTLY valid by Besag-Clifford (1989) Proposition 3.3,")
    print("regardless of whether the ReCom chain has mixed.")
    print("="*60)


# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.time()

    # seed global randomness (each spoke gets its own seed below)
    master_seed = stdlib_random.randint(0, 10**9)
    gc_random.seed(master_seed)
    print(f"Master seed: {master_seed}")

    # 1. Score the enacted map
    enacted_hisp, enacted_black, enacted_distinct, enacted_partition = score_enacted_map()

    # 2. Run hub chain  (forward L_HUB steps from enacted map)
    hub_assignment = run_hub_chain(enacted_partition)

    # 3. Run M independent spokes from hub
    spoke_df = run_spokes(hub_assignment)

    # 4. Compute exact B-C p-values
    results = compute_bc_pvalues(enacted_hisp, enacted_black, enacted_distinct, spoke_df)

    # 5. Print and save
    print_results(results, spoke_df)

    spoke_df.to_csv(DIR + f"outputs/bc_spoke_scores_{RUN_NAME}.csv", index=False)
    pd.DataFrame([results]).to_csv(DIR + f"outputs/bc_pvalues_{RUN_NAME}.csv", index=False)

    print(f"\nSpoke scores saved to: outputs/bc_spoke_scores_{RUN_NAME}.csv")
    print(f"P-values saved to    : outputs/bc_pvalues_{RUN_NAME}.csv")
    print(f"\nTotal elapsed: {(time.time() - total_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
