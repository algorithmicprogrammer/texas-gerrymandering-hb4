# -*- coding: utf-8 -*-
"""
besag_clifford_vra_functional.py
=================================

Diversity-Calibrated Minority Opportunity Functional
integrated with the Besag-Clifford (1989) parallel exact significance test.

Functional Definition
---------------------
For group g in {Black, Latino}:

    F_g(pi) = sum_d p_d^g(pi)  -  K * m_bar_g

where:
    p_d^g(pi)  = logit-model probability that group g's preferred candidate
                 wins district d under plan pi  (from final_elec_model,
                 statewide EI mode)
    K          = number of districts (36 for TX congressional)
    m_bar_g    = group g's statewide CVAP fraction  (from shapefile)

Coalition functional:
    F_joint(pi) = sum_d max(p_d^B(pi), p_d^L(pi))  -  K * m_bar_combined

Interpretation (De Grandy, 512 U.S. 997 (1994)):
    F_g < 0  -->  sub-proportional (vote dilution direction)
    F_g = 0  -->  proportional to demographic share (De Grandy baseline)
    F_g > 0  -->  supra-proportional

B-C lower-tail p-value:
    p = (# spokes with F <= F_enacted + 1) / (M + 1)
    Exactly valid by Besag-Clifford (1989) Prop 3.3, regardless of mixing time.

Changes from besag_clifford_vra.py
-----------------------------------
  NEW:      _compute_statewide_fractions()
  NEW:      compute_opportunity_functional()
  NEW:      save_distribution_plots()
  NEW:      save_latex_table()
  REPLACED: effective_districts()  -->  _functional_score_from_partition()
  REPLACED: _spoke_worker(), run_spokes(), compute_bc_pvalues(),
            print_results(), score_enacted_map(), main()
  UNCHANGED: all data loading, final_elec_model(), _build_updaters(),
             _build_proposal_and_constraint(), _run_chain_get_endpoint(),
             run_hub_chain(), all USER PARAMETERS and FIXED PARAMETERS
"""

import random as stdlib_random
import time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # headless backend -- must precede pyplot import
import matplotlib.pyplot as plt
from functools import partial
import operator
import multiprocessing as mp

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
# USER PARAMETERS  (identical to besag_clifford_vra.py)
# ============================================================

L_HUB     = 500
L_SPOKE   = 500
M_SPOKES  = 200
N_WORKERS = max(1, mp.cpu_count() - 1)

START_MAP  = 'CD'
RUN_NAME   = 'TX_BC_functional'
MODEL_MODE = 'statewide'

EFFECTIVENESS_CUTOFF = 0.6  # kept for legacy comparison columns only

# ============================================================
# FIXED PARAMETERS
# ============================================================

NUM_DISTRICTS   = 36
POP_TOL         = 0.01
PLOT_PATH       = 'TX_VTDs/TX_VTDs.shp'
DIR             = ''

TOT_POP         = 'TOTPOP_x'
WHITE_POP       = 'NH_WHITE'
CVAP            = "1_2018"
WCVAP           = "7_2018"
HCVAP           = "13_2018"
BCVAP           = "5_2018"
GEO_ID          = 'CNTYVTD'
COUNTY_SPLIT_ID = "CNTY_x"
C_X             = "C_X"
C_Y             = "C_Y"

if not os.path.exists(DIR + 'outputs'):
    os.mkdir(DIR + 'outputs')

# ============================================================
# DATA LOADING  (identical to besag_clifford_vra.py)
# ============================================================

print("Loading data...")

elec_data        = pd.read_csv("TX_elections.csv")
TX_columns       = list(pd.read_csv("TX_columns.csv")["Columns"])
dropped_elecs    = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
recency_weights  = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("ingroup_weight.csv")
cand_race_table  = pd.read_csv("Candidate_Race_Party.csv")
EI_statewide     = pd.read_csv("statewide_rxc_EI_preferences.csv")
prec_ei_df       = pd.read_csv("prec_count_quants.csv",     dtype={'CNTYVTD': 'str'})
mean_prec_counts = pd.read_csv("mean_prec_vote_counts.csv", dtype={'CNTYVTD': 'str'})
logit_params     = pd.read_csv('TX_logit_params.csv')

# ---- shapefile ----
state_gdf = gpd.read_file(PLOT_PATH)
state_gdf["CD"]        = state_gdf["CD"].astype('int')
state_gdf["Seed_Demo"] = state_gdf["Seed_Demo"].astype('int')
state_gdf.columns      = state_gdf.columns.str.replace("-", "_")
state_gdf_cols         = list(state_gdf.columns)
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
elecs_bool      = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop=True)
elec_sets       = list(set(elec_data_trunc["Election Set"]))
elections       = list(elec_data_trunc["Election"])
general_elecs   = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs   = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs    = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)
elec_cand_list  = TX_columns

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

cand_race_dict        = cand_race_table.set_index("Candidates").to_dict()["Race"]
min_cand_weights_dict = {k: min_cand_weights.to_dict()[k][0]
                          for k in min_cand_weights.to_dict().keys()}

elec_years     = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == es, 'Year']
                  .values[0].astype(str) for es in elec_sets]
recency_scores = [recency_weights[yr][0] for yr in elec_years]
recency_W1     = np.tile(recency_scores, (NUM_DISTRICTS, 1)).transpose()

(black_weight_state,  hisp_weight_state,  neither_weight_state,
 black_weight_equal,  hisp_weight_equal,  neither_weight_equal,
 black_pref_cands_prim_state,   hisp_pref_cands_prim_state,
 black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state) = \
    precompute_state_weights(
        NUM_DISTRICTS, elec_sets, elec_set_dict, recency_W1, EI_statewide,
        primary_elecs, runoff_elecs, elec_match_dict,
        min_cand_weights_dict, cand_race_dict
    )

demogs    = ['BCVAP', 'HCVAP']
bases     = {col.split('.')[0] + '.' + col.split('.')[1]
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
# STATEWIDE DEMOGRAPHIC FRACTIONS  (new)
# ============================================================

def _compute_statewide_fractions(gdf):
    """
    Compute statewide minority CVAP fractions from the shapefile.

    Called once at module load. Results stored as module-level
    constants M_BAR_B, M_BAR_L, M_BAR_COMBINED.

    Returns
    -------
    m_bar_B        : float  Black CVAP / Total CVAP
    m_bar_L        : float  Latino CVAP / Total CVAP
    m_bar_combined : float  (Black + Latino) CVAP / Total CVAP
    """
    total_cvap    = gdf[CVAP].sum()
    black_cvap    = gdf[BCVAP].sum()
    hisp_cvap     = gdf[HCVAP].sum()
    combined_cvap = black_cvap + hisp_cvap

    m_bar_B        = black_cvap    / total_cvap
    m_bar_L        = hisp_cvap     / total_cvap
    m_bar_combined = combined_cvap  / total_cvap

    print(f"\nStatewide CVAP fractions:")
    print(f"  m_bar_B        = {m_bar_B:.4f}  ({100*m_bar_B:.2f}%)")
    print(f"  m_bar_L        = {m_bar_L:.4f}  ({100*m_bar_L:.2f}%)")
    print(f"  m_bar_combined = {m_bar_combined:.4f}  ({100*m_bar_combined:.2f}%)")
    print(f"  Proportional seat targets:")
    print(f"    Black    : {NUM_DISTRICTS * m_bar_B:.2f}")
    print(f"    Latino   : {NUM_DISTRICTS * m_bar_L:.2f}")
    print(f"    Combined : {NUM_DISTRICTS * m_bar_combined:.2f}")

    return m_bar_B, m_bar_L, m_bar_combined


M_BAR_B, M_BAR_L, M_BAR_COMBINED = _compute_statewide_fractions(state_gdf)

# ============================================================
# UPDATERS  (identical to besag_clifford_vra.py)
# ============================================================

def final_elec_model(partition):
    """
    Elections model updater. Identical to besag_clifford_vra.py.

    Computes per-district win probabilities for minority-preferred
    candidates under statewide, equal, and district EI modes.

    Returns
    -------
    (final_state_prob, final_equal_prob, final_dist_prob)
    Each: {district_index: (hisp_prob, black_prob, neither_prob, combined_prob)}
    """
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

    # statewide mode
    final_state_prob_dict = compute_final_dist(
        map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
        hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
        neither_weight_state, black_weight_state, hisp_weight_state,
        dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
        candidates, elec_sets, elec_set_dict, "statewide", partition,
        logit_params, logit=True
    )

    # equal mode
    final_equal_prob_dict = compute_final_dist(
        map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,
        hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
        neither_weight_equal, black_weight_equal, hisp_weight_equal,
        dist_elec_results, dist_changes, cand_race_table, NUM_DISTRICTS,
        candidates, elec_sets, elec_set_dict, "equal", partition,
        logit_params, logit=True
    )

    # district mode
    (black_weight_dist,  hisp_weight_dist,  neither_weight_dist,
     black_pref_cands_prim_dist,   black_pref_cands_runoffs_dist,
     hisp_pref_cands_prim_dist,    hisp_pref_cands_runoffs_dist) = \
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


# ============================================================
# DIVERSITY-CALIBRATED MINORITY OPPORTUNITY FUNCTIONAL  (new)
# ============================================================

def compute_opportunity_functional(prob_dict,
                                   m_bar_B, m_bar_L, m_bar_combined,
                                   K=NUM_DISTRICTS):
    """
    Diversity-Calibrated Minority Opportunity Functional.

    F_g(pi) = sum_d p_d^g(pi)  -  K * m_bar_g

    Parameters
    ----------
    prob_dict : dict
        {district_index: (hisp_prob, black_prob, neither_prob, combined_prob)}
        Output of final_elec_model for the chosen MODEL_MODE.
    m_bar_B, m_bar_L, m_bar_combined : float
        Statewide CVAP fractions (computed once at module load).
    K : int
        Number of districts.

    Returns
    -------
    F_B     : float  Black opportunity gap (districts)
    F_L     : float  Latino opportunity gap (districts)
    F_joint : float  Coalition gap: sum_d max(p_d^B, p_d^L) - K * m_bar_combined

    Returns (nan, nan, nan) if any district has "N/A" data.
    """
    if "N/A" in prob_dict.values():
        return float('nan'), float('nan'), float('nan')

    p_B_list, p_L_list = [], []
    for d in sorted(prob_dict.keys()):
        hisp_prob, black_prob, neither_prob, combined_prob = prob_dict[d]
        p_L_list.append(hisp_prob)
        p_B_list.append(black_prob)

    p_B_arr = np.array(p_B_list)
    p_L_arr = np.array(p_L_list)

    F_B     = float(p_B_arr.sum() - K * m_bar_B)
    F_L     = float(p_L_arr.sum() - K * m_bar_L)
    F_joint = float(np.maximum(p_B_arr, p_L_arr).sum() - K * m_bar_combined)

    return F_B, F_L, F_joint


def _functional_score_from_partition(partition):
    """
    Extract F_B, F_L, F_joint from a partition using MODEL_MODE.

    Also computes legacy threshold counts (Becker et al. 2021) written
    to spoke CSV as comparison columns. NOT used as the B-C test statistic.

    Returns
    -------
    F_B, F_L, F_joint              : float
    hisp_eff, black_eff, distinct  : int  (legacy)
    """
    final_state_prob, final_equal_prob, final_dist_prob = \
        partition["final_elec_model"]

    prob_dict = (final_state_prob if MODEL_MODE == 'statewide' else
                 final_equal_prob if MODEL_MODE == 'equal'     else
                 final_dist_prob)

    F_B, F_L, F_joint = compute_opportunity_functional(
        prob_dict, M_BAR_B, M_BAR_L, M_BAR_COMBINED
    )

    if "N/A" not in prob_dict.values():
        hisp_eff  = sum(1 for v in prob_dict.values()
                        if v[0] >= EFFECTIVENESS_CUTOFF)
        black_eff = sum(1 for v in prob_dict.values()
                        if v[1] >= EFFECTIVENESS_CUTOFF)
        distinct  = len(
            set(d for d, v in prob_dict.items() if v[0] >= EFFECTIVENESS_CUTOFF) |
            set(d for d, v in prob_dict.items() if v[1] >= EFFECTIVENESS_CUTOFF)
        )
    else:
        hisp_eff = black_eff = distinct = float('nan')

    return F_B, F_L, F_joint, hisp_eff, black_eff, distinct


# ============================================================
# CHAIN INFRASTRUCTURE  (identical to besag_clifford_vra.py)
# ============================================================

def _build_updaters():
    my_updaters = {
        "population": updaters.Tally(TOT_POP, alias="population"),
        "CVAP":       updaters.Tally(CVAP,    alias="CVAP"),
        "WCVAP":      updaters.Tally(WCVAP,   alias="WCVAP"),
        "HCVAP":      updaters.Tally(HCVAP,   alias="HCVAP"),
        "BCVAP":      updaters.Tally(BCVAP,   alias="BCVAP"),
        "Sum_CX":     updaters.Tally(C_X,     alias="Sum_CX"),
        "Sum_CY":     updaters.Tally(C_Y,     alias="Sum_CY"),
        "cut_edges":  cut_edges,
        "final_elec_model": final_elec_model,
    }
    benchmark = [
        Election("PRES16", {"Democratic": 'ClintonD_16G_President',
                             "Republican": 'TrumpR_16G_President'}, alias="PRES16"),
        Election("PRES12", {"Democratic": 'ObamaD_12G_President',
                             "Republican": 'RomneyR_12G_President'}, alias="PRES12"),
        Election("SEN18",  {"Democratic": "ORourkeD_18G_US_Sen",
                             "Republican": 'CruzR_18G_US_Sen'},       alias="SEN18"),
        Election("GOV18",  {"Democratic": "ValdezD_18G_Governor",
                             "Republican": 'AbbottR_18G_Governor'},    alias="GOV18"),
    ]
    my_updaters.update({e.name: e for e in benchmark})
    my_updaters.update({e.name: e for e in [Election(j, candidates[j]) for j in elections]})
    return my_updaters


def _build_proposal_and_constraint(initial_partition):
    total_pop = state_gdf[TOT_POP].sum()
    ideal_pop = total_pop / NUM_DISTRICTS
    proposal  = partial(recom, pop_col=TOT_POP, pop_target=ideal_pop,
                        epsilon=POP_TOL, node_repeats=3)
    pop_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, POP_TOL
    )
    return proposal, pop_constraint


def _run_chain_get_endpoint(start_assignment, n_steps, seed):
    """Run ReCom for n_steps. Returns plain-dict assignment of final state."""
    gc_random.seed(seed)
    my_updaters = _build_updaters()
    init_part   = GeographicPartition(graph=graph,
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
    return dict(endpoint.assignment)


# ============================================================
# STEP 1 — Score the enacted map
# ============================================================

def score_enacted_map():
    print("\n=== Scoring enacted map ===")
    total_pop = state_gdf[TOT_POP].sum()
    ideal_pop = total_pop / NUM_DISTRICTS

    if START_MAP == 'new_seed':
        assignment = recursive_tree_part(graph, range(NUM_DISTRICTS),
                                         ideal_pop, TOT_POP, POP_TOL, 3)
    else:
        assignment = START_MAP

    my_updaters       = _build_updaters()
    enacted_partition = GeographicPartition(graph=graph,
                                            assignment=assignment,
                                            updaters=my_updaters)

    F_B, F_L, F_joint, h_eff, b_eff, d_eff = \
        _functional_score_from_partition(enacted_partition)

    print(f"\n  Enacted map functional scores ({MODEL_MODE} mode):")
    print(f"  F_B     = {F_B:+.4f} districts  "
          f"(proportional target: {NUM_DISTRICTS * M_BAR_B:.2f})")
    print(f"  F_L     = {F_L:+.4f} districts  "
          f"(proportional target: {NUM_DISTRICTS * M_BAR_L:.2f})")
    print(f"  F_joint = {F_joint:+.4f} districts  "
          f"(proportional target: {NUM_DISTRICTS * M_BAR_COMBINED:.2f})")
    print(f"\n  Realized seat sums:")
    print(f"    sum p_d^B = {F_B   + NUM_DISTRICTS * M_BAR_B:.4f}")
    print(f"    sum p_d^L = {F_L   + NUM_DISTRICTS * M_BAR_L:.4f}")
    print(f"    sum max   = {F_joint + NUM_DISTRICTS * M_BAR_COMBINED:.4f}")
    print(f"\n  Legacy threshold counts (cutoff = {EFFECTIVENESS_CUTOFF}):")
    print(f"    Latino-effective : {h_eff}")
    print(f"    Black-effective  : {b_eff}")
    print(f"    Distinct         : {d_eff}")

    return F_B, F_L, F_joint, enacted_partition


# ============================================================
# STEP 2 — Hub chain  (identical to besag_clifford_vra.py)
# ============================================================

def run_hub_chain(enacted_partition):
    """
    Run the chain forward L_HUB steps from the enacted map to reach hub X*.

    ReCom with always_accept is reversible, so running forward L steps is
    equivalent to drawing a map L steps away in state space, satisfying the
    Besag-Clifford parallel method requirement (B-C 1989, Prop 3.3).
    """
    print(f"\n=== Running hub chain ({L_HUB} steps) ===")
    t0       = time.time()
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
    print(f"  Hub reached in {time.time() - t0:.1f}s.")
    return hub_assignment


# ============================================================
# STEP 3 — Spoke chains
# ============================================================

def _spoke_worker(args):
    """
    Worker for a single spoke.
    args = (spoke_index, hub_assignment, seed)
    Returns (spoke_idx, F_B, F_L, F_joint, hisp_eff, black_eff, distinct_eff)
    """
    spoke_idx, hub_assignment, seed = args
    endpoint_assignment = _run_chain_get_endpoint(hub_assignment, L_SPOKE, seed)
    my_updaters         = _build_updaters()
    endpoint_partition  = GeographicPartition(graph=graph,
                                              assignment=endpoint_assignment,
                                              updaters=my_updaters)
    F_B, F_L, F_joint, h_eff, b_eff, d_eff = \
        _functional_score_from_partition(endpoint_partition)
    return spoke_idx, F_B, F_L, F_joint, h_eff, b_eff, d_eff


def run_spokes(hub_assignment):
    """
    Spawn M_SPOKES independent chains from hub, each of length L_SPOKE.
    Returns DataFrame with one row per spoke.
    """
    print(f"\n=== Running {M_SPOKES} spokes "
          f"(L={L_SPOKE} steps each, {N_WORKERS} workers) ===")
    t0    = time.time()
    seeds = [stdlib_random.randint(0, 10**9) for _ in range(M_SPOKES)]
    args  = [(i, hub_assignment, seeds[i]) for i in range(M_SPOKES)]
    rows  = []

    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, (idx, fb, fl, fj, he, be, de) in enumerate(
                    pool.imap_unordered(_spoke_worker, args)):
                rows.append({'spoke': idx,
                             'F_B': fb, 'F_L': fl, 'F_joint': fj,
                             'hisp_eff': he, 'black_eff': be, 'distinct_eff': de})
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{M_SPOKES} spokes done "
                          f"({time.time()-t0:.0f}s elapsed)")
    else:
        for i, arg in enumerate(args):
            idx, fb, fl, fj, he, be, de = _spoke_worker(arg)
            rows.append({'spoke': idx,
                         'F_B': fb, 'F_L': fl, 'F_joint': fj,
                         'hisp_eff': he, 'black_eff': be, 'distinct_eff': de})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{M_SPOKES} spokes done "
                      f"({time.time()-t0:.0f}s elapsed)")

    print(f"  All spokes complete in {time.time()-t0:.1f}s.")
    return pd.DataFrame(rows)


# ============================================================
# STEP 4 — Exact Besag-Clifford p-values
# ============================================================

def compute_bc_pvalues(enacted_F_B, enacted_F_L, enacted_F_joint, spoke_df):
    """
    Rank enacted map among spoke endpoints.

    Lower-tail p (vote dilution direction):
        p = (# spokes with F <= F_enacted + 1) / (M + 1)

    Exactly valid by Besag & Clifford (1989) Prop 3.3,
    regardless of ReCom mixing time.
    """
    M = len(spoke_df)

    def _pval(enacted_val, col):
        vals   = spoke_df[col].dropna()
        n_leq  = (vals <= enacted_val).sum()
        n_geq  = (vals >= enacted_val).sum()
        p_low  = (n_leq + 1) / (M + 1)
        p_high = (n_geq + 1) / (M + 1)
        rank   = int(n_leq) + 1
        return p_low, p_high, rank

    p_B_low,     p_B_high,     rank_B     = _pval(enacted_F_B,     'F_B')
    p_L_low,     p_L_high,     rank_L     = _pval(enacted_F_L,     'F_L')
    p_joint_low, p_joint_high, rank_joint = _pval(enacted_F_joint, 'F_joint')

    return {
        'mode': MODEL_MODE, 'L_hub': L_HUB, 'L_spoke': L_SPOKE,
        'M_spokes': M, 'cutoff': EFFECTIVENESS_CUTOFF,
        'm_bar_B': M_BAR_B, 'm_bar_L': M_BAR_L,
        'm_bar_combined': M_BAR_COMBINED,
        'prop_seats_B':    NUM_DISTRICTS * M_BAR_B,
        'prop_seats_L':    NUM_DISTRICTS * M_BAR_L,
        'prop_seats_comb': NUM_DISTRICTS * M_BAR_COMBINED,
        'enacted_F_B':     enacted_F_B,
        'enacted_F_L':     enacted_F_L,
        'enacted_F_joint': enacted_F_joint,
        'mean_F_B':        spoke_df['F_B'].mean(),
        'mean_F_L':        spoke_df['F_L'].mean(),
        'mean_F_joint':    spoke_df['F_joint'].mean(),
        'median_F_B':      spoke_df['F_B'].median(),
        'median_F_L':      spoke_df['F_L'].median(),
        'median_F_joint':  spoke_df['F_joint'].median(),
        'sd_F_B':          spoke_df['F_B'].std(),
        'sd_F_L':          spoke_df['F_L'].std(),
        'sd_F_joint':      spoke_df['F_joint'].std(),
        'p5_F_B':          spoke_df['F_B'].quantile(0.05),
        'p5_F_L':          spoke_df['F_L'].quantile(0.05),
        'p5_F_joint':      spoke_df['F_joint'].quantile(0.05),
        'p95_F_B':         spoke_df['F_B'].quantile(0.95),
        'p95_F_L':         spoke_df['F_L'].quantile(0.95),
        'p95_F_joint':     spoke_df['F_joint'].quantile(0.95),
        'p_B_lower':       p_B_low,     'p_B_upper':     p_B_high,
        'rank_B':          rank_B,
        'p_L_lower':       p_L_low,     'p_L_upper':     p_L_high,
        'rank_L':          rank_L,
        'p_joint_lower':   p_joint_low, 'p_joint_upper': p_joint_high,
        'rank_joint':      rank_joint,
    }


# ============================================================
# OUTPUT — Console
# ============================================================

def print_results(results, spoke_df):
    M   = results['M_spokes']
    sig = lambda p: ' *** SIGNIFICANT' if p < 0.05 else (' * marginal' if p < 0.10 else '')

    print("\n" + "=" * 65)
    print("DIVERSITY-CALIBRATED MINORITY OPPORTUNITY FUNCTIONAL")
    print("BESAG-CLIFFORD EXACT SIGNIFICANCE TEST -- TEXAS HB4")
    print("=" * 65)
    print(f"Model mode           : {results['mode']}")
    print(f"Hub length (L)       : {results['L_hub']}")
    print(f"Spoke length (L)     : {results['L_spoke']}")
    print(f"Num spokes (M)       : {M}")
    print(f"Effectiveness cutoff : {results['cutoff']}  (legacy comparison only)")
    print()
    print("Statewide CVAP fractions (De Grandy proportional baseline):")
    print(f"  m_bar_B        = {results['m_bar_B']:.4f}  "
          f"-> proportional seats = {results['prop_seats_B']:.2f}")
    print(f"  m_bar_L        = {results['m_bar_L']:.4f}  "
          f"-> proportional seats = {results['prop_seats_L']:.2f}")
    print(f"  m_bar_combined = {results['m_bar_combined']:.4f}  "
          f"-> proportional seats = {results['prop_seats_comb']:.2f}")
    print()

    for label, col, enacted, mean, sd, p_low, p_high, rank in [
        ("Black",     'F_B',
         results['enacted_F_B'],     results['mean_F_B'],     results['sd_F_B'],
         results['p_B_lower'],       results['p_B_upper'],    results['rank_B']),
        ("Latino",    'F_L',
         results['enacted_F_L'],     results['mean_F_L'],     results['sd_F_L'],
         results['p_L_lower'],       results['p_L_upper'],    results['rank_L']),
        ("Coalition", 'F_joint',
         results['enacted_F_joint'], results['mean_F_joint'], results['sd_F_joint'],
         results['p_joint_lower'],   results['p_joint_upper'], results['rank_joint']),
    ]:
        print(f"  {label} functional:")
        print(f"    Enacted F value    : {enacted:+.4f} districts")
        print(f"    Null mean +/- SD   : {mean:+.4f} +/- {sd:.4f}")
        print(f"    Rank (of {M+1})       : {rank}")
        print(f"    p-value (lower <-) : {p_low:.4f}{sig(p_low)}")
        print(f"    p-value (upper ->) : {p_high:.4f}")
        print()

    print("Interpretation:")
    print("  F_g < 0 + small lower-tail p  ->  sub-proportional minority")
    print("  opportunity relative to De Grandy baseline -- evidence of")
    print("  vote dilution under Section 2 VRA.")
    print()
    print("  p-values EXACTLY valid (Besag & Clifford 1989, Prop 3.3),")
    print("  regardless of ReCom mixing time.")
    print("=" * 65)


# ============================================================
# OUTPUT — Distribution Plots
# ============================================================

def save_distribution_plots(enacted_F_B, enacted_F_L, enacted_F_joint,
                            spoke_df, run_name):
    """
    Three-panel null distribution plot. Saves PDF + PNG.
    PDF is suitable for direct paper inclusion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Besag-Clifford Null Distribution -- Diversity-Calibrated\n'
        'Minority Opportunity Functionals  (Texas HB4)\n'
        f'({len(spoke_df)} spokes, L_spoke={L_SPOKE}, L_hub={L_HUB})',
        fontsize=12, fontweight='bold'
    )

    configs = [
        ('F_B',     enacted_F_B,     'Black',     '#2166ac',
         r'$F_B(\pi) = \sum_d p_d^B - K\bar{m}_B$'),
        ('F_L',     enacted_F_L,     'Latino',    '#d6604d',
         r'$F_L(\pi) = \sum_d p_d^L - K\bar{m}_L$'),
        ('F_joint', enacted_F_joint, 'Coalition', '#4dac26',
         r'$F_{\rm joint}(\pi) = \sum_d \max(p_d^B,p_d^L) - K\bar{m}_{\rm comb}$'),
    ]

    for ax, (col, enacted_val, label, color, formula) in zip(axes, configs):
        vals = spoke_df[col].dropna()

        ax.hist(vals, bins=30, color=color, alpha=0.65,
                edgecolor='white', linewidth=0.5,
                label='Spoke null distribution')
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--',
                   label='Proportional baseline ($F=0$)')
        ax.axvline(enacted_val, color='red', linewidth=2.0, linestyle='-',
                   label=f'Enacted ({enacted_val:+.3f})')
        ax.axvspan(vals.min(), np.percentile(vals, 5),
                   alpha=0.12, color='red', label='Lower 5% of null')

        n_leq = (vals <= enacted_val).sum()
        p_val = (n_leq + 1) / (len(vals) + 1)
        ax.text(0.03, 0.97,
                f'p = {p_val:.4f}{"*" if p_val < 0.05 else ""}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, fontweight='bold',
                color='red' if p_val < 0.05 else 'black')

        ax.set_xlabel('F value (districts)', fontsize=10)
        ax.set_ylabel('Count' if ax is axes[0] else '', fontsize=10)
        ax.set_title(f'{label}\n{formula}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    base = DIR + f'outputs/bc_functional_distributions_{run_name}'
    plt.savefig(base + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(base + '.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Distribution plots saved:")
    print(f"  {base}.pdf")
    print(f"  {base}.png")
    return base + '.pdf'


# ============================================================
# OUTPUT — LaTeX Table
# ============================================================

def save_latex_table(results, run_name):
    """
    Write a publication-ready LaTeX table.

    Include in paper with:
        \\input{outputs/bc_functional_table_<run_name>.tex}

    Requires: \\usepackage{booktabs}
    """
    def stars(p):
        if p < 0.01: return r'$^{***}$'
        if p < 0.05: return r'$^{**}$'
        if p < 0.10: return r'$^{*}$'
        return ''

    M  = results['M_spokes']
    pB = results['p_B_lower']
    pL = results['p_L_lower']
    pJ = results['p_joint_lower']

    lines = [
        r'% Auto-generated by besag_clifford_vra_functional.py',
        r'% Requires: \usepackage{booktabs}',
        r'\begin{table}[htbp]',
        r'\centering',
        (r'\caption{Diversity-Calibrated Minority Opportunity Functional: '
         r'Besag-Clifford Exact Significance Test (Texas HB4). '
         r'$F_g(\pi)=\sum_d p_d^g(\pi)-K\bar{m}_g$ measures the gap between '
         r'realized minority electoral opportunity and the De~Grandy proportional '
         r'baseline. $p_d^g(\pi)$ is the logit-model probability that group $g$\textquoteright s '
         r'preferred candidate wins district $d$ under plan $\pi$, estimated using '
         r'statewide ecological inference weights and recency-discounted election history. '
         r'$p$-values are exactly valid by \citet{besagclifford1989}, Proposition~3.3, '
         r'regardless of ReCom mixing time.}'),
        r'\label{tab:bc_functional}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        (r'& \textbf{Black} $F_B$ '
         r'& \textbf{Latino} $F_L$ '
         r'& \textbf{Coalition} $F_{\rm joint}$ \\'),
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Statewide proportional baseline}} \\',
        (f"Statewide CVAP fraction $\\bar{{m}}_g$ "
         f"& {results['m_bar_B']:.4f} "
         f"& {results['m_bar_L']:.4f} "
         f"& {results['m_bar_combined']:.4f} \\\\"),
        (f"Proportional seat target $K\\bar{{m}}_g$ "
         f"& {results['prop_seats_B']:.2f} "
         f"& {results['prop_seats_L']:.2f} "
         f"& {results['prop_seats_comb']:.2f} \\\\"),
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Enacted map}} \\',
        (f"$F_g(\\pi_{{\\rm enacted}})$ "
         f"& {results['enacted_F_B']:+.4f} "
         f"& {results['enacted_F_L']:+.4f} "
         f"& {results['enacted_F_joint']:+.4f} \\\\"),
        r'\midrule',
        (r'\multicolumn{4}{l}{\textit{Null distribution '
         r'($M=' + str(M) + r'$ spokes, $L=' + str(L_SPOKE) + r'$ steps each)}} \\'),
        (f"Mean $\\bar{{F}}_g$ "
         f"& {results['mean_F_B']:+.4f} "
         f"& {results['mean_F_L']:+.4f} "
         f"& {results['mean_F_joint']:+.4f} \\\\"),
        (f"Std.\\ dev.\\ "
         f"& {results['sd_F_B']:.4f} "
         f"& {results['sd_F_L']:.4f} "
         f"& {results['sd_F_joint']:.4f} \\\\"),
        (f"5th--95th pct.\\ "
         f"& [{results['p5_F_B']:+.3f}, {results['p95_F_B']:+.3f}] "
         f"& [{results['p5_F_L']:+.3f}, {results['p95_F_L']:+.3f}] "
         f"& [{results['p5_F_joint']:+.3f}, {results['p95_F_joint']:+.3f}] \\\\"),
        (f"Rank of enacted (of $M+1={M+1}$) "
         f"& {results['rank_B']} "
         f"& {results['rank_L']} "
         f"& {results['rank_joint']} \\\\"),
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Besag-Clifford exact $p$-values}} \\',
        (f"Lower-tail $p$ (dilution direction) "
         f"& {pB:.4f}{stars(pB)} "
         f"& {pL:.4f}{stars(pL)} "
         f"& {pJ:.4f}{stars(pJ)} \\\\"),
        (f"Upper-tail $p$ "
         f"& {results['p_B_upper']:.4f} "
         f"& {results['p_L_upper']:.4f} "
         f"& {results['p_joint_upper']:.4f} \\\\"),
        r'\bottomrule',
        r'\multicolumn{4}{p{0.95\linewidth}}{\footnotesize',
        r'\textit{Notes:} Lower-tail $p$-values test whether the enacted map provides',
        r'unusually \emph{few} minority opportunity seats relative to the null distribution',
        r'of randomly drawn alternative plans (the vote dilution direction).',
        r'$p$-values are exactly valid regardless of Markov chain mixing time',
        r'\citep{besagclifford1989}.',
        r'Significance: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (lower-tail, one-sided).',
        f'Model mode: {results["mode"]}. Hub length $L_{{\\rm hub}}={L_HUB}$.',
        r'The proportional baseline $K\bar{m}_g$ is the De~Grandy relevant fact',
        r'\citep[512 U.S.\ 997]{degrandy1994}: proportionality is not dispositive but',
        r'is evidence in the totality of circumstances.',
        r'$F_{\rm joint}$ addresses the Section~2 coalition-claim circuit split by',
        r'crediting each district to whichever group has the higher win probability.',
        r'} \\',
        r'\end{tabular}',
        r'\end{table}',
    ]

    path = DIR + f'outputs/bc_functional_table_{run_name}.tex'
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved: {path}")
    return path


# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.time()
    master_seed = stdlib_random.randint(0, 10**9)
    gc_random.seed(master_seed)
    print(f"Master seed: {master_seed}")

    # 1. Score enacted map
    enacted_F_B, enacted_F_L, enacted_F_joint, enacted_partition = \
        score_enacted_map()

    # 2. Hub chain  <-- UNCHANGED from besag_clifford_vra.py
    hub_assignment = run_hub_chain(enacted_partition)

    # 3. Spoke chains
    spoke_df = run_spokes(hub_assignment)

    # 4. B-C p-values
    results = compute_bc_pvalues(
        enacted_F_B, enacted_F_L, enacted_F_joint, spoke_df
    )

    # 5. Console output
    print_results(results, spoke_df)

    # 6. CSVs
    spoke_path = DIR + f'outputs/bc_functional_spokes_{RUN_NAME}.csv'
    pval_path  = DIR + f'outputs/bc_functional_pvalues_{RUN_NAME}.csv'
    spoke_df.to_csv(spoke_path, index=False)
    pd.DataFrame([results]).to_csv(pval_path, index=False)
    print(f"\nSpoke scores saved : {spoke_path}")
    print(f"P-values saved     : {pval_path}")

    # 7. Distribution plots (PDF + PNG)
    save_distribution_plots(
        enacted_F_B, enacted_F_L, enacted_F_joint, spoke_df, RUN_NAME
    )

    # 8. LaTeX table
    save_latex_table(results, RUN_NAME)

    print(f"\nTotal elapsed: {(time.time()-total_start)/60:.1f} minutes")
    print(f"(L_hub={L_HUB}, L_spoke={L_SPOKE}, M_spokes={M_SPOKES}, "
          f"workers={N_WORKERS})")


if __name__ == "__main__":
    main()