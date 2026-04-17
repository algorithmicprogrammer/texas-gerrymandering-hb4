"""
run_ei.py
---------
Reproduces the RxC Multinomial-Dirichlet ecological inference from
Becker, Duchin, Gold & Hirsch (2021) "Computational Redistricting and
the Voting Rights Act", Election Law Journal 20(4).

Uses pyei.r_by_c.RowByColumnEI — the Python equivalent of eiPack::ei.MD.bayes.
Both implement the same hierarchical Multinomial-Dirichlet Bayesian model
(Rosen et al. 2001) via MCMC.

Inputs (all in same directory as this script):
  tx_precincts_for_ei.csv      — precinct data from data engineering pipeline
  TX_elections.csv             — election set definitions
  recency_weights.csv          — year -> weight mapping
  dropped_elecs.csv            — election sets to exclude (may be empty)

Outputs written to ei_outputs/:
  statewide_rxc_EI_preferences.csv   — statewide candidate-of-choice by group
  mean_prec_vote_counts.csv          — precinct-level mean vote count by race
  prec_count_quants.csv              — precinct-level octile quantiles by race

Run:
  pip install pyei
  python run_ei.py
"""
from texas_gerrymandering_hb4.config import DROPPED_ELECS, RECENCY_WEIGHTS, TX_ELECTIONS, PRECINCT_DATASET_CSV

import os
import gc
import numpy as np
import pandas as pd
from pyei.r_by_c import RowByColumnEI
import pytensor

pytensor.config.mode = "NUMBA"

# -- MCMC settings (match paper: ~1000 effective draws) -----------------------
DRAWS       = 1000    # posterior draws to keep per chain
TUNE        = 100000    # tuning/burn-in steps
CHAINS      = 1       # number of chains
RANDOM_SEED = 42

# -- Demographic group columns (R=4, must sum to 1 per precinct) --------------
GROUP_COLS   = ["hisp_prop", "black_prop", "white_prop", "other_prop"]
GROUP_LABELS = ["Hispanic", "Black", "White", "Other"]

# -- Election definitions: election key -> (candidate vote cols, total col) ---
# UncommittedR_24P excluded -- no racial identity.
ELECTIONS = {

    "24G_President": {
        "cands": ["TrumpR_24G", "HarrisD_24G"],
        "total": "TOTVOTE_PRES_24G",
    },

    "24P_President": {
        "cands": [
            "BidenD_24P", "CornejoD_24P", "LockeD_24P", "LozadaD_24P",
            "PerezD_24P", "PhillipsD_24P", "UygurD_24P", "WilliamsonD_24P",
            "BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
            "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P",
        ],
        "total": None,
    },

    "24G_US_Sen": {
        "cands": ["CruzR_24G", "AllredD_24G"],
        "total": "TOTVOTE_SEN_24G",
    },

    "24P_US_Sen": {
        "cands": [
            "AllredD_24P", "GomezD_24P", "GonzalezD_24P", "GutierrezD_24P",
            "HassanD_24P", "KeoughD_24P", "PrillimanD_24P", "ShermanD_24P",
            "TchenkoD_24P",
            "CruzR_24P", "GibsonR_24P", "LopezR_24P",
        ],
        "total": None,
    },

    "24G_RR_Comm_1": {
        "cands": ["CraddickR_24G", "CulbertD_24G"],
        "total": "TOTVOTE_RRC_24G",
    },

    "24P_RR_Comm_1": {
        "cands": [
            "BurchD_24P", "CulbertD_24P",
            "ClarkR_24P", "CraddickR_24P", "HowellR_24P",
            "MatlockR_24P", "ReyesR_24P",
        ],
        "total": None,
    },
}

# -- Load supporting files -----------------------------------------------------
print("Loading input files...")

vtd     = pd.read_csv(PRECINCT_DATASET_CSV)
elec_df = pd.read_csv(TX_ELECTIONS)
recency = pd.read_csv(RECENCY_WEIGHTS)
dropped = pd.read_csv(DROPPED_ELECS)

# Parse recency weight
recency_year   = int(recency.columns[0])
recency_weight = float(recency.iloc[0, 0])
print(f"  Recency weight: year={recency_year}  weight={recency_weight:.2f}")

# Parse dropped election sets
dropped_col  = dropped.columns[0]
dropped_sets = set(dropped[dropped_col].dropna().astype(str).str.strip())
dropped_sets.discard("")

if dropped_sets:
    elec_df = elec_df[~elec_df["Election Set"].isin(dropped_sets)]
    print(f"  Dropped election sets: {dropped_sets}")

# Active elections = intersection of ELECTIONS dict keys and TX_elections.csv
active_elections = [k for k in ELECTIONS if k in elec_df["Election"].values]
print(f"  Active elections: {', '.join(active_elections)}")

# Filter vtd to CVAP > 0
vtd = vtd[vtd["CVAP"].notna() & (vtd["CVAP"] > 0)].copy()
print(f"  Precincts after CVAP>0 filter: {len(vtd)}")

# -- Output directory ----------------------------------------------------------
os.makedirs("ei_outputs", exist_ok=True)

# -- Output accumulators -------------------------------------------------------
statewide_rows    = []
mean_counts_list  = []
quant_counts_list = []

# -- Main EI loop --------------------------------------------------------------
for elec_key in active_elections:

    spec      = ELECTIONS[elec_key]
    cand_cols = spec["cands"]
    total_col = spec["total"]
    n_cands   = len(cand_cols)

    print(f"\n{'='*60}")
    print(f"Election: {elec_key}  ({n_cands} candidates)")

    # Verify all candidate columns exist
    missing = [c for c in cand_cols if c not in vtd.columns]
    if missing:
        print(f"  WARNING: Skipping {elec_key} -- missing columns: {missing}")
        continue

    # Build total column if not pre-computed
    if total_col is None:
        total_col = f"TOTAL_{elec_key}"
        vtd[total_col] = vtd[cand_cols].sum(axis=1)

    # Keep only precincts with votes in this election
    sub = vtd[vtd[total_col] > 0].copy().reset_index(drop=True)
    n   = len(sub)
    print(f"  Precincts with votes: {n}")

    if n < 50:
        print(f"  WARNING: Skipping {elec_key} -- too few precincts ({n})")
        continue

    # Round CVAP to integers (pyei requirement) and drop any that round to 0
    cvap = np.round(sub["CVAP"].values).astype(int)
    mask = cvap > 0
    sub  = sub[mask].reset_index(drop=True)
    cvap = cvap[mask]

    # -- Build pyei inputs -----------------------------------------------------
    # pyei expects shape (n_groups, n_precincts) — i.e. transposed relative to
    # the natural (n_precincts, n_groups) orientation. It validates by checking
    # group_fractions.sum(axis=0) == 1, which sums down columns (across groups
    # within each precinct). So each COLUMN must sum to 1.

    # Step 1: build row-normalised (n_precincts, n_groups), then transpose
    gf_rows = sub[GROUP_COLS].fillna(0).values.astype(float)
    row_sums = gf_rows.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    gf_rows  = gf_rows / row_sums
    # Force exact row sums to 1 via last-column adjustment
    gf_rows[:, -1] = 1.0 - gf_rows[:, :-1].sum(axis=1)
    gf_rows  = np.clip(gf_rows, 0, 1)
    # Transpose to (n_groups, n_precincts) as pyei expects
    group_fractions = gf_rows.T

    # Step 2: votes_fractions — same transposition logic
    # Use max(CVAP, total_votes) as denominator so fracs never exceed 1.
    # When turnout exceeds CVAP (ACS estimate), total_votes is used instead,
    # preserving the Abstain column for precincts where turnout < CVAP.
    vote_counts      = sub[cand_cols].values.astype(float)
    total_votes      = vote_counts.sum(axis=1)
    denom            = np.maximum(cvap.astype(float), total_votes)
    denom            = np.where(denom == 0, 1, denom)
    # precinct_pops for pyei must match the denominator used for votes_fracs
    precinct_pops_ei = np.round(denom).astype(int)
    vf_rows      = vote_counts / denom[:, None]
    # Add Abstain = remainder of CVAP not voting for any candidate
    vf_rows      = np.hstack([vf_rows, np.zeros((len(vf_rows), 1))])
    vf_rows[:, -1] = 1.0 - vf_rows[:, :-1].sum(axis=1)
    vf_rows      = np.clip(vf_rows, 0, 1)
    # Transpose to (n_cands+1, n_precincts) as pyei expects
    votes_fracs  = vf_rows.T

    all_labels = cand_cols + ["Abstain"]

    print(f"  Fitting RxC EI (draws={DRAWS}, tune={TUNE}, chains={CHAINS})...")

    ei = RowByColumnEI(model_name="multinomial-dirichlet")
    ei.fit(
        group_fractions          = group_fractions,
        votes_fractions          = votes_fracs,
        precinct_pops            = precinct_pops_ei,
        demographic_group_labels = GROUP_LABELS,
        candidate_labels         = all_labels,
        draws                    = DRAWS,
        tune                     = TUNE,
        chains                   = CHAINS,
        random_seed              = RANDOM_SEED,
    )

    # -- Extract posterior samples ---------------------------------------------
    # Raw samples from sim_trace, transposed to (n_draws, n_precincts, n_groups, n_cands+1)
    samples_all = np.transpose(
        ei.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,
        axes=(3, 0, 1, 2),
    )
    samples     = samples_all[:, :, :, :n_cands]  # real candidates only
    n_draws     = samples.shape[0]
    print(f"  Posterior samples shape (excl. Abstain): {samples.shape}  "
          f"(draws x precincts x groups x candidates)")

    # -- 1. statewide_rxc_EI_preferences --------------------------------------
    print("  Computing statewide preferences...")

    for g_idx, group in enumerate(GROUP_LABELS):
        group_pop       = sub[GROUP_COLS[g_idx]].values * precinct_pops_ei
        total_group_pop = group_pop.sum()

        if total_group_pop == 0:
            continue

        # draw_support[s, c] = state-level support for candidate c at draw s
        # weighted mean over precincts
        draw_support = np.einsum(
            "spc,p->sc",
            samples[:, :, g_idx, :],   # (n_draws, n_precincts, n_cands)
            group_pop
        ) / total_group_pop            # -> (n_draws, n_cands)

        mean_support   = draw_support.mean(axis=0)
        draw_coc       = draw_support.argmax(axis=1)
        frac_preferred = np.bincount(draw_coc, minlength=n_cands) / n_draws

        coc_idx  = int(mean_support.argmax())
        coc_name = cand_cols[coc_idx]

        statewide_rows.append({
            "election":             elec_key,
            "group":                group,
            "candidate_of_choice":  coc_name,
            "mean_support":         round(float(mean_support[coc_idx]), 6),
            "frac_draws_preferred": round(float(frac_preferred[coc_idx]), 6),
        })

    # -- 2. mean_prec_vote_counts ----------------------------------------------
    print("  Computing mean precinct vote counts...")

    for g_idx, group in enumerate(GROUP_LABELS):
        group_pop = sub[GROUP_COLS[g_idx]].values * precinct_pops_ei

        for c_idx, cand in enumerate(cand_cols):
            count_draws = samples[:, :, g_idx, c_idx] * group_pop[None, :]
            mean_counts = count_draws.mean(axis=0)

            chunk = pd.DataFrame({
                "CNTYVTD":    sub["CNTYVTD"].values,
                "election":   elec_key,
                "group":      group,
                "candidate":  cand,
                "mean_count": np.round(mean_counts, 4),
            })
            mean_counts_list.append(chunk)

    # -- 3. prec_count_quants -------------------------------------------------
    print("  Computing precinct count quantiles...")
    octile_probs = np.arange(1, 9) / 8   # 0.125, 0.25, ..., 1.0

    for g_idx, group in enumerate(GROUP_LABELS):
        group_pop = sub[GROUP_COLS[g_idx]].values * precinct_pops_ei

        for c_idx, cand in enumerate(cand_cols):
            count_draws  = samples[:, :, g_idx, c_idx] * group_pop[None, :]
            quant_matrix = np.quantile(count_draws, octile_probs, axis=0)

            for q_idx, (prob, qvals) in enumerate(
                    zip(octile_probs, quant_matrix), start=1):
                chunk = pd.DataFrame({
                    "CNTYVTD":   sub["CNTYVTD"].values,
                    "election":  elec_key,
                    "group":     group,
                    "candidate": cand,
                    "octile":    q_idx,
                    "quantile":  round(float(prob), 4),
                    "value":     np.round(qvals, 4),
                })
                quant_counts_list.append(chunk)

    print(f"  Done: {elec_key}")

    # Free memory before next election
    del ei, samples_all, samples, vote_counts, votes_fracs, group_fractions
    gc.collect()

# -- Write outputs -------------------------------------------------------------
print("\nWriting outputs to ei_outputs/...")

statewide_df = pd.DataFrame(statewide_rows)
statewide_df.to_csv("ei_outputs/statewide_rxc_EI_preferences.csv", index=False)
print(f"  statewide_rxc_EI_preferences.csv: {len(statewide_df)} rows")

mean_df = pd.concat(mean_counts_list, ignore_index=True)
mean_df.to_csv("ei_outputs/mean_prec_vote_counts.csv", index=False)
print(f"  mean_prec_vote_counts.csv: {len(mean_df)} rows")

quant_df = pd.concat(quant_counts_list, ignore_index=True)
quant_df.to_csv("ei_outputs/prec_count_quants.csv", index=False)
print(f"  prec_count_quants.csv: {len(quant_df)} rows")

print("\n=== EI complete ===")
print("Next: fill in ingroup_weight.csv using statewide_rxc_EI_preferences.csv,")
print("then proceed to TX_elections_model.py")
