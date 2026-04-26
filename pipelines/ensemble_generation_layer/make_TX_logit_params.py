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