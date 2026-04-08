"""
sensitivity.py
--------------
Sensitivity analysis for the CVAP discounting methodology.

Checks from slides (Section 9):
  1. CVAP fallback threshold: 10 vs 20 vs 50
  2. Inclusion/exclusion of "Other" folded into White
  3. Results under VAP instead of CVAP as baseline comparison

Run standalone or import individual functions into a notebook.
"""

import logging
import pandas as pd
import numpy as np
from demographics import build_block_vap, apply_cvap_discounting

log = logging.getLogger(__name__)


def run_threshold_sensitivity(
    blocks_vap: pd.DataFrame,
    acs_cvap: pd.DataFrame,
    thresholds: list = None,
) -> pd.DataFrame:
    """
    Compare CVAP estimates across fallback threshold values.

    Returns a DataFrame with one row per threshold showing summary stats.
    """
    if thresholds is None:
        thresholds = [10, 20, 50]

    results = []
    for t in thresholds:
        df = apply_cvap_discounting(blocks_vap, acs_cvap, threshold=t)
        results.append({
            "threshold":       t,
            "BCVAP_mean":      df["BCVAP"].mean(),
            "HCVAP_mean":      df["HCVAP"].mean(),
            "ACVAP_mean":      df["ACVAP"].mean(),
            "AMINCVAP_mean":   df["AMINCVAP"].mean(),
            "WCVAP_mean":      df["WCVAP"].mean(),
            "CVAP_total":      df["CVAP"].sum(),
            "n_fallback":      (df["BVAP"] < t).sum(),   # proxy for fallback count
        })
    summary = pd.DataFrame(results)
    log.info("Threshold sensitivity summary:")
    log.info(summary.to_string(index=False))
    return summary


def run_vap_vs_cvap_comparison(
    vtd_demo: "gpd.GeoDataFrame",
) -> pd.DataFrame:
    """
    Compare ecological inference inputs under VAP vs CVAP denominators.

    Returns a DataFrame with correlation stats between VAP and CVAP
    proportion columns.
    """
    import geopandas as gpd

    pairs = [
        ("BVAP", "BCVAP"),
        ("HVAP", "HCVAP"),
        ("AVAP", "ACVAP"),
        ("AMINVAP", "AMINCVAP"),
        ("WVAP",  "WCVAP"),
    ]

    cvap_total = vtd_demo["CVAP"].replace(0, np.nan)
    vap_total  = vtd_demo["VAP"].replace(0, np.nan)

    rows = []
    for vap_col, cvap_col in pairs:
        if vap_col not in vtd_demo.columns or cvap_col not in vtd_demo.columns:
            continue
        vap_prop  = vtd_demo[vap_col]  / vap_total
        cvap_prop = vtd_demo[cvap_col] / cvap_total
        corr = vap_prop.corr(cvap_prop)
        rows.append({
            "group":      vap_col.replace("VAP", ""),
            "vap_mean":   vap_prop.mean(),
            "cvap_mean":  cvap_prop.mean(),
            "correlation": corr,
            "mean_diff":  (cvap_prop - vap_prop).mean(),
        })

    summary = pd.DataFrame(rows)
    log.info("VAP vs CVAP comparison:")
    log.info(summary.to_string(index=False))
    return summary
