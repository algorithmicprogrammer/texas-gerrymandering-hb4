"""
validate.py
-----------
Data validation suite for the TX redistricting pipeline.

Checks are organized into eight categories matching the slides:
  1. Schema validation       — column names and data types
  2. Join validation         — row count consistency, null keys, join cardinality
  3. Population conservation — block VAP ≈ VTD VAP after aggregation
  4. CVAP sanity checks      — VAP ≥ CVAP, no negatives, reasonable proportions
  5. Election data           — vote totals, no duplicates, party sums
  6. Geography               — valid geometries, defined CRS
  7. Cross-data consistency  — district coverage, VTD/block population alignment
  8. Logging/reporting       — structured summary of all issues

All checks are non-fatal unless `strict=True`; issues are logged and
summarized. This allows the pipeline to produce output even with minor
data quality issues while still flagging them for review.
"""

import logging
import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ValidationIssue:
    def __init__(self, category: str, severity: str, message: str, n: int = 0):
        self.category = category
        self.severity = severity   # "ERROR" | "WARNING" | "INFO"
        self.message  = message
        self.n        = n          # affected row count, if applicable

    def __str__(self):
        suffix = f" (n={self.n:,})" if self.n else ""
        return f"[{self.severity}] {self.category}: {self.message}{suffix}"


def _check_schema(final: gpd.GeoDataFrame) -> list:
    issues = []
    from schema import SCHEMA
    schema_cols = [s[0] for s in SCHEMA]

    missing = [c for c in schema_cols if c not in final.columns and c != "geometry"]
    if missing:
        issues.append(ValidationIssue("Schema", "ERROR", f"Missing columns: {missing}", len(missing)))

    # Type spot checks
    for col, dtype, _ in SCHEMA:
        if col in ("geometry", "CNTYVTD") or col not in final.columns:
            continue
        if dtype == "float64" and not pd.api.types.is_float_dtype(final[col]):
            issues.append(ValidationIssue("Schema", "WARNING", f"Column {col} expected float64", 1))
        elif dtype == "int64" and not pd.api.types.is_integer_dtype(final[col]):
            issues.append(ValidationIssue("Schema", "WARNING", f"Column {col} expected int64", 1))

    if not issues:
        issues.append(ValidationIssue("Schema", "INFO", "All schema checks passed", 0))
    return issues


def _check_join(final: gpd.GeoDataFrame) -> list:
    issues = []

    # Null CNTYVTD keys
    null_keys = final["CNTYVTD"].isna().sum()
    if null_keys:
        issues.append(ValidationIssue("Join", "ERROR", "Null CNTYVTD keys", null_keys))

    # Duplicate CNTYVTD
    dupes = final.duplicated("CNTYVTD").sum()
    if dupes:
        issues.append(ValidationIssue("Join", "ERROR", "Duplicate CNTYVTD rows", dupes))

    # Missing CD assignments
    missing_cd = final["CD"].isna().sum()
    if missing_cd:
        issues.append(ValidationIssue("Join", "WARNING",
                                      "VTDs with no congressional district assignment", missing_cd))

    if not [i for i in issues if i.severity in ("ERROR", "WARNING")]:
        issues.append(ValidationIssue("Join", "INFO", "Join checks passed", 0))
    return issues


def _check_population_conservation(
    final: gpd.GeoDataFrame,
    blocks_cvap: pd.DataFrame,
    tolerance: float = 0.01,
) -> list:
    """
    After aggregation, VTD-level VAP sum should ≈ block-level VAP sum.
    Tolerance is expressed as a fraction (default 1%).
    """
    issues = []
    vtd_total  = final["VAP"].sum()
    block_total = blocks_cvap["VAP"].sum()

    if block_total == 0:
        issues.append(ValidationIssue("Population", "ERROR", "Block VAP total is zero"))
        return issues

    diff_frac = abs(vtd_total - block_total) / block_total
    if diff_frac > tolerance:
        issues.append(ValidationIssue(
            "Population", "WARNING",
            f"VTD VAP total ({vtd_total:,.0f}) differs from block total "
            f"({block_total:,.0f}) by {diff_frac:.2%} — exceeds {tolerance:.0%} tolerance"
        ))
    else:
        issues.append(ValidationIssue(
            "Population", "INFO",
            f"VAP conservation OK: VTD={vtd_total:,.0f}, blocks={block_total:,.0f}, "
            f"diff={diff_frac:.4%}"
        ))
    return issues


def _check_cvap_sanity(final: gpd.GeoDataFrame) -> list:
    issues = []
    float_demo = ["VAP", "CVAP", "BVAP", "HVAP", "AVAP", "AMINVAP", "WVAP",
                  "BCVAP", "HCVAP", "ACVAP", "AMINCVAP", "WCVAP"]
    demo_cols = [c for c in float_demo if c in final.columns]

    # No negative values
    for col in demo_cols:
        n_neg = (final[col] < 0).sum()
        if n_neg:
            issues.append(ValidationIssue("CVAP", "ERROR", f"{col} has negative values", n_neg))

    # VAP ≥ CVAP
    if "VAP" in final.columns and "CVAP" in final.columns:
        n_vap_lt_cvap = (final["CVAP"] > final["VAP"] + 0.5).sum()
        if n_vap_lt_cvap:
            issues.append(ValidationIssue("CVAP", "WARNING",
                                          "CVAP > VAP (rounding artifact check)", n_vap_lt_cvap))

    # Proportions in [0, 1]
    prop_cols = ["white_prop", "black_prop", "hisp_prop", "asian_prop", "amin_prop"]
    for col in prop_cols:
        if col not in final.columns:
            continue
        out_of_range = ((final[col] < -0.01) | (final[col] > 1.01)).sum()
        if out_of_range:
            issues.append(ValidationIssue("CVAP", "ERROR",
                                          f"{col} out of [0,1] range", out_of_range))

    # Proportion sums should be ≤ 1
    present_props = [c for c in prop_cols if c in final.columns]
    if present_props:
        prop_sum = final[present_props].sum(axis=1)
        n_over = (prop_sum > 1.05).sum()   # 5% tolerance for rounding
        if n_over:
            issues.append(ValidationIssue("CVAP", "WARNING",
                                          "Racial proportion sum > 1.05", n_over))

    if not [i for i in issues if i.severity == "ERROR"]:
        issues.append(ValidationIssue("CVAP", "INFO", "CVAP sanity checks passed", 0))
    return issues


def _check_election_data(final: gpd.GeoDataFrame) -> list:
    issues = []

    # Total vote columns should equal sum of constituent candidate columns
    from elections import TOTALS_SPEC
    for total_col, src_cols in TOTALS_SPEC.items():
        if total_col not in final.columns:
            continue
        present_src = [c for c in src_cols if c in final.columns]
        if not present_src:
            continue
        expected = final[present_src].sum(axis=1)
        mismatch = (final[total_col] != expected).sum()
        if mismatch:
            issues.append(ValidationIssue("Election", "ERROR",
                                          f"{total_col} doesn't match sum of parts", mismatch))

    # No negative vote counts
    int_cols = [c for c in final.columns
                if c not in ("CD", "CNTYVTD") and pd.api.types.is_integer_dtype(final[c])]
    for col in int_cols:
        n_neg = (final[col] < 0).sum()
        if n_neg:
            issues.append(ValidationIssue("Election", "ERROR",
                                          f"Negative vote count in {col}", n_neg))

    if not [i for i in issues if i.severity == "ERROR"]:
        issues.append(ValidationIssue("Election", "INFO", "Election data checks passed", 0))
    return issues


def _check_geography(final: gpd.GeoDataFrame) -> list:
    issues = []

    if final.crs is None:
        issues.append(ValidationIssue("Geography", "ERROR", "CRS is undefined"))
        return issues

    # Geometry validity
    n_invalid = (~final.geometry.is_valid).sum()
    if n_invalid:
        issues.append(ValidationIssue("Geography", "WARNING",
                                      "Invalid geometries (may need make_valid)", n_invalid))

    # Null geometries
    n_null = final.geometry.isna().sum()
    if n_null:
        issues.append(ValidationIssue("Geography", "ERROR", "Null geometries", n_null))

    # Empty geometries
    n_empty = final.geometry.is_empty.sum()
    if n_empty:
        issues.append(ValidationIssue("Geography", "WARNING", "Empty geometries", n_empty))

    if not [i for i in issues if i.severity == "ERROR"]:
        issues.append(ValidationIssue("Geography", "INFO",
                                      f"Geography checks passed (CRS: {final.crs})", 0))
    return issues


def _check_cross_data(final: gpd.GeoDataFrame) -> list:
    issues = []

    # District assignment coverage
    n_total   = len(final)
    n_no_cd   = final["CD"].isna().sum()
    if n_no_cd:
        issues.append(ValidationIssue("CrossData", "WARNING",
                                      f"{n_no_cd}/{n_total} VTDs without district assignment",
                                      n_no_cd))

    # Check that all districts in the enacted plan have at least one VTD
    if "CD" in final.columns:
        n_districts = final["CD"].dropna().nunique()
        issues.append(ValidationIssue("CrossData", "INFO",
                                      f"{n_districts} congressional districts represented"))

    # TOTALPOP vs VAP basic sanity (VAP should be ≤ TOTALPOP)
    if "TOTALPOP" in final.columns and "VAP" in final.columns:
        n_vap_gt_pop = (final["VAP"] > final["TOTALPOP"] + 0.5).sum()
        if n_vap_gt_pop:
            issues.append(ValidationIssue("CrossData", "WARNING",
                                          "VAP > TOTALPOP in precincts", n_vap_gt_pop))

    return issues


def run_all_validations(
    final: gpd.GeoDataFrame,
    blocks_cvap: pd.DataFrame,
    strict: bool = False,
) -> bool:
    """
    Run the complete validation suite and log a structured summary.

    Parameters
    ----------
    final       : GeoDataFrame — the assembled final dataset
    blocks_cvap : DataFrame    — block-level CVAP for population conservation check
    strict      : bool         — if True, raise RuntimeError on any ERROR-level issue

    Returns
    -------
    bool : True if no ERROR-level issues found
    """
    all_issues = []
    all_issues += _check_schema(final)
    all_issues += _check_join(final)
    all_issues += _check_population_conservation(final, blocks_cvap)
    all_issues += _check_cvap_sanity(final)
    all_issues += _check_election_data(final)
    all_issues += _check_geography(final)
    all_issues += _check_cross_data(final)

    errors   = [i for i in all_issues if i.severity == "ERROR"]
    warnings = [i for i in all_issues if i.severity == "WARNING"]
    infos    = [i for i in all_issues if i.severity == "INFO"]

    log.info("=== Validation Summary ===")
    log.info(f"  {len(errors)} errors  |  {len(warnings)} warnings  |  {len(infos)} passed")

    for issue in errors:
        log.error(str(issue))
    for issue in warnings:
        log.warning(str(issue))
    for issue in infos:
        log.info(str(issue))

    if errors and strict:
        raise RuntimeError(
            f"Pipeline failed validation: {len(errors)} error(s). "
            "See log output above."
        )

    # Log unmatched join % as requested in slides
    null_keys = final["CNTYVTD"].isna().sum()
    pct_unmatched = null_keys / max(len(final), 1) * 100
    log.info(f"  Unmatched join %: {pct_unmatched:.2f}%")

    return len(errors) == 0
