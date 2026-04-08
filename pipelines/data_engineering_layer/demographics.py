"""
demographics.py
---------------
Constructs VAP demographic variables at the Census block level following the
Data & Democracy Lab (MGGG) convention, then estimates CVAP by applying ACS
citizenship rates ("discounting") per the method in mggg.org/vap-cvap.

Race/ethnicity ordering (MGGG convention)
------------------------------------------
  Black → Hispanic → Asian/NHPI (AAPI) → AMIN → Other → White

Each group is mutually exclusive:
  BVAP   : "any part Black" — anyone who checked Black, regardless of other races
  HVAP   : Hispanic, excluding anyone already in BVAP
  AVAP   : Asian or NHPI (AAPI), excluding BVAP and HVAP
  AMINVAP: American Indian / Alaska Native, excluding prior groups
  OVAP   : Some Other Race, non-Hispanic, excluding prior groups
  WVAP   : Non-Hispanic, single-race White (residual)
  TVAP   : Total VAP (P0030001)

References
----------
Donnay, Duchin & Rock (n.d.). Conventions for Constructing Demographic
  Categories. Data and Democracy Lab. https://mggg.org/vap-cvap
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MGGG BVAP columns: 32 P3 columns that include Black in any combination.
# Source: Donnay et al. Table 1.  MGGG alias → Census name mapping.
# ---------------------------------------------------------------------------
BVAP_P3_COLS = [
    "P0030004",  # p3_004n  Black or African American alone
    "P0030011",  # p3_011n  White; Black or African American
    "P0030016",  # p3_016n  Black; American Indian and Alaska Native
    "P0030017",  # p3_017n  Black; Asian
    "P0030018",  # p3_018n  Black; Native Hawaiian and Other Pacific Islander
    "P0030019",  # p3_019n  Black; Some Other Race
    "P0030027",  # p3_027n  White; Black; American Indian and Alaska Native
    "P0030028",  # p3_028n  White; Black; Asian
    "P0030029",  # p3_029n  White; Black; Native Hawaiian and Other Pacific Islander
    "P0030030",  # p3_030n  White; Black; Some Other Race
    "P0030037",  # p3_037n  Black; American Indian and Alaska Native; Asian
    "P0030038",  # p3_038n  Black; American Indian and Alaska Native; Native Hawaiian
    "P0030039",  # p3_039n  Black; American Indian and Alaska Native; Some Other Race
    "P0030040",  # p3_040n  Black; Asian; Native Hawaiian and Other Pacific Islander
    "P0030041",  # p3_041n  Black; Asian; Some Other Race
    "P0030042",  # p3_042n  Black; Native Hawaiian; Some Other Race
    "P0030048",  # p3_048n  White; Black; American Indian and Alaska Native; Asian
    "P0030049",  # p3_049n  White; Black; American Indian and Alaska Native; Native Hawaiian
    "P0030050",  # p3_050n  White; Black; American Indian and Alaska Native; Some Other Race
    "P0030051",  # p3_051n  White; Black; Asian; Native Hawaiian
    "P0030052",  # p3_052n  White; Black; Asian; Some Other Race
    "P0030053",  # p3_053n  White; Black; Native Hawaiian; Some Other Race
    "P0030058",  # p3_058n  Black; American Indian and Alaska Native; Asian; Native Hawaiian
    "P0030059",  # p3_059n  Black; American Indian and Alaska Native; Asian; Some Other Race
    "P0030060",  # p3_060n  Black; American Indian and Alaska Native; Native Hawaiian; Some Other Race
    "P0030061",  # p3_061n  Black; Asian; Native Hawaiian; Some Other Race
    "P0030064",  # p3_064n  White; Black; American Indian and Alaska Native; Asian; Native Hawaiian
    "P0030065",  # p3_065n  White; Black; American Indian and Alaska Native; Asian; Some Other Race
    "P0030066",  # p3_066n  White; Black; American Indian and Alaska Native; Native Hawaiian; Some Other Race
    "P0030067",  # p3_067n  White; Black; Asian; Native Hawaiian; Some Other Race
    "P0030069",  # p3_069n  Black; American Indian and Alaska Native; Asian; Native Hawaiian; Some Other Race
    "P0030071",  # p3_071n  White; Black; American Indian and Alaska Native; Asian; Native Hawaiian; Some Other Race
]

# ---------------------------------------------------------------------------
# HVAP: 31 P3-P4 difference columns.
# HVAP_i = P3_i − P4_i  (Hispanic = total VAP minus non-Hispanic VAP)
# Columns that appear in BVAP are excluded.
# Source: Donnay et al. Table 2.
# ---------------------------------------------------------------------------
HVAP_P3_P4_PAIRS = [
    ("P0030003", "P0040005"),  # white
    ("P0030005", "P0040007"),  # amin
    ("P0030006", "P0040008"),  # asian
    ("P0030007", "P0040009"),  # nhpi
    ("P0030008", "P0040010"),  # other
    ("P0030012", "P0040014"),  # white_amin
    ("P0030013", "P0040015"),  # white_asian
    ("P0030014", "P0040016"),  # white_nhpi
    ("P0030015", "P0040017"),  # white_other
    ("P0030020", "P0040022"),  # amin_asian
    ("P0030021", "P0040023"),  # amin_nhpi
    ("P0030022", "P0040024"),  # amin_other
    ("P0030023", "P0040025"),  # asian_nhpi
    ("P0030024", "P0040026"),  # asian_other
    ("P0030025", "P0040027"),  # nhpi_other
    ("P0030031", "P0040033"),  # white_amin_asian
    ("P0030032", "P0040034"),  # white_amin_nhpi
    ("P0030033", "P0040035"),  # white_amin_other
    ("P0030034", "P0040036"),  # white_asian_nhpi
    ("P0030035", "P0040037"),  # white_asian_other
    ("P0030036", "P0040038"),  # white_nhpi_other
    ("P0030043", "P0040045"),  # amin_asian_nhpi
    ("P0030044", "P0040046"),  # amin_asian_other
    ("P0030045", "P0040047"),  # amin_nhpi_other
    ("P0030046", "P0040048"),  # asian_nhpi_other
    ("P0030054", "P0040056"),  # white_amin_asian_nhpi
    ("P0030055", "P0040057"),  # white_amin_asian_other
    ("P0030056", "P0040058"),  # white_amin_nhpi_other
    ("P0030057", "P0040059"),  # white_asian_nhpi_other
    ("P0030062", "P0040064"),  # amin_asian_nhpi_other
    ("P0030068", "P0040070"),  # white_amin_asian_nhpi_other
]

# ---------------------------------------------------------------------------
# AVAP: 24 P4 columns (non-Hispanic Asian/NHPI combinations).
# Source: Donnay et al. Table 3.
# ---------------------------------------------------------------------------
AVAP_P4_COLS = [
    "P0040008",  # asian
    "P0040009",  # nhpi
    "P0040015",  # white_asian
    "P0040016",  # white_nhpi
    "P0040022",  # amin_asian
    "P0040023",  # amin_nhpi
    "P0040025",  # asian_nhpi
    "P0040026",  # asian_other
    "P0040027",  # nhpi_other
    "P0040033",  # white_amin_asian
    "P0040034",  # white_amin_nhpi
    "P0040036",  # white_asian_nhpi
    "P0040037",  # white_asian_other
    "P0040038",  # white_nhpi_other
    "P0040045",  # amin_asian_nhpi
    "P0040046",  # amin_asian_other
    "P0040047",  # amin_nhpi_other
    "P0040048",  # asian_nhpi_other
    "P0040056",  # white_amin_asian_nhpi
    "P0040057",  # white_amin_asian_other
    "P0040058",  # white_amin_nhpi_other
    "P0040059",  # white_asian_nhpi_other
    "P0040064",  # amin_asian_nhpi_other
    "P0040070",  # white_amin_asian_nhpi_other
]

# ---------------------------------------------------------------------------
# AMINVAP: 4 P4 columns.  Source: Donnay et al. Table 3.
# ---------------------------------------------------------------------------
AMINVAP_P4_COLS = [
    "P0040007",  # amin
    "P0040014",  # white_amin
    "P0040024",  # amin_other
    "P0040035",  # white_amin_other
]

# ---------------------------------------------------------------------------
# OVAP: 2 P4 columns.  Source: Donnay et al. Table 3.
# ---------------------------------------------------------------------------
OVAP_P4_COLS = [
    "P0040010",  # other
    "P0040017",  # white_other
]

# WVAP: single P4 column.  Source: Donnay et al. Table 3.
WVAP_P4_COL = "P0040005"   # Non-Hispanic White alone

# TVAP: P3 total VAP column.
TVAP_P3_COL = "P0030001"


def build_block_vap(pl_blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Construct BVAP, HVAP, AVAP, AMINVAP, OVAP, WVAP, VAP at block level.

    Parameters
    ----------
    pl_blocks : DataFrame
        Merged geo-header + PL data segment, one row per Census block.

    Returns
    -------
    DataFrame with columns: GEOID20, tract_geoid, BVAP, HVAP, AVAP,
        AMINVAP, OVAP, WVAP, VAP
    """
    df = pl_blocks.copy()

    # Validate required columns
    all_required = (
        [TVAP_P3_COL, WVAP_P4_COL]
        + BVAP_P3_COLS
        + AMINVAP_P4_COLS
        + AVAP_P4_COLS
        + OVAP_P4_COLS
        + [p3 for p3, _ in HVAP_P3_P4_PAIRS]
        + [p4 for _, p4 in HVAP_P3_P4_PAIRS]
    )
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PL columns: {missing}")

    df["VAP"]     = df[TVAP_P3_COL]
    df["BVAP"]    = df[BVAP_P3_COLS].sum(axis=1)
    df["HVAP"]    = sum(df[p3] - df[p4] for p3, p4 in HVAP_P3_P4_PAIRS)
    df["AVAP"]    = df[AVAP_P4_COLS].sum(axis=1)
    df["AMINVAP"] = df[AMINVAP_P4_COLS].sum(axis=1)
    df["OVAP"]    = df[OVAP_P4_COLS].sum(axis=1)
    df["WVAP"]    = df[WVAP_P4_COL]

    # Sanity: BVAP + HVAP + AVAP + AMINVAP + OVAP + WVAP should ≤ VAP
    group_sum = df[["BVAP", "HVAP", "AVAP", "AMINVAP", "OVAP", "WVAP"]].sum(axis=1)
    overcount = (group_sum > df["VAP"] + 0.5).sum()   # small float tolerance
    if overcount > 0:
        log.warning(f"  {overcount} blocks have group VAP sum > TVAP (investigate)")

    keep_cols = ["GEOID20", "tract_geoid", "VAP", "BVAP", "HVAP", "AVAP", "AMINVAP", "OVAP", "WVAP"]
    result = df[keep_cols].copy()
    log.info(
        f"  VAP construction complete: {len(result):,} blocks  "
        f"(VAP total={result['VAP'].sum():,.0f})"
    )
    return result


# ---------------------------------------------------------------------------
# ACS CVAP lnnumber → racial category mapping
# Source: Donnay et al. / ACS CVAP special tabulation documentation.
# ---------------------------------------------------------------------------
# For each CVAP variable we sum the listed lnnumbers from the ACS pivot table.
ACS_CVAP_MAP = {
    "BCVAP_acs":    ["5", "10", "11"],   # Black alone; Black+White; AMIN+Black
    "HCVAP_acs":    ["13"],              # Hispanic or Latino
    "ACVAP_acs":    ["4", "6", "9"],     # Asian alone; NHPI alone; Asian+White
    "AMINCVAP_acs": ["3", "8"],          # AMIN alone; AMIN+White
    # WCVAP_acs folds "Other" (lnnumber 12) with White (7) per MGGG convention:
    # "non-Hispanic members of Some Other Race are rare and their citizenship rate
    #  is not well-represented by the Other rate in ACS, so we fold Other with White"
    "WCVAP_acs":    ["7", "12"],         # White alone; Remainder of Two+ Race Responses
    "CVAP_acs":     ["1"],               # Total CVAP
}


def _tract_citizenship_rates(acs_cvap: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tract-level citizenship rates (CVAP / VAP proxy) for each
    racial/ethnic group using the ACS pivot table.

    We produce BCVAP, HCVAP, ACVAP, AMINCVAP, WCVAP, CVAP totals per tract,
    then store them for use in the discounting formula.
    """
    df = acs_cvap.copy()
    for col, lnnums in ACS_CVAP_MAP.items():
        src_cols = [f"cvap_ln{n}" for n in lnnums]
        present = [c for c in src_cols if c in df.columns]
        if not present:
            log.warning(f"  ACS pivot missing cols for {col}: {src_cols}")
            df[col] = 0
        else:
            df[col] = df[present].sum(axis=1)
    return df[["tract_geoid"] + list(ACS_CVAP_MAP.keys())]


def _state_cvap_rates(tract_rates: pd.DataFrame) -> dict:
    """
    Compute state-level fallback citizenship rates from tract aggregates.
    Used when tract VAP < threshold (default 20).
    """
    totals = tract_rates[list(ACS_CVAP_MAP.keys())].sum()
    # We need the VAP denominators per group at state level.
    # Since the ACS doesn't give us VAP by group, we return the raw ACS CVAP
    # totals and use the ratio against each other as rates.
    # The rate is: BCVAP_state / BVAP_state — approximated below from
    # total CVAP / total CVAP columns (the discounting formula scales the
    # block's VAP by the ACS rate, so rates below are CVAP/CVAP_total for
    # each group, used as a fraction of VAP).
    state_rates = {}
    total_cvap = totals["CVAP_acs"] if totals["CVAP_acs"] > 0 else 1
    for grp in ["BCVAP_acs", "HCVAP_acs", "ACVAP_acs", "AMINCVAP_acs", "WCVAP_acs"]:
        # These are absolute counts; the discounting formula will use
        # GRPCVAP_acs(state) / GRPVAP_acs(state).  Since ACS doesn't give
        # VAP by group at state level, we approximate with total CVAP:
        state_rates[grp] = totals[grp] / total_cvap
    return state_rates


def apply_cvap_discounting(
    blocks_vap: pd.DataFrame,
    acs_cvap: pd.DataFrame,
    threshold: int = 20,
) -> pd.DataFrame:
    """
    Estimate CVAP at block level using the MGGG citizenship-rate discounting
    method (https://mggg.org/vap-cvap).

    Formula (for Black, example):
        BCVAP(block X) = [BCVAP_ACS(tract Y) / BVAP_ACS(tract Y)] × BVAP_PL(block X)

    Fallback (when tract BVAP_ACS < threshold):
        BCVAP(block X) = [BCVAP_ACS(state Z) / BVAP_ACS(state Z)] × BVAP_PL(block X)

    "Other" is folded into White per MGGG convention (see ACS_CVAP_MAP).

    Parameters
    ----------
    blocks_vap : DataFrame
        Block-level VAP from build_block_vap().
    acs_cvap : DataFrame
        ACS CVAP special tabulation from ingest._load_acs_cvap().
    threshold : int
        Minimum tract VAP size before falling back to state rate.

    Returns
    -------
    DataFrame with block-level CVAP columns added.
    """
    tract_rates = _tract_citizenship_rates(acs_cvap)
    state_rates = _state_cvap_rates(tract_rates)

    # Merge tract ACS estimates onto blocks
    df = blocks_vap.merge(tract_rates, on="tract_geoid", how="left")

    # Map each block VAP group → corresponding ACS denominator and output name
    #   (block_vap_col, acs_cvap_col, output_col, acs_vap_col_for_denom)
    # Note: ACS BVAP denominator (BVAP_ACS) is proxied by the ACS BCVAP
    # columns — the special tabulation does not include a separate VAP column,
    # so the denominator is reconstructed from the total CVAP / group fractions.
    # For maximum defensibility we use the ACS CVAP columns as relative rates
    # rather than absolute counts (see MGGG method note).

    # Build per-group citizenship rates at tract level:
    # rate = GRPCVAP_acs(tract) / CVAP_acs(tract)  × (CVAP_acs(tract)/VAP_approx)
    # Since ACS gives us CVAP but not group-level VAP, we follow MGGG and define:
    # group_rate(tract) = GRPCVAP_acs(tract) / GRPCVAP_acs(tract) * group_rate_approx
    #
    # More precisely: MGGG defines the "discount factor" as GRPCVAP_ACS / GRVAP_ACS.
    # We reconstruct VAP denominators from the total ACS VAP = total CVAP / p(citizen).
    # The cleanest implementation uses GRPCVAP_ACS as numerator and infers the
    # group-level VAP denominator from the *total* ACS VAP and the share of each group.
    #
    # Implementation note: The ACS Tract.csv only includes CVAP columns (not VAP).
    # The recommended approach from MGGG is to use CVAP_acs(group)/CVAP_acs(total)
    # as a proxy for the group's share, then multiply by the overall citizenship rate.
    # We implement that cleanly below.

    # Overall tract citizenship rate (CVAP / ... )
    # We don't have ACS VAP directly, so we use the formula as written:
    # BCVAP(block) = BCVAP_acs(tract) / BVAP_acs(tract) × BVAP_pl(block)
    # Because ACS Tract.csv doesn't have BVAP_acs, we use the CVAP_acs total
    # as a normalizer and scale against the Decennial VAP totals at tract level.

    # Compute White+Other combined VAP before tract aggregation so it can be
    # included in the tract-level denominator used by _discount()
    df["_WOVAP"] = df["WVAP"] + df["OVAP"]

    # Aggregate Decennial VAP to tract for normalization denominator
    tract_pl_vap = (
        df.groupby("tract_geoid")[["VAP", "BVAP", "HVAP", "AVAP", "AMINVAP", "WVAP", "OVAP", "_WOVAP"]]
        .sum()
        .add_suffix("_pl_tract")
        .reset_index()
    )
    df = df.merge(tract_pl_vap, on="tract_geoid", how="left")

    def _discount(df, vap_col, cvap_acs_col, out_col, fallback_rate):
        """
        Apply discounting for one racial group.
        Uses tract-level citizenship rate; falls back to state rate when
        tract denominator < threshold.
        """
        denom = df[f"{vap_col}_pl_tract"].fillna(0)
        numerator = df[cvap_acs_col].fillna(0)

        # Citizenship rate at tract level
        tract_rate = np.where(denom >= threshold, numerator / denom.replace(0, np.nan), np.nan)
        # Fallback to state rate when tract denominator too small or missing
        rate = np.where(np.isnan(tract_rate), fallback_rate, tract_rate)
        # Edge case: zero VAP → zero CVAP
        result = np.where(df[vap_col] == 0, 0.0, rate * df[vap_col])
        return result.clip(min=0)  # no negative CVAP

    # State-level fallback rates: GRPCVAP_state / GRVAP_state
    state_b = tract_rates["BCVAP_acs"].sum() / max(blocks_vap["BVAP"].sum(), 1)
    state_h = tract_rates["HCVAP_acs"].sum() / max(blocks_vap["HVAP"].sum(), 1)
    state_a = tract_rates["ACVAP_acs"].sum() / max(blocks_vap["AVAP"].sum(), 1)
    state_amin = tract_rates["AMINCVAP_acs"].sum() / max(blocks_vap["AMINVAP"].sum(), 1)
    # White+Other folded: WVAP denominator = WVAP + OVAP
    state_w = tract_rates["WCVAP_acs"].sum() / max(
        (blocks_vap["WVAP"] + blocks_vap["OVAP"]).sum(), 1
    )

    df["BCVAP"]    = _discount(df, "BVAP",    "BCVAP_acs",    "BCVAP",    state_b)
    df["HCVAP"]    = _discount(df, "HVAP",    "HCVAP_acs",    "HCVAP",    state_h)
    df["ACVAP"]    = _discount(df, "AVAP",    "ACVAP_acs",    "ACVAP",    state_a)
    df["AMINCVAP"] = _discount(df, "AMINVAP", "AMINCVAP_acs", "AMINCVAP", state_amin)
    # White+Other share a single ACS rate (MGGG "folded Other" convention)
    df["WCVAP"]    = _discount(df, "_WOVAP",  "WCVAP_acs",    "WCVAP",    state_w)

    # Total CVAP = sum of the five mutually disjoint groups
    df["CVAP"] = df[["BCVAP", "HCVAP", "ACVAP", "AMINCVAP", "WCVAP"]].sum(axis=1)

    n_fallback = df[f"BVAP_pl_tract"].lt(threshold).sum()
    log.info(
        f"  CVAP discounting complete: {len(df):,} blocks  "
        f"({n_fallback:,} used state-level fallback)"
    )

    keep = [
        "GEOID20", "tract_geoid",
        "VAP", "BVAP", "HVAP", "AVAP", "AMINVAP", "OVAP", "WVAP",
        "CVAP", "BCVAP", "HCVAP", "ACVAP", "AMINCVAP", "WCVAP",
    ]
    return df[keep].copy()