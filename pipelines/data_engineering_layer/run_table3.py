"""
run_table3.py
-------------
Standalone runner that reproduces the data-engineering pipeline (steps 1-8 of
pipeline.py) and prints the Table 3 validation diagnostics. Edits nothing in
your existing code.

PLACEMENT
    Put this file in the SAME directory as pipeline.py
    (and table3_diagnostics.py beside it too).

RUN  (exactly how you already run pipeline.py)
    From the command line, from your repo root:
        python -m data_engineering_layer.run_table3
    or in PyCharm: right-click run_table3.py -> Run.

It reads the same data your real pipeline reads (via texas_gerrymandering_hb4.config),
so it only works on a machine where that data exists — i.e. yours.
"""

import logging

# --- same imports pipeline.py uses ---
from ingest import load_all_sources
from demographics import build_block_vap, apply_cvap_discounting
from spatial import spatial_crosswalk_blocks_to_vtd
from elections import load_election_data
from join import assemble_final_dataset
from schema import enforce_schema, compute_derived_columns

from texas_gerrymandering_hb4.config import (
    CENSUS_DEMOGRAPHICS_PL,
    CENSUS_DEMOGRAPHICS_GEOHEADER,
    ACS_CSV_FILE,
    VTDS_SHP_FILE,
    PLANC2333_SHP_FILE,
    CENSUS_GEO_SHP_FILE,
    GEN_ELECTION_CSV,
    DEM_PRIMARY_CSV,
    REP_PRIMARY_CSV,
)

from table3_diagnostics import collect_de_diagnostics, render_table3

logging.basicConfig(level=logging.INFO)

# --- same constants pipeline.py uses ---
CVAP_FALLBACK_THRESHOLD = 20
TX_CRS = "EPSG:3083"
BLOCK_SUMLEV = "750"

PATHS = {
    "pl_data":     CENSUS_DEMOGRAPHICS_PL,
    "pl_geo":      CENSUS_DEMOGRAPHICS_GEOHEADER,
    "acs_cvap":    ACS_CSV_FILE,
    "vtd_shp":     VTDS_SHP_FILE,
    "planc":       PLANC2333_SHP_FILE,
    "blocks_shp":  CENSUS_GEO_SHP_FILE,
    "gen_returns": GEN_ELECTION_CSV,
    "dem_primary": DEM_PRIMARY_CSV,
    "rep_primary": REP_PRIMARY_CSV,
}


def main():
    # 1. Ingest
    sources = load_all_sources(PATHS, block_sumlev=BLOCK_SUMLEV)

    # 2-3. Block VAP + CVAP discounting
    blocks_vap = build_block_vap(sources["pl_blocks"])
    blocks_cvap = apply_cvap_discounting(
        blocks_vap, sources["acs_cvap"], threshold=CVAP_FALLBACK_THRESHOLD
    )

    # 4. Spatial crosswalk blocks -> VTDs
    vtd_demo = spatial_crosswalk_blocks_to_vtd(
        blocks_cvap, sources["blocks_shp"], sources["vtd_shp"], tx_crs=TX_CRS
    )

    # 5. Elections
    elections = load_election_data(
        sources["gen_returns"], sources["dem_primary"], sources["rep_primary"]
    )

    # 6-8. Assemble + derive + schema
    final = assemble_final_dataset(vtd_demo, elections, sources["planc"])
    final = compute_derived_columns(final)
    final = enforce_schema(final)

    import re

    # General-election vote columns — adjust the pattern to your naming if needed
    gen_cols = [
        c
        for c in final.columns
        if re.search(r"(PRES|SEN|GOV|G24|GEN)", c, re.I) and final[c].dtype.kind in "if"
    ]
    print("General-election columns detected:", gen_cols)

    row_tot = final[gen_cols].sum(axis=1)
    empty_all = int((row_tot == 0).sum())  # zero across ALL general races
    print(f"VTDs with zero votes across ALL general-election cols: {empty_all}")
    print(f"VTDs with zero in the first col only: {int((final[gen_cols[0]] == 0).sum())}")

    gen_only = ["TOTVOTE_PRES_24G", "TOTVOTE_SEN_24G"]  # actual general-election cols
    z = final[final[gen_only].sum(axis=1) == 0]
    print(f"Zero general-election votes: {len(z)} VTDs")
    print(f"  VAP == 0 (unpopulated, expected):        {(z['VAP'] == 0).sum()}")
    print(f"  VAP  > 0 (populated -> real data gap):   {(z['VAP'] > 0).sum()}")
    print(f"  VAP  > 100:                              {(z['VAP'] > 100).sum()}")
    print(f"  total VAP sitting in zero-vote VTDs:     {z['VAP'].sum():,.0f}")

    # --- Table 3 ---
    diag = collect_de_diagnostics(
        final=final,
        blocks_cvap=blocks_cvap,
        blocks_shp=sources["blocks_shp"],
        vtd_shp=sources["vtd_shp"],
        elections=elections,
        vtd_demo=vtd_demo,
    )
    print("\n" + "=" * 60)
    print(render_table3(diag))
    print("=" * 60)


if __name__ == "__main__":
    main()