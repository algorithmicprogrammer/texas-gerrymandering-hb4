"""
Texas Precinct-Level Data Engineering Pipeline
===============================================
Produces a cleaned, analysis-ready precinct shapefile for ecological inference
and VRA ReCom ensemble work.

Pipeline steps:
  1. Ingest raw Census (PL 94-171), ACS CVAP, and election data
  2. Construct demographic variables (VAP -> CVAP via MGGG discounting)
  3. Disaggregate / reaggregate Census blocks -> VTDs
  4. Normalize join keys
  5. Join election + geography
  6. Validate data integrity
  7. Output final analysis-ready dataset

References
----------
Becker et al. (2021). Computational Redistricting and the VRA.
  https://doi.org/10.1089/elj.2020.0704
Donnay, Duchin & Rock (n.d.). Conventions for Constructing Demographic
  Categories. https://mggg.org/vap-cvap
MGGG (n.d.). VAP and CVAP. https://mggg.org/vap-cvap
"""

import logging
import sys
from pathlib import Path

from ingest import load_all_sources
from demographics import build_block_vap, apply_cvap_discounting
from spatial import spatial_crosswalk_blocks_to_vtd
from elections import load_election_data
from join import assemble_final_dataset
from validate import run_all_validations
from schema import enforce_schema, compute_derived_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path configuration – edit these to match your directory layout
# ---------------------------------------------------------------------------
RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    # Source 1 – PL 94-171 data segment
    "pl_data": RAW / "tx_pl2020/tx000022020.pl",
    # Source 2 – PL 94-171 geo header
    "pl_geo": RAW / "tx_pl2020/txgeo2020.pl",
    # Source 3 – ACS 5-year CVAP special tabulation (tract level)
    "acs_cvap": RAW / "CVAP_2020-2024_ACS_csv_files/Tract.csv",
    # Source 4 – 2024 primary + general VTD shapefile (geometry + CNTYKEY)
    "vtd_shp": RAW / "vtds_24pg/VTDs_24PG.shp",
    # Source 5 – PLANC2333 enacted congressional district shapefile
    "planc": RAW / "PLANC2333/PLANC2333.shp",
    # Source 6 – TIGER/LINE 2020 Census block shapefile
    "blocks_shp": RAW / "tl_2020_48_tabblock20/tl_2020_48_tabblock20.shp",
    # Source 7 – 2024 general election returns (VTD-level CSV)
    "gen_returns": RAW / "2024-general-vtds-election-data/2024_General_Election_Returns.csv",
    # Source 8 – 2024 Democratic primary returns
    "dem_primary": RAW / "2024-general-vtds-election-data/2024_Democratic_Primary_Election_Returns.csv",
    # Source 9 – 2024 Republican primary returns
    "rep_primary": RAW / "2024-general-vtds-election-data/2024_Republican_Primary_Election_Returns.csv",
}

# MGGG discounting: use state-level citizenship rate when tract VAP < this
CVAP_FALLBACK_THRESHOLD = 20

# Texas EPSG for planar area/projection operations
TX_CRS = "EPSG:3083"

# Summary level code for Census blocks in geo header
BLOCK_SUMLEV = "750"


def run_pipeline():
    log.info("=== TX Redistricting Data Engineering Pipeline ===")

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    log.info("Step 1 — Ingesting raw data sources")
    sources = load_all_sources(PATHS, block_sumlev=BLOCK_SUMLEV)

    # ------------------------------------------------------------------
    # 2. Build block-level VAP demographic variables
    # ------------------------------------------------------------------
    log.info("Step 2 — Constructing block-level VAP variables (MGGG convention)")
    blocks_vap = build_block_vap(sources["pl_blocks"])

    # ------------------------------------------------------------------
    # 3. CVAP via discounting (MGGG method)
    # ------------------------------------------------------------------
    log.info("Step 3 — Estimating CVAP via ACS citizenship-rate discounting")
    blocks_cvap = apply_cvap_discounting(
        blocks_vap,
        sources["acs_cvap"],
        threshold=CVAP_FALLBACK_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # 4. Spatial crosswalk: blocks -> VTDs
    # ------------------------------------------------------------------
    log.info("Step 4 — Spatial crosswalk: disaggregate blocks -> VTDs")
    vtd_demo = spatial_crosswalk_blocks_to_vtd(
        blocks_cvap,
        sources["blocks_shp"],
        sources["vtd_shp"],
        tx_crs=TX_CRS,
    )

    # ------------------------------------------------------------------
    # 5. Load and normalize election data
    # ------------------------------------------------------------------
    log.info("Step 5 — Loading and normalizing election returns")
    elections = load_election_data(
        sources["gen_returns"],
        sources["dem_primary"],
        sources["rep_primary"],
    )

    # ------------------------------------------------------------------
    # 6. Assemble final dataset (join election + geography + districts)
    # ------------------------------------------------------------------
    log.info("Step 6 — Joining election data + congressional district assignments")
    final = assemble_final_dataset(vtd_demo, elections, sources["planc"])

    # ------------------------------------------------------------------
    # 7. Derived proportion columns
    # ------------------------------------------------------------------
    log.info("Step 7 — Computing derived proportion columns")
    final = compute_derived_columns(final)

    # ------------------------------------------------------------------
    # 8. Enforce final schema column order / types
    # ------------------------------------------------------------------
    log.info("Step 8 — Enforcing final schema")
    final = enforce_schema(final)

    # ------------------------------------------------------------------
    # 9. Validate
    # ------------------------------------------------------------------
    log.info("Step 9 — Running data validation suite")
    run_all_validations(final, blocks_cvap)

    # ------------------------------------------------------------------
    # 10. Output
    # ------------------------------------------------------------------
    out_path = OUT / "tx_precincts_final.gpkg"
    log.info(f"Step 10 — Writing output to {out_path}")
    final.to_file(out_path, driver="GPKG", layer="precincts")

    log.info("=== Pipeline complete ===")
    return final


if __name__ == "__main__":
    run_pipeline()
