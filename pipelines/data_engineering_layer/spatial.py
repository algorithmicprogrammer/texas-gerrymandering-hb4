"""
spatial.py
----------
Disaggregate block-level demographic data to VTDs via area-weighted
spatial intersection.

Strategy
--------
  1. Reproject both layers to EPSG:3083 (Texas Centric Albers Equal Area)
     for accurate area computation.
  2. Spatial-intersect Census blocks with VTDs.
  3. For each block fragment, compute area weight = fragment_area / block_area.
  4. Multiply each VAP/CVAP count by the weight and aggregate to VTD.

This is a standard population-weighted areal interpolation approach; it
assumes uniform population distribution within blocks, which is a reasonable
assumption given that Census blocks are already very small geographic units.

References
----------
Becker et al. (2021). Computational Redistricting and the VRA.
  https://doi.org/10.1089/elj.2020.0704
"""

import logging
import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

# Planar Texas CRS for area measurement
TX_CRS = "EPSG:3083"

# Demographic columns to aggregate by area weight
DEMO_COLS = [
    "POP20",
    "VAP", "BVAP", "HVAP", "AVAP", "AMINVAP", "OVAP", "WVAP",
    "CVAP", "BCVAP", "HCVAP", "ACVAP", "AMINCVAP", "WCVAP",
]


def _fix_geometry(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """
    Validate and repair geometries.
    - Uses make_valid if available (Shapely ≥ 2.0), otherwise buffer(0).
    - Applies a second zero-width buffer to clean residual topology artifacts.
    - Raises ValueError if CRS is undefined.
    """
    if gdf.crs is None:
        raise ValueError(f"{name}: CRS is undefined — cannot continue.")

    try:
        from shapely.validation import make_valid
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )
    except ImportError:
        gdf["geometry"] = gdf["geometry"].buffer(0)

    # Second pass zero-width buffer for silvers / residual artifacts
    gdf["geometry"] = gdf["geometry"].buffer(0)

    n_invalid = (~gdf.geometry.is_valid).sum()
    if n_invalid > 0:
        log.warning(f"  {name}: {n_invalid} geometries still invalid after repair")

    return gdf


def _normalize_vtd_join_key(vtd_shp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Rebuild the CNTYVTD join key using CNTY (Census FIPS) + VTD so it matches
    the cntyvtd column in the election return CSVs.

    The raw shapefile has CNTYVTD = CNTYKEY + VTD which uses a non-FIPS county
    code.  The election CSVs use FIPS + VTD.  We rebuild the key here.
    """
    vtd = vtd_shp.copy()
    vtd["CNTY_FIPS"] = vtd["CNTY"].astype(str).str.zfill(3)
    vtd["CNTYVTD"]   = vtd["CNTY_FIPS"] + vtd["VTD"].astype(str).str.strip()
    log.info(f"  VTD join key rebuilt: {vtd['CNTYVTD'].nunique():,} unique CNTYVTD values")
    return vtd


def spatial_crosswalk_blocks_to_vtd(
    blocks_cvap: pd.DataFrame,
    blocks_shp: gpd.GeoDataFrame,
    vtd_shp: gpd.GeoDataFrame,
    tx_crs: str = TX_CRS,
) -> gpd.GeoDataFrame:
    """
    Area-weighted crosswalk from Census blocks to VTDs.

    Parameters
    ----------
    blocks_cvap : DataFrame
        Block-level VAP + CVAP from demographics.apply_cvap_discounting().
    blocks_shp  : GeoDataFrame
        TIGER/LINE 2020 Census block geometries with GEOID20.
    vtd_shp     : GeoDataFrame
        2024 VTD shapefile with geometry, VTD, CNTY, CNTYKEY.
    tx_crs      : str
        Planar CRS string for accurate area computation.

    Returns
    -------
    GeoDataFrame with one row per VTD containing CNTYVTD, geometry, and all
    aggregated demographic columns.
    """
    # --- Fix and reproject geometries ---
    blocks_geo = _fix_geometry(blocks_shp[["GEOID20", "geometry"]].copy(), "Census blocks")
    vtd_geo    = _fix_geometry(vtd_shp.copy(), "VTDs")

    blocks_geo = blocks_geo.to_crs(tx_crs)
    vtd_geo    = vtd_geo.to_crs(tx_crs)

    # --- Rebuild VTD join key ---
    vtd_geo = _normalize_vtd_join_key(vtd_geo)

    # --- Join demographic data onto block geometries ---
    blocks_geo = blocks_geo.merge(blocks_cvap, on="GEOID20", how="inner")
    log.info(f"  Blocks with demographics: {len(blocks_geo):,}")

    # Compute block areas (before intersection)
    blocks_geo["block_area"] = blocks_geo.geometry.area

    # --- Spatial intersection ---
    log.info("  Running spatial intersection (blocks ∩ VTDs) — may take a moment …")
    intersected = gpd.overlay(blocks_geo, vtd_geo[["CNTYVTD", "geometry"]], how="intersection")
    intersected["frag_area"] = intersected.geometry.area

    # Area weight = fragment area / original block area
    intersected = intersected.merge(
        blocks_geo[["GEOID20", "block_area"]], on="GEOID20", how="left", suffixes=("", "_orig")
    )
    intersected["area_weight"] = intersected["frag_area"] / intersected["block_area_orig"].replace(0, 1)
    intersected["area_weight"] = intersected["area_weight"].clip(0, 1)

    # Apply weights
    for col in DEMO_COLS:
        intersected[col] = intersected[col] * intersected["area_weight"]

    # Aggregate to VTD
    vtd_demo = (
        intersected.groupby("CNTYVTD")[DEMO_COLS]
        .sum()
        .reset_index()
    )

    # Attach VTD geometry (dissolve in case of any duplicates)
    vtd_geom = vtd_geo[["CNTYVTD", "geometry"]].dissolve(by="CNTYVTD").reset_index()
    vtd_demo = vtd_geom.merge(vtd_demo, on="CNTYVTD", how="left")
    vtd_demo = vtd_demo.rename(columns={"POP20": "TOTALPOP"})

    log.info(
        f"  Crosswalk complete: {len(vtd_demo):,} VTDs  "
        f"(VAP total={vtd_demo['VAP'].sum():,.0f})"
    )
    return vtd_demo
