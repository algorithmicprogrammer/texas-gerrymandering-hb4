"""
join.py
-------
Assemble the final analysis-ready dataset by joining:
  1. VTD demographic data (from spatial crosswalk)
  2. Election returns (from elections.py)
  3. Congressional district assignments from PLANC2333

Join key throughout: CNTYVTD (County FIPS + VTD, string).
"""

import logging
import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

TX_CRS = "EPSG:3083"


def _assign_congressional_districts(
    vtd_demo: gpd.GeoDataFrame,
    planc: gpd.GeoDataFrame,
    tx_crs: str = TX_CRS,
) -> gpd.GeoDataFrame:
    """
    Assign each VTD to a congressional district via largest-area overlap
    (point-in-polygon on centroid, with area-overlap fallback).

    Using centroid is fast and correct for the vast majority of VTDs.
    The area-overlap fallback handles edge cases where a centroid lands on a
    district boundary.
    """
    vtd = vtd_demo.to_crs(tx_crs).copy()
    districts = planc.to_crs(tx_crs)[["District", "geometry"]].copy()

    # Centroid join (fast path)
    centroids = vtd[["CNTYVTD", "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin(
        centroids,
        districts.rename(columns={"District": "CD"}),
        how="left",
        predicate="within",
    )

    # Some centroids may fall outside any district (boundary artifacts)
    unmatched = joined["CD"].isna()
    if unmatched.any():
        n = unmatched.sum()
        log.warning(
            f"  {n} VTD centroids outside all districts — "
            "falling back to area-overlap for those"
        )
        # Area overlap fallback for unmatched VTDs
        unmatched_vtds = vtd[vtd["CNTYVTD"].isin(joined.loc[unmatched, "CNTYVTD"])].copy()
        overlap = gpd.overlay(unmatched_vtds[["CNTYVTD", "geometry"]], districts, how="intersection")
        overlap["area"] = overlap.geometry.area
        best = (
            overlap.sort_values("area", ascending=False)
            .drop_duplicates("CNTYVTD")[["CNTYVTD", "District"]]
            .rename(columns={"District": "CD"})
        )
        # Patch in the fallback assignments
        cd_map = joined.set_index("CNTYVTD")["CD"].to_dict()
        for _, row in best.iterrows():
            cd_map[row["CNTYVTD"]] = row["CD"]
        joined_cd = pd.DataFrame(list(cd_map.items()), columns=["CNTYVTD", "CD"])
    else:
        joined_cd = joined[["CNTYVTD", "CD"]].drop_duplicates("CNTYVTD")

    vtd_demo = vtd_demo.merge(joined_cd, on="CNTYVTD", how="left")
    vtd_demo["CD"] = pd.to_numeric(vtd_demo["CD"], errors="coerce").astype("Int64")

    n_assigned = vtd_demo["CD"].notna().sum()
    log.info(f"  Congressional district assignment: {n_assigned:,} / {len(vtd_demo):,} VTDs")
    return vtd_demo


def assemble_final_dataset(
    vtd_demo: gpd.GeoDataFrame,
    elections: pd.DataFrame,
    planc: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Join election data and congressional district assignments to VTD demographics.

    Parameters
    ----------
    vtd_demo   : GeoDataFrame — VTD-level demographics from spatial crosswalk
    elections  : DataFrame    — wide election returns from elections.py
    planc      : GeoDataFrame — PLANC2333 congressional district boundaries

    Returns
    -------
    GeoDataFrame with all demographic, election, and district columns.
    """
    log.info("  Assigning congressional districts …")
    vtd_demo = _assign_congressional_districts(vtd_demo, planc)

    log.info("  Joining election returns …")
    n_before = len(vtd_demo)
    final = vtd_demo.merge(elections, on="CNTYVTD", how="left")

    # Fill missing election data (precincts with no votes recorded) with 0
    election_cols = [c for c in elections.columns if c != "CNTYVTD"]
    final[election_cols] = final[election_cols].fillna(0).astype("int64")

    n_unmatched = final[election_cols[0]].eq(0).sum()
    log.info(
        f"  Election join: {n_before:,} → {len(final):,} VTDs  "
        f"({n_unmatched:,} with zero votes — check for unmatched precincts)"
    )

    # Log any CNTYVTD in elections not present in VTD shapefile
    missing_in_shp = set(elections["CNTYVTD"]) - set(vtd_demo["CNTYVTD"])
    if missing_in_shp:
        log.warning(
            f"  {len(missing_in_shp):,} election CNTYVTD keys not found in VTD "
            f"shapefile — these precincts are dropped. Sample: "
            f"{sorted(missing_in_shp)[:5]}"
        )

    return final
