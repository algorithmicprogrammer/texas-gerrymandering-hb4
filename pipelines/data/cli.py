#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from .io import mkdir_p, stdcols, read_any, ensure_crs, assert_projected_planar, write_parquet, dataset_key
from .keys import infer_vtd_width_from_series, normalize_cntyvtd_flexible, digits_only_cntyvtd
from .elections import clean_vtd_election_returns
from .demographics import ensure_geoid20_str, unify_pl94_schema, pick_pop_columns
from .districts import pick_district_id_col

# -----------------------------
# Canonical processed outputs
# -----------------------------
def out_paths(processed_dir: Path):
    return {
        "geo_vtd": processed_dir / "geo_vtd.parquet",
        "elections": processed_dir / "elections.parquet",
        "returns_vtd": processed_dir / "election_returns_vtd.parquet",
        "plans": processed_dir / "plans.parquet",
        "plan_map": processed_dir / "plan_district_vtd.parquet",
    }

def build_elections_meta(processed_dir: Path, election_id: str, year: int, office: str, stage: str, notes: str="") -> None:
    df = pd.DataFrame([{
        "election_id": election_id,
        "year": int(year),
        "office": office,
        "stage": stage,
        "notes": notes or None,
    }])
    write_parquet(df, processed_dir / "elections.parquet")

def build_plans_meta(processed_dir: Path, plan_id: str, cycle: str, chamber: str, ensemble_id: str) -> None:
    df = pd.DataFrame([{
        "plan_id": plan_id,
        "plan_type": "ENACTED",
        "cycle": cycle,
        "chamber": chamber,
        "ensemble_id": ensemble_id,
        "generator": "enacted",
        "seed": None,
        "constraints_json": None,
        "created_at": None,
    }])
    write_parquet(df, processed_dir / "plans.parquet")

def _construct_vtd_geoid_from_cntykey_vtdkey(vtds: 'gpd.GeoDataFrame') -> pd.Series:
    """Construct a project-stable VTD GEOID from (state_fips, CNTYKEY, VTDKEY).

    Your VTD shapefile columns (post-stdcols) typically include:
      - cntykey: county numeric key (1..254)
      - vtdkey: VTD numeric key

    We define: vtd_geoid := '48' + zfill(cntykey,3) + zfill(vtdkey,6)
    This is not necessarily TIGER's official GEOID, but it is deterministic and unique within TX.
    """
    if "cntykey" not in vtds.columns or "vtdkey" not in vtds.columns:
        raise ValueError(
            "VTD shapefile is missing cntykey/vtdkey needed to construct vtd_geoid. "
            "Expected columns like CNTYKEY and VTDKEY."
        )
    cnty = pd.to_numeric(vtds["cntykey"], errors="coerce")
    vtdk = pd.to_numeric(vtds["vtdkey"], errors="coerce")
    if cnty.isna().any() or vtdk.isna().any():
        raise ValueError("Could not parse cntykey/vtdkey as numbers for vtd_geoid construction.")
    return (
        "48"
        + cnty.astype("int64").astype(str).str.zfill(3)
        + vtdk.astype("int64").astype(str).str.zfill(6)
    )


def _infer_or_build_vtd_geoid(vtds: 'gpd.GeoDataFrame') -> pd.Series:
    # Prefer TIGER-style GEOID columns if present
    for cand in ["geoid20", "geoid", "vtdgeoid", "vtd_geoid", "geoid_20"]:
        if cand in vtds.columns:
            return vtds[cand].astype("string").str.strip()
    # Otherwise construct from CNTYKEY/VTDKEY (your provided schema)
    return _construct_vtd_geoid_from_cntykey_vtdkey(vtds).astype("string")

def build_processed_inputs(
    districts_path: Path,
    census_blocks_path: Path,
    vtds_path: Path,
    pl94_path: Path,
    elections_path: Path,
    processed_dir: Path,
    plan_id: str,
    ensemble_id: str,
    cycle: str,
    chamber: str,
    election_id: str,
    election_year: int,
    election_office: str,
    election_stage: str,
    elections_office_filter: str | None,
):
    if gpd is None:
        raise ImportError("geopandas required for this pipeline (districts/census/vtds are geospatial).")

    mkdir_p(processed_dir)
    outs = out_paths(processed_dir)

    # -----------------------------
    # Read raw inputs
    # -----------------------------
    districts = ensure_crs(stdcols(read_any(districts_path)))
    vtds = ensure_crs(stdcols(read_any(vtds_path)))
    blocks = ensure_crs(stdcols(read_any(census_blocks_path)))
    pl = stdcols(read_any(pl94_path))
    elect_raw = stdcols(read_any(elections_path))

    # -----------------------------
    # Elections: clean wide by CNTYVTD
    # -----------------------------
    elect_wide = clean_vtd_election_returns(elect_raw, office_filter=elections_office_filter)

    # -----------------------------
    # VTD keys: build vtd_geoid + helper cntyvtd variants for robust join
    # -----------------------------
    vtds = vtds.copy()
    vtds["vtd_geoid"] = _infer_or_build_vtd_geoid(vtds)

    # Your existing elections join works through CNTYVTD variants, so keep that as helper columns
    # If your VTD shapefile already includes CNTYVTD-like keys, use them; else derive from county+vtd fields.
    if "cntyvtd" in vtds.columns:
        raw_cnty = vtds["cntyvtd"]
    else:
        # attempt from common fields: county + vtd
        possible = None
        for c in ["cntyvtd", "cntyvtd20", "countyvtd", "cnty_vtd"]:
            if c in vtds.columns:
                possible = vtds[c]
                break
        if possible is None:
            raise ValueError("VTD shapefile is missing CNTYVTD-like identifier needed to join your elections. "
                             "Add a column 'cntyvtd' matching the election returns universe, or change elections to use GEOID.")
        raw_cnty = possible

    vtd_width = infer_vtd_width_from_series(raw_cnty, default=6)
    vtds["cntyvtd_std_5"] = normalize_cntyvtd_flexible(raw_cnty, vtd_width=5)
    vtds["cntyvtd_std_6"] = normalize_cntyvtd_flexible(raw_cnty, vtd_width=6)
    vtds["cntyvtd_dig_5"] = digits_only_cntyvtd(vtds["cntyvtd_std_5"], vtd_width=5)
    vtds["cntyvtd_dig_6"] = digits_only_cntyvtd(vtds["cntyvtd_std_6"], vtd_width=6)

    elect_wide["k_std_5"] = normalize_cntyvtd_flexible(elect_wide["cntyvtd"], vtd_width=5)
    elect_wide["k_std_6"] = normalize_cntyvtd_flexible(elect_wide["cntyvtd"], vtd_width=6)
    elect_wide["k_dig_5"] = digits_only_cntyvtd(elect_wide["k_std_5"], vtd_width=5)
    elect_wide["k_dig_6"] = digits_only_cntyvtd(elect_wide["k_std_6"], vtd_width=6)

    # Choose best join variant (same logic as your existing pipeline)
    variants = [
        ("cntyvtd_std_6", "k_std_6"),
        ("cntyvtd_std_5", "k_std_5"),
        ("cntyvtd_dig_6", "k_dig_6"),
        ("cntyvtd_dig_5", "k_dig_5"),
    ]
    candidates = []
    for left_col, right_col in variants:
        left = set(vtds[left_col].dropna().astype(str))
        right = set(elect_wide[right_col].dropna().astype(str))
        candidates.append((left_col, right_col, len(left & right)))
    candidates.sort(key=lambda t: t[2], reverse=True)
    best_left, best_right, overlap = candidates[0]
    if overlap == 0:
        raise ValueError("No overlap between elections CNTYVTD keys and VTD shapefile keys after normalization. "
                         "Fix key universe or supply VTDs with matching CNTYVTD.")

    # Build election returns at VTD_GEOID
    joined = vtds[["vtd_geoid", best_left]].merge(
        elect_wide[[best_right, "dem_votes", "rep_votes", "third_party_votes", "total_votes"]],
        left_on=best_left, right_on=best_right, how="left"
    )
    for c in ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0).astype("int64")

    returns_vtd = pd.DataFrame({
        "election_id": election_id,
        "vtd_geoid": joined["vtd_geoid"].astype("string"),
        "votes_total": joined["total_votes"].astype("int64"),
        "votes_dem": joined["dem_votes"].astype("int64"),
        "votes_rep": joined["rep_votes"].astype("int64"),
        "votes_other": joined["third_party_votes"].astype("int64"),
    })
    returns_vtd["dem_share"] = np.where(returns_vtd["votes_total"] > 0, returns_vtd["votes_dem"] / returns_vtd["votes_total"], np.nan)
    write_parquet(returns_vtd, outs["returns_vtd"])

    # -----------------------------
    # District assignment: plan_district_vtd (enacted)
    # -----------------------------
    districts = districts.copy()
    vtds = vtds.copy()

    # Project for area computations
    # Use a projected CRS if currently geographic
    assert_projected_planar(districts, "districts")
    assert_projected_planar(vtds, "vtds")

    # Overlay and take dominant intersection area
    d = districts.reset_index(drop=True).copy()
    d["district_idx"] = np.arange(len(d))
    v = vtds.reset_index(drop=True).copy()
    v["vtd_idx"] = np.arange(len(v))

    id_col = pick_district_id_col(d)
    inter = gpd.overlay(v[["vtd_idx","geometry"]], d[["district_idx","geometry"]], how="intersection", keep_geom_type=True)
    if inter.empty:
        raise ValueError("VTD↔district overlay produced 0 intersections. Check CRS/geometry validity.")
    inter["inter_area"] = inter.geometry.area
    best = inter.sort_values("inter_area", ascending=False).drop_duplicates("vtd_idx")[["vtd_idx","district_idx"]]
    if id_col is not None:
        best = best.merge(d[["district_idx", id_col]], on="district_idx", how="left").rename(columns={id_col:"district_id"})
    else:
        best["district_id"] = best["district_idx"] + 1

    v_assign = v.merge(best[["vtd_idx","district_id"]], on="vtd_idx", how="left")
    plan_map = pd.DataFrame({
        "plan_id": plan_id,
        "vtd_geoid": v_assign["vtd_geoid"].astype("string"),
        "district_id": v_assign["district_id"].astype("string"),
    })
    write_parquet(plan_map, outs["plan_map"])

    # -----------------------------
    # Demographics: blocks -> VTD (counts) => geo_vtd.parquet
    # -----------------------------
    # Normalize GEOID on blocks and construct GEOID on PL table if needed
    blocks = ensure_geoid20_str(blocks, col="geoid20")
    pl = ensure_geoid20_str(unify_pl94_schema(pl), col="geoid20")

    # Join PL attributes onto blocks
    pl["geoid20"] = pl["geoid20"].astype("string").str.strip().str.zfill(15)
    blocks["geoid20"] = blocks["geoid20"].astype("string").str.strip().str.zfill(15)

    blocks2 = blocks.merge(pl, on="geoid20", how="left")

    total_col, race_map, mode = pick_pop_columns(blocks2)

    if not race_map:
        print("[WARN] No race VAP columns inferred; geo_vtd will include only vap_total. "
              "Check your PL file columns (expected anglovap/blackvap/hispvap/asianvap).")

    # Intersect blocks with VTDs and area-weight
    blk = blocks2[["geoid20", "geometry"]].copy()

    # Build list of attribute columns to carry through overlay
    attr_cols = [total_col] + list(race_map.values())
    attrs = blocks2[["geoid20"] + attr_cols].copy()

    inter2 = gpd.overlay(blk, v[["vtd_idx", "geometry"]], how="intersection", keep_geom_type=False)
    if inter2.empty:
        raise ValueError("blocks→VTD overlay returned 0 rows (CRS/geometry mismatch).")

    inter2 = inter2.merge(attrs, on="geoid20", how="left")

    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    inter2["blk_area"] = blk_area.reindex(inter2["geoid20"]).values
    inter2["inter_area"] = inter2.geometry.area
    inter2 = inter2.loc[inter2["blk_area"] > 0].copy()
    inter2["w"] = (inter2["inter_area"] / inter2["blk_area"]).clip(0, 1)

    for c in attr_cols:
        inter2[c] = pd.to_numeric(inter2[c], errors="coerce").fillna(0) * inter2["w"]

    agg = inter2.groupby("vtd_idx", observed=True)[attr_cols].sum().reindex(v["vtd_idx"], fill_value=0)
    agg = agg.apply(lambda s: np.rint(s).astype("int64"))

    geo = pd.DataFrame({"vtd_geoid": v["vtd_geoid"].astype("string")})
    geo["vap_total"] = agg[total_col].to_numpy()

    for out_name, src_col in race_map.items():
        geo[out_name] = agg[src_col].to_numpy()

    # vap_other = total - known groups (clip at 0)
    known_cols = [c for c in ["vap_nh_white", "vap_nh_black", "vap_hisp", "vap_nh_asian"] if c in geo.columns]
    if known_cols:
        geo["vap_other"] = (geo["vap_total"] - geo[known_cols].sum(axis=1)).clip(lower=0).astype("int64")
    else:
        geo["vap_other"] = 0

    geo["state_fips"] = "48"
    write_parquet(geo, outs["geo_vtd"])


    # -----------------------------
    # Metadata tables
    # -----------------------------
    build_elections_meta(processed_dir, election_id, election_year, election_office, election_stage)
    build_plans_meta(processed_dir, plan_id, cycle, chamber, ensemble_id)

    print("[OK] Wrote processed inputs:")
    for k, pth in outs.items():
        print(f"  {k}: {pth}")

def main():
    ap = argparse.ArgumentParser(description="Build processed inputs needed by the redistricting analysis pipeline (VTD+VAP, GEOID keys).")
    ap.add_argument("--districts", type=Path, required=True, help="District polygons (enacted plan).")
    ap.add_argument("--census", type=Path, required=True, help="Block geometries with geoid20 + demographics joins.")
    ap.add_argument(
        "--vtds",
        type=Path,
        required=True,
        help=(
            "VTD polygons. If GEOID/GEOID20 is present it will be used; otherwise vtd_geoid is constructed "
            "as '48'+zfill(CNTYKEY,3)+zfill(VTDKEY,6)."
        ),
    )
    ap.add_argument("--pl94", type=Path, required=True, help="Block-level attributes keyed by geoid20. Should include VAP if available.")
    ap.add_argument("--elections", type=Path, required=True, help="Election returns keyed by CNTYVTD (tall or wide).")
    ap.add_argument("--out", type=Path, required=True, help="Output directory (data/processed).")

    ap.add_argument("--plan-id", default="ENACTED_TXCD_2021")
    ap.add_argument("--ensemble-id", default="ENS_TXCD_2021_recom_v1")
    ap.add_argument("--cycle", default="2021")
    ap.add_argument("--chamber", default="USCD")

    ap.add_argument("--election-id", default="TX_PRES_2020_GEN")
    ap.add_argument("--election-year", type=int, default=2020)
    ap.add_argument("--election-office", default="PRES")
    ap.add_argument("--election-stage", default="GENERAL")

    ap.add_argument(
        "--elections-office-filter",
        default=None,
        help=(
            "If the elections file contains multiple contests (Office column), filter to this exact Office value "
            "(case-insensitive), e.g. 'President' or 'U.S. Sen'. If omitted and multiple offices exist, the pipeline errors."
        ),
    )

    args = ap.parse_args()

    build_processed_inputs(
        districts_path=args.districts,
        census_blocks_path=args.census,
        vtds_path=args.vtds,
        pl94_path=args.pl94,
        elections_path=args.elections,
        processed_dir=args.out,
        plan_id=args.plan_id,
        ensemble_id=args.ensemble_id,
        cycle=args.cycle,
        chamber=args.chamber,
        election_id=args.election_id,
        election_year=args.election_year,
        election_office=args.election_office,
        election_stage=args.election_stage,
        elections_office_filter=args.elections_office_filter,
    )

if __name__ == "__main__":
    main()
