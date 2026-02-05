# pipelines/data/cli.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from .io import mkdir_p, stdcols, read_any, ensure_crs, assert_projected_planar, write_parquet
from .elections import clean_vtd_election_returns
from .demographics import (
    ensure_geoid20_str,
    unify_pl94_schema,
    pick_pop_columns,
    pick_total_pop_column,
)
from .districts import pick_district_id_col


# -----------------------------
# Canonical processed outputs
# -----------------------------
def out_paths(processed_dir: Path):
    return {
        # tabular
        "geo_vtd": processed_dir / "geo_vtd.parquet",
        "elections": processed_dir / "elections.parquet",
        "returns_vtd": processed_dir / "election_returns_vtd.parquet",
        "plans": processed_dir / "plans.parquet",
        "plan_map": processed_dir / "plan_district_vtd.parquet",
        # geospatial
        "vtds_geo": processed_dir / "geospatial" / "vtds.parquet",
    }


def build_elections_meta(
    processed_dir: Path,
    election_id: str,
    year: int,
    office: str,
    stage: str,
    notes: str = "",
) -> None:
    df = pd.DataFrame(
        [
            {
                "election_id": election_id,
                "year": int(year),
                "office": office,
                "stage": stage,
                "notes": notes or None,
            }
        ]
    )
    write_parquet(df, processed_dir / "elections.parquet")


def build_plans_meta(processed_dir: Path, plan_id: str, cycle: str, chamber: str, ensemble_id: str) -> None:
    df = pd.DataFrame(
        [
            {
                "plan_id": plan_id,
                "plan_type": "ENACTED",
                "cycle": cycle,
                "chamber": chamber,
                "ensemble_id": ensemble_id,
                "generator": "enacted",
                "seed": None,
                "constraints_json": None,
                "created_at": None,
            }
        ]
    )
    write_parquet(df, processed_dir / "plans.parquet")


def _construct_vtd_geoid_from_cntykey_vtdkey(vtds: "gpd.GeoDataFrame") -> pd.Series:
    """Construct a stable VTD GEOID from CNTYKEY/VTDKEY (Texas-specific)."""
    if "cntykey" not in vtds.columns or "vtdkey" not in vtds.columns:
        raise ValueError(
            "VTD shapefile is missing cntykey/vtdkey needed to construct vtd_geoid. "
            "Expected columns like CNTYKEY and VTDKEY."
        )
    cnty = pd.to_numeric(vtds["cntykey"], errors="coerce")
    vtdk = pd.to_numeric(vtds["vtdkey"], errors="coerce")
    if cnty.isna().any() or vtdk.isna().any():
        raise ValueError("Could not parse cntykey/vtdkey as numbers for vtd_geoid construction.")
    return ("48" + cnty.astype("int64").astype(str).str.zfill(3) + vtdk.astype("int64").astype(str).str.zfill(6))


def _infer_or_build_vtd_geoid(vtds: "gpd.GeoDataFrame") -> pd.Series:
    # Prefer TIGER-style GEOID columns if present
    for cand in ["geoid20", "geoid", "vtdgeoid", "vtd_geoid", "geoid_20"]:
        if cand in vtds.columns:
            return vtds[cand].astype("string").str.strip()
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
    # VTD keys: build vtd_geoid and numeric vtdkey (for elections join)
    # -----------------------------
    vtds = vtds.copy()
    vtds["vtd_geoid"] = _infer_or_build_vtd_geoid(vtds)

    if "vtdkey" not in vtds.columns:
        raise ValueError("VTD shapefile is missing VTDKEY (expected column vtdkey after stdcols).")
    vtds["vtdkey"] = pd.to_numeric(vtds["vtdkey"], errors="coerce").astype("Int64")
    if vtds["vtdkey"].isna().any():
        raise ValueError("Some VTDs have missing/non-numeric vtdkey; cannot join elections by vtdkey.")

    vtds = vtds.reset_index(drop=True).copy()
    vtds["vtd_idx"] = np.arange(len(vtds))

    # -----------------------------
    # Elections: clean wide returns (prefer vtdkeyvalue key)
    # -----------------------------
    elect_wide = clean_vtd_election_returns(elect_raw, office_filter=elections_office_filter, prefer_key="vtdkey")
    if "vtdkey" not in elect_wide.columns:
        raise ValueError(
            "Election file did not yield vtdkey after cleaning. "
            "Your election file should include vtdkeyvalue or a compatible key."
        )

    elect_wide = elect_wide.copy()
    elect_wide["vtdkey"] = pd.to_numeric(elect_wide["vtdkey"], errors="coerce").astype("Int64")

    # Aggregate duplicates
    vote_cols = ["dem_votes", "rep_votes", "third_party_votes", "total_votes"]
    elect_wide = elect_wide.groupby("vtdkey", as_index=False)[vote_cols].sum()

    # Join elections to VTD universe
    joined = vtds[["vtd_geoid", "vtdkey"]].merge(elect_wide, on="vtdkey", how="left")

    # Preserve missingness (do NOT fill NAs with 0)
    for c in vote_cols:
        joined[c] = pd.to_numeric(joined[c], errors="coerce")

    # Store votes as nullable integers to match DuckDB BIGINT schema
    returns_vtd = pd.DataFrame(
        {
            "election_id": election_id,
            "vtd_geoid": joined["vtd_geoid"].astype("string"),
            "votes_total": joined["total_votes"].round().astype("Int64"),
            "votes_dem": joined["dem_votes"].round().astype("Int64"),
            "votes_rep": joined["rep_votes"].round().astype("Int64"),
            "votes_other": joined["third_party_votes"].round().astype("Int64"),
        }
    )

    returns_vtd["dem_share"] = pd.NA
    mask = returns_vtd["votes_total"].notna() & (returns_vtd["votes_total"] > 0)
    returns_vtd.loc[mask, "dem_share"] = (
        returns_vtd.loc[mask, "votes_dem"].astype("Float64") / returns_vtd.loc[mask, "votes_total"].astype("Float64")
    )

    write_parquet(returns_vtd, outs["returns_vtd"])

    # -----------------------------
    # District assignment: plan_district_vtd (enacted)
    # FIX: centroid-within + nearest fallback (robust to overlaps/slivers)
    # -----------------------------
    assert_projected_planar(districts, "districts")
    assert_projected_planar(vtds, "vtds")

    d = districts.reset_index(drop=True).copy()
    d["district_idx"] = np.arange(len(d))

    v = vtds.reset_index(drop=True).copy()  # already has vtd_idx
    id_col = pick_district_id_col(d)

    d_clean = d[["district_idx", "geometry"]].copy()
    d_clean["geometry"] = d_clean.geometry.buffer(0)

    v_cent = v[["vtd_idx", "geometry"]].copy()
    v_cent["geometry"] = v_cent.geometry.buffer(0).centroid

    j = gpd.sjoin(v_cent, d_clean, predicate="within", how="left")[["vtd_idx", "district_idx"]]

    missing = int(j["district_idx"].isna().sum())
    if missing:
        try:
            miss_mask = j["district_idx"].isna()
            miss = v_cent.loc[miss_mask, ["vtd_idx", "geometry"]].copy()

            near = gpd.sjoin_nearest(miss, d_clean, how="left", distance_col="dist")[["vtd_idx", "district_idx"]]
            near_map = near.dropna(subset=["district_idx"]).set_index("vtd_idx")["district_idx"]

            j.loc[miss_mask, "district_idx"] = j.loc[miss_mask, "vtd_idx"].map(near_map)

            missing2 = int(j["district_idx"].isna().sum())
            if missing2:
                examples = j.loc[j["district_idx"].isna(), "vtd_idx"].head(10).tolist()
                raise ValueError(
                    f"{missing2} VTD centroids could not be assigned to any district (even nearest). "
                    f"Examples vtd_idx: {examples}"
                )
        except Exception as e:
            raise ValueError(f"{missing} VTD centroids not within any district and nearest fallback failed: {e}")

    best = j.drop_duplicates("vtd_idx")[["vtd_idx", "district_idx"]]

    if id_col is not None:
        best = best.merge(d[["district_idx", id_col]], on="district_idx", how="left").rename(columns={id_col: "district_id"})
    else:
        best["district_id"] = best["district_idx"] + 1

    v_assign = v.merge(best[["vtd_idx", "district_id"]], on="vtd_idx", how="left")
    plan_map = pd.DataFrame(
        {
            "plan_id": plan_id,
            "vtd_geoid": v_assign["vtd_geoid"].astype("string"),
            "district_id": v_assign["district_id"].astype("string"),
        }
    )
    write_parquet(plan_map, outs["plan_map"])

    # -----------------------------
    # Demographics: blocks -> VTD (counts) => geo_vtd.parquet
    #
    # IMPORTANT:
    #   - TOTAL POP uses centroid assignment (no area weights) to avoid double-counting when VTD polygons overlap.
    #   - VAP (and race-VAP) keeps the existing area-weighted overlay.
    # -----------------------------
    blocks = ensure_geoid20_str(blocks, col="geoid20")
    pl = ensure_geoid20_str(unify_pl94_schema(pl), col="geoid20")

    pl["geoid20"] = pl["geoid20"].astype("string").str.strip().str.zfill(15)
    blocks["geoid20"] = blocks["geoid20"].astype("string").str.strip().str.zfill(15)

    blocks2 = blocks.merge(pl, on="geoid20", how="left")

    # VAP schema (existing)
    total_col, race_map, _mode = pick_pop_columns(blocks2)

    # TOTAL POP column (new)
    total_pop_col = pick_total_pop_column(blocks2)

    # Geometry + attributes
    blk = blocks2[["geoid20", "geometry"]].copy()

    vap_cols = [total_col] + list(race_map.values())
    attrs = blocks2[["geoid20", total_pop_col] + vap_cols].copy()

    if blk.crs is None:
        raise ValueError("Blocks CRS is None; cannot overlay. Assign CRS first.")
    if vtds.crs is None:
        raise ValueError("VTD CRS is None; cannot overlay. Assign CRS first.")
    if blk.crs != vtds.crs:
        blk = blk.to_crs(vtds.crs)

    assert_projected_planar(vtds, "VTDs")

    # -----------------------------
    # TOTAL POP: centroid-based block -> VTD assignment
    # FIX: fill missing rows only (do NOT drop vtd_idx for everyone)
    # -----------------------------
    v_clean = vtds[["vtd_idx", "geometry"]].copy()
    v_clean["geometry"] = v_clean.geometry.buffer(0)

    blk_cent = blk.copy()
    blk_cent["geometry"] = blk_cent.geometry.buffer(0).centroid

    cent_join = gpd.sjoin(
        blk_cent[["geoid20", "geometry"]],
        v_clean,
        predicate="within",
        how="left",
    )

    cent_join = cent_join.merge(attrs[["geoid20", total_pop_col]], on="geoid20", how="left")
    cent_join[total_pop_col] = pd.to_numeric(cent_join[total_pop_col], errors="coerce").fillna(0)

    missing = int(cent_join["vtd_idx"].isna().sum())
    if missing:
        try:
            miss_mask = cent_join["vtd_idx"].isna()
            miss = cent_join.loc[miss_mask, ["geoid20", "geometry"]].copy()

            near = gpd.sjoin_nearest(miss, v_clean, how="left", distance_col="dist")[["geoid20", "vtd_idx"]]
            near_map = near.dropna(subset=["vtd_idx"]).set_index("geoid20")["vtd_idx"]

            cent_join.loc[miss_mask, "vtd_idx"] = cent_join.loc[miss_mask, "geoid20"].map(near_map)

            missing2 = int(cent_join["vtd_idx"].isna().sum())
            if missing2:
                examples = cent_join.loc[cent_join["vtd_idx"].isna(), "geoid20"].head(10).tolist()
                raise ValueError(
                    f"{missing2} block centroids could not be assigned to any VTD (even nearest). "
                    f"Examples geoid20: {examples}"
                )
        except Exception as e:
            raise ValueError(
                f"{missing} block centroids were not within any VTD, and nearest-join fallback failed. Error: {e}"
            )

    total_pop_by_vtd = (
        cent_join.groupby("vtd_idx", observed=True)[total_pop_col]
        .sum()
        .reindex(vtds["vtd_idx"], fill_value=0)
        .astype("int64")
    )

    # -----------------------------
    # VAP: area-weighted block -> VTD overlay
    # -----------------------------
    inter2 = gpd.overlay(blk, vtds[["vtd_idx", "geometry"]], how="intersection", keep_geom_type=False)
    if inter2.empty:
        raise ValueError("blocks→VTD overlay returned 0 rows (CRS/geometry mismatch).")

    inter2 = inter2.merge(attrs[["geoid20"] + vap_cols], on="geoid20", how="left")

    blk_area = blk.set_index("geoid20").geometry.area.rename("blk_area")
    inter2["blk_area"] = blk_area.reindex(inter2["geoid20"]).values
    inter2["inter_area"] = inter2.geometry.area
    inter2 = inter2.loc[inter2["blk_area"] > 0].copy()
    inter2["w"] = (inter2["inter_area"] / inter2["blk_area"]).clip(0, 1)

    for c in vap_cols:
        inter2[c] = pd.to_numeric(inter2[c], errors="coerce").fillna(0) * inter2["w"]

    agg_vap = inter2.groupby("vtd_idx", observed=True)[vap_cols].sum().reindex(vtds["vtd_idx"], fill_value=0)
    agg_vap = agg_vap.apply(lambda s: np.rint(s).astype("int64"))

    # -----------------------------
    # Build geo_vtd output
    # -----------------------------
    geo = pd.DataFrame({"vtd_geoid": vtds["vtd_geoid"].astype("string")})

    # TOTAL POP (new)
    geo["total_pop"] = total_pop_by_vtd.to_numpy()

    # VAP (existing)
    geo["vap_total"] = agg_vap[total_col].to_numpy()
    for out_name, src_col in race_map.items():
        geo[out_name] = agg_vap[src_col].to_numpy()

    # Ensure all expected race columns exist for downstream (including nh_native)
    for col in ["vap_nh_white", "vap_nh_black", "vap_hisp", "vap_nh_asian", "vap_nh_native"]:
        if col not in geo.columns:
            geo[col] = 0

    known_cols = [
        c for c in ["vap_nh_white", "vap_nh_black", "vap_hisp", "vap_nh_asian", "vap_nh_native"] if c in geo.columns
    ]
    geo["vap_other"] = (geo["vap_total"] - geo[known_cols].sum(axis=1)).clip(lower=0).astype("int64")

    geo["state_fips"] = "48"
    write_parquet(geo, outs["geo_vtd"])

    # -----------------------------
    # Write geospatial VTDs keyed by vtd_geoid (geometry + pop columns)
    # -----------------------------
    vtds_geo_path = outs["vtds_geo"]
    vtds_geo_path.parent.mkdir(parents=True, exist_ok=True)

    vtds_geo = vtds[["vtd_geoid", "vtdkey", "geometry"]].copy()
    vtds_geo = vtds_geo.merge(geo, on="vtd_geoid", how="left")
    vtds_geo = gpd.GeoDataFrame(vtds_geo, geometry="geometry", crs=vtds.crs)

    if vtds_geo.geometry.isna().any():
        n_missing = int(vtds_geo.geometry.isna().sum())
        raise RuntimeError(f"vtds_geo has {n_missing} missing geometries (unexpected).")
    if vtds_geo["total_pop"].isna().any():
        n_missing = int(vtds_geo["total_pop"].isna().sum())
        raise ValueError(f"vtds_geo missing total_pop for {n_missing} rows after merge (unexpected).")
    if vtds_geo["vap_total"].isna().any():
        n_missing = int(vtds_geo["vap_total"].isna().sum())
        raise ValueError(f"vtds_geo missing vap_total for {n_missing} rows after merge (unexpected).")

    vtds_geo.to_parquet(vtds_geo_path, index=False)
    print(f"[write] geospatial VTDs -> {vtds_geo_path.resolve()}")

    # -----------------------------
    # Metadata tables
    # -----------------------------
    build_elections_meta(processed_dir, election_id, election_year, election_office, election_stage)
    build_plans_meta(processed_dir, plan_id, cycle, chamber, ensemble_id)

    print("[OK] Wrote processed inputs:")
    for k, pth in outs.items():
        print(f"  {k}: {pth}")


def main():
    ap = argparse.ArgumentParser(
        description="Build processed inputs needed by the redistricting analysis pipeline (VTD+demographics, GEOID keys)."
    )
    ap.add_argument("--districts", type=Path, required=True, help="District polygons (enacted plan).")
    ap.add_argument("--census", type=Path, required=True, help="Block geometries (needs geoid20).")
    ap.add_argument("--vtds", type=Path, required=True, help="VTD polygons.")
    ap.add_argument(
        "--pl94",
        type=Path,
        required=True,
        help="Block-level attributes keyed by geoid20 (should include PL94 total pop and VAP columns if available).",
    )
    ap.add_argument(
        "--elections",
        type=Path,
        required=True,
        help="Election returns (tall or wide). Prefer files with vtdkeyvalue.",
    )
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
        help="If the elections file contains multiple contests (Office column), filter to this exact Office value (case-insensitive).",
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

