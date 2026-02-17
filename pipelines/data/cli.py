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

try:
    import maup
except Exception:  # pragma: no cover
    maup = None

from .io import mkdir_p, stdcols, read_any, ensure_crs, assert_projected_planar, write_parquet
from .elections import clean_vtd_election_returns
from .demographics import (
    ensure_geoid20_str,
    unify_pl94_schema,
    pick_pop_columns,
    pick_total_pop_column,
    pick_total_race_columns,
)
from .districts import pick_district_id_col


def out_paths(processed_dir: Path):
    return {
        "geo_vtd": processed_dir / "geo_vtd.parquet",
        "elections": processed_dir / "elections.parquet",
        "returns_vtd": processed_dir / "election_returns_vtd.parquet",
        "plans": processed_dir / "plans.parquet",
        "plan_map": processed_dir / "plan_district_vtd.parquet",
        "vtds_geo": processed_dir / "geospatial" / "vtds.parquet",
    }


# -----------------------------
# CVAP Special Tabulation helpers (ACS 5-year)
# -----------------------------
def load_cvap_block_groups(cvap_blockgr_path: Path, state_fips: str) -> pd.DataFrame:
    """
    Load Census Bureau CVAP Special Tabulation (ACS 5-year) Block Group file (BlockGr.csv)
    and return wide CVAP estimates keyed by 12-digit block-group GEOID.

    Output columns:
      geoid_bg
      cvap_total
      cvap_hisp
      cvap_nh_white
      cvap_nh_black
      cvap_nh_asian
      cvap_nh_native
      cvap_nh_pi
      cvap_other
    """
    df = pd.read_csv(cvap_blockgr_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    needed = {"geoid", "lntitle", "cvap_est"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CVAP BlockGr file missing required columns: {sorted(missing)}")

    df["geoid_bg"] = df["geoid"].astype("string").str[-12:]
    df = df.loc[df["geoid_bg"].str.startswith(state_fips, na=False)].copy()

    df["cvap_est"] = pd.to_numeric(df["cvap_est"], errors="coerce").fillna(0)

    wide = (
        df.pivot_table(index="geoid_bg", columns="lntitle", values="cvap_est", aggfunc="first")
        .reset_index()
    )

    def col(name: str) -> pd.Series:
        return wide[name] if name in wide.columns else pd.Series(0, index=wide.index, dtype="float64")

    out = pd.DataFrame({"geoid_bg": wide["geoid_bg"].astype("string")})

    out["cvap_total"] = col("Total")
    out["cvap_hisp"] = col("Hispanic or Latino")
    out["cvap_nh_white"] = col("White Alone")
    out["cvap_nh_black"] = col("Black or African American Alone")
    out["cvap_nh_asian"] = col("Asian Alone")
    out["cvap_nh_native"] = col("American Indian or Alaska Native Alone")
    out["cvap_nh_pi"] = col("Native Hawaiian or Other Pacific Islander Alone")

    for c in [c for c in out.columns if c != "geoid_bg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # "Other" for CVAP file itself (fine to keep as full residual across all non-Hisp NH groups)
    known = ["cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi"]
    out["cvap_other"] = (out["cvap_total"] - out[known].sum(axis=1)).clip(lower=0)

    return out


def build_elections_meta(processed_dir: Path, election_id: str, year: int, office: str, stage: str):
    outs = out_paths(processed_dir)
    elections = pd.DataFrame([{"election_id": election_id, "year": int(year), "office": office, "stage": stage}])
    write_parquet(elections, outs["elections"])


def build_plans_meta(processed_dir: Path, plan_id: str, cycle: str, chamber: str, ensemble_id: str):
    outs = out_paths(processed_dir)
    plans = pd.DataFrame(
        [{"plan_id": plan_id, "cycle": cycle, "chamber": chamber, "ensemble_id": ensemble_id, "is_enacted": True}]
    )
    write_parquet(plans, outs["plans"])


def build_processed_inputs(
    districts_path: Path,
    census_blocks_path: Path,
    vtds_path: Path,
    pl94_path: Path,
    cvap_blockgr_path: Path,
    elections_path: Path,
    processed_dir: Path,
    state_fips: str,
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
        raise ImportError("geopandas is required for this pipeline.")

    processed_dir = Path(processed_dir)
    mkdir_p(processed_dir)
    outs = out_paths(processed_dir)

    # -----------------------------
    # Load geospatial inputs
    # -----------------------------
    districts = read_any(districts_path)
    blocks = read_any(census_blocks_path)
    vtds = read_any(vtds_path)

    if not isinstance(districts, gpd.GeoDataFrame):
        districts = gpd.GeoDataFrame(districts, geometry="geometry")
    if not isinstance(blocks, gpd.GeoDataFrame):
        blocks = gpd.GeoDataFrame(blocks, geometry="geometry")
    if not isinstance(vtds, gpd.GeoDataFrame):
        vtds = gpd.GeoDataFrame(vtds, geometry="geometry")

    districts = stdcols(ensure_crs(districts))
    blocks = stdcols(ensure_crs(blocks))
    vtds = stdcols(ensure_crs(vtds))

    if districts.crs != vtds.crs:
        districts = districts.to_crs(vtds.crs)
    if blocks.crs != vtds.crs:
        blocks = blocks.to_crs(vtds.crs)

    if "geoid20" not in blocks.columns:
        raise ValueError("Blocks file must contain geoid20.")

    if "vtdkey" not in vtds.columns:
        if "vtd" in vtds.columns:
            vtds = vtds.rename(columns={"vtd": "vtdkey"})
        elif "vtd_key" in vtds.columns:
            vtds = vtds.rename(columns={"vtd_key": "vtdkey"})
    if "vtdkey" not in vtds.columns:
        raise ValueError("VTDs must include vtdkey (or a column renameable to vtdkey).")

    vtds = vtds.copy()
    vtds["vtd_idx"] = np.arange(len(vtds), dtype="int64")
    vtds["vtd_geoid"] = vtds["vtdkey"].astype("string")

    assert_projected_planar(vtds, "VTDs")

    # -----------------------------
    # Plan map: assign VTD -> district (maup.assign)
    d = districts.copy()
    d["district_idx"] = np.arange(len(d), dtype="int64")
    id_col = pick_district_id_col(districts)

    assert_projected_planar(d, "Districts")

    if maup is None:
        raise ImportError("maup is required for assignment/aggregation. Install with `pip install maup`.")

    # Repair geometries to reduce topological errors before assignment
    if not maup.doctor(d, silent=True):
        d = maup.smart_repair(d)
    if not maup.doctor(vtds, silent=True):
        vtds = maup.smart_repair(vtds)

    d = d.set_index("district_idx", drop=False)
    vtds = vtds.set_index("vtd_idx", drop=False)

    vtd_to_district = maup.assign(vtds, d)

    # Build plan_map output (one row per VTD)
    plan_map = pd.DataFrame({
        "vtd_geoid": vtds["vtd_geoid"].astype("string"),
        "district_id": vtd_to_district.map(d[id_col]).astype("string"),
        "district_idx": vtd_to_district.astype("int64"),
    })

    best = j.drop_duplicates("vtd_idx")[["vtd_idx", "district_idx"]]
    if id_col is not None:
        best = best.merge(d[["district_idx", id_col]], on="district_idx", how="left").rename(columns={id_col: "district_id"})
    else:
        best["district_id"] = best["district_idx"] + 1

    plan_map = pd.DataFrame(
        {
            "plan_id": plan_id,
            "vtd_geoid": vtds["vtd_geoid"].astype("string"),
            "district_id": best["district_id"].astype("string"),
        }
    )
    write_parquet(plan_map, outs["plan_map"])

    # -----------------------------
    # Demographics: blocks -> VTD using CONSISTENT centroid assignment
    # -----------------------------
    blocks = ensure_geoid20_str(blocks, col="geoid20")

    pl = read_any(pl94_path)
    pl = ensure_geoid20_str(unify_pl94_schema(pl), col="geoid20")

    pl["geoid20"] = pl["geoid20"].astype("string").str.strip().str.zfill(15)
    blocks["geoid20"] = blocks["geoid20"].astype("string").str.strip().str.zfill(15)

    blocks2 = blocks.merge(pl, on="geoid20", how="left")

    # Identify block attribute columns
    total_pop_col = pick_total_pop_column(blocks2)
    vap_total_col, vap_race_map, _ = pick_pop_columns(blocks2)
    total_race_map = pick_total_race_columns(blocks2)

    # Ensure numeric
    numeric_cols = [total_pop_col, vap_total_col] + list(vap_race_map.values()) + list(total_race_map.values())
    for c in numeric_cols:
        if c in blocks2.columns:
            blocks2[c] = pd.to_numeric(blocks2[c], errors="coerce").fillna(0)

    # Prepare geometries for centroid join
    
    # -----------------------------
    # Assign blocks -> VTDs (maup.assign) and aggregate block-level pop/VAP to VTDs
    # -----------------------------
    if maup is None:
        raise ImportError("maup is required for assignment/aggregation. Install with `pip install maup`.")

    # Ensure CRS match and is planar/projected (maup behaves badly in geographic CRS)
    blk = blocks2[["geoid20", "geometry"]].copy()
    if blk.crs != vtds.crs:
        blk = blk.to_crs(vtds.crs)

    assert_projected_planar(blk, "Blocks")
    assert_projected_planar(vtds, "VTDs")

    # Fix invalid geometries (blocks) and topological issues (VTDs) before assignment
    blk = blk.copy()
    blk["geometry"] = maup.repair.make_valid_polygons(blk.geometry, force_polygons=True)

    if not maup.doctor(vtds, silent=True):
        vtds = maup.smart_repair(vtds)

    vtds = vtds.set_index("vtd_idx", drop=False)

    with maup.progress():
        blocks_to_vtd = maup.assign(blk, vtds)

    # Attach attributes we will aggregate
    total_race_cols = list(total_race_map.values())
    vap_race_cols = list(vap_race_map.values())
    centroid_attr_cols = [total_pop_col] + total_race_cols + [vap_total_col] + vap_race_cols

    for c in centroid_attr_cols:
        blocks2[c] = pd.to_numeric(blocks2[c], errors="coerce").fillna(0)

    # Total pop
    total_pop_by_vtd = (
        blocks2[total_pop_col]
        .groupby(blocks_to_vtd, observed=True)
        .sum()
        .reindex(vtds.index, fill_value=0)
        .astype("int64")
    )

    # Total pop by race
    total_by_vtd = None
    if total_race_cols:
        total_by_vtd = (
            blocks2[total_race_cols]
            .groupby(blocks_to_vtd, observed=True)
            .sum()
            .reindex(vtds.index, fill_value=0)
            .apply(lambda s: np.rint(s).astype("int64"))
        )

    # VAP totals and race buckets
    vap_by_vtd = (
        blocks2[[vap_total_col] + vap_race_cols]
        .groupby(blocks_to_vtd, observed=True)
        .sum()
        .reindex(vtds.index, fill_value=0)
        .apply(lambda s: np.rint(s).astype("int64"))
    )

    # -----------------------------
    # CVAP: disaggregate BG -> blocks (maup.prorate), then aggregate blocks -> VTDs
    # -----------------------------
    cvap_bg = load_cvap_block_groups(cvap_blockgr_path, state_fips=state_fips)

    blocks2["geoid_bg"] = blocks2["geoid20"].astype("string").str.slice(0, 12)

    # Build block-group geometries by dissolving blocks (exact nesting in Census)
    bgs = gpd.GeoDataFrame(
        blocks2[["geoid_bg", "geometry"]]
        .dissolve(by="geoid_bg", as_index=False),
        crs=blocks2.crs,
    )

    # Join BG-level CVAP to BG geometries
    bgs = bgs.merge(cvap_bg, on="geoid_bg", how="left")
    cvap_cols_bg = [
        "cvap_total",
        "cvap_hisp",
        "cvap_nh_white",
        "cvap_nh_black",
        "cvap_nh_asian",
        "cvap_nh_native",
        "cvap_nh_pi",
        "cvap_other",
    ]
    for c in cvap_cols_bg:
        if c not in bgs.columns:
            bgs[c] = 0
        bgs[c] = pd.to_numeric(bgs[c], errors="coerce").fillna(0)

    # Repair BG tiling if needed (rare, but dissolve can introduce slivers)
    if not maup.doctor(bgs, silent=True, accept_holes=True):
        bgs = maup.smart_repair(bgs)

    bgs = bgs.set_index("geoid_bg", drop=False)

    # Assign each block to its BG container
    with maup.progress():
        blocks_to_bg = maup.assign(blk, bgs)

    # Choose weights for disaggregation (paper is ambiguous; VAP-weighted is typically preferable)
    # You can swap to total_pop_col or to equal-area weights by changing weight_col.
    weight_col = vap_total_col if vap_total_col in blocks2.columns else total_pop_col
    w = pd.to_numeric(blocks2[weight_col], errors="coerce").fillna(0)

    # Normalize weights within each BG as in maup docs for disaggregation via prorate
    denom = blocks_to_bg.map(w.groupby(blocks_to_bg, observed=True).sum())
    weights = (w / denom.replace(0, np.nan)).fillna(0)

    # Prorate CVAP from BGs down to blocks
    block_cvap = maup.prorate(blocks_to_bg, bgs[cvap_cols_bg], weights)

    # Rename to match downstream expectations
    block_cvap = block_cvap.rename(columns={c: f"{c}_blk" for c in cvap_cols_bg})
    cvap_blk_cols = list(block_cvap.columns)

    # Aggregate block CVAP up to VTDs
    cvap_by_vtd = (
        block_cvap
        .groupby(blocks_to_vtd, observed=True)
        .sum()
        .reindex(vtds.index, fill_value=0)
        .apply(lambda s: np.rint(s).astype("int64"))
    )
# -----------------------------
    # Build geo_vtd output
    # -----------------------------
    geo = pd.DataFrame({"vtd_geoid": vtds["vtd_geoid"].astype("string")})

    geo["total_pop"] = total_pop_by_vtd.to_numpy()

    # total pop by race buckets
    if total_by_vtd is not None:
        inv_total = {v: k for k, v in total_race_map.items()}
        for src in total_by_vtd.columns:
            out_name = inv_total.get(src)
            if out_name is not None:
                geo[out_name] = total_by_vtd[src].to_numpy()

    # Ensure required total columns exist
    for col in ["total_hisp", "total_nh_white", "total_nh_black"]:
        if col not in geo.columns:
            geo[col] = 0

    # VAP totals and race buckets
    geo["vap_total"] = vap_by_vtd[vap_total_col].to_numpy()
    for out_name, src_col in vap_race_map.items():
        geo[out_name] = vap_by_vtd[src_col].to_numpy()

    for col in ["vap_hisp", "vap_nh_white", "vap_nh_black"]:
        if col not in geo.columns:
            geo[col] = 0

    # CVAP totals and buckets
    geo["cvap_total"] = cvap_by_vtd["cvap_total_blk"].to_numpy()
    geo["cvap_hisp"] = cvap_by_vtd["cvap_hisp_blk"].to_numpy()
    geo["cvap_nh_white"] = cvap_by_vtd["cvap_nh_white_blk"].to_numpy()
    geo["cvap_nh_black"] = cvap_by_vtd["cvap_nh_black_blk"].to_numpy()

    # --- HERE IS THE FIX: define "Other" as residual of (Latino + NH White + NH Black)
    geo["total_other"] = (
        geo["total_pop"]
        - geo["total_hisp"]
        - geo["total_nh_white"]
        - geo["total_nh_black"]
    ).clip(lower=0).astype("int64")

    geo["vap_other"] = (
        geo["vap_total"]
        - geo["vap_hisp"]
        - geo["vap_nh_white"]
        - geo["vap_nh_black"]
    ).clip(lower=0).astype("int64")

    geo["cvap_other"] = (
        geo["cvap_total"]
        - geo["cvap_hisp"]
        - geo["cvap_nh_white"]
        - geo["cvap_nh_black"]
    ).clip(lower=0).astype("int64")

    geo["state_fips"] = state_fips

    write_parquet(geo, outs["geo_vtd"])

    # geospatial VTD output
    outs["vtds_geo"].parent.mkdir(parents=True, exist_ok=True)
    vtds_geo = vtds[["vtd_geoid", "vtdkey", "geometry"]].copy()
    vtds_geo = vtds_geo.merge(geo, on="vtd_geoid", how="left")
    vtds_geo = gpd.GeoDataFrame(vtds_geo, geometry="geometry", crs=vtds.crs)
    vtds_geo.to_parquet(outs["vtds_geo"], index=False)
    print(f"[write] geospatial VTDs -> {outs['vtds_geo'].resolve()}")

    # Election returns
    returns = read_any(elections_path)
    returns = stdcols(returns)

    wide = clean_vtd_election_returns(
        returns,
        office_filter=elections_office_filter,
        prefer_key="vtdkey",
    )

    vtd_key_num = pd.to_numeric(vtds["vtdkey"], errors="coerce").astype("Int64")
    vtd_key_map = (
        pd.DataFrame({"vtdkey": vtd_key_num, "vtd_geoid": vtds["vtd_geoid"].astype("string")})
        .dropna(subset=["vtdkey"])
        .drop_duplicates(subset=["vtdkey"])
    )

    if "vtdkey" not in wide.columns:
        raise ValueError(
            "Election returns did not produce a 'vtdkey' column. "
            "Provide elections with vtdkeyvalue or extend join logic."
        )

    merged = wide.merge(vtd_key_map, on="vtdkey", how="inner")
    if merged.empty:
        raise ValueError(
            "After joining election returns to VTDs on vtdkey, no rows matched. "
            "Check that elections vtdkeyvalue matches the VTD shapefile vtdkey."
        )

    returns_vtd = pd.DataFrame(
        {
            "election_id": election_id,
            "vtd_geoid": merged["vtd_geoid"].astype("string"),
            "votes_total": pd.to_numeric(merged["total_votes"], errors="coerce").fillna(0).astype("int64"),
            "votes_dem": pd.to_numeric(merged["dem_votes"], errors="coerce").fillna(0).astype("int64"),
        }
    )
    write_parquet(returns_vtd, outs["returns_vtd"])

    # Metadata
    build_elections_meta(processed_dir, election_id, election_year, election_office, election_stage)
    build_plans_meta(processed_dir, plan_id, cycle, chamber, ensemble_id)

    print("[OK] Wrote processed inputs:")
    for k, pth in outs.items():
        print(f"  {k}: {pth}")


def main():
    ap = argparse.ArgumentParser(description="Build processed VTD inputs (demographics + elections + plan map).")
    ap.add_argument("--districts", type=Path, required=True, help="District polygons (enacted plan).")
    ap.add_argument("--census", type=Path, required=True, help="Block geometries (needs geoid20).")
    ap.add_argument("--vtds", type=Path, required=True, help="VTD polygons.")
    ap.add_argument("--pl94", type=Path, required=True, help="Block-level PL/derived attributes keyed by geoid20.")
    ap.add_argument("--cvap-blockgr", type=Path, required=True, help="ACS CVAP Special Tabulation BlockGr.csv.")
    ap.add_argument("--elections", type=Path, required=True, help="Election returns file.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory (data/processed).")
    ap.add_argument("--state-fips", default="48", help="2-digit state FIPS, e.g., 48 for Texas.")

    ap.add_argument("--plan-id", default="ENACTED_TXCD_2021")
    ap.add_argument("--ensemble-id", default="ENS_TXCD_2021_recom_v1")
    ap.add_argument("--cycle", default="2021")
    ap.add_argument("--chamber", default="USCD")

    ap.add_argument("--election-id", default="TX_PRES_2020_GEN")
    ap.add_argument("--election-year", type=int, default=2020)
    ap.add_argument("--election-office", default="PRES")
    ap.add_argument("--election-stage", default="GENERAL")
    ap.add_argument("--elections-office-filter", default=None)

    args = ap.parse_args()

    build_processed_inputs(
        districts_path=args.districts,
        census_blocks_path=args.census,
        vtds_path=args.vtds,
        pl94_path=args.pl94,
        cvap_blockgr_path=args.cvap_blockgr,
        elections_path=args.elections,
        processed_dir=args.out,
        state_fips=str(args.state_fips).zfill(2),
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


