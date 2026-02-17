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

    # Extract 12-digit block-group GEOID (state+county+tract+bg)
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

    # Non-Hispanic race lines in this tabulation
    out["cvap_nh_white"] = col("White Alone")
    out["cvap_nh_black"] = col("Black or African American Alone")
    out["cvap_nh_asian"] = col("Asian Alone")
    out["cvap_nh_native"] = col("American Indian or Alaska Native Alone")
    out["cvap_nh_pi"] = col("Native Hawaiian or Other Pacific Islander Alone")

    for c in [c for c in out.columns if c != "geoid_bg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

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

    # project into VTD CRS
    if districts.crs != vtds.crs:
        districts = districts.to_crs(vtds.crs)
    if blocks.crs != vtds.crs:
        blocks = blocks.to_crs(vtds.crs)

    if "geoid20" not in blocks.columns:
        raise ValueError("Blocks file must contain geoid20.")

    # VTD key normalization
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
    # Plan map: assign VTD -> district (centroid join)
    # -----------------------------
    d = districts.copy()
    d["district_idx"] = np.arange(len(d), dtype="int64")
    id_col = pick_district_id_col(districts)

    v_cent = vtds[["vtd_idx", "vtd_geoid", "vtdkey", "geometry"]].copy()
    v_cent["geometry"] = v_cent.geometry.buffer(0).centroid

    j = gpd.sjoin(v_cent, d[["district_idx", "geometry"]], predicate="within", how="left")
    missing = int(j["district_idx"].isna().sum())
    if missing:
        miss = j.loc[j["district_idx"].isna(), ["vtd_idx", "geometry"]].copy()
        near = gpd.sjoin_nearest(miss, d[["district_idx", "geometry"]], how="left", distance_col="dist")
        j.loc[j["district_idx"].isna(), "district_idx"] = near["district_idx"].to_numpy()
        if int(j["district_idx"].isna().sum()):
            raise ValueError("Some VTDs could not be assigned to any district (even nearest).")

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
    #   * total_pop and total-by-race (centroid)
    #   * VAP and VAP-by-race (centroid)   <-- FIX
    #   * CVAP (BG->block allocation using block VAP weights, then centroid)
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

    # Ensure numeric on relevant cols
    numeric_cols = [total_pop_col, vap_total_col] + list(vap_race_map.values()) + list(total_race_map.values())
    for c in numeric_cols:
        if c in blocks2.columns:
            blocks2[c] = pd.to_numeric(blocks2[c], errors="coerce").fillna(0)

    # Prepare geometries for centroid join
    blk = blocks2[["geoid20", "geometry"]].copy()
    if blk.crs != vtds.crs:
        blk = blk.to_crs(vtds.crs)

    v_clean = vtds[["vtd_idx", "geometry"]].copy()
    v_clean["geometry"] = v_clean.geometry.buffer(0)

    blk_cent = blk.copy()
    blk_cent["geometry"] = blk_cent.geometry.buffer(0).centroid

    cent_join = gpd.sjoin(blk_cent[["geoid20", "geometry"]], v_clean, predicate="within", how="left")

    # nearest fallback for unassigned block centroids
    missing = int(cent_join["vtd_idx"].isna().sum())
    if missing:
        miss_mask = cent_join["vtd_idx"].isna()
        miss = cent_join.loc[miss_mask, ["geoid20", "geometry"]].copy()
        near = gpd.sjoin_nearest(miss, v_clean, how="left", distance_col="dist")[["geoid20", "vtd_idx"]]
        near_map = near.dropna(subset=["vtd_idx"]).set_index("geoid20")["vtd_idx"]
        cent_join.loc[miss_mask, "vtd_idx"] = cent_join.loc[miss_mask, "geoid20"].map(near_map)
        if int(cent_join["vtd_idx"].isna().sum()):
            raise ValueError("Some block centroids could not be assigned to any VTD (even nearest).")

    # Attach attributes used for centroid aggregation (total, total-race, vap, vap-race)
    total_race_cols = list(total_race_map.values())
    vap_race_cols = list(vap_race_map.values())

    centroid_attr_cols = [total_pop_col] + total_race_cols + [vap_total_col] + vap_race_cols
    centroid_attrs = blocks2[["geoid20"] + centroid_attr_cols].copy()

    cent = cent_join.merge(centroid_attrs, on="geoid20", how="left")
    for c in centroid_attr_cols:
        cent[c] = pd.to_numeric(cent[c], errors="coerce").fillna(0)

    # --- Total pop (centroid)
    total_pop_by_vtd = (
        cent.groupby("vtd_idx", observed=True)[total_pop_col]
        .sum()
        .reindex(vtds["vtd_idx"], fill_value=0)
        .astype("int64")
    )

    # --- Total pop by race (centroid)
    total_by_vtd = None
    if total_race_cols:
        total_by_vtd = (
            cent.groupby("vtd_idx", observed=True)[total_race_cols]
            .sum()
            .reindex(vtds["vtd_idx"], fill_value=0)
            .apply(lambda s: np.rint(s).astype("int64"))
        )

    # --- VAP (centroid)  <-- FIXED HERE
    vap_by_vtd = (
        cent.groupby("vtd_idx", observed=True)[[vap_total_col] + vap_race_cols]
        .sum()
        .reindex(vtds["vtd_idx"], fill_value=0)
        .apply(lambda s: np.rint(s).astype("int64"))
    )

    # --- CVAP: BG->block allocation (weights: block VAP share in BG), then centroid aggregation
    cvap_bg = load_cvap_block_groups(cvap_blockgr_path, state_fips=state_fips)

    blocks2["geoid_bg"] = blocks2["geoid20"].astype("string").str.slice(0, 12)
    blocks2 = blocks2.merge(cvap_bg, on="geoid_bg", how="left")

    blocks2[vap_total_col] = pd.to_numeric(blocks2[vap_total_col], errors="coerce").fillna(0)
    denom = blocks2.groupby("geoid_bg", observed=True)[vap_total_col].transform("sum")
    w_bg = (blocks2[vap_total_col] / denom.replace(0, np.nan)).fillna(0)

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
        if c not in blocks2.columns:
            blocks2[c] = 0
        blocks2[c] = pd.to_numeric(blocks2[c], errors="coerce").fillna(0)
        blocks2[c + "_blk"] = blocks2[c] * w_bg

    cvap_blk_cols = [c + "_blk" for c in cvap_cols_bg]
    cvap_attrs = blocks2[["geoid20"] + cvap_blk_cols].copy()

    cent_cvap = cent_join.merge(cvap_attrs, on="geoid20", how="left")
    for c in cvap_blk_cols:
        cent_cvap[c] = pd.to_numeric(cent_cvap[c], errors="coerce").fillna(0)

    cvap_by_vtd = (
        cent_cvap.groupby("vtd_idx", observed=True)[cvap_blk_cols]
        .sum()
        .reindex(vtds["vtd_idx"], fill_value=0)
        .apply(lambda s: np.rint(s).astype("int64"))
    )

    # -----------------------------
    # Build geo_vtd output
    # -----------------------------
    geo = pd.DataFrame({"vtd_geoid": vtds["vtd_geoid"].astype("string")})

    # Total pop
    geo["total_pop"] = total_pop_by_vtd.to_numpy()

    # Total pop by race buckets + residual other
    if total_by_vtd is not None:
        inv_total = {v: k for k, v in total_race_map.items()}
        for src in total_by_vtd.columns:
            out_name = inv_total.get(src)
            if out_name is not None:
                geo[out_name] = total_by_vtd[src].to_numpy()

        for col in ["total_hisp", "total_nh_white", "total_nh_black", "total_nh_asian", "total_nh_native", "total_nh_pi"]:
            if col not in geo.columns:
                geo[col] = 0

        # --- Define "Other" for the table as: not Latino, not NH White, not NH Black
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

    else:
        for col in ["total_hisp", "total_nh_white", "total_nh_black", "total_nh_asian", "total_nh_native", "total_nh_pi", "total_other"]:
            geo[col] = 0

    # VAP totals and race buckets
    geo["vap_total"] = vap_by_vtd[vap_total_col].to_numpy()
    for out_name, src_col in vap_race_map.items():
        geo[out_name] = vap_by_vtd[src_col].to_numpy()

    for col in ["vap_nh_white", "vap_nh_black", "vap_hisp", "vap_nh_asian", "vap_nh_native"]:
        if col not in geo.columns:
            geo[col] = 0
    known_vap = ["vap_nh_white", "vap_nh_black", "vap_hisp", "vap_nh_asian", "vap_nh_native"]
    geo["vap_other"] = (geo["vap_total"] - geo[known_vap].sum(axis=1)).clip(lower=0).astype("int64")

    # CVAP totals and race buckets
    geo["cvap_total"] = cvap_by_vtd["cvap_total_blk"].to_numpy()
    geo["cvap_hisp"] = cvap_by_vtd["cvap_hisp_blk"].to_numpy()
    geo["cvap_nh_white"] = cvap_by_vtd["cvap_nh_white_blk"].to_numpy()
    geo["cvap_nh_black"] = cvap_by_vtd["cvap_nh_black_blk"].to_numpy()
    geo["cvap_nh_asian"] = cvap_by_vtd["cvap_nh_asian_blk"].to_numpy()
    geo["cvap_nh_native"] = cvap_by_vtd["cvap_nh_native_blk"].to_numpy()
    geo["cvap_nh_pi"] = cvap_by_vtd["cvap_nh_pi_blk"].to_numpy()

    known_cvap = ["cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi"]
    geo["cvap_other"] = (geo["cvap_total"] - geo[known_cvap].sum(axis=1)).clip(lower=0).astype("int64")

    geo["state_fips"] = state_fips

    write_parquet(geo, outs["geo_vtd"])

    # -----------------------------
    # Write geospatial VTDs parquet (geometry + columns)
    # -----------------------------
    outs["vtds_geo"].parent.mkdir(parents=True, exist_ok=True)
    vtds_geo = vtds[["vtd_geoid", "vtdkey", "geometry"]].copy()
    vtds_geo = vtds_geo.merge(geo, on="vtd_geoid", how="left")
    vtds_geo = gpd.GeoDataFrame(vtds_geo, geometry="geometry", crs=vtds.crs)
    vtds_geo.to_parquet(outs["vtds_geo"], index=False)
    print(f"[write] geospatial VTDs -> {outs['vtds_geo'].resolve()}")

    # -----------------------------
    # Election returns -> returns_vtd.parquet (match elections.py signature)
    # -----------------------------
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
            "Your elections file likely lacks vtdkeyvalue. Provide a file with vtdkeyvalue, "
            "or extend the join logic to use cntyvtd."
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

