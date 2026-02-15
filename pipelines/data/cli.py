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


# -----------------------------
# CVAP Special Tabulation helpers (ACS 5-year)
# -----------------------------
def load_cvap_block_groups(cvap_blockgr_path: Path, state_fips: str) -> pd.DataFrame:
    """Load Census Bureau CVAP Special Tabulation (Block Group) file and return wide CVAP estimates.

    The CVAP special tabulation is delivered in *long* format with one row per
    (block group, race/ethnicity line). We pivot it to wide format and produce a
    standardized set of columns:

        geoid_bg (12-digit block group GEOID, e.g. '481130201001')
        cvap_total
        cvap_hisp
        cvap_nh_white
        cvap_nh_black
        cvap_nh_asian
        cvap_nh_native
        cvap_nh_pi
        cvap_other

    Notes:
      - 'geoid' in the file looks like '1500000US<12-digit GEOID>'; we strip to the last 12 digits.
      - Race lines such as 'White Alone' in this file are within the 'Not Hispanic or Latino' section,
        so they correspond to *non-Hispanic* race categories.
      - We keep only the estimate columns (cvap_est). MOE can be added later if needed.

    Args:
        cvap_blockgr_path: Path to BlockGr.csv from the CVAP special tabulation release.
        state_fips: 2-digit state FIPS (e.g., '48' for Texas).

    Returns:
        DataFrame with one row per block group and standardized CVAP columns.
    """
    df = pd.read_csv(cvap_blockgr_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"geoid", "lntitle", "cvap_est"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CVAP BlockGr file missing required columns: {sorted(missing)}")

    # Extract 12-digit BG GEOID
    df["geoid_bg"] = df["geoid"].astype("string").str[-12:]
    df = df.loc[df["geoid_bg"].str.startswith(state_fips, na=False)].copy()

    # Numeric cvap_est
    df["cvap_est"] = pd.to_numeric(df["cvap_est"], errors="coerce").fillna(0)

    wide = (
        df.pivot_table(index="geoid_bg", columns="lntitle", values="cvap_est", aggfunc="first")
        .reset_index()
    )

    def col(name: str) -> pd.Series:
        if name in wide.columns:
            return wide[name]
        return pd.Series(0, index=wide.index, dtype="float64")

    out = pd.DataFrame({"geoid_bg": wide["geoid_bg"].astype("string")})

    # Core lines
    out["cvap_total"] = col("Total")
    out["cvap_hisp"] = col("Hispanic or Latino")

    # Non-Hispanic race lines (as labeled in CVAP file)
    out["cvap_nh_white"] = col("White Alone")
    out["cvap_nh_black"] = col("Black or African American Alone")
    out["cvap_nh_asian"] = col("Asian Alone")
    out["cvap_nh_native"] = col("American Indian or Alaska Native Alone")
    out["cvap_nh_pi"] = col("Native Hawaiian or Other Pacific Islander Alone")

    # Fill NA -> 0, round to int later after allocation/aggregation
    for c in [c for c in out.columns if c != "geoid_bg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # Derive 'other' as residual from total
    known = ["cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi"]
    out["cvap_other"] = (out["cvap_total"] - out[known].sum(axis=1)).clip(lower=0)

    return out


def build_elections_meta(
    processed_dir: Path,
    election_id: str,
    election_year: int,
    election_office: str,
    election_stage: str,
):
    outs = out_paths(processed_dir)
    elections = pd.DataFrame(
        [
            {
                "election_id": election_id,
                "year": int(election_year),
                "office": election_office,
                "stage": election_stage,
            }
        ]
    )
    write_parquet(elections, outs["elections"])


def build_plans_meta(processed_dir: Path, plan_id: str, cycle: str, chamber: str, ensemble_id: str):
    outs = out_paths(processed_dir)
    plans = pd.DataFrame(
        [
            {
                "plan_id": plan_id,
                "cycle": cycle,
                "chamber": chamber,
                "ensemble_id": ensemble_id,
                "is_enacted": True,
            }
        ]
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

    districts = ensure_crs(districts)
    blocks = ensure_crs(blocks)
    vtds = ensure_crs(vtds)

    # Make sure everything is in same CRS as VTDs
    if districts.crs != vtds.crs:
        districts = districts.to_crs(vtds.crs)
    if blocks.crs != vtds.crs:
        blocks = blocks.to_crs(vtds.crs)

    # Basic sanity
    if "geoid20" not in [c.lower() for c in blocks.columns]:
        # try standardize
        blocks = stdcols(blocks)
    blocks = stdcols(blocks)
    districts = stdcols(districts)
    vtds = stdcols(vtds)

    if "geoid20" not in blocks.columns:
        raise ValueError("Blocks file must contain geoid20.")

    # Identify district id column (optional)
    id_col = pick_district_id_col(districts)

    # -----------------------------
    # Build plan map: assign VTD -> district
    # -----------------------------
    if "vtdkey" not in vtds.columns:
        # attempt common alternatives
        if "vtd" in vtds.columns:
            vtds = vtds.rename(columns={"vtd": "vtdkey"})
        elif "vtd_key" in vtds.columns:
            vtds = vtds.rename(columns={"vtd_key": "vtdkey"})

    if "vtdkey" not in vtds.columns:
        raise ValueError("VTDs must include vtdkey (or a column renameable to vtdkey).")

    # canonical vtd index + geoid
    vtds = vtds.copy()
    vtds["vtd_idx"] = np.arange(len(vtds), dtype="int64")
    vtds["vtd_geoid"] = vtds["vtdkey"].astype("string")

    # canonical district index
    d = districts.copy()
    d["district_idx"] = np.arange(len(d), dtype="int64")

    # centroid-based spatial join: VTD centroid within district polygon
    v = vtds[["vtd_idx", "vtd_geoid", "vtdkey", "geometry"]].copy()
    v["geometry"] = v.geometry.buffer(0).centroid

    j = gpd.sjoin(v, d[["district_idx", "geometry"]], predicate="within", how="left")
    missing = int(j["district_idx"].isna().sum())
    if missing:
        # fallback: nearest district for missing
        try:
            miss = j.loc[j["district_idx"].isna(), ["vtd_idx", "geometry"]].copy()
            near = gpd.sjoin_nearest(miss, d[["district_idx", "geometry"]], how="left", distance_col="dist")
            j.loc[j["district_idx"].isna(), "district_idx"] = near["district_idx"].to_numpy()
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
    #   - CVAP is taken from the Census Bureau's CVAP Special Tabulation at BLOCK GROUP level (ACS 5-year),
    #     allocated down to blocks (by block VAP share within block group), then aggregated to VTD via centroid assignment.
    #   - We still require PL94 VAP columns *only* to compute allocation weights from block group -> block.
    # -----------------------------
    blocks = ensure_geoid20_str(blocks, col="geoid20")
    pl = read_any(pl94_path)
    pl = ensure_geoid20_str(unify_pl94_schema(pl), col="geoid20")

    pl["geoid20"] = pl["geoid20"].astype("string").str.strip().str.zfill(15)
    blocks["geoid20"] = blocks["geoid20"].astype("string").str.strip().str.zfill(15)

    blocks2 = blocks.merge(pl, on="geoid20", how="left")

    # VAP schema (used only for allocation weights)
    total_col, race_map, _mode = pick_pop_columns(blocks2)

    # TOTAL POP column
    total_pop_col = pick_total_pop_column(blocks2)

    # Load CVAP special tabulation (Block Group level) and merge to blocks
    cvap_bg = load_cvap_block_groups(cvap_blockgr_path, state_fips="48")
    blocks2["geoid_bg"] = blocks2["geoid20"].astype("string").str.slice(0, 12)
    blocks2 = blocks2.merge(cvap_bg, on="geoid_bg", how="left")

    # Allocation weights: block VAP share within block group
    blocks2[total_col] = pd.to_numeric(blocks2[total_col], errors="coerce").fillna(0)
    bg_denom = blocks2.groupby("geoid_bg", observed=True)[total_col].transform("sum")
    w = (blocks2[total_col] / bg_denom.replace(0, np.nan)).fillna(0)

    # Allocate BG CVAP estimates to blocks (float; round after aggregating to VTD)
    cvap_cols_bg = ["cvap_total", "cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi", "cvap_other"]
    for c in cvap_cols_bg:
        if c not in blocks2.columns:
            blocks2[c] = 0
        blocks2[c] = pd.to_numeric(blocks2[c], errors="coerce").fillna(0)
        blocks2[c + "_blk"] = blocks2[c] * w

    # Geometry + attributes
    blk = blocks2[["geoid20", "geometry"]].copy()
    attrs = blocks2[["geoid20", total_pop_col] + [c + "_blk" for c in cvap_cols_bg]].copy()

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
    # CVAP: centroid-based block -> VTD aggregation (from allocated block CVAP)
    # -----------------------------
    cvap_blk_cols = [c + "_blk" for c in ["cvap_total", "cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi", "cvap_other"]]
    cent_cvap = cent_join.merge(attrs[["geoid20"] + cvap_blk_cols], on="geoid20", how="left")

    for c in cvap_blk_cols:
        cent_cvap[c] = pd.to_numeric(cent_cvap[c], errors="coerce").fillna(0)

    cvap_by_vtd = cent_cvap.groupby("vtd_idx", observed=True)[cvap_blk_cols].sum().reindex(vtds["vtd_idx"], fill_value=0)
    cvap_by_vtd = cvap_by_vtd.apply(lambda s: np.rint(s).astype("int64"))

    # -----------------------------
    # Build geo_vtd output
    # -----------------------------
    geo = pd.DataFrame({"vtd_geoid": vtds["vtd_geoid"].astype("string")})

    # TOTAL POP
    geo["total_pop"] = total_pop_by_vtd.to_numpy()

    # CVAP (ACS 5-year; from allocated BG CVAP)
    geo["cvap_total"] = cvap_by_vtd["cvap_total_blk"].to_numpy()
    geo["cvap_hisp"] = cvap_by_vtd["cvap_hisp_blk"].to_numpy()
    geo["cvap_nh_white"] = cvap_by_vtd["cvap_nh_white_blk"].to_numpy()
    geo["cvap_nh_black"] = cvap_by_vtd["cvap_nh_black_blk"].to_numpy()
    geo["cvap_nh_asian"] = cvap_by_vtd["cvap_nh_asian_blk"].to_numpy()
    geo["cvap_nh_native"] = cvap_by_vtd["cvap_nh_native_blk"].to_numpy()
    geo["cvap_nh_pi"] = cvap_by_vtd["cvap_nh_pi_blk"].to_numpy()

    known_cols = ["cvap_hisp", "cvap_nh_white", "cvap_nh_black", "cvap_nh_asian", "cvap_nh_native", "cvap_nh_pi"]
    geo["cvap_other"] = (geo["cvap_total"] - geo[known_cols].sum(axis=1)).clip(lower=0).astype("int64")

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
    if vtds_geo["cvap_total"].isna().any():
        n_missing = int(vtds_geo["cvap_total"].isna().sum())
        raise ValueError(f"vtds_geo missing cvap_total for {n_missing} rows after merge (unexpected).")

    vtds_geo.to_parquet(vtds_geo_path, index=False)
    print(f"[write] geospatial VTDs -> {vtds_geo_path.resolve()}")

    # -----------------------------
    # Elections: clean returns keyed to VTDs
    # -----------------------------
    returns = read_any(elections_path)
    returns = stdcols(returns)

    # elections.py expects: clean_vtd_election_returns(df, office_filter=..., prefer_key=...)
    wide = clean_vtd_election_returns(
        returns,
        office_filter=elections_office_filter,
        prefer_key="vtdkey",
    )

    # Build a numeric VTD key from the VTD shapefile to join
    vtd_key_num = pd.to_numeric(vtds["vtdkey"], errors="coerce").astype("Int64")
    vtd_key_map = (
        pd.DataFrame(
            {
                "vtdkey": vtd_key_num,
                "vtd_geoid": vtds["vtd_geoid"].astype("string"),
            }
        )
        .dropna(subset=["vtdkey"])
        .drop_duplicates(subset=["vtdkey"])
    )

    if "vtdkey" not in wide.columns:
        raise ValueError(
            "Election returns did not produce a 'vtdkey' column. "
            "Your elections file likely lacks 'vtdkeyvalue'. "
            "Either provide a file with vtdkeyvalue, or extend the pipeline to join on cntyvtd."
        )

    merged = wide.merge(vtd_key_map, on="vtdkey", how="inner")

    if merged.empty:
        raise ValueError(
            "After joining election returns to VTDs on vtdkey, no rows matched. "
            "Check that elections vtdkeyvalue matches the VTD shapefile's vtdkey."
        )

    # Standard output schema expected downstream (EI expects votes_total and votes_dem)
    returns_vtd = pd.DataFrame(
        {
            "election_id": election_id,
            "vtd_geoid": merged["vtd_geoid"].astype("string"),
            "votes_total": pd.to_numeric(merged["total_votes"], errors="coerce").fillna(0).astype("int64"),
            "votes_dem": pd.to_numeric(merged["dem_votes"], errors="coerce").fillna(0).astype("int64"),
        }
    )

    write_parquet(returns_vtd, outs["returns_vtd"])

    # -----------------------------
    # Elections: clean returns keyed to VTDs
    # -----------------------------
    returns = read_any(elections_path)
    returns = stdcols(returns)

    # elections.py expects: clean_vtd_election_returns(df, office_filter=..., prefer_key=...)
    wide = clean_vtd_election_returns(
        returns,
        office_filter=elections_office_filter,
        prefer_key="vtdkey",
    )

    # Build a numeric VTD key from the VTD shapefile to join
    vtd_key_num = pd.to_numeric(vtds["vtdkey"], errors="coerce").astype("Int64")
    vtd_key_map = (
        pd.DataFrame(
            {
                "vtdkey": vtd_key_num,
                "vtd_geoid": vtds["vtd_geoid"].astype("string"),
            }
        )
        .dropna(subset=["vtdkey"])
        .drop_duplicates(subset=["vtdkey"])
    )

    if "vtdkey" not in wide.columns:
        raise ValueError(
            "Election returns did not produce a 'vtdkey' column. "
            "Your elections file likely lacks 'vtdkeyvalue'. "
            "Either provide a file with vtdkeyvalue, or extend the pipeline to join on cntyvtd."
        )

    merged = wide.merge(vtd_key_map, on="vtdkey", how="inner")

    if merged.empty:
        raise ValueError(
            "After joining election returns to VTDs on vtdkey, no rows matched. "
            "Check that elections vtdkeyvalue matches the VTD shapefile's vtdkey."
        )

    # Standard output schema expected downstream (EI expects votes_total and votes_dem)
    returns_vtd = pd.DataFrame(
        {
            "election_id": election_id,
            "vtd_geoid": merged["vtd_geoid"].astype("string"),
            "votes_total": pd.to_numeric(merged["total_votes"], errors="coerce").fillna(0).astype("int64"),
            "votes_dem": pd.to_numeric(merged["dem_votes"], errors="coerce").fillna(0).astype("int64"),
        }
    )

    write_parquet(returns_vtd, outs["returns_vtd"])


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
        "--cvap-blockgr",
        type=Path,
        required=True,
        help="ACS CVAP Special Tabulation Block Group file (BlockGr.csv) for the ACS 5-year period you are using.",
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
        cvap_blockgr_path=args.cvap_blockgr,
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

