from __future__ import annotations

from texas_gerrymandering_hb4.config import (
    PLANC2333_SHP_FILE,
    VTDS_SHP_FILE,
    GEN_ELECTION_CSV,
    CENSUS_GEO_SHP_FILE,
    ACS_CSV_FILE,
    DEM_PRIMARY_CSV,
    REP_PRIMARY_CSV,
    PROCESSED_DATA_DIR,
)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import maup  # type: ignore
except ImportError:  # pragma: no cover
    maup = None

try:
    from shapely.validation import make_valid  # type: ignore
    HAS_MAKE_VALID = True
except ImportError:  # pragma: no cover
    HAS_MAKE_VALID = False


@dataclass
class Config:
    cvap_csv: Path
    vtd_shp: Path
    enacted_cd_shp: Path
    census_blocks_shp: Path
    general_returns_csv: Path
    dem_primary_returns_csv: Path
    rep_primary_returns_csv: Path

    output_dir: Path
    output_basename: str = "tx_precincts_analysis_ready"

    target_crs: Optional[str] = None
    use_maup_for_block_to_precinct_assignment: bool = True
    write_geoparquet: bool = True
    write_geopackage: bool = False
    write_csv_without_geometry: bool = True

    debug: bool = True
    fail_on_duplicate_join_matches: bool = True
    min_expected_general_match_rate: float = 0.95
    min_expected_primary_match_rate: float = 0.95


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def debug_print(config: Config, *args) -> None:
    if config.debug:
        print(*args)


def normalize_cntyvtd(series: pd.Series) -> pd.Series:
    """
    Conservative CNTYVTD normalization.

    Keeps alphanumeric VTD suffixes intact, strips whitespace, uppercases, and
    removes a trailing '.0' artifact if one was introduced by CSV parsing.
    """
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
    return s


def build_cntyvtd_from_county_and_vtd(
    county_series: pd.Series,
    vtd_series: pd.Series,
) -> pd.Series:
    """
    Build join key as county FIPS + VTD, preserving VTD leading zeros/suffixes.

    Based on the slideshow, election cntyvtd is FIPS + VTD, so the shapefile key
    should be rebuilt from CNTY + VTD rather than using the shapefile CNTYVTD as-is.
    """
    county = pd.to_numeric(county_series, errors="coerce").astype("Int64").astype(str)
    county = county.replace("<NA>", pd.NA)

    vtd = vtd_series.astype(str).str.strip().str.upper()
    vtd = vtd.str.replace(r"\.0$", "", regex=True)
    vtd = vtd.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})

    out = county + vtd
    out = out.where(county.notna() & vtd.notna(), pd.NA)
    return out


def clean_office_name(value: str) -> Optional[str]:
    v = str(value).strip().lower()

    if v in {"president", "pres"}:
        return "PRES"

    if v in {"u.s. sen", "u.s. senate", "us senate", "senate", "u.s sen"}:
        return "SEN"

    if v in {
        "rr comm 1",
        "railroad commissioner",
        "rr comm",
        "railroad comm",
        "rr commissioner",
    }:
        return "RRC"

    return None


def normalize_candidate_token(name: str) -> str:
    token = str(name).strip()
    token = token.replace(" ", "")
    token = token.replace("-", "")
    token = token.replace(".", "")
    token = token.replace("'", "")
    return token


def safe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repair_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    config: Config,
    label: str,
) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if out.geometry.isna().any():
        n_missing = int(out.geometry.isna().sum())
        raise ValueError(f"{label} contains {n_missing} missing geometries.")

    invalid_mask = ~out.is_valid
    n_invalid = int(invalid_mask.sum())
    debug_print(config, f"[GEOMETRY DIAGNOSTIC] {label} invalid before repair:", n_invalid)

    if n_invalid == 0:
        return out

    if HAS_MAKE_VALID:
        out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].apply(make_valid)
    else:
        out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].buffer(0)

    invalid_mask = ~out.is_valid
    n_invalid_after_first = int(invalid_mask.sum())
    debug_print(config, f"[GEOMETRY DIAGNOSTIC] {label} invalid after first repair pass:", n_invalid_after_first)

    if n_invalid_after_first > 0:
        out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].buffer(0)

    invalid_mask = ~out.is_valid
    n_invalid_final = int(invalid_mask.sum())
    debug_print(config, f"[GEOMETRY DIAGNOSTIC] {label} invalid after final repair:", n_invalid_final)

    if n_invalid_final > 0:
        bad_ids = list(out.index[invalid_mask][:10])
        raise ValueError(
            f"{label} still has {n_invalid_final} invalid geometries after repair. "
            f"Sample row indices: {bad_ids}"
        )

    return out


# -----------------------------------------------------------------------------
# CVAP processing
# -----------------------------------------------------------------------------

def build_bg_cvap_table(cvap_csv: Path) -> pd.DataFrame:
    cvap = pd.read_csv(cvap_csv, dtype={"geoid": str, "lnnumber": str, "lntitle": str})
    required = {"geoid", "lnnumber", "cvap_est"}
    missing = required - set(cvap.columns)
    if missing:
        raise ValueError(f"CVAP CSV missing required columns: {sorted(missing)}")

    cvap = cvap[["geoid", "lnnumber", "cvap_est"]].copy()
    cvap["geoid"] = cvap["geoid"].astype(str)
    cvap["lnnumber"] = cvap["lnnumber"].astype(str).str.strip()
    cvap["cvap_est"] = pd.to_numeric(cvap["cvap_est"], errors="coerce").fillna(0)

    cvap["bg_geoid"] = cvap["geoid"].str.replace("1500000US", "", regex=False)

    wide = (
        cvap.pivot_table(
            index="bg_geoid",
            columns="lnnumber",
            values="cvap_est",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for needed in ["1", "5", "7", "13"]:
        if needed not in wide.columns:
            wide[needed] = 0

    out = pd.DataFrame({
        "bg_geoid": wide["bg_geoid"],
        "CVAP": wide["1"].astype(float),
        "HCVAP": wide["13"].astype(float),
        "BCVAP": wide["5"].astype(float),
        "WCVAP": wide["7"].astype(float),
    })
    out["OCVAP"] = out["CVAP"] - out["HCVAP"] - out["BCVAP"] - out["WCVAP"]
    out["OCVAP"] = out["OCVAP"].clip(lower=0)

    return out


def allocate_bg_cvap_to_blocks(blocks_gdf: gpd.GeoDataFrame, bg_cvap: pd.DataFrame) -> gpd.GeoDataFrame:
    blocks = blocks_gdf.copy()
    if "GEOID20" not in blocks.columns or "POP20" not in blocks.columns:
        raise ValueError("Block shapefile must contain GEOID20 and POP20 columns.")

    blocks["GEOID20"] = blocks["GEOID20"].astype(str)
    blocks["bg_geoid"] = blocks["GEOID20"].str[:12]
    blocks["POP20"] = pd.to_numeric(blocks["POP20"], errors="coerce").fillna(0)

    blocks = blocks.merge(bg_cvap, on="bg_geoid", how="left", validate="m:1")
    for col in ["CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]:
        blocks[col] = pd.to_numeric(blocks[col], errors="coerce").fillna(0)

    bg_pop = blocks.groupby("bg_geoid", dropna=False)["POP20"].transform("sum")
    n_blocks = blocks.groupby("bg_geoid", dropna=False)["GEOID20"].transform("count")

    weight = np.where(bg_pop > 0, blocks["POP20"] / bg_pop, 1 / n_blocks)
    blocks["alloc_weight"] = weight

    for col in ["CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]:
        blocks[col] = blocks[col] * blocks["alloc_weight"]

    return blocks


# -----------------------------------------------------------------------------
# Geographic joins
# -----------------------------------------------------------------------------

def read_geodata(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No records found in {path}")
    return gdf


def standardize_vtds(
    vtds: gpd.GeoDataFrame,
    config: Config,
    target_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    gdf = vtds.copy()

    required_cols = {"CNTY", "VTD", "geometry"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(f"VTD shapefile missing required columns for rebuilt join key: {sorted(missing)}")

    raw_shapefile_cntyvtd = gdf["CNTYVTD"].astype(str) if "CNTYVTD" in gdf.columns else pd.Series(dtype=str)

    rebuilt_ids = build_cntyvtd_from_county_and_vtd(gdf["CNTY"], gdf["VTD"])
    rebuilt_ids = normalize_cntyvtd(rebuilt_ids)

    debug_print(config, "[VTD DIAGNOSTIC] rows:", len(gdf))
    if "CNTYVTD" in gdf.columns:
        debug_print(config, "[VTD DIAGNOSTIC] raw shapefile CNTYVTD unique:", raw_shapefile_cntyvtd.nunique(dropna=False))
    debug_print(config, "[VTD DIAGNOSTIC] rebuilt CNTY+VTD unique:", rebuilt_ids.nunique(dropna=False))

    gdf["CNTYVTD"] = rebuilt_ids

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    out = gdf[["CNTYVTD", "geometry"]].copy()

    dupes = out[out["CNTYVTD"].duplicated(keep=False)].sort_values("CNTYVTD")
    debug_print(config, "[VTD DIAGNOSTIC] duplicate CNTYVTD row count after rebuild:", len(dupes))
    if not dupes.empty:
        debug_print(config, "[VTD DIAGNOSTIC] sample duplicate CNTYVTD values after rebuild:")
        debug_print(config, dupes[["CNTYVTD"]].head(20))

    missing_key_rows = int(out["CNTYVTD"].isna().sum())
    debug_print(config, "[VTD DIAGNOSTIC] missing rebuilt CNTYVTD rows:", missing_key_rows)
    if missing_key_rows > 0:
        raise ValueError(f"Shapefile has {missing_key_rows} rows with missing rebuilt CNTYVTD values.")

    return out


def attach_enacted_cd(
    vtds: gpd.GeoDataFrame,
    cds: gpd.GeoDataFrame,
    config: Config,
    target_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    precincts = vtds.copy()
    districts = cds.copy()

    district_col = None
    for cand in ["District", "DISTRICT", "district", "CD", "cd"]:
        if cand in districts.columns:
            district_col = cand
            break
    if district_col is None:
        raise ValueError("Congressional district shapefile must contain a district number column.")

    if target_crs is not None:
        precincts = precincts.to_crs(target_crs)
        districts = districts.to_crs(target_crs)
    elif precincts.crs is not None and districts.crs != precincts.crs:
        districts = districts.to_crs(precincts.crs)

    debug_print(config, "[CD JOIN DIAGNOSTIC] precinct rows:", len(precincts))
    debug_print(config, "[CD JOIN DIAGNOSTIC] unique precinct CNTYVTD:", precincts["CNTYVTD"].nunique(dropna=False))

    precinct_dupes = precincts[precincts["CNTYVTD"].duplicated(keep=False)].sort_values("CNTYVTD")
    debug_print(config, "[CD JOIN DIAGNOSTIC] duplicate precinct CNTYVTD row count:", len(precinct_dupes))
    if not precinct_dupes.empty:
        debug_print(config, "[CD JOIN DIAGNOSTIC] sample duplicate precinct CNTYVTD values:")
        debug_print(config, precinct_dupes[["CNTYVTD"]].head(20))

    pts = precincts[["CNTYVTD", "geometry"]].copy()
    pts["geometry"] = pts.representative_point()

    joined = gpd.sjoin(
        pts,
        districts[[district_col, "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.rename(columns={district_col: "CD"})[["CNTYVTD", "CD"]]
    joined["CD"] = pd.to_numeric(joined["CD"], errors="coerce").astype("Int64")

    debug_print(config, "[CD JOIN DIAGNOSTIC] joined rows:", len(joined))
    debug_print(config, "[CD JOIN DIAGNOSTIC] unique joined CNTYVTD:", joined["CNTYVTD"].nunique(dropna=False))

    joined_dupes = joined[joined["CNTYVTD"].duplicated(keep=False)].sort_values(["CNTYVTD", "CD"])
    debug_print(config, "[CD JOIN DIAGNOSTIC] duplicate joined CNTYVTD row count:", len(joined_dupes))
    if not joined_dupes.empty:
        debug_print(config, "[CD JOIN DIAGNOSTIC] sample duplicate joined CNTYVTD/CD pairs:")
        debug_print(config, joined_dupes[["CNTYVTD", "CD"]].head(30))

        cd_counts = (
            joined.groupby("CNTYVTD")["CD"]
            .nunique(dropna=True)
            .reset_index(name="n_cd")
        )
        multi_cd = cd_counts[cd_counts["n_cd"] > 1]
        debug_print(config, "[CD JOIN DIAGNOSTIC] precincts matching multiple distinct CDs:", len(multi_cd))
        if not multi_cd.empty:
            debug_print(config, "[CD JOIN DIAGNOSTIC] sample precincts matching multiple CDs:")
            debug_print(config, multi_cd.head(20))

        if config.fail_on_duplicate_join_matches and not multi_cd.empty:
            raise ValueError(
                "Spatial join produced precincts matching multiple congressional districts. "
                f"Examples: {multi_cd['CNTYVTD'].head(10).tolist()}"
            )

    joined = joined.drop_duplicates(subset=["CNTYVTD", "CD"])
    joined = joined.drop_duplicates(subset=["CNTYVTD"])

    debug_print(config, "[CD JOIN DIAGNOSTIC] joined rows after dedup:", len(joined))
    debug_print(config, "[CD JOIN DIAGNOSTIC] unique joined CNTYVTD after dedup:", joined["CNTYVTD"].nunique(dropna=False))

    out = precincts.merge(joined, on="CNTYVTD", how="left")

    debug_print(config, "[CD JOIN DIAGNOSTIC] merged precinct rows:", len(out))
    debug_print(config, "[CD JOIN DIAGNOSTIC] merged unique CNTYVTD:", out["CNTYVTD"].nunique(dropna=False))

    merged_dupes = out[out["CNTYVTD"].duplicated(keep=False)].sort_values("CNTYVTD")
    debug_print(config, "[CD JOIN DIAGNOSTIC] duplicate merged CNTYVTD row count:", len(merged_dupes))
    if not merged_dupes.empty:
        debug_print(config, "[CD JOIN DIAGNOSTIC] sample duplicate merged rows:")
        debug_print(config, merged_dupes[["CNTYVTD", "CD"]].head(30))

    return out


def assign_blocks_to_precincts(
    blocks: gpd.GeoDataFrame,
    precincts: gpd.GeoDataFrame,
    use_maup: bool = True,
) -> gpd.GeoDataFrame:
    blk = blocks.copy()
    pct = precincts.copy()

    if blk.crs != pct.crs:
        blk = blk.to_crs(pct.crs)

    if use_maup and maup is not None:
        assignment = maup.assign(blk, pct)
        blk["CNTYVTD"] = pct.loc[assignment, "CNTYVTD"].values
    else:
        pts = blk[["GEOID20", "geometry"]].copy()
        pts["geometry"] = pts.representative_point()
        joined = gpd.sjoin(
            pts,
            pct[["CNTYVTD", "geometry"]],
            how="left",
            predicate="within",
        )
        blk = blk.merge(
            joined[["GEOID20", "CNTYVTD"]],
            on="GEOID20",
            how="left",
            suffixes=("", "_joined"),
        )

    return blk


def aggregate_block_demographics_to_precincts(blocks_assigned: gpd.GeoDataFrame) -> pd.DataFrame:
    needed = ["CNTYVTD", "POP20", "CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]
    missing = [c for c in needed if c not in blocks_assigned.columns]
    if missing:
        raise ValueError(f"Missing required columns on assigned blocks: {missing}")

    demo = (
        blocks_assigned.groupby("CNTYVTD", dropna=False)[["POP20", "CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]]
        .sum()
        .reset_index()
        .rename(columns={"POP20": "TOTALPOP"})
    )

    return demo


# -----------------------------------------------------------------------------
# Election processing
# -----------------------------------------------------------------------------

def read_election_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cntyvtd", "Office", "Name", "Party", "Votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Election returns file {path} missing columns: {sorted(missing)}")

    out = df[["cntyvtd", "Office", "Name", "Party", "Votes"]].copy()
    out["cntyvtd"] = normalize_cntyvtd(out["cntyvtd"])
    out["Office"] = out["Office"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Party"] = out["Party"].astype(str).str.strip().str.upper()
    out["Votes"] = pd.to_numeric(out["Votes"], errors="coerce").fillna(0).astype(int)

    missing_keys = int(out["cntyvtd"].isna().sum())
    if missing_keys > 0:
        raise ValueError(f"Election file {path} has {missing_keys} missing cntyvtd values.")

    return out


def build_wide_votes_general(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["office_code"] = data["Office"].map(clean_office_name)

    dropped = data[data["office_code"].isna()]["Office"].drop_duplicates().tolist()
    if dropped:
        print("[ELECTION DIAGNOSTIC] Dropping unsupported general-election offices:")
        print(dropped)

    data = data[data["office_code"].notna()].copy()

    data["candidate_token"] = data["Name"].map(normalize_candidate_token)
    data["vote_col"] = data["candidate_token"] + data["Party"] + "_24G"

    candidate_wide = (
        data.pivot_table(
            index="cntyvtd",
            columns="vote_col",
            values="Votes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    totals = data.groupby(["cntyvtd", "office_code"], as_index=False)["Votes"].sum()
    totals["total_col"] = "TOTVOTE_" + totals["office_code"] + "_24G"

    totals_wide = (
        totals.pivot(index="cntyvtd", columns="total_col", values="Votes")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    return candidate_wide, totals_wide


def build_wide_votes_primary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["office_code"] = data["Office"].map(clean_office_name)

    dropped = data[data["office_code"].isna()]["Office"].drop_duplicates().tolist()
    if dropped:
        print("[ELECTION DIAGNOSTIC] Dropping unsupported primary-election offices:")
        print(dropped)

    data = data[data["office_code"].notna()].copy()

    data["candidate_token"] = data["Name"].map(normalize_candidate_token)
    data["vote_col"] = data["candidate_token"] + data["Party"] + "_24P"

    candidate_wide = (
        data.pivot_table(
            index="cntyvtd",
            columns="vote_col",
            values="Votes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    totals = data.groupby(["cntyvtd", "office_code", "Party"], as_index=False)["Votes"].sum()
    totals["total_col"] = "TOTVOTE_" + totals["office_code"] + totals["Party"] + "_24P"

    totals_wide = (
        totals.pivot(index="cntyvtd", columns="total_col", values="Votes")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    return candidate_wide, totals_wide


def build_primary_combined(dem_df: pd.DataFrame, rep_df: pd.DataFrame) -> pd.DataFrame:
    dem = dem_df.copy()
    rep = rep_df.copy()
    dem["Party"] = "D"
    rep["Party"] = "R"
    return pd.concat([dem, rep], ignore_index=True)


def check_election_join_coverage(
    precincts: gpd.GeoDataFrame,
    election_df: pd.DataFrame,
    key_left: str,
    key_right: str,
    label: str,
    min_expected_match_rate: float,
) -> None:
    left_ids = set(precincts[key_left].dropna().astype(str))
    right_ids = set(election_df[key_right].dropna().astype(str))
    matched = left_ids & right_ids
    match_rate = len(matched) / max(len(left_ids), 1)

    print(f"[ELECTION MATCH DIAGNOSTIC] {label} precinct keys: {len(left_ids)}")
    print(f"[ELECTION MATCH DIAGNOSTIC] {label} election keys: {len(right_ids)}")
    print(f"[ELECTION MATCH DIAGNOSTIC] {label} matched keys: {len(matched)}")
    print(f"[ELECTION MATCH DIAGNOSTIC] {label} match rate: {match_rate:.4f}")

    if match_rate < min_expected_match_rate:
        only_left = list(sorted(left_ids - right_ids))[:10]
        only_right = list(sorted(right_ids - left_ids))[:10]
        raise ValueError(
            f"{label} election join coverage is too low ({match_rate:.4f}). "
            f"Sample keys only in precincts: {only_left}. "
            f"Sample keys only in election data: {only_right}."
        )


# -----------------------------------------------------------------------------
# Validation / final assembly
# -----------------------------------------------------------------------------

def validate_final_dataset(final_gdf: gpd.GeoDataFrame) -> None:
    required = [
        "CNTYVTD",
        "geometry",
        "CD",
        "TOTALPOP",
        "CVAP",
        "HCVAP",
        "BCVAP",
        "WCVAP",
        "OCVAP",
    ]
    missing = [c for c in required if c not in final_gdf.columns]
    if missing:
        raise ValueError(f"Final dataset missing required columns: {missing}")

    if final_gdf["CNTYVTD"].isna().any():
        raise ValueError("Missing CNTYVTD values in final dataset.")
    if final_gdf["CNTYVTD"].duplicated().any():
        dupes = final_gdf.loc[final_gdf["CNTYVTD"].duplicated(), "CNTYVTD"].tolist()[:10]
        raise ValueError(f"Duplicate CNTYVTD values in final dataset, e.g. {dupes}")

    if final_gdf.geometry.isna().any():
        raise ValueError("Missing geometry values in final dataset.")

    invalid_count = int((~final_gdf.is_valid).sum())
    if invalid_count > 0:
        raise ValueError(f"Final dataset contains {invalid_count} invalid geometries.")

    for col in ["TOTALPOP", "CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]:
        if (final_gdf[col] < 0).any():
            raise ValueError(f"Negative values detected in {col}")

    cvap_sum = final_gdf[["HCVAP", "BCVAP", "WCVAP", "OCVAP"]].sum(axis=1)
    if not np.allclose(final_gdf["CVAP"].fillna(0), cvap_sum.fillna(0), atol=1e-6):
        max_diff = (final_gdf["CVAP"] - cvap_sum).abs().max()
        raise ValueError(f"CVAP != HCVAP+BCVAP+WCVAP+OCVAP for some rows; max diff = {max_diff}")


def write_outputs(final_gdf: gpd.GeoDataFrame, config: Config) -> None:
    safe_make_dir(config.output_dir)
    stem = config.output_basename

    if config.write_geoparquet:
        out = config.output_dir / f"{stem}.parquet"
        final_gdf.to_parquet(out)

    if config.write_geopackage:
        out = config.output_dir / f"{stem}.gpkg"
        final_gdf.to_file(out, layer=stem, driver="GPKG")

    if config.write_csv_without_geometry:
        out = config.output_dir / f"{stem}.csv"
        final_gdf.drop(columns="geometry").to_csv(out, index=False)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def build_analysis_ready_dataset(config: Config) -> gpd.GeoDataFrame:
    vtds_raw = read_geodata(config.vtd_shp)
    cds_raw = read_geodata(config.enacted_cd_shp)
    blocks_raw = read_geodata(config.census_blocks_shp)

    vtds_raw = repair_invalid_geometries(vtds_raw, config, label="VTD shapefile")
    cds_raw = repair_invalid_geometries(cds_raw, config, label="Congressional district shapefile")
    blocks_raw = repair_invalid_geometries(blocks_raw, config, label="Census block shapefile")

    target_crs = config.target_crs or vtds_raw.crs
    precincts = standardize_vtds(vtds_raw, config=config, target_crs=target_crs)
    precincts = attach_enacted_cd(precincts, cds_raw, config=config, target_crs=target_crs)

    blocks = blocks_raw.copy()
    if target_crs is not None:
        blocks = blocks.to_crs(target_crs)

    bg_cvap = build_bg_cvap_table(config.cvap_csv)
    blocks = allocate_bg_cvap_to_blocks(blocks, bg_cvap)

    blocks = assign_blocks_to_precincts(
        blocks,
        precincts,
        use_maup=config.use_maup_for_block_to_precinct_assignment,
    )
    demo = aggregate_block_demographics_to_precincts(blocks)

    precincts = precincts.merge(demo, on="CNTYVTD", how="left", validate="1:1")
    for col in ["TOTALPOP", "CVAP", "HCVAP", "BCVAP", "WCVAP", "OCVAP"]:
        precincts[col] = pd.to_numeric(precincts[col], errors="coerce").fillna(0).astype(float)

    general = read_election_returns(config.general_returns_csv)
    dem_primary = read_election_returns(config.dem_primary_returns_csv)
    rep_primary = read_election_returns(config.rep_primary_returns_csv)
    primary = build_primary_combined(dem_primary, rep_primary)

    check_election_join_coverage(
        precincts=precincts,
        election_df=general,
        key_left="CNTYVTD",
        key_right="cntyvtd",
        label="GENERAL",
        min_expected_match_rate=config.min_expected_general_match_rate,
    )
    check_election_join_coverage(
        precincts=precincts,
        election_df=primary,
        key_left="CNTYVTD",
        key_right="cntyvtd",
        label="PRIMARY",
        min_expected_match_rate=config.min_expected_primary_match_rate,
    )

    general_votes, general_totals = build_wide_votes_general(general)
    primary_votes, primary_totals = build_wide_votes_primary(primary)

    final_gdf = precincts.merge(general_votes, left_on="CNTYVTD", right_on="cntyvtd", how="left")
    if "cntyvtd" in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=["cntyvtd"])

    final_gdf = final_gdf.merge(general_totals, left_on="CNTYVTD", right_on="cntyvtd", how="left")
    if "cntyvtd" in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=["cntyvtd"])

    final_gdf = final_gdf.merge(primary_votes, left_on="CNTYVTD", right_on="cntyvtd", how="left")
    if "cntyvtd" in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=["cntyvtd"])

    final_gdf = final_gdf.merge(primary_totals, left_on="CNTYVTD", right_on="cntyvtd", how="left")
    if "cntyvtd" in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=["cntyvtd"])

    non_geo_cols = [c for c in final_gdf.columns if c != "geometry"]
    vote_like = [
        c for c in non_geo_cols
        if c.endswith("_24G") or c.endswith("_24P") or c.startswith("TOTVOTE_")
    ]
    for col in vote_like:
        final_gdf[col] = pd.to_numeric(final_gdf[col], errors="coerce").fillna(0).astype(int)

    preferred_prefix = [
        "CNTYVTD",
        "geometry",
        "CD",
        "TOTALPOP",
        "CVAP",
        "HCVAP",
        "WCVAP",
        "BCVAP",
        "OCVAP",
    ]
    ordered = [c for c in preferred_prefix if c in final_gdf.columns] + [
        c for c in final_gdf.columns if c not in preferred_prefix
    ]
    final_gdf = final_gdf[ordered].copy()

    final_gdf = repair_invalid_geometries(final_gdf, config, label="Final analysis-ready dataset")
    validate_final_dataset(final_gdf)

    return final_gdf


def main() -> None:
    config = Config(
        cvap_csv=Path(ACS_CSV_FILE),
        vtd_shp=Path(VTDS_SHP_FILE),
        enacted_cd_shp=Path(PLANC2333_SHP_FILE),
        census_blocks_shp=Path(CENSUS_GEO_SHP_FILE),
        general_returns_csv=Path(GEN_ELECTION_CSV),
        dem_primary_returns_csv=Path(DEM_PRIMARY_CSV),
        rep_primary_returns_csv=Path(REP_PRIMARY_CSV),
        output_dir=Path(PROCESSED_DATA_DIR),
        output_basename="tx_vtds_analysis_ready",
        target_crs=None,
        use_maup_for_block_to_precinct_assignment=True,
        write_geoparquet=True,
        write_geopackage=False,
        write_csv_without_geometry=True,
        debug=True,
        fail_on_duplicate_join_matches=True,
        min_expected_general_match_rate=0.95,
        min_expected_primary_match_rate=0.95,
    )

    final_gdf = build_analysis_ready_dataset(config)
    write_outputs(final_gdf, config)

    print("Built final dataset:")
    print(final_gdf.head())
    print(f"Rows: {len(final_gdf):,}")
    print(f"Columns: {len(final_gdf.columns):,}")
    print("Invalid geometries in final dataset:", int((~final_gdf.is_valid).sum()))
    print("Output directory:", config.output_dir)


if __name__ == "__main__":
    main()