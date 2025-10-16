"""Tests for the ETL pipeline utilities and end-to-end flow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import pytest

from etl_pipeline import (
    stdcols,
    coerce_types_light,
    clean_vtd_election_returns,
    run_etl,
    build_final,
    dataset_key,
    build_cntyvtd_from_parts,
    build_cntyvtd_from_fips_vtd,
)


def test_stdcols_and_coerce_types_light_handles_common_cases() -> None:
    df = pd.DataFrame(
        {
            " Column Name ": [" value ", "", "NA"],
            "Event Date": ["2020-01-05", "2020-01-06", "bad"],
            "numeric ": ["1", " 2 ", "x"],
        }
    )

    processed = coerce_types_light(stdcols(df))

    assert list(processed.columns) == ["column_name", "event_date", "numeric"]
    # Leading/trailing whitespace and NA markers are normalized to pd.NA
    assert pd.isna(processed.loc[1, "column_name"])
    assert processed.loc[2, "column_name"] == "NA"
    # Datetime inference kicks in for *_date columns
    assert np.issubdtype(processed["event_date"].dtype, np.datetime64)
    # Numeric coercion works for mostly numeric columns
    numeric = processed["numeric"].tolist()
    assert numeric[0] == 1.0
    assert numeric[1] == 2.0
    assert pd.isna(numeric[2])


def test_clean_vtd_election_returns_tall_to_wide_conversion() -> None:
    raw = pd.DataFrame(
        {
            "countyfips": ["48001", "48001", "48003", "48003"],
            "Precinct": ["1", "1", "2", "2"],
            "office": ["Mock Senate"] * 4,
            "party": ["D", "R", "D", "R"],
            "votes": [100, 80, 90, 70],
        }
    )

    raw["cntyvtd"] = ["00100001", "00100001", "00300002", "00300002"]

    wide = clean_vtd_election_returns(raw, target_office="Mock Senate")

    assert set(["cntyvtd", "dem_votes", "rep_votes", "total_votes"]) <= set(wide.columns)

    first_key = "00100001"  # county=001, precinct=1 (zfilled)
    second_key = "00300002"

    first = wide.loc[wide["cntyvtd"] == first_key].iloc[0]
    second = wide.loc[wide["cntyvtd"] == second_key].iloc[0]

    assert first["dem_votes"] == 100 and first["rep_votes"] == 80
    assert second["dem_votes"] == 90 and second["rep_votes"] == 70
    assert first["total_votes"] == 180
    assert pytest.approx(first["two_party_dem_share"], rel=1e-6) == 100 / (100 + 80)


def test_cntyvtd_builders_require_both_parts() -> None:
    cnty = pd.Series(["1", None, "003", None])
    vtd = pd.Series(["5", "12", None, "7A"])

    built = build_cntyvtd_from_parts(cnty, vtd)
    expected = pd.Series(["00100005", pd.NA, pd.NA, pd.NA], dtype="string")
    pd.testing.assert_series_equal(built.astype("string"), expected)

    fips = pd.Series(["48001", "", "48005", None])
    built_fips = build_cntyvtd_from_fips_vtd(fips, vtd)
    pd.testing.assert_series_equal(built_fips.astype("string"), expected)


def _create_mock_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_tab = tmp_path / "processed_tabular"
    processed_geo = tmp_path / "processed_geo"
    sqlite_path = tmp_path / "warehouse.sqlite"

    num_districts = 38
    counties = ["001", "003", "005", "007"]

    district_records = []
    vtd_records = []
    block_records = []
    pl94_records = []
    election_records = []

    for idx in range(num_districts):
        x0 = idx * 1.5
        geom = box(x0, 0, x0 + 1, 1)
        district_records.append({"district": idx + 1, "geometry": geom})

        county = counties[idx % len(counties)]
        vtd_code = f"{idx + 1:05d}"
        cntyvtd = county + vtd_code

        vtd_records.append({"cnty": county, "vtd": vtd_code, "geometry": geom})

        geoid_suffix = f"{idx + 1:010d}"
        geoid = f"48{county}{geoid_suffix}"
        block_records.append(
            {
                "statefp20": "48",
                "countyfp20": county,
                "tractce20": geoid_suffix[:6],
                "blockce20": geoid_suffix[6:],
                "geoid20": geoid,
                "name20": f"Block {idx + 1}",
                "aland20": 100,
                "awater20": 0,
                "intptlat20": 0.0,
                "intptlon20": 0.0,
                "geometry": geom,
            }
        )

        vap_total = 1000 + idx
        pl94_records.append(
            {
                "geoid20": geoid,
                "vap_total": vap_total,
                "nh_white_vap": int(vap_total * 0.6),
                "nh_black_vap": int(vap_total * 0.2),
                "nh_asian_vap": int(vap_total * 0.1),
                "hispanic_vap": vap_total - int(vap_total * 0.6) - int(vap_total * 0.2) - int(vap_total * 0.1),
            }
        )

        dem_votes = 120
        rep_votes = 80
        election_records.extend(
            [
                {
                    "countyfips": f"48{county}",
                    "Precinct": str(idx + 1),
                    "office": "Mock Senate",
                    "party": "D",
                    "votes": dem_votes,
                    "cntyvtd": cntyvtd,
                },
                {
                    "countyfips": f"48{county}",
                    "Precinct": str(idx + 1),
                    "office": "Mock Senate",
                    "party": "R",
                    "votes": rep_votes,
                    "cntyvtd": cntyvtd,
                },
            ]
        )

    districts_gdf = gpd.GeoDataFrame(district_records, geometry="geometry", crs="EPSG:4326")
    vtds_gdf = gpd.GeoDataFrame(vtd_records, geometry="geometry", crs="EPSG:4326")
    blocks_gdf = gpd.GeoDataFrame(block_records, geometry="geometry", crs="EPSG:4326")
    pl94_df = pd.DataFrame(pl94_records)
    elections_df = pd.DataFrame(election_records)

    districts_path = raw_dir / "districts.parquet"
    census_path = raw_dir / "census_blocks.parquet"
    vtds_path = raw_dir / "vtds.parquet"
    pl94_path = raw_dir / "pl94.parquet"
    elections_path = raw_dir / "elections.csv"

    districts_gdf.to_parquet(districts_path)
    blocks_gdf.to_parquet(census_path)
    vtds_gdf.to_parquet(vtds_path)
    pl94_df.to_parquet(pl94_path)
    elections_df.to_csv(elections_path, index=False)

    return [districts_path, census_path, vtds_path, pl94_path, elections_path], processed_tab, processed_geo, sqlite_path


def test_run_etl_and_build_final_end_to_end(tmp_path: Path) -> None:
    inputs, processed_tab, processed_geo, sqlite_path = _create_mock_inputs(tmp_path)

    run_etl(inputs, processed_tab, processed_geo, sqlite_path, elections_office="Mock Senate")

    # Ensure Stage 1 produced expected artifacts
    geo_indices = {0, 1, 2}
    for idx, path in enumerate(inputs[:-1]):
        key = dataset_key(path)
        if idx in geo_indices:
            assert (processed_geo / f"{key}.parquet").exists()
        else:
            assert (processed_tab / f"{key}.parquet").exists()

    elect_key = dataset_key(inputs[-1])
    assert (processed_tab / f"{elect_key}.parquet").exists()
    assert (processed_tab / f"{elect_key}_all.parquet").exists()

    dk, ck, vk, pk, ek = [dataset_key(p) for p in inputs]
    final = build_final(processed_tab, processed_geo, dk, ck, vk, pk, ek)

    assert len(final) == 38
    assert set(["district_id", "polsby_popper", "pct_white", "pct_black", "dem_share", "rep_share"]).issubset(final.columns)

    # Racial composition is inherited from the PL-94 mock data (60% white, 20% black, 10% asian, 10% hispanic)
    assert pytest.approx(final["pct_white"].iloc[0], rel=1e-6) == 0.6
    assert pytest.approx(final["pct_black"].iloc[0], rel=1e-6) == 0.2
    assert pytest.approx(final["pct_asian"].iloc[0] + final["pct_hispanic"].iloc[0], rel=1e-6) == 0.2

    # Votes were split 120 Dem / 80 Rep for every district → 60% / 40%
    assert all(pytest.approx(v, rel=1e-6) == 0.6 for v in final["dem_share"].tolist())
    assert all(pytest.approx(v, rel=1e-6) == 0.4 for v in final["rep_share"].tolist())
