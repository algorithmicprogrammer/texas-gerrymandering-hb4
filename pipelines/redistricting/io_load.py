from __future__ import annotations

import duckdb


def load_geo_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.execute("DELETE FROM geo_vtd;")
    con.execute(
        f"""
        INSERT INTO geo_vtd (
            vtd_geoid,
            state_fips,
            county_fips,
            vap_total,
            vap_nh_white,
            vap_nh_black,
            vap_hisp,
            vap_nh_asian,
            vap_nh_native,
            vap_other
        )
        SELECT
            CAST(vtd_geoid AS VARCHAR) AS vtd_geoid,
            CAST(state_fips AS VARCHAR) AS state_fips,
            CASE
                WHEN length(CAST(vtd_geoid AS VARCHAR)) >= 5
                THEN substr(CAST(vtd_geoid AS VARCHAR), 3, 3)
                ELSE NULL
            END AS county_fips,
            CAST(vap_total AS BIGINT)      AS vap_total,
            CAST(vap_nh_white AS BIGINT)   AS vap_nh_white,
            CAST(vap_nh_black AS BIGINT)   AS vap_nh_black,
            CAST(vap_hisp AS BIGINT)       AS vap_hisp,
            CAST(vap_nh_asian AS BIGINT)   AS vap_nh_asian,
            CAST(vap_nh_native AS BIGINT)  AS vap_nh_native,
            CAST(vap_other AS BIGINT)      AS vap_other
        FROM read_parquet('{path}');
        """
    )


def load_election(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.execute("DELETE FROM election;")
    con.execute(f"INSERT INTO election SELECT * FROM read_parquet('{path}');")


def load_election_returns_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.execute("DELETE FROM election_returns_vtd;")
    con.execute(f"INSERT INTO election_returns_vtd SELECT * FROM read_parquet('{path}');")


def load_plan(con: duckdb.DuckDBPyConnection, path: str) -> None:
    """
    Load enacted plan metadata (often 1 row).
    """
    con.execute("DELETE FROM plan;")
    con.execute(f"INSERT INTO plan SELECT * FROM read_parquet('{path}');")


def load_ensemble_plans(con: duckdb.DuckDBPyConnection, path: str) -> None:
    """
    Append ensemble plan metadata rows into plan, skipping duplicates
    (plan.plan_id is a primary key).
    """
    # Use an anti-join to avoid inserting plan_ids that already exist.
    con.execute(
        f"""
        INSERT INTO plan
        SELECT ep.*
        FROM read_parquet('{path}') ep
        LEFT JOIN plan p
          ON p.plan_id = ep.plan_id
        WHERE p.plan_id IS NULL;
        """
    )


def load_plan_district_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.execute("DELETE FROM plan_district_vtd;")
    con.execute(f"INSERT INTO plan_district_vtd SELECT * FROM read_parquet('{path}');")
