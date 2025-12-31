from __future__ import annotations
import duckdb

def build_district_demo_vap(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM district_demo_vap;")
    con.execute("""
    INSERT INTO district_demo_vap
    SELECT
        pd.plan_id,
        pd.district_id,
        SUM(g.vap_total) AS vap_total,
        SUM(g.vap_nh_white) AS vap_nh_white,
        SUM(g.vap_nh_black) AS vap_nh_black,
        SUM(g.vap_hisp) AS vap_hisp,
        SUM(g.vap_nh_asian) AS vap_nh_asian,
        SUM(g.vap_nh_native) AS vap_nh_native,
        SUM(g.vap_other) AS vap_other,
        CASE WHEN SUM(g.vap_total) > 0 THEN SUM(g.vap_nh_white)::DOUBLE / SUM(g.vap_total) ELSE NULL END AS share_white_vap,
        CASE WHEN SUM(g.vap_total) > 0 THEN SUM(g.vap_nh_black)::DOUBLE / SUM(g.vap_total) ELSE NULL END AS share_black_vap,
        CASE WHEN SUM(g.vap_total) > 0 THEN SUM(g.vap_hisp)::DOUBLE / SUM(g.vap_total) ELSE NULL END AS share_hisp_vap,
        CASE WHEN SUM(g.vap_total) > 0 THEN (1.0 - SUM(g.vap_nh_white)::DOUBLE / SUM(g.vap_total)) ELSE NULL END AS share_minority_vap
    FROM plan_district_vtd pd
    JOIN geo_vtd g ON g.vtd_geoid = pd.vtd_geoid
    GROUP BY pd.plan_id, pd.district_id;
    """)

def build_district_returns(con: duckdb.DuckDBPyConnection) -> None:
    """
    Aggregate VTD-level returns to districts.

    IMPORTANT:
    - Uses LEFT JOIN so districts are not dropped when some VTDs have missing returns.
    - Missing votes remain NULL at the VTD level; SUM ignores NULLs in DuckDB,
      so totals represent "sum over reported VTDs" rather than silently dropping entire districts.
    """
    con.execute("DELETE FROM district_returns;")

    # LEFT JOIN keeps all (plan_id, district_id, election_id) combos that exist in returns,
    # but does not drop districts with partial coverage.
    # We include election_id from returns; if you want districts present even when *no* VTDs
    # have returns for an election, you’d need to CROSS JOIN election table—usually unnecessary.
    con.execute("""
    INSERT INTO district_returns
    SELECT
        pd.plan_id,
        pd.district_id,
        er.election_id,
        SUM(er.votes_total) AS votes_total,
        SUM(er.votes_dem) AS votes_dem,
        SUM(er.votes_rep) AS votes_rep,
        SUM(er.votes_other) AS votes_other,
        CASE
            WHEN SUM(er.votes_total) > 0 THEN SUM(er.votes_dem)::DOUBLE / SUM(er.votes_total)
            ELSE NULL
        END AS dem_share
    FROM plan_district_vtd pd
    LEFT JOIN election_returns_vtd er
      ON er.vtd_geoid = pd.vtd_geoid
    WHERE er.election_id IS NOT NULL
    GROUP BY pd.plan_id, pd.district_id, er.election_id;
    """)

    # Optional: quick coverage diagnostics (prints nothing unless you query it)
    # Example:
    # con.execute("""
    #   SELECT plan_id, district_id, election_id,
    #          COUNT(*) AS n_vtd_assigned,
    #          SUM(CASE WHEN votes_total IS NOT NULL THEN 1 ELSE 0 END) AS n_vtd_with_returns
    #   FROM (
    #     SELECT pd.plan_id, pd.district_id, er.election_id, er.votes_total
    #     FROM plan_district_vtd pd
    #     LEFT JOIN election_returns_vtd er ON er.vtd_geoid = pd.vtd_geoid
    #   )
    #   WHERE election_id IS NOT NULL
    #   GROUP BY plan_id, district_id, election_id
    #   ORDER BY n_vtd_with_returns / NULLIF(n_vtd_assigned,0) ASC
    # """).df()
