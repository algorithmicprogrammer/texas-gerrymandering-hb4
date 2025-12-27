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
    con.execute("DELETE FROM district_returns;")
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
        CASE WHEN SUM(er.votes_total) > 0 THEN SUM(er.votes_dem)::DOUBLE / SUM(er.votes_total) ELSE NULL END AS dem_share
    FROM plan_district_vtd pd
    JOIN election_returns_vtd er ON er.vtd_geoid = pd.vtd_geoid
    GROUP BY pd.plan_id, pd.district_id, er.election_id;
    """)
