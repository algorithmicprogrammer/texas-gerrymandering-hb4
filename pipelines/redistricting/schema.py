from __future__ import annotations

import pandas as pd
import duckdb
from .config import OpportunityDef


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS geo_vtd (
        vtd_geoid TEXT PRIMARY KEY,
        state_fips TEXT,
        county_fips TEXT,
        county_name TEXT,
        area_km2 DOUBLE,
        total_pop BIGINT,
        vap_total BIGINT,
        vap_nh_white BIGINT,
        vap_nh_black BIGINT,
        vap_hisp BIGINT,
        vap_nh_asian BIGINT,
        vap_nh_native BIGINT,
        vap_other BIGINT,
        cvap_total BIGINT,
        cvap_nh_white BIGINT,
        cvap_nh_black BIGINT,
        cvap_hisp BIGINT,
        cvap_nh_asian BIGINT,
        cvap_nh_native BIGINT,
        cvap_other BIGINT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS election (
        election_id TEXT PRIMARY KEY,
        year INTEGER,
        office TEXT,
        stage TEXT,
        notes TEXT,
        weight DOUBLE DEFAULT 1.0
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS election_returns_vtd (
        election_id TEXT,
        vtd_geoid TEXT,
        votes_total BIGINT,
        votes_dem BIGINT,
        votes_rep BIGINT,
        votes_other BIGINT,
        dem_share DOUBLE,
        PRIMARY KEY (election_id, vtd_geoid)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS election_candidate (
        election_id TEXT,
        candidate_id TEXT,
        candidate_name TEXT,
        party TEXT,
        candidate_race TEXT,
        is_general BOOLEAN,
        PRIMARY KEY (election_id, candidate_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS election_candidate_returns_vtd (
        election_id TEXT,
        vtd_geoid TEXT,
        candidate_id TEXT,
        votes BIGINT,
        PRIMARY KEY (election_id, vtd_geoid, candidate_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS plan (
        plan_id TEXT PRIMARY KEY,
        plan_type TEXT,
        cycle TEXT,
        chamber TEXT,
        ensemble_id TEXT,
        generator TEXT,
        seed BIGINT,
        constraints_json TEXT,
        created_at TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS plan_district_vtd (
        plan_id TEXT,
        vtd_geoid TEXT,
        district_id TEXT,
        PRIMARY KEY (plan_id, vtd_geoid)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS district_demo_vap (
        plan_id TEXT,
        district_id TEXT,
        vap_total BIGINT,
        vap_nh_white BIGINT,
        vap_nh_black BIGINT,
        vap_hisp BIGINT,
        vap_nh_asian BIGINT,
        vap_nh_native BIGINT,
        vap_other BIGINT,
        share_white_vap DOUBLE,
        share_black_vap DOUBLE,
        share_hisp_vap DOUBLE,
        share_asian_vap DOUBLE,
        share_native_vap DOUBLE,
        share_minority_vap DOUBLE,
        PRIMARY KEY (plan_id, district_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS district_demo_cvap (
        plan_id TEXT,
        district_id TEXT,
        cvap_total BIGINT,
        cvap_nh_white BIGINT,
        cvap_nh_black BIGINT,
        cvap_hisp BIGINT,
        cvap_nh_asian BIGINT,
        cvap_nh_native BIGINT,
        cvap_other BIGINT,
        share_white_cvap DOUBLE,
        share_black_cvap DOUBLE,
        share_hisp_cvap DOUBLE,
        share_asian_cvap DOUBLE,
        share_native_cvap DOUBLE,
        share_minority_cvap DOUBLE,
        PRIMARY KEY (plan_id, district_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS district_returns (
        plan_id TEXT,
        district_id TEXT,
        election_id TEXT,
        votes_total BIGINT,
        votes_dem BIGINT,
        votes_rep BIGINT,
        votes_other BIGINT,
        dem_share DOUBLE,
        PRIMARY KEY (plan_id, district_id, election_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS opp_def (
        opp_def_id TEXT PRIMARY KEY,
        group_label TEXT,
        group_share_col TEXT,
        threshold DOUBLE,
        notes TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS district_opportunity (
        plan_id TEXT,
        district_id TEXT,
        opp_def_id TEXT,
        opp_value DOUBLE,
        is_opportunity BOOLEAN,
        PRIMARY KEY (plan_id, district_id, opp_def_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS plan_metrics (
        plan_id TEXT,
        opp_def_id TEXT,
        n_opportunity_districts BIGINT,
        mean_group_share DOUBLE,
        PRIMARY KEY (plan_id, opp_def_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ensemble_distribution (
        ensemble_id TEXT,
        opp_def_id TEXT,
        metric_name TEXT,
        n_plans BIGINT,
        mean DOUBLE,
        sd DOUBLE,
        p01 DOUBLE, p05 DOUBLE, p10 DOUBLE, p25 DOUBLE, p50 DOUBLE,
        p75 DOUBLE, p90 DOUBLE, p95 DOUBLE, p99 DOUBLE,
        min DOUBLE,
        max DOUBLE,
        PRIMARY KEY (ensemble_id, opp_def_id, metric_name)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ensemble_distribution_draws (
        ensemble_id TEXT,
        opp_def_id TEXT,
        metric_name TEXT,
        plan_id TEXT,
        metric_value DOUBLE,
        PRIMARY KEY (ensemble_id, opp_def_id, metric_name, plan_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS plan_vs_ensemble (
        plan_id TEXT,
        ensemble_id TEXT,
        opp_def_id TEXT,
        metric_name TEXT,
        plan_value DOUBLE,
        percentile DOUBLE,
        z_score DOUBLE,
        tail_prob_low DOUBLE,
        tail_prob_high DOUBLE,
        delta_from_mean DOUBLE,
        PRIMARY KEY (plan_id, ensemble_id, opp_def_id, metric_name)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_run (
        ei_run_id TEXT PRIMARY KEY,
        ensemble_id TEXT,
        election_id TEXT,
        model_spec_json TEXT,
        created_at TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_posterior_group (
        ei_run_id TEXT,
        group_id TEXT,
        param_name TEXT,
        mean DOUBLE,
        sd DOUBLE,
        q05 DOUBLE,
        q50 DOUBLE,
        q95 DOUBLE,
        PRIMARY KEY (ei_run_id, group_id, param_name)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_posterior_district (
        ei_run_id TEXT,
        plan_id TEXT,
        district_id TEXT,
        minority_preferred_win_prob DOUBLE,
        dem_share_pred_mean DOUBLE,
        dem_share_pred_q05 DOUBLE,
        dem_share_pred_q95 DOUBLE,
        PRIMARY KEY (ei_run_id, plan_id, district_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS district_outcome_summary (
        ei_run_id TEXT,
        plan_id TEXT,
        district_id TEXT,
        minority_cvap_share DOUBLE,
        is_majority_minority_cvap BOOLEAN,
        minority_preferred_win_prob DOUBLE,
        dem_share_pred_mean DOUBLE,
        dem_share_pred_q05 DOUBLE,
        dem_share_pred_q95 DOUBLE,
        PRIMARY KEY (ei_run_id, plan_id, district_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS plan_opportunity_score (
        ei_run_id TEXT,
        plan_id TEXT,
        opportunity_score DOUBLE,
        n_districts BIGINT,
        PRIMARY KEY (ei_run_id, plan_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_multielection_run (
        ei_run_id TEXT PRIMARY KEY,
        ensemble_id TEXT,
        vra_config_json TEXT,
        model_spec_json TEXT,
        created_at TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_multielection_district (
        ei_run_id TEXT,
        plan_id TEXT,
        district_id TEXT,
        group_id TEXT,
        effectiveness_score DOUBLE,
        is_effective BOOLEAN,
        PRIMARY KEY (ei_run_id, plan_id, district_id, group_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ei_plan_vs_ensemble (
        ei_run_id TEXT,
        plan_id TEXT,
        ensemble_id TEXT,
        opportunity_score DOUBLE,
        ensemble_mean DOUBLE,
        ensemble_sd DOUBLE,
        percentile DOUBLE,
        p_value_left DOUBLE,
        p_value_right DOUBLE,
        z_score DOUBLE,
        delta_from_mean DOUBLE,
        PRIMARY KEY (ei_run_id, plan_id, ensemble_id)
    );
    """)


def upsert_opp_defs(con: duckdb.DuckDBPyConnection, opp_defs: list[OpportunityDef]) -> None:
    df = pd.DataFrame([{
        "opp_def_id": od.opp_def_id,
        "group_label": od.group_label,
        "group_share_col": od.group_share_col,
        "threshold": od.threshold,
        "notes": None,
    } for od in opp_defs])

    con.register("tmp_opp_defs", df)
    con.execute("""
        INSERT OR REPLACE INTO opp_def
        SELECT opp_def_id, group_label, group_share_col, threshold, notes
        FROM tmp_opp_defs
    """)
    con.unregister("tmp_opp_defs")
