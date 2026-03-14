from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd


def build_district_opportunity(con: duckdb.DuckDBPyConnection) -> None:
    """
    Prong-1 style demographic opportunity table.
    Uses district_demo_cvap rather than district_demo_vap.
    """
    con.execute("DELETE FROM district_opportunity;")

    opp_df = con.execute("SELECT opp_def_id, group_share_col, threshold FROM opp_def").df()
    demo_df = con.execute("SELECT * FROM district_demo_cvap").df()
    if demo_df.empty:
        raise ValueError("district_demo_cvap is empty; run aggregates first.")

    out_rows = []
    for _, od in opp_df.iterrows():
        col = od["group_share_col"]
        thr = float(od["threshold"])
        if col not in demo_df.columns:
            raise ValueError(f"district_demo_cvap missing share col required by opp_def: {col}")
        vals = pd.to_numeric(demo_df[col], errors="coerce")
        tmp = pd.DataFrame({
            "plan_id": demo_df["plan_id"],
            "district_id": demo_df["district_id"],
            "opp_def_id": od["opp_def_id"],
            "opp_value": vals,
            "is_opportunity": vals >= thr,
        })
        out_rows.append(tmp)

    out = pd.concat(out_rows, ignore_index=True)
    con.register("tmp_district_opp", out)
    con.execute("""
        INSERT INTO district_opportunity
        SELECT plan_id, district_id, opp_def_id, opp_value, is_opportunity
        FROM tmp_district_opp
    """)
    con.unregister("tmp_district_opp")


def build_plan_metrics(con: duckdb.DuckDBPyConnection) -> None:
    """
    Diagnostic plan-level metrics for Prong 1:
      - number of CVAP opportunity districts
      - mean group share (from the selected opp share column)
    """
    con.execute("DELETE FROM plan_metrics;")

    opp_def = con.execute("SELECT opp_def_id, group_share_col FROM opp_def").df()
    demo_df = con.execute("SELECT * FROM district_demo_cvap").df()
    opp_df = con.execute("SELECT * FROM district_opportunity").df()

    rows = []
    for _, od in opp_def.iterrows():
        oid = od["opp_def_id"]
        col = od["group_share_col"]
        if col not in demo_df.columns:
            raise ValueError(f"district_demo_cvap missing required group_share_col={col}")

        dsub = demo_df[["plan_id", "district_id", col]].copy()
        osub = opp_df.loc[opp_df["opp_def_id"] == oid, ["plan_id", "district_id", "is_opportunity"]].copy()

        merged = dsub.merge(osub, on=["plan_id", "district_id"], how="left")
        merged["is_opportunity"] = merged["is_opportunity"].fillna(False)
        grouped = merged.groupby("plan_id", dropna=False)

        rows.extend(
            {
                "plan_id": pid,
                "opp_def_id": oid,
                "n_opportunity_districts": int(g["is_opportunity"].sum()),
                "mean_group_share": float(pd.to_numeric(g[col], errors="coerce").mean()),
            }
            for pid, g in grouped
        )

    out = pd.DataFrame(rows)
    con.register("tmp_plan_metrics", out)
    con.execute("""
        INSERT INTO plan_metrics
        SELECT plan_id, opp_def_id, n_opportunity_districts, mean_group_share
        FROM tmp_plan_metrics
    """)
    con.unregister("tmp_plan_metrics")


def build_district_outcome_summary(con: duckdb.DuckDBPyConnection, ei_run_id: str) -> None:
    con.execute("DELETE FROM district_outcome_summary WHERE ei_run_id = ?", [ei_run_id])
    con.execute("""
        INSERT INTO district_outcome_summary
        SELECT
            e.ei_run_id,
            e.plan_id,
            e.district_id,
            c.share_minority_cvap AS minority_cvap_share,
            (c.share_minority_cvap >= 0.50) AS is_majority_minority_cvap,
            e.minority_preferred_win_prob,
            e.dem_share_pred_mean,
            e.dem_share_pred_q05,
            e.dem_share_pred_q95
        FROM ei_posterior_district e
        JOIN district_demo_cvap c
          ON c.plan_id = e.plan_id
         AND c.district_id = e.district_id
        WHERE e.ei_run_id = ?
    """, [ei_run_id])


def build_plan_opportunity_score(con: duckdb.DuckDBPyConnection, ei_run_id: str) -> None:
    """
    Main paper statistic:
        O(pi) = sum_k rho_k(pi)
    where rho_k(pi) is minority_preferred_win_prob.
    """
    con.execute("DELETE FROM plan_opportunity_score WHERE ei_run_id = ?", [ei_run_id])
    con.execute("""
        INSERT INTO plan_opportunity_score
        SELECT
            ei_run_id,
            plan_id,
            SUM(minority_preferred_win_prob) AS opportunity_score,
            COUNT(*) AS n_districts
        FROM ei_posterior_district
        WHERE ei_run_id = ?
        GROUP BY ei_run_id, plan_id
    """, [ei_run_id])
