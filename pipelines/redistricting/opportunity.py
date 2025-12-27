from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

def build_district_opportunity(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM district_opportunity;")

    opp_df = con.execute("SELECT opp_def_id, group_share_col, threshold FROM opp_def").df()
    demo_df = con.execute("SELECT * FROM district_demo_vap").df()
    if demo_df.empty:
        raise ValueError("district_demo_vap is empty; run aggregates first.")

    out_rows = []
    for _, od in opp_df.iterrows():
        col = od["group_share_col"]
        thr = float(od["threshold"])
        if col not in demo_df.columns:
            raise ValueError(f"district_demo_vap missing share col required by opp_def: {col}")
        vals = pd.to_numeric(demo_df[col], errors="coerce")
        tmp = pd.DataFrame({
            "plan_id": demo_df["plan_id"],
            "district_id": demo_df["district_id"],
            "opp_def_id": od["opp_def_id"],
            "opp_value": vals,
            "is_opportunity": vals >= thr
        })
        out_rows.append(tmp)

    out = pd.concat(out_rows, ignore_index=True)
    con.register("tmp_opp", out)
    con.execute("""
        INSERT INTO district_opportunity
        SELECT plan_id, district_id, opp_def_id, opp_value, is_opportunity
        FROM tmp_opp
    """)
    con.unregister("tmp_opp")

def build_plan_metrics(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM plan_metrics;")
    con.execute("""
    INSERT INTO plan_metrics
    SELECT
        d.plan_id,
        d.opp_def_id,
        SUM(CASE WHEN d.is_opportunity THEN 1 ELSE 0 END) AS n_opportunity_districts,
        AVG(dd.share_minority_vap) AS mean_minority_share
    FROM district_opportunity d
    JOIN district_demo_vap dd
      ON dd.plan_id = d.plan_id AND dd.district_id = d.district_id
    GROUP BY d.plan_id, d.opp_def_id;
    """)
