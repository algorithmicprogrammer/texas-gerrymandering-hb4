from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

def project_ei_to_districts(
    con: duckdb.DuckDBPyConnection,
    ei_run_id: str,
    ensemble_id: str,
    groups: list[str],
    support_draws: np.ndarray,  # (samples, groups)
    dem_is_minority_preferred: bool = True,
) -> None:
    """
    Projects EI posterior draws (group Dem support) onto every plan's district composition.
    Writes ei_posterior_district for plans in ensemble_id.
    """
    con.execute("DELETE FROM ei_posterior_district WHERE ei_run_id = ?", [ei_run_id])

    demo = con.execute("""
        SELECT d.plan_id, d.district_id,
               d.share_white_vap, d.share_black_vap, d.share_hisp_vap, d.share_minority_vap
        FROM district_demo_vap d
        JOIN plan p ON p.plan_id = d.plan_id
        WHERE p.ensemble_id = ?
    """, [ensemble_id]).df()

    if demo.empty:
        raise ValueError(f"No district_demo_vap rows for ensemble_id={ensemble_id}")

    # Approximate remaining minority shares as "OTHER" bucket
    demo = demo.copy()
    demo["share_other_minor"] = np.clip(
        demo["share_minority_vap"] - demo["share_black_vap"] - demo["share_hisp_vap"],
        0.0, 1.0
    )

    group_to_col = {
        "WHITE_NH": "share_white_vap",
        "BLACK_NH": "share_black_vap",
        "HISP": "share_hisp_vap",
        "ASIAN_NH": "share_other_minor",
        "NATIVE_NH": "share_other_minor",
        "OTHER": "share_other_minor",
    }

    Xd = np.vstack([demo[group_to_col[g]].to_numpy() for g in groups]).T
    rs = Xd.sum(axis=1, keepdims=True)
    Xd = np.where(rs > 0, Xd / rs, Xd)

    # (n_districts, n_samples)
    p_draws = Xd @ support_draws.T

    win_prob = np.mean(p_draws > 0.5, axis=1)
    minority_preferred_win_prob = win_prob if dem_is_minority_preferred else 1.0 - win_prob

    out = pd.DataFrame({
        "ei_run_id": ei_run_id,
        "plan_id": demo["plan_id"].values,
        "district_id": demo["district_id"].values,
        "minority_preferred_win_prob": minority_preferred_win_prob,
        "dem_share_pred_mean": np.mean(p_draws, axis=1),
        "dem_share_pred_q05": np.quantile(p_draws, 0.05, axis=1),
        "dem_share_pred_q95": np.quantile(p_draws, 0.95, axis=1),
    })

    con.register("tmp_ei_d", out)
    con.execute("""
        INSERT OR REPLACE INTO ei_posterior_district
        SELECT * FROM tmp_ei_d
    """)
    con.unregister("tmp_ei_d")
