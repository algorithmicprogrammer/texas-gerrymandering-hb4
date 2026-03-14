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
    Project EI posterior draws onto district compositions.

    Key change:
      use district CVAP shares rather than district VAP shares so that the projection
      aligns with the slideshow's CVAP-based district opportunity language.
    """
    con.execute("DELETE FROM ei_posterior_district WHERE ei_run_id = ?", [ei_run_id])

    demo = con.execute("""
        SELECT
            d.plan_id,
            d.district_id,
            d.share_white_cvap,
            d.share_black_cvap,
            d.share_hisp_cvap,
            d.share_asian_cvap,
            d.share_native_cvap,
            d.share_minority_cvap
        FROM district_demo_cvap d
        JOIN plan p ON p.plan_id = d.plan_id
        WHERE p.ensemble_id = ?
    """, [ensemble_id]).df()

    if demo.empty:
        raise ValueError(f"No district_demo_cvap rows for ensemble_id={ensemble_id}")

    demo = demo.copy()
    demo["share_other_cvap"] = np.clip(
        demo["share_minority_cvap"]
        - demo["share_black_cvap"]
        - demo["share_hisp_cvap"]
        - demo["share_asian_cvap"]
        - demo["share_native_cvap"],
        0.0,
        1.0,
    )

    group_to_col = {
        "WHITE_NH": "share_white_cvap",
        "BLACK_NH": "share_black_cvap",
        "HISP": "share_hisp_cvap",
        "ASIAN_NH": "share_asian_cvap",
        "NATIVE_NH": "share_native_cvap",
        "OTHER": "share_other_cvap",
    }

    missing = [g for g in groups if g not in group_to_col]
    if missing:
        raise ValueError(f"Unknown EI groups in projection: {missing}")

    Xd = np.vstack([demo[group_to_col[g]].to_numpy(dtype=float) for g in groups]).T
    rs = Xd.sum(axis=1, keepdims=True)
    Xd = np.divide(Xd, rs, out=np.zeros_like(Xd), where=(rs > 0))

    p_draws = Xd @ support_draws.T

    dem_win_prob = np.mean(p_draws > 0.5, axis=1)
    minority_preferred_win_prob = dem_win_prob if dem_is_minority_preferred else 1.0 - dem_win_prob

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
