from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

def build_ensemble_distribution(
    con: duckdb.DuckDBPyConnection,
    metric_name: str = "n_opportunity_districts",
) -> None:
    con.execute("DELETE FROM ensemble_distribution;")

    df = con.execute(f"""
        SELECT p.ensemble_id, pm.opp_def_id, pm.{metric_name} AS value
        FROM plan_metrics pm
        JOIN plan p ON p.plan_id = pm.plan_id
        WHERE p.plan_type = 'ENSEMBLE'
    """).df()

    if df.empty:
        raise ValueError("No ensemble plans found. Check plan.plan_type and that plan_metrics is built.")

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []

    for (ensemble_id, opp_def_id), g in df.groupby(["ensemble_id", "opp_def_id"]):
        vals = pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        qs = np.quantile(vals, quantiles)
        rows.append({
            "ensemble_id": ensemble_id,
            "opp_def_id": opp_def_id,
            "metric_name": metric_name,
            "n_plans": int(len(vals)),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "p01": float(qs[0]),
            "p05": float(qs[1]),
            "p10": float(qs[2]),
            "p25": float(qs[3]),
            "p50": float(qs[4]),
            "p75": float(qs[5]),
            "p90": float(qs[6]),
            "p95": float(qs[7]),
            "p99": float(qs[8]),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    out = pd.DataFrame(rows)
    con.register("tmp_ensdist", out)
    con.execute("INSERT INTO ensemble_distribution SELECT * FROM tmp_ensdist")
    con.unregister("tmp_ensdist")


def build_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    metric_name: str = "n_opportunity_districts",
) -> None:
    con.execute("DELETE FROM plan_vs_ensemble;")

    ens_vals = con.execute(f"""
        SELECT p.ensemble_id, pm.opp_def_id, pm.{metric_name} AS value
        FROM plan_metrics pm
        JOIN plan p ON p.plan_id = pm.plan_id
        WHERE p.plan_type = 'ENSEMBLE'
    """).df()

    ref = con.execute("""
        SELECT ensemble_id, opp_def_id, metric_name, mean, sd
        FROM ensemble_distribution
    """).df()

    plans = con.execute(f"""
        SELECT p.plan_id, p.ensemble_id, pm.opp_def_id, pm.{metric_name} AS plan_value
        FROM plan p
        JOIN plan_metrics pm ON pm.plan_id = p.plan_id
    """).df()

    if ens_vals.empty or ref.empty or plans.empty:
        raise ValueError("Missing required inputs for plan_vs_ensemble.")

    # Precompute sorted arrays
    grouped_sorted = {}
    for (ensemble_id, opp_def_id), g in ens_vals.groupby(["ensemble_id", "opp_def_id"]):
        arr = np.sort(pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy())
        grouped_sorted[(ensemble_id, opp_def_id)] = arr

    ref_map = {(r.ensemble_id, r.opp_def_id): (float(r.mean), float(r.sd)) for r in ref.itertuples(index=False)}

    out_rows = []
    for r in plans.itertuples(index=False):
        key = (r.ensemble_id, r.opp_def_id)
        arr = grouped_sorted.get(key)
        if arr is None or len(arr) == 0:
            continue

        pv = float(r.plan_value)
        right = np.searchsorted(arr, pv, side="right")
        left = np.searchsorted(arr, pv, side="left")

        percentile = 100.0 * right / len(arr)
        tail_low = right / len(arr)
        tail_high = 1.0 - (left / len(arr))

        mean, sd = ref_map.get(key, (float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0))
        z = (pv - mean) / sd if sd > 0 else np.nan

        out_rows.append({
            "plan_id": r.plan_id,
            "ensemble_id": r.ensemble_id,
            "opp_def_id": r.opp_def_id,
            "metric_name": metric_name,
            "plan_value": pv,
            "percentile": percentile,
            "z_score": z,
            "tail_prob_low": tail_low,
            "tail_prob_high": tail_high,
            "delta_from_mean": pv - mean,
        })

    out = pd.DataFrame(out_rows)
    con.register("tmp_pve", out)
    con.execute("INSERT INTO plan_vs_ensemble SELECT * FROM tmp_pve")
    con.unregister("tmp_pve")
