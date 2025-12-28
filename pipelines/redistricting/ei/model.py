from __future__ import annotations
import json
import numpy as np
import pandas as pd
import duckdb
from ..config import RACE_COLS_VAP
from .project import project_ei_to_districts

def fit_hierarchical_ei_vtd(
    con: duckdb.DuckDBPyConnection,
    election_id: str,
    ei_run_id: str,
    ensemble_id: str,
    draws: int = 1500,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.9,
) -> None:
    """
    Minimal hierarchical EI on VTDs, then project to district compositions across all plans in ensemble_id.
    Imports pymc/arviz only when you run EI stage.
    """
    try:
        import pymc as pm
    except ImportError as e:
        raise ImportError("Install PyMC: pip install pymc arviz") from e

    df = con.execute("""
        SELECT
            g.vtd_geoid,
            g.vap_total,
            g.vap_nh_white,
            g.vap_nh_black,
            g.vap_hisp,
            g.vap_nh_asian,
            g.vap_nh_native,
            g.vap_other,
            er.votes_total,
            er.votes_dem
        FROM geo_vtd g
        JOIN election_returns_vtd er
          ON er.vtd_geoid = g.vtd_geoid
        WHERE er.election_id = ?
    """, [election_id]).df()

    df = df[df["votes_total"] > 0].copy()

    # shares in VTD
    for col in RACE_COLS_VAP:
        df[col + "_share"] = np.where(df["vap_total"] > 0, df[col] / df["vap_total"], 0.0)

    group_map = {
        "WHITE_NH": "vap_nh_white_share",
        "BLACK_NH": "vap_nh_black_share",
        "HISP": "vap_hisp_share",
        "ASIAN_NH": "vap_nh_asian_share",
        "NATIVE_NH": "vap_nh_native_share",
        "OTHER": "vap_other_share",
    }
    groups = list(group_map.keys())

    X = df[[group_map[g] for g in groups]].to_numpy().astype(float)
    rs = X.sum(axis=1, keepdims=True)
    X = np.where(rs > 0, X / rs, X)

    y = df["votes_dem"].to_numpy().astype(int)
    n = df["votes_total"].to_numpy().astype(int)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.5, shape=len(groups))
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=len(groups))
        theta = pm.Normal("theta", mu=mu, sigma=sigma, shape=len(groups))
        support = pm.Deterministic("support", pm.math.sigmoid(theta))
        p = pm.math.dot(X, support)
        pm.Binomial("y_obs", n=n, p=p, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, progressbar=True)

    # extract draws: (samples, groups)
    support_draws = idata.posterior["support"].stack(sample=("chain", "draw")).values
    if support_draws.shape[0] == len(groups):
        support_draws = support_draws.T

    spec = {
        "model": "minimal_hierarchical_ei_vtd",
        "election_id": election_id,
        "groups": groups,
        "priors": {"mu": "Normal(0,1.5)", "sigma": "HalfNormal(1.0)"},
        "notes": "Fit on VTDs; projected to plan districts via district composition.",
    }
    con.execute("""
        INSERT OR REPLACE INTO ei_run (ei_run_id, ensemble_id, election_id, model_spec_json, created_at)
        VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S', now()))
    """, [ei_run_id, ensemble_id, election_id, json.dumps(spec)])

    # store group summaries
    rows = []
    for gi, gname in enumerate(groups):
        s = support_draws[:, gi]
        rows.append({
            "ei_run_id": ei_run_id,
            "group_id": gname,
            "param_name": "support_dem",
            "mean": float(np.mean(s)),
            "sd": float(np.std(s, ddof=1)),
            "q05": float(np.quantile(s, 0.05)),
            "q50": float(np.quantile(s, 0.50)),
            "q95": float(np.quantile(s, 0.95)),
        })
    out = pd.DataFrame(rows)
    con.register("tmp_ei_group", out)
    con.execute("INSERT OR REPLACE INTO ei_posterior_group SELECT * FROM tmp_ei_group")
    con.unregister("tmp_ei_group")

    # project to every plan in ensemble_id
    project_ei_to_districts(
        con=con,
        ei_run_id=ei_run_id,
        ensemble_id=ensemble_id,
        groups=groups,
        support_draws=support_draws,
    )
