import numpy as np
import pandas as pd
from typing import Dict, List

def enacted_opportunity_count(cd_posteriors: pd.DataFrame, opportunity_prob_thresh: float) -> int:
    return int((cd_posteriors["win_prob_dem"] > opportunity_prob_thresh).sum())

def score_plan_opportunities_global_mu(
    plans_df: pd.DataFrame,
    vtd_units: pd.DataFrame,
    idata,
    vtd_id_col: str,
    pop_col: str,
    race_cols: List[str],
    opportunity_prob_thresh: float,
) -> pd.DataFrame:
    """
    Scores simulated plans using global race propensities (mu) from the Bayesian EI model.
    Conservative and publishable; can be upgraded to simulate district-level variability.
    """
    mu = idata.posterior["mu"].stack(sample=("chain", "draw")).values  # (S,R)
    theta_global = 1.0 / (1.0 + np.exp(-mu))                           # (S,R)

    df_units = vtd_units[[vtd_id_col, pop_col] + race_cols].copy()
    merged = plans_df.merge(df_units, on=vtd_id_col, how="left")

    out = []
    for plan_id, g in merged.groupby("plan_id"):
        grp = g.groupby("cd_sim")

        M_sim = []
        for _, h in grp:
            w = h[pop_col].values.astype(float)
            M = np.average(h[race_cols].values.astype(float), weights=w, axis=0) if np.sum(w) > 0 else np.full(len(race_cols), np.nan)
            M_sim.append(M)

        M_sim = np.vstack(M_sim)  # (K,R)

        # (S,R) @ (R,K) -> (S,K)
        Yhat = theta_global @ M_sim.T
        win_prob = (Yhat > 0.5).mean(axis=0)  # (K,)

        out.append({
            "plan_id": int(plan_id),
            "opp_districts": int((win_prob > opportunity_prob_thresh).sum()),
            "mean_win_prob": float(np.nanmean(win_prob)),
        })

    return pd.DataFrame(out)

def opportunity_loss_summary(plan_scores: pd.DataFrame, enacted_opp: int) -> Dict[str, float]:
    L = plan_scores["opp_districts"].values - enacted_opp
    return {
        "enacted_opp": int(enacted_opp),
        "ensemble_mean_opp": float(plan_scores["opp_districts"].mean()),
        "ensemble_ci95_opp_low": float(np.quantile(plan_scores["opp_districts"], 0.025)),
        "ensemble_ci95_opp_high": float(np.quantile(plan_scores["opp_districts"], 0.975)),
        "P(L>0)": float((L > 0).mean()),
        "L_mean": float(L.mean()),
        "L_ci95_low": float(np.quantile(L, 0.025)),
        "L_ci95_high": float(np.quantile(L, 0.975)),
    }
