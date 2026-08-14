# -*- coding: utf-8 -*-
"""
threshold_sweep.py
==================

Threshold sensitivity analysis for the minority-opportunity functional.

Motivation
----------
The canonical functional used in the paper is continuous:

    O_g(pi) = sum_d p_d^g(pi)

The *legacy* literature statistic instead counts districts whose
effectiveness score clears a hard cutoff tau:

    C_g^(tau)(pi) = sum_d I[ p_d^g(pi) >= tau ]

besag_clifford_vra_opportunity.py reports C at a single cutoff
(EFFECTIVENESS_CUTOFF = 0.6) as a legacy comparison column. A reviewer
objection to that column is standard and fair: the 0.6 cutoff has no
citation behind it, and no alternative cutoff was tested. This module
answers the objection directly by sweeping tau across the full plausible
range (default 0.40 to 0.80 in steps of 0.01) and reporting, for every
threshold and every group:

  * the enacted plan's threshold count C_g^(tau)(enacted);
  * the enacted plan's rank among all M+1 plans (enacted + M spokes);
  * the Besag-Clifford lower-tail exact p-value;
  * the number of DISTINCT values C_g^(tau) takes across the M spokes;
  * the percentage of spokes sitting in the modal bin;
  * Spearman and Pearson association between C_g^(tau) and O_g across spokes.

The last three quantify *how much information the thresholding throws
away*. When 0.40 <= tau <= 0.80 makes the enacted rank swing across most
of the ensemble, or when the spoke distribution collapses onto two or
three integers, the conclusion is not "0.6 happens to give a silly
result" but "threshold-based inference is structurally unstable across
reasonable cutoffs, which is exactly why the paper's primary statistic is
the continuous functional O_g."

No new EI runs and no new ReCom simulations are needed: the sweep is a
pure re-reduction of per-district probabilities that the chain already
computed.

Inputs
------
Preferred: the tidy per-district probability table written by
besag_clifford_vra_opportunity.py,

    outputs/bc_district_probs_{RUN_NAME}.parquet
    columns: plan (str), district (int), p_L, p_B, p_N (float)

with one row per (plan, district); ``plan`` is ``"enacted"`` for the
enacted map and ``"spoke_000" ... "spoke_{M-1}"`` for the ensemble.

Fallback (``--from-assignments``): for runs that finished BEFORE the
probability table existed, the same table is reconstructed from

    outputs/bc_plan_assignments_{RUN_NAME}.parquet

by re-scoring each saved plan. This re-runs only the (cheap, exact)
scoring layer -- it does NOT re-run the reversible-ReCom chains, which
are the expensive part -- so the ensemble is bit-for-bit the same one the
p-values were computed from. The reconstructed table is cached to the
parquet path above so the sweep is instant on subsequent runs.

Usage
-----
    # sweep an existing run (probability table already saved)
    python pipelines/ensemble_generation_layer/threshold_sweep.py

    # sweep an older run that only has plan assignments saved
    python pipelines/ensemble_generation_layer/threshold_sweep.py \
        --from-assignments --workers 8

    # custom grid
    python pipelines/ensemble_generation_layer/threshold_sweep.py \
        --tau-min 0.30 --tau-max 0.90 --tau-step 0.05

Outputs (in outputs/)
---------------------
    threshold_sweep_{RUN_NAME}.csv             tidy sweep table (all stats)
    threshold_sweep_baseline_{RUN_NAME}.csv    continuous O_g reference row
    threshold_sweep_{RUN_NAME}.tex             publication LaTeX table
    threshold_sweep_rank_{RUN_NAME}.{pdf,png}  MAIN figure: rank vs tau
    threshold_sweep_diagnostics_{RUN_NAME}.{pdf,png}   discreteness panels
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend -- must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scipy carries the exact permutation-free correlation p-values. It is in
# requirements.txt, but the sweep is useful without it, so fall back to
# pandas' correlation (coefficient only, no p-value) rather than dying.
try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised only on a scipy-less env
    _scipy_stats = None


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_RUN_NAME = "TX_BC_functional"

# Sweep grid. 0.40-0.80 brackets every cutoff we could find in the
# effectiveness-score literature and in expert-witness practice; the paper's
# 0.60 sits in the middle of it.
DEFAULT_TAU_MIN = 0.40
DEFAULT_TAU_MAX = 0.80
DEFAULT_TAU_STEP = 0.01

# Row spacing of the LaTeX table (the CSV keeps the full fine grid).
DEFAULT_LATEX_STEP = 0.05

# The cutoff the pipeline hard-codes; drawn as a reference rule on the
# figures and used for the legacy cross-check.
PAPER_CUTOFF = 0.60

ENACTED_LABEL = "enacted"

# Categorical series colors, validated for deuteranopia/protanopia/tritanopia
# separation against a light surface. Every series also carries a distinct
# marker and a direct label, so identity never rests on color alone (the
# figures are printed in grayscale often enough to matter).
COLOR_L = "#2a78d6"  # blue
COLOR_B = "#eb6834"  # orange
COLOR_E = "#1baf7a"  # aqua

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "0.90"


@dataclass(frozen=True)
class GroupSpec:
    """One group whose opportunity is measured.

    ``prob_cols`` lists the per-district probability columns that define the
    group. A district counts as effective for the group when the MAXIMUM
    over those columns clears tau, so the single-group cases reduce to the
    obvious thing and the coalition case reproduces the pipeline's
    ``distinct_eff`` (the union of Latino-effective and Black-effective
    districts) exactly.
    """

    key: str
    label: str
    prob_cols: tuple
    o_col: str        # matching continuous functional in the spokes CSV
    legacy_col: str   # matching legacy tau=0.6 count in the spokes CSV
    color: str
    marker: str


GROUPS = (
    GroupSpec("L", "Latino", ("p_L",), "O_L", "hisp_eff", COLOR_L, "o"),
    GroupSpec("B", "Black", ("p_B",), "O_B", "black_eff", COLOR_B, "s"),
    GroupSpec("either", "Either (distinct)", ("p_L", "p_B"), "O_joint",
              "distinct_eff", COLOR_E, "^"),
)

PROB_COLUMNS = ("p_L", "p_B")


# ============================================================
# GRID
# ============================================================


def tau_grid(tau_min=DEFAULT_TAU_MIN, tau_max=DEFAULT_TAU_MAX,
             tau_step=DEFAULT_TAU_STEP):
    """Build the threshold grid in integer thousandths, then divide.

    Doing this as ``np.arange(0.40, 0.81, 0.01)`` accumulates floating-point
    error, and the 21st element comes out as 0.6000000000000001 rather than
    0.6. That matters here and nowhere else: a district whose probability is
    exactly 0.6 would be counted at the literal cutoff 0.6 but NOT at the
    drifted one, so the sweep's tau=0.60 row would silently fail to
    reproduce the pipeline's legacy counts. Integer division is correctly
    rounded, so ``600 / 1000`` is the same double as the literal ``0.6``.
    """
    lo, hi, step = (int(round(x * 1000)) for x in (tau_min, tau_max, tau_step))
    if step <= 0:
        raise ValueError(f"tau_step must be positive, got {tau_step}")
    if hi < lo:
        raise ValueError(f"tau_max ({tau_max}) is below tau_min ({tau_min})")
    return np.array([t / 1000 for t in range(lo, hi + 1, step)], dtype=float)


# ============================================================
# INPUT
# ============================================================


def load_district_probs(path):
    """Load the tidy per-district probability table (parquet or csv)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Either re-run "
            f"besag_clifford_vra_opportunity.py (which now writes this table), "
            f"or pass --from-assignments to rebuild it from the saved plan "
            f"assignments without re-running any chains."
        )
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    missing = [c for c in ("plan", "district", *PROB_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}.")
    return df


def probability_matrices(probs_df, enacted_label=ENACTED_LABEL):
    """Pivot the tidy table into per-group (n_plans x n_districts) matrices.

    Returns ``(mats, plan_labels)`` where ``mats[col]`` is an ndarray whose
    row 0 is the enacted plan and whose remaining rows are the spokes in
    sorted plan-label order.

    Plans carrying any missing probability are dropped: ``final_elec_model``
    emits "N/A" when an election set has no minority-preferred candidate, and
    the pipeline's own scoring treats such a plan as unscorable rather than
    scoring it as zero. Dropping keeps the sweep consistent with the
    p-values, which are computed on ``dropna()``'d columns.
    """
    if enacted_label not in set(probs_df["plan"]):
        raise ValueError(
            f"No plan labelled {enacted_label!r} in the probability table; "
            f"the enacted map is required as the test statistic."
        )

    mats = {}
    for col in PROB_COLUMNS:
        wide = probs_df.pivot(index="plan", columns="district", values=col)
        mats[col] = wide.sort_index(axis=1)

    any_frame = mats[PROB_COLUMNS[0]]
    good = np.ones(len(any_frame), dtype=bool)
    for frame in mats.values():
        good &= frame.notna().all(axis=1).to_numpy()

    dropped = [p for p, ok in zip(any_frame.index, good) if not ok]
    if enacted_label in dropped:
        raise ValueError(
            "The enacted plan has missing per-district probabilities; the "
            "sweep cannot rank a plan whose statistic is undefined."
        )
    if dropped:
        print(f"  NOTE: dropped {len(dropped)} plan(s) with missing "
              f"probabilities (e.g. {dropped[:3]}).")

    spoke_labels = sorted(p for p, ok in zip(any_frame.index, good)
                          if ok and p != enacted_label)
    order = [enacted_label] + spoke_labels

    return ({col: frame.loc[order].to_numpy(dtype=float)
             for col, frame in mats.items()},
            order)


def group_matrix(mats, group):
    """Per-district effectiveness probability for one group.

    For a single-column group this is that column. For the coalition group
    it is the elementwise maximum, which makes ``count >= tau`` the union
    "Latino-effective OR Black-effective" and makes the continuous score
    ``sum_d max(p_d^B, p_d^L)`` -- both matching the pipeline's definitions.
    """
    stack = np.stack([mats[c] for c in group.prob_cols], axis=0)
    return stack.max(axis=0)


# ============================================================
# STATISTICS
# ============================================================


def _besag_clifford_stats(spoke_vals, enacted_val):
    """Rank and exact lower/upper-tail p-values for the enacted statistic.

    Conventions match ``besag_clifford_vra_opportunity.compute_bc_pvalues``:

        rank  = #{spokes <= enacted} + 1
        p_low = (#{spokes <= enacted} + 1) / (M + 1)

    Threshold counts are integers, so ties are the rule rather than the
    exception. Counting ties as "<=" is the conservative choice for a
    lower-tail test (it inflates p rather than deflating it) and it is what
    keeps the test exact under the exchangeability argument. ``rank_min``
    and ``n_tied`` are also returned so the figures can show the tie band
    instead of implying a precision the discrete statistic does not have.
    """
    spoke_vals = np.asarray(spoke_vals, dtype=float)
    m = spoke_vals.size
    n_leq = int((spoke_vals <= enacted_val).sum())
    n_lt = int((spoke_vals < enacted_val).sum())
    n_geq = int((spoke_vals >= enacted_val).sum())
    return {
        "rank": n_leq + 1,
        "rank_min": n_lt + 1,
        "n_tied": n_leq - n_lt,
        "p_lower": (n_leq + 1) / (m + 1),
        "p_upper": (n_geq + 1) / (m + 1),
        "n_plans": m + 1,
    }


def _association(x, y):
    """Spearman rho and Pearson r (with p-values when scipy is available).

    A constant input -- every spoke landing on the same threshold count, which
    is precisely the failure mode the sweep is looking for -- leaves both
    coefficients undefined. Return NaN rather than letting scipy warn its way
    to a NaN, so the CSV says "undefined" for an unambiguous reason.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nan = {"spearman_rho": np.nan, "spearman_p": np.nan,
           "pearson_r": np.nan, "pearson_p": np.nan}
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return nan
    if _scipy_stats is None:  # pragma: no cover - scipy is in requirements
        s = pd.Series(x).corr(pd.Series(y), method="spearman")
        p = pd.Series(x).corr(pd.Series(y), method="pearson")
        return {"spearman_rho": float(s), "spearman_p": np.nan,
                "pearson_r": float(p), "pearson_p": np.nan}
    rho, rho_p = _scipy_stats.spearmanr(x, y)
    r, r_p = _scipy_stats.pearsonr(x, y)
    return {"spearman_rho": float(rho), "spearman_p": float(rho_p),
            "pearson_r": float(r), "pearson_p": float(r_p)}


def _modal_bin(spoke_vals):
    """Modal threshold count and the share of spokes sitting in it.

    On a frequency tie the smaller count wins, purely so the column is a
    deterministic function of the ensemble.
    """
    counts = Counter(int(v) for v in spoke_vals)
    top = max(counts.values())
    modal_value = min(v for v, c in counts.items() if c == top)
    return modal_value, 100.0 * top / len(spoke_vals)


def compute_sweep(probs_df, taus=None, enacted_label=ENACTED_LABEL):
    """Run the threshold sweep.

    Returns ``(sweep_df, baseline_df)``:

    sweep_df   one row per (group, tau) with the enacted count, its rank and
               exact p-value, the discreteness diagnostics, and the
               association with the continuous functional.
    baseline_df  one row per group for the continuous functional O_g itself
               -- the reference the thresholded statistic is being compared
               against, and the horizontal reference line on the figures.
    """
    if taus is None:
        taus = tau_grid()
    taus = np.asarray(taus, dtype=float)

    mats, plan_labels = probability_matrices(probs_df, enacted_label)
    n_spokes = len(plan_labels) - 1
    if n_spokes < 1:
        raise ValueError("Need at least one spoke plan to rank the enacted map.")

    sweep_rows, baseline_rows = [], []

    for group in GROUPS:
        p_mat = group_matrix(mats, group)          # (n_plans, n_districts)

        # Continuous functional O_g(pi) = sum_d p_d^g, recomputed here from
        # the same probabilities so the baseline and the sweep can never
        # disagree about which ensemble they describe.
        o_vals = p_mat.sum(axis=1)
        o_enacted, o_spokes = o_vals[0], o_vals[1:]
        o_stats = _besag_clifford_stats(o_spokes, o_enacted)
        o_sd = float(np.std(o_spokes, ddof=1)) if n_spokes > 1 else np.nan
        baseline_rows.append({
            "group": group.key,
            "group_label": group.label,
            "statistic": group.o_col,
            "enacted": float(o_enacted),
            "spoke_mean": float(np.mean(o_spokes)),
            "spoke_sd": o_sd,
            "spoke_median": float(np.median(o_spokes)),
            "spoke_p05": float(np.quantile(o_spokes, 0.05)),
            "spoke_p95": float(np.quantile(o_spokes, 0.95)),
            "z": float((o_enacted - np.mean(o_spokes)) / o_sd)
                 if o_sd and not np.isnan(o_sd) and o_sd > 0 else np.nan,
            "n_distinct_spokes": int(len(np.unique(o_spokes))),
            **o_stats,
        })

        # C_g^(tau) for every plan and every threshold at once:
        # (n_plans, n_districts, 1) >= (1, 1, n_taus) -> sum over districts.
        counts = (p_mat[:, :, None] >= taus[None, None, :]).sum(axis=1)
        enacted_counts, spoke_counts = counts[0], counts[1:]

        for j, tau in enumerate(taus):
            c_enacted = int(enacted_counts[j])
            c_spokes = spoke_counts[:, j].astype(float)
            modal_value, pct_modal = _modal_bin(c_spokes)
            stats = _besag_clifford_stats(c_spokes, c_enacted)
            sweep_rows.append({
                "group": group.key,
                "group_label": group.label,
                "tau": float(tau),
                "enacted_count": c_enacted,
                **stats,
                "spoke_mean": float(c_spokes.mean()),
                "spoke_sd": float(c_spokes.std(ddof=1)) if n_spokes > 1 else np.nan,
                "spoke_median": float(np.median(c_spokes)),
                "spoke_p05": float(np.quantile(c_spokes, 0.05)),
                "spoke_p95": float(np.quantile(c_spokes, 0.95)),
                "spoke_min": int(c_spokes.min()),
                "spoke_max": int(c_spokes.max()),
                "n_distinct_spokes": int(len(np.unique(c_spokes))),
                "modal_count": int(modal_value),
                "pct_spokes_modal": pct_modal,
                **_association(c_spokes, o_spokes),
                "o_enacted": float(o_enacted),
                "o_rank": o_stats["rank"],
                "o_p_lower": o_stats["p_lower"],
            })

    sweep_df = pd.DataFrame(sweep_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    return sweep_df, baseline_df


# ============================================================
# CROSS-CHECK AGAINST THE PIPELINE'S OWN tau = 0.6 COLUMNS
# ============================================================


def crosscheck_legacy_counts(probs_df, spokes_df, enacted_label=ENACTED_LABEL,
                             cutoff=PAPER_CUTOFF):
    """Verify the sweep reproduces the chain's legacy cutoff columns.

    ``besag_clifford_vra_opportunity`` computes ``hisp_eff`` / ``black_eff`` /
    ``distinct_eff`` inside the chain at EFFECTIVENESS_CUTOFF. Recomputing
    them here from the saved probabilities must give identical integers for
    every spoke; if it does not, the probability table and the scores CSV
    describe different ensembles and nothing downstream should be trusted.

    Returns a list of human-readable mismatch descriptions (empty == clean).
    """
    mats, plan_labels = probability_matrices(probs_df, enacted_label)
    label_to_row = {label: i for i, label in enumerate(plan_labels)}
    problems = []

    for group in GROUPS:
        if group.legacy_col not in spokes_df.columns:
            continue
        p_mat = group_matrix(mats, group)
        recomputed = (p_mat >= cutoff).sum(axis=1)
        for _, row in spokes_df.iterrows():
            label = f"spoke_{int(row['spoke']):03d}"
            if label not in label_to_row:
                continue
            expected = row[group.legacy_col]
            if pd.isna(expected):
                continue
            got = int(recomputed[label_to_row[label]])
            if int(expected) != got:
                problems.append(
                    f"{label} {group.legacy_col}: chain={int(expected)} "
                    f"sweep={got}"
                )
    return problems


# ============================================================
# OUTPUT -- FIGURES
# ============================================================


def _style_axis(ax, ylabel, xlabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_PRIMARY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK_PRIMARY)


def plot_rank_vs_threshold(sweep_df, baseline_df, out_base, alpha=0.05):
    """MAIN figure: enacted rank against the threshold, one panel per group.

    Within a panel:

      * the solid line is the enacted plan's rank under the thresholded count
        C_g^(tau) -- if threshold-based inference were stable, it would be
        roughly flat;
      * the shaded band spans rank_min..rank, the range of ranks the enacted
        plan could be assigned given ties. Ties are unavoidable once the
        statistic is an integer count shared between 1000 spokes, and the
        band is often wider than the whole significance region, which is
        itself part of the argument;
      * the dotted horizontal line is where the CONTINUOUS functional O_g
        puts the same plan -- one number, no cutoff, no ties;
      * the gray strip along the bottom is the region where the exact
        lower-tail test rejects at alpha.

    Groups are faceted rather than overlaid: three tie bands on one axis
    overplot into mush, and the comparison that matters is each group's line
    against its own dotted reference, not against the other groups.
    """
    n_plans = int(sweep_df["n_plans"].iloc[0])
    tau_lo, tau_hi = sweep_df["tau"].min(), sweep_df["tau"].max()
    sig_rank = alpha * n_plans  # p = rank / (M + 1) <= alpha

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(11.5, 4.3),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, group in zip(axes, GROUPS):
        sub = sweep_df[sweep_df["group"] == group.key].sort_values("tau")
        base = baseline_df[baseline_df["group"] == group.key].iloc[0]

        ax.axhspan(0, sig_rank, color="#f0f0ec", zorder=0)
        ax.axvline(PAPER_CUTOFF, color="0.75", linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.fill_between(sub["tau"], sub["rank_min"], sub["rank"],
                        color=group.color, alpha=0.18, linewidth=0, zorder=2,
                        label="rank range across ties")
        ax.plot(sub["tau"], sub["rank"], color=group.color, linewidth=2.0,
                marker=group.marker, markersize=4.5, markevery=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                label="thresholded count $C_g^{(\\tau)}$")
        ax.axhline(base["rank"], color=INK_PRIMARY, linewidth=1.3,
                   linestyle=(0, (1, 2)), zorder=3,
                   label="continuous functional $O_g$")

        _style_axis(ax, "", "Effectiveness threshold τ")
        ax.set_title(group.label, fontsize=10.5, color=group.color,
                     fontweight="bold", loc="left")
        ax.set_xlim(tau_lo, tau_hi)
        ax.set_ylim(0, n_plans * 1.02)

    axes[0].set_ylabel(f"Rank of enacted plan among {n_plans} plans\n"
                       "(1 = most extreme)", fontsize=10, color=INK_PRIMARY)
    axes[0].text(tau_lo + 0.01, sig_rank, f"p ≤ {alpha:g}", va="bottom",
                 ha="left", fontsize=8, color=INK_SECONDARY)
    axes[0].annotate("τ = 0.60\n(cutoff in paper)",
                     xy=(PAPER_CUTOFF, n_plans * 0.99),
                     xytext=(3, 0), textcoords="offset points",
                     va="top", ha="left", fontsize=8, color=INK_SECONDARY)
    # Shared legend under the panels: an in-axes legend collides with the
    # tie band, which fills most of the plotting area at large M.
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("thresholded count $C_g^{(\\tau)}$"),
             labels.index("rank range across ties"),
             labels.index("continuous functional $O_g$")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
               frameon=False, fontsize=9)

    return _save(fig, out_base)


def plot_diagnostics(sweep_df, baseline_df, out_base):
    """Supporting figure: what the thresholding does to the statistic.

    (a) enacted count vs the spoke 5-95% band -- the substantive comparison;
    (b) share of spokes in the modal bin -- how collapsed the null is;
    (c) number of distinct spoke values -- the resolution actually available;
    (d) Spearman association with the continuous O_g -- how much of the
        continuous signal survives the cutoff.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    for group in GROUPS:
        sub = sweep_df[sweep_df["group"] == group.key].sort_values("tau")
        kw = dict(color=group.color, linewidth=2.0, marker=group.marker,
                  markersize=4.0, markevery=5, markeredgecolor="white",
                  markeredgewidth=0.8, label=group.label)

        # 5th-95th percentile of the spokes rather than mean +/- sd: the
        # statistic is a bounded integer count, so a symmetric normal band
        # runs below zero and overstates the null's spread.
        ax_a.fill_between(sub["tau"], sub["spoke_p05"], sub["spoke_p95"],
                          color=group.color, alpha=0.13, linewidth=0)
        ax_a.plot(sub["tau"], sub["enacted_count"], **kw)
        ax_a.plot(sub["tau"], sub["spoke_median"], color=group.color,
                  linewidth=1.1, linestyle=(0, (1, 2)))

        ax_b.plot(sub["tau"], sub["pct_spokes_modal"], **kw)
        ax_c.plot(sub["tau"], sub["n_distinct_spokes"], **kw)
        ax_d.plot(sub["tau"], sub["spearman_rho"], **kw)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.axvline(PAPER_CUTOFF, color="0.75", linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)

    _style_axis(ax_a, "Districts clearing τ", "Effectiveness threshold τ")
    ax_a.set_title("Enacted count (solid) vs ensemble median "
                   "(dotted, 5–95% band)",
                   fontsize=9.5, color=INK_PRIMARY, loc="left")
    _style_axis(ax_b, "% of spokes in modal bin", "Effectiveness threshold τ")
    ax_b.set_title("Collapse of the null distribution", fontsize=9.5,
                   color=INK_PRIMARY, loc="left")
    ax_b.set_ylim(0, 104)
    _style_axis(ax_c, "Distinct values across spokes", "Effectiveness threshold τ")
    ax_c.set_title("Resolution available to the threshold test", fontsize=9.5,
                   color=INK_PRIMARY, loc="left")
    _style_axis(ax_d, "Spearman ρ with $O_g$", "Effectiveness threshold τ")
    ax_d.set_title("Signal retained from the continuous functional",
                   fontsize=9.5, color=INK_PRIMARY, loc="left")
    ax_d.set_ylim(-0.05, 1.05)
    ax_a.legend(fontsize=8.5, loc="best", frameon=False)

    return _save(fig, out_base)


def _save(fig, out_base):
    out_base = Path(out_base)
    pdf, png = out_base.with_suffix(".pdf"), out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")
    return pdf, png


# ============================================================
# OUTPUT -- LaTeX
# ============================================================


def save_latex_table(sweep_df, baseline_df, out_path,
                     latex_step=DEFAULT_LATEX_STEP):
    """Publication table: one panel per group, one row per threshold."""
    out_path = Path(out_path)
    n_plans = int(sweep_df["n_plans"].iloc[0])
    m_spokes = n_plans - 1

    step_thousandths = int(round(latex_step * 1000))
    keep = sweep_df[
        (sweep_df["tau"] * 1000).round().astype(int) % step_thousandths == 0
    ]

    lines = [
        r"% Auto-generated by pipelines/ensemble_generation_layer/threshold_sweep.py",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        (r"\caption{Sensitivity of the thresholded opportunity count "
         r"$C_g^{(\tau)}(\pi)=\sum_d \mathbb{I}[p_{gd}(\pi)\ge\tau]$ to the "
         r"effectiveness cutoff $\tau$. Rank and $p$ are the Besag--Clifford "
         rf"exact lower-tail statistics over {m_spokes} reversible-ReCom "
         rf"spokes plus the enacted plan ({n_plans} plans). ``Distinct'' is "
         r"the number of distinct values $C_g^{(\tau)}$ takes across the "
         r"spokes and ``Modal'' the share of spokes in its most common bin: "
         r"both measure how much resolution the cutoff destroys. $\rho$ is "
         r"the Spearman association between $C_g^{(\tau)}$ and the continuous "
         r"functional $O_g$ across the spokes.}"),
        r"\label{tab:threshold-sweep}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (r"$\tau$ & $C^{(\tau)}$ enacted & Rank & $p_{\text{lower}}$ & "
         r"Distinct & Modal (\%) & $\rho$ with $O_g$ \\"),
    ]

    for group in GROUPS:
        base = baseline_df[baseline_df["group"] == group.key].iloc[0]
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textbf{{{group.label}}} "
            rf"-- continuous $O_g={base['enacted']:.3f}$, "
            rf"rank {int(base['rank'])}, $p={base['p_lower']:.4f}$}} \\")
        lines.append(r"\midrule")
        sub = keep[keep["group"] == group.key].sort_values("tau")
        for _, row in sub.iterrows():
            rho = "--" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
            lines.append(
                f"{row['tau']:.2f} & {int(row['enacted_count'])} & "
                f"{int(row['rank'])} & {row['p_lower']:.4f} & "
                f"{int(row['n_distinct_spokes'])} & "
                f"{row['pct_spokes_modal']:.1f} & {rho} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# OUTPUT -- Console
# ============================================================


def print_sweep_summary(sweep_df, baseline_df, alpha=0.05):
    n_plans = int(sweep_df["n_plans"].iloc[0])
    print("\n" + "=" * 72)
    print("THRESHOLD SENSITIVITY SWEEP")
    print("=" * 72)
    print(f"Plans ranked        : {n_plans} (enacted + {n_plans - 1} spokes)")
    print(f"Threshold grid      : {sweep_df['tau'].min():.2f} to "
          f"{sweep_df['tau'].max():.2f} "
          f"({sweep_df['tau'].nunique()} values)")

    for group in GROUPS:
        sub = sweep_df[sweep_df["group"] == group.key].sort_values("tau")
        base = baseline_df[baseline_df["group"] == group.key].iloc[0]
        at_paper = sub[np.isclose(sub["tau"], PAPER_CUTOFF)]
        n_sig = int((sub["p_lower"] <= alpha).sum())

        print("\n" + "-" * 72)
        print(f"{group.label}")
        print("-" * 72)
        print(f"  Continuous  O_g      : {base['enacted']:.4f} "
              f"(rank {int(base['rank'])}/{n_plans}, p = {base['p_lower']:.4f}, "
              f"Z = {base['z']:.2f})")
        if not at_paper.empty:
            row = at_paper.iloc[0]
            print(f"  Thresholded τ = 0.60 : C = {int(row['enacted_count'])} "
                  f"(rank {int(row['rank'])}/{n_plans}, "
                  f"p = {row['p_lower']:.4f})")
        print(f"  Enacted count range  : {int(sub['enacted_count'].min())} to "
              f"{int(sub['enacted_count'].max())} districts across τ")
        print(f"  Enacted rank range   : {int(sub['rank'].min())} to "
              f"{int(sub['rank'].max())} of {n_plans}")
        print(f"  p-value range        : {sub['p_lower'].min():.4f} to "
              f"{sub['p_lower'].max():.4f}")
        print(f"  Significant at {alpha:g}   : {n_sig}/{len(sub)} thresholds"
              + ("  <-- CONCLUSION FLIPS ACROSS τ"
                 if 0 < n_sig < len(sub) else ""))
        print(f"  Distinct spoke values: {int(sub['n_distinct_spokes'].min())} to "
              f"{int(sub['n_distinct_spokes'].max())} "
              f"(continuous O_g: {int(base['n_distinct_spokes'])})")
        print(f"  Modal-bin share      : {sub['pct_spokes_modal'].min():.1f}% to "
              f"{sub['pct_spokes_modal'].max():.1f}% of spokes")
        rho = sub["spearman_rho"].dropna()
        if not rho.empty:
            print(f"  Spearman ρ with O_g  : {rho.min():.3f} to {rho.max():.3f}")
        else:
            print("  Spearman ρ with O_g  : undefined (constant across spokes)")
    print("=" * 72)


# ============================================================
# DRIVER
# ============================================================


def run_sweep(probs_df, run_name=DEFAULT_RUN_NAME, out_dir="outputs",
              taus=None, spokes_df=None, latex_step=DEFAULT_LATEX_STEP,
              alpha=0.05):
    """Compute the sweep, write every artifact, and return the tables.

    Called both by the CLI below and by the tail of
    besag_clifford_vra_opportunity.main(), so a production chain run emits
    the sweep without a second command.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if spokes_df is not None:
        problems = crosscheck_legacy_counts(probs_df, spokes_df)
        if problems:
            print(f"  WARNING: {len(problems)} spoke(s) disagree with the "
                  f"chain's own τ = {PAPER_CUTOFF} counts. The probability "
                  f"table and the scores CSV may come from different runs. "
                  f"First few: {problems[:3]}")
        else:
            print(f"  Cross-check OK: sweep reproduces the chain's "
                  f"τ = {PAPER_CUTOFF} counts for every spoke.")

    sweep_df, baseline_df = compute_sweep(probs_df, taus=taus)
    print_sweep_summary(sweep_df, baseline_df, alpha=alpha)

    sweep_path = out_dir / f"threshold_sweep_{run_name}.csv"
    base_path = out_dir / f"threshold_sweep_baseline_{run_name}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    baseline_df.to_csv(base_path, index=False)
    print(f"\n  Saved: {sweep_path}")
    print(f"  Saved: {base_path}")

    plot_rank_vs_threshold(sweep_df, baseline_df,
                           out_dir / f"threshold_sweep_rank_{run_name}",
                           alpha=alpha)
    plot_diagnostics(sweep_df, baseline_df,
                     out_dir / f"threshold_sweep_diagnostics_{run_name}")
    save_latex_table(sweep_df, baseline_df,
                     out_dir / f"threshold_sweep_{run_name}.tex",
                     latex_step=latex_step)
    return sweep_df, baseline_df


def probs_from_assignments(plans_path, workers=1):
    """Rebuild the per-district probability table from saved plan assignments.

    This is the path for a chain run that finished before the probability
    table existed. It re-runs ONLY the scoring layer (``final_elec_model``
    plus the opportunity reduction) on plans that are already on disk; the
    reversible-ReCom chains -- the part that costs days -- are not touched,
    and the EI posterior is read from the same CSVs the chain read, so the
    reconstructed probabilities are the ones the published p-values came
    from.

    Importing besag_clifford_vra_opportunity loads the full precinct
    parquet and EI inputs at module scope, so this import is deliberately
    local to the function: the fast path must not pay for it.
    """
    import multiprocessing as mp

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import besag_clifford_vra_opportunity as bc  # heavy: loads data at import

    plans_df = pd.read_parquet(plans_path)
    nodes_order = list(bc.graph.nodes())
    if len(nodes_order) != len(plans_df):
        raise ValueError(
            f"{plans_path} has {len(plans_df)} precinct rows but the graph has "
            f"{len(nodes_order)} nodes. The plan parquet was written against a "
            f"different precinct vintage than {bc.PLOT_PATH}."
        )

    # main() wrote the parquet with rows in state_gdf order and columns read
    # off list(graph.nodes()), so position i of a column belongs to node
    # nodes_order[i]. Verify that alignment through the GEO_ID the graph
    # carries rather than trusting position blindly.
    graph_geo_ids = [bc.graph.nodes[n][bc.GEO_ID] for n in nodes_order]
    if list(plans_df.index.astype(str)) != [str(g) for g in graph_geo_ids]:
        raise ValueError(
            f"Row order of {plans_path} does not match the graph's node order "
            f"by {bc.GEO_ID}; refusing to guess an alignment."
        )

    plan_columns = list(plans_df.columns)
    print(f"  Re-scoring {len(plan_columns)} saved plans "
          f"({workers} worker(s))... no chains are re-run.")

    global _RECONSTRUCT_STATE
    _RECONSTRUCT_STATE = (bc, nodes_order, plans_df)

    rows = []
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            for i, part in enumerate(
                    pool.imap_unordered(_score_one_plan, plan_columns)):
                rows.extend(part)
                if (i + 1) % 25 == 0:
                    print(f"    {i + 1}/{len(plan_columns)} plans scored")
    else:
        for i, col in enumerate(plan_columns):
            rows.extend(_score_one_plan(col))
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(plan_columns)} plans scored")

    return pd.DataFrame(rows)


# Module-level handle so forked pool workers inherit the loaded pipeline
# module and the plan table without pickling them per task.
_RECONSTRUCT_STATE = None


def _score_one_plan(plan_column):
    """Score one saved plan; returns its per-district probability rows."""
    bc, nodes_order, plans_df = _RECONSTRUCT_STATE
    from gerrychain import GeographicPartition

    assignment = {node: int(val) for node, val
                  in zip(nodes_order, plans_df[plan_column].to_numpy())}
    partition = GeographicPartition(graph=bc.graph, assignment=assignment,
                                    updaters=bc._build_updaters())
    return bc.district_prob_rows(partition, plan_column)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity sweep for the opportunity functional.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--out-dir", default="outputs",
                        help="Directory holding chain outputs (default: outputs)")
    parser.add_argument("--probs", default=None,
                        help="Explicit path to the district-probability table.")
    parser.add_argument("--from-assignments", action="store_true",
                        help="Rebuild probabilities from the saved plan "
                             "assignments (re-scores plans; no chains re-run).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processes for --from-assignments re-scoring.")
    parser.add_argument("--tau-min", type=float, default=DEFAULT_TAU_MIN)
    parser.add_argument("--tau-max", type=float, default=DEFAULT_TAU_MAX)
    parser.add_argument("--tau-step", type=float, default=DEFAULT_TAU_STEP)
    parser.add_argument("--latex-step", type=float, default=DEFAULT_LATEX_STEP,
                        help="Row spacing of the LaTeX table (default 0.05).")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    probs_path = Path(args.probs) if args.probs else \
        out_dir / f"bc_district_probs_{args.run_name}.parquet"

    if args.from_assignments and not probs_path.exists():
        plans_path = out_dir / f"bc_plan_assignments_{args.run_name}.parquet"
        probs_df = probs_from_assignments(plans_path, workers=args.workers)
        out_dir.mkdir(parents=True, exist_ok=True)
        probs_df.to_parquet(probs_path, index=False)
        print(f"  Cached reconstructed probabilities: {probs_path}")
    else:
        if args.from_assignments:
            print(f"  {probs_path} already exists; using it instead of "
                  f"re-scoring saved plans.")
        probs_df = load_district_probs(probs_path)

    spokes_path = out_dir / f"bc_opportunity_spokes_{args.run_name}.csv"
    spokes_df = pd.read_csv(spokes_path) if spokes_path.exists() else None
    if spokes_df is None:
        print(f"  NOTE: {spokes_path} not found; skipping the legacy "
              f"τ = {PAPER_CUTOFF} cross-check.")

    run_sweep(
        probs_df,
        run_name=args.run_name,
        out_dir=out_dir,
        taus=tau_grid(args.tau_min, args.tau_max, args.tau_step),
        spokes_df=spokes_df,
        latex_step=args.latex_step,
        alpha=args.alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
