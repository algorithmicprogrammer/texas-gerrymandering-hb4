# -*- coding: utf-8 -*-
"""
Tests for pipelines/ensemble_generation_layer/threshold_sweep.py.

The sweep is a pure re-reduction of per-district probabilities, so every
quantity it reports can be checked against a hand-computed example. The
cases below pin the three things a reviewer would poke at: the exact
Besag-Clifford rank/p-value conventions, the >= tau boundary (a district
whose probability is exactly tau must count at tau), and the union
semantics of the coalition group.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "ensemble_generation_layer"))

import threshold_sweep as ts  # noqa: E402


def _probs_frame(plan_probs):
    """Build the tidy probability table from {plan: [(p_L, p_B), ...]}."""
    rows = []
    for plan, districts in plan_probs.items():
        for d, (p_l, p_b) in enumerate(districts):
            rows.append({"plan": plan, "district": d,
                         "p_L": p_l, "p_B": p_b, "p_N": np.nan})
    return pd.DataFrame(rows)


# ============================================================
# Grid
# ============================================================


def test_tau_grid_hits_the_paper_cutoff_exactly():
    grid = ts.tau_grid()
    assert len(grid) == 41
    assert grid[0] == 0.40 and grid[-1] == 0.80
    # Not "close to" 0.6 -- the same double as the literal, otherwise a
    # district scoring exactly 0.6 would silently drop out at tau = 0.60.
    assert 0.6 in set(grid.tolist())
    assert grid[20] == 0.6


def test_tau_grid_custom_step():
    grid = ts.tau_grid(0.40, 0.80, 0.05)
    assert np.allclose(grid, [0.40, 0.45, 0.50, 0.55, 0.60,
                              0.65, 0.70, 0.75, 0.80])


def test_tau_grid_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ts.tau_grid(0.4, 0.8, 0.0)
    with pytest.raises(ValueError):
        ts.tau_grid(0.8, 0.4, 0.05)


# ============================================================
# Counting
# ============================================================


def test_threshold_count_includes_probabilities_exactly_at_tau():
    probs = _probs_frame({
        "enacted": [(0.60, 0.10), (0.59, 0.10), (0.61, 0.10)],
        "spoke_000": [(0.10, 0.10), (0.10, 0.10), (0.10, 0.10)],
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    latino = sweep[sweep["group"] == "L"].iloc[0]
    # 0.60 and 0.61 clear the bar; 0.59 does not.
    assert latino["enacted_count"] == 2


def test_coalition_group_is_the_union_of_the_two_groups():
    probs = _probs_frame({
        # d0 Latino-only, d1 Black-only, d2 both, d3 neither.
        "enacted": [(0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.1, 0.1)],
        "spoke_000": [(0.1, 0.1)] * 4,
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    by_group = sweep.set_index("group")["enacted_count"]
    assert by_group["L"] == 2
    assert by_group["B"] == 2
    assert by_group["either"] == 3  # union, not the sum


def test_continuous_baseline_matches_the_pipeline_definitions():
    probs = _probs_frame({
        "enacted": [(0.4, 0.2), (0.3, 0.8)],
        "spoke_000": [(0.1, 0.1), (0.2, 0.2)],
    })
    _, baseline = ts.compute_sweep(probs, taus=[0.60])
    base = baseline.set_index("group")["enacted"]
    assert base["L"] == pytest.approx(0.7)          # sum of p_L
    assert base["B"] == pytest.approx(1.0)          # sum of p_B
    assert base["either"] == pytest.approx(1.2)     # sum of max(p_L, p_B)


# ============================================================
# Besag-Clifford conventions
# ============================================================


def test_rank_and_pvalue_follow_the_chain_convention_with_ties():
    # Enacted has 1 Latino district; spokes have 1, 1, 2, 3.
    probs = _probs_frame({
        "enacted": [(0.9, 0.0), (0.1, 0.0), (0.1, 0.0)],
        "spoke_000": [(0.9, 0.0), (0.1, 0.0), (0.1, 0.0)],
        "spoke_001": [(0.9, 0.0), (0.1, 0.0), (0.1, 0.0)],
        "spoke_002": [(0.9, 0.0), (0.9, 0.0), (0.1, 0.0)],
        "spoke_003": [(0.9, 0.0), (0.9, 0.0), (0.9, 0.0)],
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]

    assert row["enacted_count"] == 1
    assert row["n_plans"] == 5                    # enacted + 4 spokes
    # Ties count toward the lower tail: #{spokes <= 1} = 2, so rank 3 and
    # p = (2 + 1) / (4 + 1), matching compute_bc_pvalues in the chain script.
    assert row["rank"] == 3
    assert row["p_lower"] == pytest.approx(3 / 5)
    # Tie band: without the two ties the enacted plan would rank 1st.
    assert row["rank_min"] == 1
    assert row["n_tied"] == 2
    # #{spokes >= 1} = 4 -> upper tail (4 + 1) / 5.
    assert row["p_upper"] == pytest.approx(1.0)


def test_most_extreme_plan_gets_rank_one_and_the_pvalue_floor():
    probs = _probs_frame({
        "enacted": [(0.1, 0.1), (0.1, 0.1)],
        **{f"spoke_{i:03d}": [(0.9, 0.1), (0.9, 0.1)] for i in range(9)},
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]
    assert row["rank"] == 1
    assert row["p_lower"] == pytest.approx(1 / 10)  # 1 / (M + 1)


# ============================================================
# Discreteness diagnostics
# ============================================================


def test_distinct_and_modal_bin_diagnostics():
    # Spoke counts at tau = 0.6: 1, 1, 1, 2 -> 2 distinct values, mode 1 at 75%.
    probs = _probs_frame({
        "enacted": [(0.9, 0.0), (0.1, 0.0)],
        "spoke_000": [(0.9, 0.0), (0.1, 0.0)],
        "spoke_001": [(0.9, 0.0), (0.1, 0.0)],
        "spoke_002": [(0.9, 0.0), (0.1, 0.0)],
        "spoke_003": [(0.9, 0.0), (0.9, 0.0)],
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]
    assert row["n_distinct_spokes"] == 2
    assert row["modal_count"] == 1
    assert row["pct_spokes_modal"] == pytest.approx(75.0)


def test_association_is_nan_when_every_spoke_lands_in_one_bin():
    # Every spoke has exactly 1 Latino district at tau = 0.6, but the
    # continuous O_L differs between them: the cutoff has destroyed all the
    # variation, so the association is undefined rather than 0.
    probs = _probs_frame({
        "enacted": [(0.9, 0.0), (0.1, 0.0)],
        "spoke_000": [(0.90, 0.0), (0.10, 0.0)],
        "spoke_001": [(0.95, 0.0), (0.20, 0.0)],
        "spoke_002": [(0.99, 0.0), (0.30, 0.0)],
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]
    assert row["n_distinct_spokes"] == 1
    assert np.isnan(row["spearman_rho"])
    assert np.isnan(row["pearson_r"])


def test_association_recovers_a_monotone_relationship():
    plans = {"enacted": [(0.10, 0.0), (0.10, 0.0), (0.10, 0.0)]}
    # Spoke k has k Latino-effective districts and a matching larger O_L.
    for k in range(4):
        plans[f"spoke_{k:03d}"] = [(0.9 if d < k else 0.1, 0.0)
                                   for d in range(3)]
    sweep, _ = ts.compute_sweep(_probs_frame(plans), taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]
    assert row["spearman_rho"] == pytest.approx(1.0)
    assert row["pearson_r"] == pytest.approx(1.0)


# ============================================================
# Input handling
# ============================================================


def test_plans_with_missing_probabilities_are_dropped():
    probs = _probs_frame({
        "enacted": [(0.9, 0.1), (0.1, 0.1)],
        "spoke_000": [(0.9, 0.1), (0.1, 0.1)],
        "spoke_001": [(np.nan, np.nan), (0.1, 0.1)],   # unscorable plan
    })
    sweep, _ = ts.compute_sweep(probs, taus=[0.60])
    row = sweep[sweep["group"] == "L"].iloc[0]
    assert row["n_plans"] == 2  # enacted + the one scorable spoke


def test_missing_enacted_plan_is_an_error():
    probs = _probs_frame({"spoke_000": [(0.9, 0.1)]})
    with pytest.raises(ValueError, match="enacted"):
        ts.compute_sweep(probs, taus=[0.60])


def test_load_district_probs_round_trips_parquet(tmp_path):
    probs = _probs_frame({
        "enacted": [(0.9, 0.1)],
        "spoke_000": [(0.1, 0.1)],
    })
    path = tmp_path / "probs.parquet"
    probs.to_parquet(path, index=False)
    loaded = ts.load_district_probs(path)
    pd.testing.assert_frame_equal(loaded, probs)


def test_load_district_probs_reports_a_missing_file():
    with pytest.raises(FileNotFoundError, match="from-assignments"):
        ts.load_district_probs("does/not/exist.parquet")


# ============================================================
# Cross-check against the chain's own tau = 0.6 columns
# ============================================================


def _legacy_fixture():
    probs = _probs_frame({
        "enacted": [(0.9, 0.1), (0.1, 0.1)],
        "spoke_000": [(0.9, 0.7), (0.1, 0.1)],
        "spoke_001": [(0.1, 0.1), (0.1, 0.1)],
    })
    spokes = pd.DataFrame([
        {"spoke": 0, "hisp_eff": 1, "black_eff": 1, "distinct_eff": 1},
        {"spoke": 1, "hisp_eff": 0, "black_eff": 0, "distinct_eff": 0},
    ])
    return probs, spokes


def test_crosscheck_passes_on_consistent_inputs():
    probs, spokes = _legacy_fixture()
    assert ts.crosscheck_legacy_counts(probs, spokes) == []


def test_crosscheck_flags_a_mismatched_scores_csv():
    probs, spokes = _legacy_fixture()
    spokes.loc[0, "hisp_eff"] = 4
    problems = ts.crosscheck_legacy_counts(probs, spokes)
    assert len(problems) == 1
    assert "spoke_000 hisp_eff: chain=4 sweep=1" in problems[0]


# ============================================================
# End-to-end artifacts
# ============================================================


def test_run_sweep_writes_every_artifact(tmp_path):
    rng = np.random.default_rng(0)
    plans = {"enacted": [(0.2, 0.1)] * 8}
    for k in range(40):
        plans[f"spoke_{k:03d}"] = [
            (float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
            for _ in range(8)
        ]
    sweep, baseline = ts.run_sweep(
        _probs_frame(plans), run_name="TEST", out_dir=tmp_path,
        taus=ts.tau_grid(0.40, 0.80, 0.05),
    )

    assert len(sweep) == len(ts.GROUPS) * 9
    assert len(baseline) == len(ts.GROUPS)
    for name in [
        "threshold_sweep_TEST.csv",
        "threshold_sweep_baseline_TEST.csv",
        "threshold_sweep_TEST.tex",
        "threshold_sweep_rank_TEST.pdf",
        "threshold_sweep_rank_TEST.png",
        "threshold_sweep_diagnostics_TEST.pdf",
        "threshold_sweep_diagnostics_TEST.png",
    ]:
        assert (tmp_path / name).exists(), f"missing artifact {name}"

    tex = (tmp_path / "threshold_sweep_TEST.tex").read_text()
    assert r"\begin{tabular}" in tex and r"\end{table}" in tex
    # The column header, plus one panel header and nine threshold rows per group.
    assert tex.count(r"\\") == 1 + len(ts.GROUPS) * (1 + 9)
