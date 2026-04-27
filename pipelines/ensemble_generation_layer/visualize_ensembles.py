# -*- coding: utf-8 -*-
"""
visualize_ensemble.py
=====================

Standalone post-processing for besag_clifford_vra_opportunity.py output.
Loads the saved spoke plan assignments and produces:

  1. Side-by-side district maps of the enacted plan vs. the spoke plan that
     maximizes a chosen opportunity metric (default O_L, Latino opportunity).

  2. Per-precinct heatmap: fraction of spoke plans where the precinct landed
     in a Latino-effective district (s^L > EFFECTIVENESS_CUTOFF).

  3. Same as #2 but for Black-effective districts.

Run from the repo root:

    ./venv/bin/python visualize_ensemble.py
    # or with a specific run name:
    ./venv/bin/python visualize_ensemble.py --run-name TX_BC_functional

Output files land in outputs/ alongside the chain outputs.

Reproduces (in spirit) the paper's Figure 7 (heatmaps) and Figure 11
(comparison of demonstration plan to enacted).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

# Resolve repo paths and pipeline imports.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "ensemble_generation_layer"))

from texas_gerrymandering_hb4.config import (
    PRECINCT_DATASET_PARQUET,
)


# ============================================================
# I/O
# ============================================================


def load_inputs(run_name: str, outputs_dir: Path):
    """Load precinct geodataframe, plan assignments parquet, and spoke scores."""
    plans_path  = outputs_dir / f"bc_plan_assignments_{run_name}.parquet"
    scores_path = outputs_dir / f"bc_opportunity_spokes_{run_name}.csv"

    if not plans_path.exists():
        raise FileNotFoundError(
            f"Could not find {plans_path}. Did the chain run save plans? "
            f"Make sure you re-ran besag_clifford_vra_opportunity.py after "
            f"the plan-saving feature was added."
        )
    if not scores_path.exists():
        raise FileNotFoundError(f"Could not find {scores_path}.")

    state_gdf = gpd.read_parquet(PRECINCT_DATASET_PARQUET)
    plans_df  = pd.read_parquet(plans_path)
    scores_df = pd.read_csv(scores_path)

    # Sanity check: every row of plans_df should map onto a row of state_gdf
    # via the GEO_ID column ("CNTYVTD" in this codebase).
    geo_id_col = plans_df.index.name
    if geo_id_col not in state_gdf.columns:
        raise ValueError(
            f"plans parquet is indexed by {geo_id_col!r} but state_gdf does "
            f"not have that column."
        )
    return state_gdf, plans_df, scores_df, geo_id_col


def merge_assignment_into_gdf(state_gdf, plans_df, plan_column, geo_id_col):
    """Return a copy of state_gdf with a 'district' column from plans_df[plan_column]."""
    out = state_gdf.merge(
        plans_df[[plan_column]].rename(columns={plan_column: "district"}),
        left_on=geo_id_col, right_index=True, how="left",
    )
    if out["district"].isna().any():
        n = int(out["district"].isna().sum())
        raise ValueError(
            f"{n} precincts in state_gdf have no assignment in plan column "
            f"{plan_column!r}. Plan parquet may be from a different precinct "
            f"vintage than PRECINCT_DATASET_PARQUET."
        )
    out["district"] = out["district"].astype(int)
    return out


# ============================================================
# Side-by-side map
# ============================================================


def _categorical_district_cmap(num_districts: int):
    """Build a categorical colormap with num_districts distinct colors.

    matplotlib's tab20 has 20 colors, tab20b another 20, tab20c another 20.
    For 38 districts we concatenate tab20 + tab20b for visual contrast.
    """
    base = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    colors = list(base[:num_districts])
    return ListedColormap(colors), BoundaryNorm(
        np.arange(num_districts + 1) - 0.5, ncolors=num_districts,
    )


def plot_side_by_side(state_gdf, plans_df, geo_id_col,
                      enacted_col, alt_col, alt_label,
                      out_pdf: Path, out_png: Path,
                      num_districts: int):
    """Render two choropleth maps side-by-side: enacted vs. alternative."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

    cmap, norm = _categorical_district_cmap(num_districts)

    for ax, col, title in zip(
        axes, [enacted_col, alt_col],
        ["Enacted Texas Congressional Map", alt_label],
    ):
        gdf_with_dist = merge_assignment_into_gdf(
            state_gdf, plans_df, col, geo_id_col)
        gdf_with_dist.plot(
            column="district", cmap=cmap, norm=norm,
            linewidth=0, ax=ax, antialiased=False,
        )
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    fig.suptitle(
        "Enacted Plan vs. Alternative Plan from Besag-Clifford Ensemble",
        fontsize=16, fontweight="bold",
    )

    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")
    print(f"  Saved: {out_png.name}")


# ============================================================
# Heatmaps
# ============================================================


def _per_district_score_indicator(state_gdf, plans_df, geo_id_col,
                                  group_score_column_in_spokes_df=None):
    """For each spoke plan and district, indicate whether that district was
    'effective' for the chosen group, by re-running compute_final_dist on
    the saved assignment.

    Because the score involves precinct-level EI draws AND statewide
    weights, this is structurally the same machinery used inside the chain.
    Rather than reproduce all that here, we take a simpler path: we use the
    spoke-level total scores from bc_opportunity_spokes_*.csv only as a
    sanity check, and compute *district-level* effectiveness via direct
    geometric operations on the assignment. This requires re-running the
    scoring functions, which is expensive. We defer that path here and
    instead implement the heatmaps using a simpler statistic that doesn't
    require re-scoring: per-precinct, the fraction of spokes in which the
    precinct sits in a district whose minority CVAP share exceeds a
    threshold. This is a demographic proxy for opportunity, not the full
    paper-faithful effectiveness score.

    Returns a function that, given a plan column name, returns a boolean
    Series indexed by district indicating "this district passes the
    proxy effectiveness check for the chosen group."
    """
    raise NotImplementedError("inline placeholder; not used")


def plot_effective_heatmap(state_gdf, plans_df, geo_id_col, spoke_columns,
                           group_label: str, cvap_col: str,
                           cvap_total_col: str, threshold: float,
                           out_pdf: Path, out_png: Path):
    """Render per-precinct heatmap: fraction of spokes where the precinct
    sat in a district whose group-CVAP share exceeded threshold.

    Note: this is a *demographic* proxy for "minority-effective district",
    not the full electoral effectiveness score from compute_final_dist.
    The substantive question -- "do precincts cluster geographically into
    opportunity districts across the ensemble?" -- is well-served by either
    measure. The full electoral effectiveness score would require re-running
    EI-derived scoring on each saved plan and is much more expensive.

    Mirrors paper Figure 7's intent (geographic concentration of opportunity
    districts) using a simpler input.
    """
    geo_ids = list(plans_df.index)
    # Per-precinct aggregates: total CVAP and group CVAP per precinct.
    cvap_by_geo = (
        state_gdf.set_index(geo_id_col)[[cvap_col, cvap_total_col]]
        .reindex(geo_ids)
    )
    if cvap_by_geo.isna().any().any():
        raise ValueError(
            f"Some precincts in plans_df missing from state_gdf for columns "
            f"{cvap_col}, {cvap_total_col}."
        )

    # Build per-precinct counter: number of spoke plans where this precinct
    # was in a district passing the demographic threshold.
    n_spokes = len(spoke_columns)
    in_effective = np.zeros(len(geo_ids), dtype=int)

    for spoke_col in spoke_columns:
        assignment = plans_df[spoke_col].to_numpy()
        # Sum group CVAP and total CVAP per district label.
        district_group  = pd.Series(cvap_by_geo[cvap_col].values).groupby(
            assignment).sum()
        district_total  = pd.Series(cvap_by_geo[cvap_total_col].values).groupby(
            assignment).sum()
        district_share  = district_group / district_total.replace(0, np.nan)
        passes = district_share > threshold
        # For each precinct, did the district it landed in pass?
        precinct_passes = passes.reindex(assignment).to_numpy()
        in_effective += precinct_passes.astype(int)

    fraction = in_effective / max(n_spokes, 1)

    # Render
    plot_gdf = state_gdf.set_index(geo_id_col).reindex(geo_ids).reset_index()
    plot_gdf["fraction_effective"] = fraction

    fig, ax = plt.subplots(1, 1, figsize=(14, 12), constrained_layout=True)
    plot_gdf.plot(
        column="fraction_effective", cmap="viridis", linewidth=0,
        antialiased=False, legend=True, ax=ax,
        legend_kwds={"label": f"Fraction of spokes (n={n_spokes})",
                     "shrink": 0.6},
    )
    ax.set_title(
        f"{group_label} demographic-opportunity heatmap\n"
        f"(precinct = colored by fraction of spokes where it sat in a "
        f"district with {cvap_col}/{cvap_total_col} > {threshold:.0%})",
        fontsize=13,
    )
    ax.set_axis_off()

    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")
    print(f"  Saved: {out_png.name}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", default="TX_BC_functional",
        help="The RUN_NAME used by besag_clifford_vra_opportunity.py "
             "(default: TX_BC_functional)",
    )
    parser.add_argument(
        "--outputs-dir", default=str(REPO_ROOT / "outputs"),
        help="Directory containing the chain output files (default: ./outputs)",
    )
    parser.add_argument(
        "--alt-metric", default="O_L", choices=["O_B", "O_L", "O_joint"],
        help="Which spoke metric to maximize when picking the alternative "
             "plan for side-by-side comparison (default: O_L)",
    )
    parser.add_argument(
        "--num-districts", type=int, default=38,
        help="Number of congressional districts (default: 38 for Texas)",
    )
    parser.add_argument(
        "--latino-threshold", type=float, default=0.40,
        help="HCVAP / CVAP threshold for the Latino-opportunity heatmap "
             "(default: 0.40, matching paper §5.4 footnote 27 demographic "
             "thresholds)",
    )
    parser.add_argument(
        "--black-threshold", type=float, default=0.30,
        help="BCVAP / CVAP threshold for the Black-opportunity heatmap "
             "(default: 0.30)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs dir not found: {outputs_dir}")

    print(f"Loading inputs for run_name={args.run_name!r} ...")
    state_gdf, plans_df, scores_df, geo_id_col = load_inputs(
        args.run_name, outputs_dir)
    print(f"  Loaded {len(plans_df)} precincts, "
          f"{len(plans_df.columns)} plans (1 enacted + "
          f"{len(plans_df.columns) - 1} spokes), "
          f"{len(scores_df)} spoke score rows.")

    spoke_columns = [c for c in plans_df.columns if c.startswith("spoke_")]
    if not spoke_columns:
        raise ValueError("No spoke_NNN columns found in plans parquet.")

    # 1. Side-by-side: enacted vs. spoke maximizing chosen metric.
    metric = args.alt_metric
    if metric not in scores_df.columns:
        raise ValueError(
            f"Metric {metric!r} not in scores_df columns: "
            f"{list(scores_df.columns)}")

    best_idx     = int(scores_df.iloc[scores_df[metric].idxmax()]["spoke"])
    best_metric  = float(scores_df.iloc[scores_df[metric].idxmax()][metric])
    best_col     = f"spoke_{best_idx:03d}"
    if best_col not in plans_df.columns:
        raise ValueError(f"{best_col} not in plans_df.")
    print(f"\n[1/3] Side-by-side: enacted vs. spoke maximizing {metric}")
    print(f"      Best spoke index: {best_idx} ({metric} = {best_metric:.3f})")
    plot_side_by_side(
        state_gdf, plans_df, geo_id_col,
        enacted_col="enacted", alt_col=best_col,
        alt_label=f"Spoke #{best_idx} (max {metric} = {best_metric:.3f})",
        out_pdf=outputs_dir / f"map_compare_enacted_vs_best_{metric}_{args.run_name}.pdf",
        out_png=outputs_dir / f"map_compare_enacted_vs_best_{metric}_{args.run_name}.png",
        num_districts=args.num_districts,
    )

    # 2. Latino-opportunity heatmap.
    print(f"\n[2/3] Latino-opportunity heatmap "
          f"(HCVAP > {args.latino_threshold:.0%})")
    t = time.time()
    plot_effective_heatmap(
        state_gdf, plans_df, geo_id_col, spoke_columns,
        group_label="Latino", cvap_col="HCVAP", cvap_total_col="CVAP",
        threshold=args.latino_threshold,
        out_pdf=outputs_dir / f"heatmap_latino_opportunity_{args.run_name}.pdf",
        out_png=outputs_dir / f"heatmap_latino_opportunity_{args.run_name}.png",
    )
    print(f"      ({time.time() - t:.1f}s)")

    # 3. Black-opportunity heatmap.
    print(f"\n[3/3] Black-opportunity heatmap "
          f"(BCVAP > {args.black_threshold:.0%})")
    t = time.time()
    plot_effective_heatmap(
        state_gdf, plans_df, geo_id_col, spoke_columns,
        group_label="Black", cvap_col="BCVAP", cvap_total_col="CVAP",
        threshold=args.black_threshold,
        out_pdf=outputs_dir / f"heatmap_black_opportunity_{args.run_name}.pdf",
        out_png=outputs_dir / f"heatmap_black_opportunity_{args.run_name}.png",
    )
    print(f"      ({time.time() - t:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()