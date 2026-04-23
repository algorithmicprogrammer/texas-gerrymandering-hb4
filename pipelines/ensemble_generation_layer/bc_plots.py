# -*- coding: utf-8 -*-
"""
bc_plots.py
===========
Visualize the results of besag_clifford_vra.py.

Produces:
  1. Distribution histograms for each effectiveness metric,
     with the enacted map's score marked and p-values annotated.
  2. A summary table suitable for inclusion in a report.

Usage:
  python bc_plots.py --run TX_BC_parallel --mode statewide
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DIR = ''

def load_results(run_name):
    spoke_df  = pd.read_csv(DIR + f"outputs/bc_spoke_scores_{run_name}.csv")
    pval_df   = pd.read_csv(DIR + f"outputs/bc_pvalues_{run_name}.csv")
    results   = pval_df.iloc[0].to_dict()
    return spoke_df, results


def plot_distributions(spoke_df, results, run_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Besag-Clifford Exact Significance Test — Texas Congressional Districts\n"
        f"Model mode: {results['mode']}  |  "
        f"M={int(results['M_spokes'])} spokes  |  "
        f"L_hub={int(results['L_hub'])}, L_spoke={int(results['L_spoke'])}",
        fontsize=11, y=1.02
    )

    configs = [
        ('hisp',     'Latino', '#1a7abf'),
        ('black',    'Black',  '#c0392b'),
        ('distinct', 'Distinct (B or L)', '#27ae60'),
    ]

    for ax, (key, label, color) in zip(axes, configs):
        enacted = results[f'enacted_{key}']
        p_low   = results[f'p_{key}_lower']
        p_high  = results[f'p_{key}_upper']
        values  = spoke_df[key]

        # histogram
        bins = range(int(values.min()), int(values.max()) + 2)
        counts, edges = np.histogram(values, bins=bins)
        ax.bar(edges[:-1], counts / len(values), color=color, alpha=0.6,
               width=0.8, align='edge', label='Spoke endpoints')

        # enacted map marker
        ax.axvline(enacted, color='black', linewidth=2.5, linestyle='--', label=f'Enacted map ({enacted})')

        # annotation
        pct_below = 100 * (values < enacted).mean()
        pct_above = 100 * (values > enacted).mean()

        ax.set_title(f"{label}-effective districts", fontsize=11)
        ax.set_xlabel("# effective districts", fontsize=10)
        ax.set_ylabel("Proportion of spokes", fontsize=10)

        textstr = (
            f"Enacted: {enacted}\n"
            f"Mean: {values.mean():.2f}  Median: {values.median():.1f}\n"
            f"p (lower ←): {p_low:.4f}\n"
            f"p (upper →): {p_high:.4f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=8.5,
                verticalalignment='top', bbox=props)

        # significance shading
        if p_low < 0.05:
            ax.set_facecolor('#fff0f0')
            ax.text(0.5, 0.5, 'SIGNIFICANT\n(lower tail)',
                    transform=ax.transAxes, fontsize=9, color='red',
                    ha='center', va='center', alpha=0.25,
                    fontweight='bold')

        ax.legend(fontsize=8)
        ax.set_xticks(list(bins))

    plt.tight_layout()
    out_path = DIR + f"outputs/bc_distributions_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {out_path}")
    plt.close()


def print_summary_table(results):
    """Print a clean ASCII table suitable for copy-paste into a report."""
    M = int(results['M_spokes'])
    print()
    print("=" * 72)
    print(f"{'BESAG-CLIFFORD EXACT TEST — SUMMARY TABLE':^72}")
    print("=" * 72)
    print(f"{'Metric':<22} {'Enacted':>8} {'Mean':>8} {'Median':>8} "
          f"{'p (lower)':>12} {'p (upper)':>12}")
    print("-" * 72)
    for key, label in [('hisp', 'Latino effective'),
                        ('black', 'Black effective'),
                        ('distinct', 'Distinct effective')]:
        enacted = results[f'enacted_{key}']
        mean    = results[f'mean_{key}']
        median  = results[f'median_{key}']
        p_low   = results[f'p_{key}_lower']
        p_high  = results[f'p_{key}_upper']
        sig_low  = " **" if p_low  < 0.05 else ""
        sig_high = " **" if p_high < 0.05 else ""
        print(f"{label:<22} {enacted:>8} {mean:>8.2f} {median:>8.1f} "
              f"{p_low:>10.4f}{sig_low:<2} {p_high:>10.4f}{sig_high:<2}")
    print("-" * 72)
    print(f"  ** p < 0.05   |   M = {M} spokes   |   Mode: {results['mode']}")
    print(f"  p-values are exact by Besag-Clifford (1989) Prop. 3.3")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',  default='TX_BC_parallel', help='run name')
    args = parser.parse_args()

    spoke_df, results = load_results(args.run)
    print_summary_table(results)
    plot_distributions(spoke_df, results, args.run)


if __name__ == "__main__":
    main()
