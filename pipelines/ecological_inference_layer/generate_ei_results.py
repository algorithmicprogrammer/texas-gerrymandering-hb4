#!/usr/bin/env python3
"""
generate_ei_results.py
----------------------
Build publication-ready tables and figures for the EI results section, from
the long-form CSVs produced by run_ei.py.

Inputs (from --ei-dir, default ./ei_outputs):
    statewide_rxc_EI_preferences.csv   (election, group, candidate_of_choice,
                                        mean_support, frac_draws_preferred)
    mean_prec_vote_counts.csv          (CNTYVTD, election, group, candidate,
                                        mean_count)
    prec_count_quants.csv              (CNTYVTD, election, group, candidate,
                                        octile, quantile, value)

Outputs (to --out-dir, default ./results):
    tables/
        coc_summary.{csv,tex,md}              candidate-of-choice by group x election
        cohesion_summary.{csv,tex,md}         mean support for each group's CoC
        polarization_summary.{csv,tex,md}     full group x candidate matrix for generals
    figures/
        polarization_general_elections.{pdf,png}
        coc_support_by_group.{pdf,png}
        cohesion_heatmap.{pdf,png}
        within_precinct_uncertainty.{pdf,png}

Run:
    python generate_ei_results.py
    python generate_ei_results.py --ei-dir ei_outputs --out-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

# Logical group ordering for tables/plots
GROUP_ORDER = ["Hispanic", "Black", "White", "Asian", "AMIN"]

# Pretty labels for the AMIN group in tables/figures
GROUP_LABEL = {
    "Hispanic": "Hispanic",
    "Black":    "Black",
    "White":    "White",
    "Asian":    "Asian",
    "AMIN":     "AI/AN",      # American Indian / Alaska Native
}

# Okabe–Ito colorblind-safe palette, mapped to groups
GROUP_COLORS = {
    "Hispanic": "#E69F00",  # orange
    "Black":    "#0072B2",  # blue
    "White":    "#999999",  # gray
    "Asian":    "#D55E00",  # vermilion
    "AMIN":     "#009E73",  # bluish green
}

# Election ordering: generals first (left-to-right by office),
# then Dem primaries, then Rep primaries.
ELECTION_ORDER = [
    "24G_President", "24G_US_Sen", "24G_RR_Comm_1",
    "24P_PresD",     "24P_SenD",   "24P_RRCD",
    "24P_PresR",     "24P_SenR",   "24P_RRCR",
]

ELECTION_LABEL = {
    "24G_President": "2024 General — President",
    "24G_US_Sen":    "2024 General — U.S. Senate",
    "24G_RR_Comm_1": "2024 General — Railroad Comm.",
    "24P_PresD":     "2024 Dem Primary — President",
    "24P_PresR":     "2024 Rep Primary — President",
    "24P_SenD":      "2024 Dem Primary — U.S. Senate",
    "24P_SenR":      "2024 Rep Primary — U.S. Senate",
    "24P_RRCD":      "2024 Dem Primary — Railroad Comm.",
    "24P_RRCR":      "2024 Rep Primary — Railroad Comm.",
}

# Short labels for compact tables/axes
ELECTION_SHORT = {
    "24G_President": "Pres (G)",
    "24G_US_Sen":    "Senate (G)",
    "24G_RR_Comm_1": "RR Comm (G)",
    "24P_PresD":     "Pres (DP)",
    "24P_PresR":     "Pres (RP)",
    "24P_SenD":      "Senate (DP)",
    "24P_SenR":      "Senate (RP)",
    "24P_RRCD":      "RR Comm (DP)",
    "24P_RRCR":      "RR Comm (RP)",
}

# Matplotlib styling for academic papers
plt.rcParams.update({
    "font.family":        "DejaVu Sans",  # broadly available; swap for Times if installed
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.titlesize":   12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
    "pdf.fonttype":       42,   # editable text in LaTeX/Illustrator
    "ps.fonttype":        42,
})


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _ordered_elections(present: set[str]) -> list[str]:
    """Return ELECTION_ORDER intersected with elections actually in the data."""
    ordered = [e for e in ELECTION_ORDER if e in present]
    extra = sorted(present - set(ELECTION_ORDER))
    return ordered + extra


def _candidate_short(cand: str) -> str:
    """Trim the trailing _24G / _24P from candidate names for display."""
    for suf in ("_24G", "_24P"):
        if cand.endswith(suf):
            return cand[: -len(suf)]
    return cand


def _wrap_small(latex: str) -> str:
    """Wrap the tabular in \\small{...} so it fits paper-column widths."""
    # Pandas emits \begin{table} ... \begin{tabular}{...} ... \end{tabular} ... \end{table}.
    # Wrapping the tabular in {\small ...} is the least invasive way to shrink it.
    latex = latex.replace("\\begin{tabular}", "{\\small\n\\begin{tabular}")
    latex = latex.replace("\\end{tabular}", "\\end{tabular}\n}")
    return latex


def _to_markdown_table(df: pd.DataFrame, *, index: bool = False) -> str:
    """Minimal GitHub-flavoured Markdown table writer (no `tabulate` needed).

    Handles MultiIndex columns by joining levels with " / ". Cells are
    str()-ified; NaN renders as an em-dash. Pure left-alignment.
    """
    if isinstance(df.columns, pd.MultiIndex):
        headers = [" / ".join(str(x) for x in tup if x != "") for tup in df.columns]
    else:
        headers = [str(c) for c in df.columns]

    if index:
        idx_names = [str(n) if n is not None else "" for n in df.index.names]
        headers = idx_names + headers

    def _fmt(x) -> str:
        if pd.isna(x):
            return "—"
        return str(x)

    rows = []
    for tup, row in df.iterrows():
        cells = [_fmt(v) for v in row.tolist()]
        if index:
            idx_vals = tup if isinstance(tup, tuple) else (tup,)
            cells = [_fmt(v) for v in idx_vals] + cells
        rows.append(cells)

    # Column widths (cap to keep file readable on disk)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    widths = [min(w, 60) for w in widths]

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"

    out = [_line(headers), sep] + [_line(r) for r in rows]
    return "\n".join(out) + "\n"


def _save_table(df: pd.DataFrame, stem: Path, *, index: bool = False,
                caption: str = "", label: str = "",
                column_format: str | None = None,
                float_format: str | None = "%.2f",
                small: bool = True) -> None:
    """Write CSV, LaTeX (booktabs), and Markdown versions of a table.

    column_format
        Optional explicit LaTeX column spec (e.g. "llllll"). If None, defaults
        to left-align all columns — the right thing for text-heavy tables.
    float_format
        Format string passed to to_latex for numeric cells. Default "%.2f".
        Set to None to use pandas defaults.
    small
        If True (default), wraps the tabular in {\\small ...} so wide tables
        fit a single-column page width.
    """
    df.to_csv(stem.with_suffix(".csv"), index=index)

    if column_format is None:
        n_idx = df.index.nlevels if index else 0
        n_col = df.shape[1]
        column_format = "l" * (n_idx + n_col)

    latex = df.to_latex(
        index=index,
        escape=False,
        longtable=False,
        column_format=column_format,
        caption=caption or None,
        label=label or None,
        float_format=float_format,
        na_rep="--",
    )
    if small:
        latex = _wrap_small(latex)
    stem.with_suffix(".tex").write_text(latex)

    stem.with_suffix(".md").write_text(_to_markdown_table(df, index=index))


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_inputs(ei_dir: Path):
    sw = pd.read_csv(ei_dir / "statewide_rxc_EI_preferences.csv")
    mc = pd.read_csv(ei_dir / "mean_prec_vote_counts.csv",
                     dtype={"CNTYVTD": str})
    qd_path = ei_dir / "prec_count_quants.csv"
    qd = pd.read_csv(qd_path, dtype={"CNTYVTD": str}) if qd_path.exists() else None

    # Drop any groups not in our ordering (e.g. if you added one)
    sw = sw[sw["group"].isin(GROUP_ORDER)].copy()
    mc = mc[mc["group"].isin(GROUP_ORDER)].copy()
    if qd is not None:
        qd = qd[qd["group"].isin(GROUP_ORDER)].copy()

    # --- Defensive dedup -----------------------------------------------------
    # run_ei.py's resume logic appends to existing CSVs without dedup, so if
    # any election was re-run after a partial crash you can end up with
    # multiple rows for the same key. Keep the LAST occurrence (= most recent
    # run) and warn loudly so this doesn't silently bias the plots.

    sw_dups = sw.duplicated(subset=["election", "group"], keep=False)
    if sw_dups.any():
        examples = (sw[sw_dups]
                    .groupby(["election", "group"]).size()
                    .reset_index(name="n").sort_values("n", ascending=False)
                    .head(5))
        print(f"  WARNING: statewide_rxc_EI_preferences has {int(sw_dups.sum())} "
              f"rows with duplicate (election, group) keys. Keeping the LAST "
              f"occurrence per key. Top duplicates:")
        print(examples.to_string(index=False))
        sw = sw.drop_duplicates(subset=["election", "group"], keep="last").copy()

    mc_keys = ["CNTYVTD", "election", "group", "candidate"]
    n_dupes = len(mc) - mc.drop_duplicates(subset=mc_keys).shape[0]
    if n_dupes > 0:
        print(f"  WARNING: mean_prec_vote_counts has {n_dupes:,} duplicate "
              f"(CNTYVTD, election, group, candidate) rows. Keeping last.")
        mc = mc.drop_duplicates(subset=mc_keys, keep="last").copy()

    if qd is not None:
        qd_keys = ["CNTYVTD", "election", "group", "candidate", "octile"]
        n_dupes = len(qd) - qd.drop_duplicates(subset=qd_keys).shape[0]
        if n_dupes > 0:
            print(f"  WARNING: prec_count_quants has {n_dupes:,} duplicate "
                  f"rows. Keeping last.")
            qd = qd.drop_duplicates(subset=qd_keys, keep="last").copy()

    return sw, mc, qd


def compute_full_support(mean_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean precinct counts to statewide group-by-candidate vote shares.

    Returns long-form DataFrame: election, group, candidate, support
        where support = sum_p mean_count(p,g,c) / sum_c sum_p mean_count(p,g,c)

    This is the within-group vote share among voters (excluding abstain), which
    is the standard metric for racially-polarized-voting analysis.
    """
    grouped = (
        mean_counts.groupby(["election", "group", "candidate"], as_index=False)
        ["mean_count"].sum()
    )
    totals = (
        grouped.groupby(["election", "group"], as_index=False)
        ["mean_count"].sum().rename(columns={"mean_count": "group_total"})
    )
    out = grouped.merge(totals, on=["election", "group"])
    # avoid divide-by-zero
    out["support"] = np.where(out["group_total"] > 0,
                              out["mean_count"] / out["group_total"], np.nan)
    return out[["election", "group", "candidate", "support"]]


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #

def table_coc_summary(sw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Wide CoC table: rows = elections, columns = groups.
    Each cell shows "<Candidate> (<support>%)". The literal % gets escaped on
    the LaTeX side so that to_latex with escape=False still compiles.
    """
    elections = _ordered_elections(set(sw["election"]))
    rows_csv = []  # CSV/MD version: plain "%" is fine
    rows_tex = []  # TeX version: escape the percent sign
    for elec in elections:
        elec_label = ELECTION_SHORT.get(elec, elec)
        row_csv = {"Election": elec_label}
        row_tex = {"Election": elec_label}
        for g in GROUP_ORDER:
            sel = sw[(sw["election"] == elec) & (sw["group"] == g)]
            if sel.empty:
                row_csv[GROUP_LABEL[g]] = "—"
                row_tex[GROUP_LABEL[g]] = "--"
                continue
            r = sel.iloc[0]
            cand = _candidate_short(r["candidate_of_choice"])
            sup = 100 * r["mean_support"]
            row_csv[GROUP_LABEL[g]] = f"{cand} ({sup:.0f}%)"
            row_tex[GROUP_LABEL[g]] = f"{cand} ({sup:.0f}\\%)"
        rows_csv.append(row_csv)
        rows_tex.append(row_tex)

    df_csv = pd.DataFrame(rows_csv)
    df_tex = pd.DataFrame(rows_tex)

    # Write CSV/MD from the unescaped version
    df_csv.to_csv(out_dir / "coc_summary.csv", index=False)
    (out_dir / "coc_summary.md").write_text(_to_markdown_table(df_csv, index=False))

    # Write LaTeX from the escaped version, with explicit l-alignment
    n_cols = df_tex.shape[1]
    latex = df_tex.to_latex(
        index=False, escape=False, longtable=False,
        column_format="l" * n_cols,
        caption=("Statewide candidate of choice for each demographic group across "
                 "elections, with mean posterior support (\\%) in parentheses. "
                 "Estimates are from the RxC Multinomial--Dirichlet model."),
        label="tab:coc_summary",
        na_rep="--",
    )
    latex = _wrap_small(latex)
    (out_dir / "coc_summary.tex").write_text(latex)

    return df_csv


def table_cohesion_summary(sw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Cohesion: mean support of each group's CoC, with posterior probability that
    candidate is the CoC. Rows = elections, columns = groups (multi-level).
    """
    elections = _ordered_elections(set(sw["election"]))
    # Build with flat string column names first, then promote to MultiIndex.
    flat_cols = ["Election"]
    for g in GROUP_ORDER:
        flat_cols.append(f"{GROUP_LABEL[g]}__support")
        flat_cols.append(f"{GROUP_LABEL[g]}__P(CoC)")

    records = []
    for elec in elections:
        rec = {"Election": ELECTION_SHORT.get(elec, elec)}
        for g in GROUP_ORDER:
            sel = sw[(sw["election"] == elec) & (sw["group"] == g)]
            if sel.empty:
                rec[f"{GROUP_LABEL[g]}__support"] = "--"
                rec[f"{GROUP_LABEL[g]}__P(CoC)"] = "--"
            else:
                r = sel.iloc[0]
                # Pre-format as strings so to_latex doesn't homogenise precision
                rec[f"{GROUP_LABEL[g]}__support"] = f"{100 * r['mean_support']:.1f}"
                rec[f"{GROUP_LABEL[g]}__P(CoC)"]  = f"{r['frac_draws_preferred']:.2f}"
        records.append(rec)

    out = pd.DataFrame(records, columns=flat_cols)
    # Promote to a MultiIndex header: ("", "Election") and (Group, sub-metric)
    tuples = [("", "Election")] + [
        (GROUP_LABEL[g], k) for g in GROUP_ORDER for k in ("support", "P(CoC)")
    ]
    out.columns = pd.MultiIndex.from_tuples(tuples)

    n_cols = 1 + 2 * len(GROUP_ORDER)
    _save_table(
        out, out_dir / "cohesion_summary",
        column_format="l" + "r" * (n_cols - 1),
        float_format=None,
        caption=("Mean within-group support (\\%) for each group's candidate of "
                 "choice, with the posterior probability $P(\\mathrm{CoC})$ that the "
                 "candidate is the group's most-preferred. Higher cohesion and "
                 "$P(\\mathrm{CoC}) \\to 1$ together imply strong group consensus."),
        label="tab:cohesion_summary",
    )

    # Center the group-header multicolumns (pandas defaults to {r}, which puts
    # the group name over only the second sub-column).
    tex_path = out_dir / "cohesion_summary.tex"
    tex_path.write_text(
        tex_path.read_text().replace("\\multicolumn{2}{r}", "\\multicolumn{2}{c}")
    )
    return out


def table_polarization_summary(full_support: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    For 2-candidate general elections: stacked table with Republican / Democrat
    columns shared across elections. Each row is (Election, Group); cells are
    within-group vote shares (excluding abstain).

    Inferring R vs D: candidate names follow the pattern "<Name><R|D>_24G",
    so the party letter immediately precedes the underscore.
    """
    generals = [e for e in _ordered_elections(set(full_support["election"]))
                if e.startswith("24G_")]

    rows = []
    for elec in generals:
        sub = full_support[full_support["election"] == elec]
        cands = list(sub["candidate"].unique())

        # Identify R and D candidates
        def party(c: str) -> str:
            stem = c.split("_")[0]      # e.g. HarrisD
            return stem[-1]              # 'D' or 'R'

        d_cands = [c for c in cands if party(c) == "D"]
        r_cands = [c for c in cands if party(c) == "R"]
        if len(d_cands) != 1 or len(r_cands) != 1:
            # Skip non-2-candidate generals — they don't fit the schema
            continue
        d_cand, r_cand = d_cands[0], r_cands[0]

        for g in GROUP_ORDER:
            d_share = sub[(sub["group"] == g) & (sub["candidate"] == d_cand)]
            r_share = sub[(sub["group"] == g) & (sub["candidate"] == r_cand)]
            if d_share.empty or r_share.empty:
                continue
            d_pct = 100 * float(d_share["support"].iloc[0])
            r_pct = 100 * float(r_share["support"].iloc[0])
            rows.append({
                "Election": ELECTION_LABEL.get(elec, elec),
                "Group":    GROUP_LABEL[g],
                "Democrat": f"{_candidate_short(d_cand)} ({d_pct:.1f})",
                "Republican": f"{_candidate_short(r_cand)} ({r_pct:.1f})",
                # ASCII minus in CSV/MD; LaTeX gets a math-mode column header
                "RPV gap": f"{d_pct - r_pct:+.1f}",
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    n_cols = out.shape[1]
    _save_table(
        out, out_dir / "polarization_summary",
        column_format="ll" + "r" * (n_cols - 2),
        float_format=None,
        caption=("Within-group vote shares (\\%) by candidate in the 2024 general "
                 "elections. Estimates are derived by aggregating posterior-mean "
                 "precinct vote counts to the statewide level. The right-most "
                 "column is the Democratic minus Republican vote-share gap; "
                 "large positive (negative) gaps for one group alongside large "
                 "negative (positive) gaps for another are direct evidence of "
                 "racially polarized voting."),
        label="tab:polarization_summary",
    )

    # In the LaTeX output, dress up the "RPV gap" column header with math.
    tex_path = out_dir / "polarization_summary.tex"
    tex_path.write_text(
        tex_path.read_text().replace(
            "RPV gap", "RPV gap ($D\\!-\\!R$)"
        )
    )
    return out


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def fig_polarization_generals(full_support: pd.DataFrame, out_dir: Path) -> None:
    """
    Headline figure: for each general election, horizontal bars showing each
    group's vote share for the two candidates. One panel per general.
    """
    generals = [e for e in _ordered_elections(set(full_support["election"]))
                if e.startswith("24G_")]
    if not generals:
        return

    n = len(generals)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, elec in zip(axes, generals):
        sub = full_support[full_support["election"] == elec]
        cands = sorted(sub["candidate"].unique())
        # Try to put R first, D second for canonical orientation
        cands = sorted(cands, key=lambda c: ("D" not in c.split("_")[0], c))

        y_pos = np.arange(len(GROUP_ORDER))
        width = 0.38

        for i, cand in enumerate(cands):
            offset = (-width / 2) if i == 0 else (width / 2)
            shares = [
                float(sub[(sub["group"] == g) & (sub["candidate"] == cand)]
                      ["support"].iloc[0]) * 100 if not sub[
                    (sub["group"] == g) & (sub["candidate"] == cand)
                ].empty else np.nan
                for g in GROUP_ORDER
            ]
            color = "#0072B2" if "D" in cand.split("_")[0] else "#D55E00"
            bars = ax.barh(y_pos + offset, shares, height=width,
                           color=color, edgecolor="white", linewidth=0.6,
                           label=_candidate_short(cand))
            for bar, v in zip(bars, shares):
                if np.isnan(v):
                    continue
                ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_xlabel("Within-group vote share")
        ax.set_title(ELECTION_LABEL.get(elec, elec), pad=8)
        ax.legend(loc="lower right", frameon=False)
        ax.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Racially polarized voting — 2024 Texas general elections",
        y=1.01, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "polarization_general_elections.pdf")
    fig.savefig(out_dir / "polarization_general_elections.png")
    plt.close(fig)


def fig_coc_support_by_group(sw: pd.DataFrame, out_dir: Path) -> None:
    """
    Compact dot plot: for each (election, group), plot the mean support level
    of the group's CoC, colored by group. Annotated with frac_draws_preferred.
    """
    elections = _ordered_elections(set(sw["election"]))

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(elections) + 1.3))

    y_positions = np.arange(len(elections))[::-1]  # top-to-bottom
    n_groups = len(GROUP_ORDER)
    dy = 0.14

    for gi, g in enumerate(GROUP_ORDER):
        offset = (gi - (n_groups - 1) / 2) * dy
        x = []
        y = []
        labels = []
        for ei, elec in enumerate(elections):
            sel = sw[(sw["election"] == elec) & (sw["group"] == g)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            x.append(100 * r["mean_support"])
            y.append(y_positions[ei] + offset)
            labels.append(_candidate_short(r["candidate_of_choice"]))

        # Outline thickness encodes posterior probability of CoC
        sizes = []
        edge = []
        for ei, elec in enumerate(elections):
            sel = sw[(sw["election"] == elec) & (sw["group"] == g)]
            if sel.empty:
                continue
            p = float(sel.iloc[0]["frac_draws_preferred"])
            sizes.append(60 + 80 * p)   # base 60, max 140
            edge.append("black" if p >= 0.95 else "0.6")

        ax.scatter(x, y, s=sizes, color=GROUP_COLORS[g],
                   edgecolor=edge, linewidths=0.8, label=GROUP_LABEL[g], zorder=3)

        # Candidate labels
        for xi, yi, lbl in zip(x, y, labels):
            ax.text(xi + 1.5, yi, lbl, va="center", fontsize=7.5,
                    color="0.25")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([ELECTION_SHORT.get(e, e) for e in elections])
    ax.set_xlim(0, 110)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Mean posterior support for candidate of choice")
    ax.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=n_groups, frameon=False)

    ax.set_title(
        "Candidate of choice and posterior support, by group and election",
        pad=10, fontweight="bold",
    )
    # Footnote
    ax.text(
        1.0, -0.22,
        "Marker size scales with $P(\\mathrm{CoC})$, the posterior probability "
        "that the named candidate is the group's most-preferred.",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="0.4",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "coc_support_by_group.pdf")
    fig.savefig(out_dir / "coc_support_by_group.png")
    plt.close(fig)


def fig_cohesion_heatmap(sw: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of CoC support (cohesion) across elections × groups."""
    elections = _ordered_elections(set(sw["election"]))

    M = np.full((len(elections), len(GROUP_ORDER)), np.nan)
    for i, elec in enumerate(elections):
        for j, g in enumerate(GROUP_ORDER):
            sel = sw[(sw["election"] == elec) & (sw["group"] == g)]
            if not sel.empty:
                M[i, j] = 100 * sel.iloc[0]["mean_support"]

    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(elections) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=30, vmax=100)

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
    ax.set_yticks(range(len(elections)))
    ax.set_yticklabels([ELECTION_SHORT.get(e, e) for e in elections])

    for i in range(len(elections)):
        for j in range(len(GROUP_ORDER)):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 70 else "0.15", fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean support for CoC (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Cohesion — support for the group's candidate of choice",
                 pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "cohesion_heatmap.pdf")
    fig.savefig(out_dir / "cohesion_heatmap.png")
    plt.close(fig)


def fig_within_precinct_uncertainty(quant_df: pd.DataFrame | None,
                                    out_dir: Path) -> None:
    """
    Posterior precinct-level support distribution for one headline general
    election. For each group × candidate, draw 12.5/87.5% (or 25/75%) bands
    against the median, summarised across all precincts.
    """
    if quant_df is None or quant_df.empty:
        return

    elec = "24G_President"
    if elec not in set(quant_df["election"]):
        # fall back to first general election present
        gens = [e for e in _ordered_elections(set(quant_df["election"]))
                if e.startswith("24G_")]
        if not gens:
            return
        elec = gens[0]

    sub = quant_df[quant_df["election"] == elec].copy()

    # For each (group, candidate), compute the within-group support fraction at
    # the precinct level by normalising counts across candidates at each
    # quantile. Then summarise the distribution of precinct-level supports.
    cands = sorted(sub["candidate"].unique())
    # Choose median (octile=4, quantile≈0.5) for the central estimate
    med = sub[sub["octile"] == 4].copy()

    # Pivot to (precinct, group) -> {candidate: median_count}
    summary_rows = []
    for g in GROUP_ORDER:
        gsub = med[med["group"] == g]
        if gsub.empty:
            continue
        wide = gsub.pivot_table(index="CNTYVTD", columns="candidate",
                                values="value", aggfunc="first").fillna(0)
        wide = wide.reindex(columns=cands, fill_value=0)
        totals = wide.sum(axis=1).replace(0, np.nan)
        shares = wide.div(totals, axis=0)
        for cand in cands:
            vals = shares[cand].dropna().values * 100
            if vals.size == 0:
                continue
            summary_rows.append({
                "group": g,
                "candidate": cand,
                "p10": np.nanpercentile(vals, 10),
                "p25": np.nanpercentile(vals, 25),
                "p50": np.nanpercentile(vals, 50),
                "p75": np.nanpercentile(vals, 75),
                "p90": np.nanpercentile(vals, 90),
            })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return

    # Plot: one panel per candidate, x = group, y = within-group support
    cands_present = [c for c in cands if c in set(summary["candidate"])]
    fig, axes = plt.subplots(1, len(cands_present),
                             figsize=(4.0 * len(cands_present), 4.0),
                             sharey=True)
    if len(cands_present) == 1:
        axes = [axes]

    for ax, cand in zip(axes, cands_present):
        cdf = summary[summary["candidate"] == cand].set_index("group")
        cdf = cdf.reindex([g for g in GROUP_ORDER if g in cdf.index])

        x = np.arange(len(cdf))
        # 80% band
        ax.vlines(x, cdf["p10"], cdf["p90"], color="0.7", linewidth=2.0)
        # 50% band (IQR)
        ax.vlines(x, cdf["p25"], cdf["p75"], color="0.35", linewidth=4.5)
        # median
        for xi, gi in zip(x, cdf.index):
            ax.plot(xi, cdf.loc[gi, "p50"], "o",
                    color=GROUP_COLORS[gi], markersize=8,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g] for g in cdf.index], rotation=0)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_title(_candidate_short(cand), pad=6)
        ax.grid(axis="y", color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Precinct-level within-group vote share")
    fig.suptitle(
        f"Across-precinct distribution of within-group support — "
        f"{ELECTION_LABEL.get(elec, elec)}",
        y=1.02, fontweight="bold",
    )
    # Caption
    fig.text(
        0.5, -0.04,
        "Thick bars: 25–75th percentiles across precincts. Thin bars: 10–90th. "
        "Dot: median precinct.",
        ha="center", fontsize=8, color="0.4",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "within_precinct_uncertainty.pdf")
    fig.savefig(out_dir / "within_precinct_uncertainty.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ei-dir", type=Path, default=Path("ei_outputs"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EI outputs from {args.ei_dir}/ ...")
    sw, mc, qd = load_inputs(args.ei_dir)
    print(f"  statewide:  {len(sw):,} rows  ({sw['election'].nunique()} elections)")
    print(f"  mean cnts:  {len(mc):,} rows")
    print(f"  quantiles:  {0 if qd is None else len(qd):,} rows")

    print("Computing full statewide group×candidate support matrix ...")
    full_support = compute_full_support(mc)

    print("Writing tables ...")
    table_coc_summary(sw, tables_dir)
    table_cohesion_summary(sw, tables_dir)
    table_polarization_summary(full_support, tables_dir)
    print(f"  -> {tables_dir}/")

    print("Writing figures ...")
    fig_polarization_generals(full_support, figures_dir)
    fig_coc_support_by_group(sw, figures_dir)
    fig_cohesion_heatmap(sw, figures_dir)
    fig_within_precinct_uncertainty(qd, figures_dir)
    print(f"  -> {figures_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
