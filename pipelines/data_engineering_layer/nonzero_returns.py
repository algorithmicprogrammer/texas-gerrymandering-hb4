"""
nonzero_returns.py
------------------
Compute the "Election units with nonzero returns" diagnostic for Table 1.

Definition
----------
An election unit (VTD / precinct, keyed on CNTYVTD) has *nonzero returns* if
its recorded vote total across the tracked contests is > 0. This is one notch
stricter than "matched": a unit can join to a geometry (matched) yet carry no
votes (empty VTD -- airport, park, unpopulated industrial polygon).

It is computed on the same TOTVOTE_* contest-total columns that elections.py
builds and ei_export.py consumes, so it is consistent with the rest of the
data-engineering layer.

Two ways to use it
------------------
1. Inside the pipeline, on the merged `elections` frame (the object
   load_election_data() returns), optionally restricted to the matched keys:

       from nonzero_returns import nonzero_returns_diagnostic
       d.update(nonzero_returns_diagnostic(elections, matched_keys=matched))

2. Standalone, on the exported EI CSV:

       python nonzero_returns.py path/to/ei_input.csv
"""

from __future__ import annotations

import pandas as pd

# The nine contest totals (mirror ei_export.TOTVOTE_COLS). If your column names
# ever drift, the fallback below auto-detects any "TOTVOTE_*" columns present.
TOTVOTE_COLS = [
    "TOTVOTE_PRES_24G",
    "TOTVOTE_SEN_24G",
    "TOTVOTE_RRC_24G",
    "TOTVOTE_PRESD_24P",
    "TOTVOTE_PRESR_24P",
    "TOTVOTE_SEND_24P",
    "TOTVOTE_SENR_24P",
    "TOTVOTE_RRCD_24P",
    "TOTVOTE_RRCR_24P",
]


def _contest_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in TOTVOTE_COLS if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c.startswith("TOTVOTE_")]
    if not cols:
        raise ValueError(
            "No TOTVOTE_* contest columns found. Pass the merged elections "
            "frame from load_election_data() or the EI export CSV."
        )
    return cols


def nonzero_returns_diagnostic(
    elections: pd.DataFrame,
    matched_keys: set | None = None,
    key: str = "CNTYVTD",
) -> dict:
    """
    Returns a dict with:
      election_units_nonzero        : units with any contest total > 0
      election_units_zero_returns   : units that joined but recorded no votes
      election_units_denominator    : the population it was counted over
      nonzero_returns_by_contest    : per-contest nonzero counts (sanity check)

    If `matched_keys` is given (e.g. the `matched` set from
    collect_de_diagnostics), the count is restricted to matched units so the
    denominator lines up with the "Election units matched" row.
    """
    df = elections
    if matched_keys is not None:
        df = df[df[key].isin(matched_keys)]

    cols = _contest_cols(df)
    totals = df[cols].fillna(0)

    any_nonzero = (totals.sum(axis=1) > 0)
    n_nonzero = int(any_nonzero.sum())
    n_units = int(len(df))

    return {
        "election_units_nonzero": n_nonzero,
        "election_units_zero_returns": n_units - n_nonzero,
        "election_units_denominator": n_units,
        "nonzero_returns_by_contest": {
            c: int((totals[c] > 0).sum()) for c in cols
        },
    }


# ---------------------------------------------------------------------------
# Two-line integration into table3_diagnostics.py
# ---------------------------------------------------------------------------
# In collect_de_diagnostics(), right after rows 3 & 4 are computed:
#
#     from nonzero_returns import nonzero_returns_diagnostic
#     d.update(nonzero_returns_diagnostic(elections, matched_keys=matched))
#
# In render_table3(), add to the LaTeX body (between the matched/unmatched
# rows and "Invalid geometries repaired"):
#
#     f"Election units with nonzero returns & "
#     f"{_fmt_count_of(d.get('election_units_nonzero', 0), "
#     f"d.get('election_units_matched'))} \\\\",
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python nonzero_returns.py <ei_input.csv | elections.parquet>")
        sys.exit(1)

    path = sys.argv[1]
    df = (pd.read_parquet(path) if path.endswith(".parquet")
          else pd.read_csv(path))

    out = nonzero_returns_diagnostic(df)
    print(f"Election units (denominator):     {out['election_units_denominator']:,}")
    print(f"Election units with NONZERO returns: {out['election_units_nonzero']:,}")
    print(f"Election units with ZERO returns:    {out['election_units_zero_returns']:,}")
    print("\nPer-contest nonzero counts (sanity check):")
    for contest, n in out["nonzero_returns_by_contest"].items():
        print(f"  {contest:<22} {n:,}")