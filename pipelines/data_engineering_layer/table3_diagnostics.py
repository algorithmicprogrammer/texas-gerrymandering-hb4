"""
table3_diagnostics.py
---------------------
Emit every value in "Table 3: Validation diagnostics for the spatial graph
processing pipeline" in a single pass.

Rows 1-5 come from the data engineering layer and are computed here directly
from the same intermediate objects that pipeline.run_pipeline() already builds.
Rows 6-8 come from the ReCom / Besag-Clifford ensemble step and are computed by
`ensemble_constraint_diagnostics` (wire in your gerrychain graph + saved plans).

Usage (inside the repo, after a pipeline run)
---------------------------------------------
    from table3_diagnostics import collect_de_diagnostics, render_table3

    # `final`, `blocks_cvap`, `elections`, `vtd_demo` are the objects from
    # run_pipeline(); `blocks_shp` / `vtd_shp` are the raw shapefiles from
    # ingest (sources["blocks_shp"], sources["vtd_shp"]).
    diag = collect_de_diagnostics(
        final=final,
        blocks_cvap=blocks_cvap,
        blocks_shp=blocks_shp,
        vtd_shp=vtd_shp,
        elections=elections,
        vtd_demo=vtd_demo,
    )
    diag.update(ensemble_constraint_diagnostics(...))   # rows 6-8
    print(render_table3(diag))                          # plain + LaTeX
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------------
# Row 5 helper — replicate spatial._fix_geometry and count what it repairs
# ---------------------------------------------------------------------------
def _repair_count(gdf: gpd.GeoDataFrame) -> tuple[int, int]:
    """
    Returns (n_invalid_before, n_still_invalid_after) using the SAME repair
    strategy as spatial._fix_geometry: make_valid (Shapely >= 2.0) then a
    zero-width buffer pass. Run on a copy so the caller's data is untouched.
    """
    g = gdf.geometry
    n_before = int((~g.is_valid).sum())

    try:
        from shapely.validation import make_valid
        repaired = g.apply(lambda x: make_valid(x) if x is not None and not x.is_valid else x)
    except ImportError:
        repaired = g.buffer(0)

    repaired = repaired.buffer(0)
    n_after = int((~repaired.is_valid).sum())
    return n_before, n_after


# ---------------------------------------------------------------------------
# Rows 1-5 — data engineering layer
# ---------------------------------------------------------------------------
def collect_de_diagnostics(
    final: gpd.GeoDataFrame,
    blocks_cvap: pd.DataFrame,
    blocks_shp: gpd.GeoDataFrame,
    vtd_shp: gpd.GeoDataFrame,
    elections: pd.DataFrame,
    vtd_demo: gpd.GeoDataFrame,
) -> dict:
    d: dict = {}

    # --- Row 1: Population preserved after aggregation ---------------------
    # Primary metric: total population (POP20 -> TOTALPOP).
    block_pop = float(blocks_shp["POP20"].sum()) if "POP20" in blocks_shp.columns else None
    vtd_pop = float(final["TOTALPOP"].sum()) if "TOTALPOP" in final.columns else None
    if block_pop and vtd_pop is not None:
        d["pop_preserved_pct"] = 100.0 * vtd_pop / block_pop
        d["pop_preserved_detail"] = f"{vtd_pop:,.0f} / {block_pop:,.0f}"
    # Secondary: VAP (matches validate._check_population_conservation).
    block_vap = float(blocks_cvap["VAP"].sum())
    vtd_vap = float(final["VAP"].sum())
    if block_vap:
        d["vap_preserved_pct"] = 100.0 * vtd_vap / block_vap
        d["vap_preserved_detail"] = f"{vtd_vap:,.0f} / {block_vap:,.0f}"

    # --- Row 2: Precincts with district assignment -------------------------
    n_total = len(final)
    n_assigned = int(final["CD"].notna().sum())
    d["precincts_with_cd"] = n_assigned
    d["precincts_total"] = n_total
    d["precincts_with_cd_pct"] = 100.0 * n_assigned / max(n_total, 1)

    # --- Rows 3 & 4: Election units matched / unmatched --------------------
    elec_keys = set(elections["CNTYVTD"])
    geom_keys = set(vtd_demo["CNTYVTD"])
    matched = elec_keys & geom_keys
    unmatched = elec_keys - geom_keys
    d["election_units_total"] = len(elec_keys)
    d["election_units_matched"] = len(matched)
    d["election_units_unmatched"] = len(unmatched)

    # --- Row 5: Invalid geometries repaired --------------------------------
    # Count across BOTH layers (blocks + VTDs), as both pass through
    # spatial._fix_geometry in the crosswalk.
    b_before, b_after = _repair_count(blocks_shp)
    v_before, v_after = _repair_count(vtd_shp)
    invalid_before = b_before + v_before
    still_invalid = b_after + v_after
    d["invalid_geoms_before"] = invalid_before
    d["invalid_geoms_repaired"] = invalid_before - still_invalid
    d["invalid_geoms_still_invalid"] = still_invalid

    return d


# ---------------------------------------------------------------------------
# Rows 6-8 — ensemble constraint pass rates
# ---------------------------------------------------------------------------
def ensemble_constraint_diagnostics(
    plans: list,                       # iterable of gerrychain Partition objects (or assignment dicts)
    *,
    graph,                             # the gerrychain Graph (for cut-edge / pop / county evaluation)
    pop_col: str = "TOTALPOP",
    county_col: str = "COUNTY",
    ideal_pop: float,                  # total population / number of districts
    pop_tolerance: float = 0.02,       # +/- fraction allowed per district
    enacted_cut_edges: int,            # cut-edge count of PLANC2333
    compactness_factor: float = 1.5,   # cut-edges must be <= factor * enacted
    max_county_splits: int | None = None,  # None => use enacted split count as ceiling
    enacted_county_splits: int | None = None,
) -> dict:
    """
    Evaluate the three ensemble constraints as POST-HOC pass rates. If your
    ReCom run enforces these as HARD acceptance filters, every accepted plan
    passes by construction and each count equals len(plans); running this
    confirms that and gives you the denominator. If you accept plans more
    loosely and report constraints as diagnostics, this gives the true rate.

    `plans` should be gerrychain Partitions; if you saved assignments to
    Parquet (per your pipeline), reconstruct Partition(graph, assignment)
    before passing them in.
    """
    from gerrychain import Partition  # noqa: F401  (import here to keep module light)
    from gerrychain.updaters import Tally, cut_edges

    n = 0
    pass_pop = pass_compact = pass_county = 0
    lo = ideal_pop * (1 - pop_tolerance)
    hi = ideal_pop * (1 + pop_tolerance)
    county_ceiling = max_county_splits if max_county_splits is not None else enacted_county_splits

    for part in plans:
        n += 1

        # Population balance: every district within tolerance
        pops = part["population"].values() if "population" in part.updaters else [
            sum(graph.nodes[v][pop_col] for v in dist) for dist in part.parts.values()
        ]
        if all(lo <= p <= hi for p in pops):
            pass_pop += 1

        # Compactness: cut-edge count <= factor * enacted
        n_cut = len(part["cut_edges"]) if "cut_edges" in part.updaters else len(part.cut_edges)
        if n_cut <= compactness_factor * enacted_cut_edges:
            pass_compact += 1

        # County splits: number of counties touching >1 district
        if county_ceiling is not None:
            split = 0
            seen: dict = {}
            for node, dist in part.assignment.items():
                c = graph.nodes[node][county_col]
                seen.setdefault(c, set()).add(dist)
            split = sum(1 for dists in seen.values() if len(dists) > 1)
            if split <= county_ceiling:
                pass_county += 1

    return {
        "plans_total": n,
        "plans_pop_balanced": pass_pop,
        "plans_compact": pass_compact,
        "plans_county_split_ok": pass_county if county_ceiling is not None else None,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_count_of(num, den):
    if den is None:
        return f"{num:,}"
    return f"{num:,} / {den:,} ({100.0 * num / max(den, 1):.1f}\\%)"


def render_table3(d: dict) -> str:
    """Return a plain-text summary plus paste-ready LaTeX table body."""
    plain = [
        "=== Table 3 diagnostics ===",
        f"1. Population preserved (total pop): {d.get('pop_preserved_pct', float('nan')):.3f}%"
        f"  [{d.get('pop_preserved_detail', 'n/a')}]",
        f"   Population preserved (VAP):       {d.get('vap_preserved_pct', float('nan')):.3f}%"
        f"  [{d.get('vap_preserved_detail', 'n/a')}]",
        f"2. Precincts with district assignment: "
        f"{d.get('precincts_with_cd', 0):,} / {d.get('precincts_total', 0):,} "
        f"({d.get('precincts_with_cd_pct', 0):.1f}%)",
        f"3. Election units matched:   {d.get('election_units_matched', 0):,} "
        f"of {d.get('election_units_total', 0):,}",
        f"4. Election units unmatched: {d.get('election_units_unmatched', 0):,}",
        f"5. Invalid geometries repaired: {d.get('invalid_geoms_repaired', 0):,} "
        f"(of {d.get('invalid_geoms_before', 0):,} invalid; "
        f"{d.get('invalid_geoms_still_invalid', 0):,} still invalid)",
        f"6. Plans satisfying population balance: "
        f"{d.get('plans_pop_balanced', 'TODO')} / {d.get('plans_total', 'TODO')}",
        f"7. Plans satisfying compactness:        "
        f"{d.get('plans_compact', 'TODO')} / {d.get('plans_total', 'TODO')}",
        f"8. Plans satisfying county-split:       "
        f"{d.get('plans_county_split_ok', 'TODO')} / {d.get('plans_total', 'TODO')}",
    ]

    pp = f"{d.get('pop_preserved_pct', float('nan')):.2f}\\%"
    # Ensemble rows stay "TODO" until real plan counts are supplied, so an
    # un-run ensemble never silently pastes "0" into the paper.
    have_ens = "plans_total" in d
    pop_bal = _fmt_count_of(d["plans_pop_balanced"], d["plans_total"]) if have_ens else "TODO"
    compact = _fmt_count_of(d["plans_compact"], d["plans_total"]) if have_ens else "TODO"
    cty = (_fmt_count_of(d["plans_county_split_ok"], d["plans_total"])
           if have_ens and d.get("plans_county_split_ok") is not None else "TODO")
    latex = [
        "% --- paste into Table 3 body ---",
        f"Population preserved after aggregation & {pp} \\\\",
        f"Precincts with district assignment & {_fmt_count_of(d.get('precincts_with_cd', 0), d.get('precincts_total'))} \\\\",
        f"Election units matched & {d.get('election_units_matched', 0):,} \\\\",
        f"Election units unmatched & {d.get('election_units_unmatched', 0):,} \\\\",
        f"Invalid geometries repaired & {d.get('invalid_geoms_repaired', 0):,} \\\\",
        f"Generated plans satisfying population balance & {pop_bal} \\\\",
        f"Generated plans satisfying compactness constraint & {compact} \\\\",
        f"Generated plans satisfying county-split constraint & {cty} \\\\",
    ]
    return "\n".join(plain) + "\n\n" + "\n".join(latex)