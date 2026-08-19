# -*- coding: utf-8 -*-
"""
add_second_plan.py
==================

Add a second enacted plan (e.g. Texas's 2021 congressional map PLANC2193) to
the precinct dataset as an extra district-label column, so the ensemble script
can be pointed at it with BC_START_MAP=<column>.

The point of this script is that the expensive layers do not have to be rerun:
the graph, the demographics, the election returns, and above all the ecological
inference posterior are all plan-independent. Evaluating a second enacted plan
costs one district assignment plus one ensemble.

    cd pipelines/data_engineering_layer
    python add_second_plan.py \
        --plan-shapefile ../../data/raw/PLANC2193/PLANC2193.shp \
        --column CD2021

Two things to check before using the result:

  1. The plan shapefile must have a district-label column; pass --label-field
     if it is not called "District".
  2. Contiguity repair is PLAN-SPECIFIC. The bridge edges in
     ensemble_generation_layer/outputs/bridge_edges.json were built so that
     PLANC2333's districts induce connected subgraphs. A different enacted plan
     needs its own repair pass, and the resulting graph is a (slightly)
     different graph -- so the finite-sample statement for the second plan is
     conditional on that graph, and the paper must say so.
"""

import argparse

import geopandas as gpd
import pandas as pd

from texas_gerrymandering_hb4.config import PRECINCT_DATASET_PARQUET
from join import _assign_congressional_districts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-shapefile', required=True)
    ap.add_argument('--column', required=True,
                    help='name for the new district-label column, e.g. CD2021')
    ap.add_argument('--label-field', default='District',
                    help='district-label field in the plan shapefile')
    ap.add_argument('--parquet', default=str(PRECINCT_DATASET_PARQUET))
    ap.add_argument('--out', default=None,
                    help='output parquet (default: overwrite --parquet)')
    args = ap.parse_args()

    gdf = gpd.read_parquet(args.parquet)
    plan = gpd.read_file(args.plan_shapefile)
    if args.label_field != 'District':
        plan = plan.rename(columns={args.label_field: 'District'})

    assigned = _assign_congressional_districts(
        gdf[['CNTYVTD', 'geometry']].copy(), plan)
    labels = assigned[['CNTYVTD', 'CD']].rename(columns={'CD': args.column})

    merged = gdf.merge(labels, on='CNTYVTD', how='left')
    n_missing = int(merged[args.column].isna().sum())
    n_districts = merged[args.column].nunique(dropna=True)
    print(f"Assigned {len(merged) - n_missing:,}/{len(merged):,} precincts to "
          f"{n_districts} districts in column {args.column!r}")
    if n_missing:
        raise SystemExit(
            f"{n_missing} precincts unassigned -- the ensemble script requires "
            "a complete assignment. Inspect these precincts before continuing.")

    # Report the crosswalk's population deviation under the new plan: if it
    # exceeds POP_TOL the ensemble script will reject the plan as an initial
    # state, exactly as documented for the enacted plan.
    if 'TOTALPOP' in merged.columns:
        pops = merged.groupby(args.column)['TOTALPOP'].sum()
        ideal = pops.sum() / len(pops)
        dev = (pops.max() - pops.min()) / ideal
        print(f"Top-to-bottom population deviation under {args.column}: "
              f"{100 * dev:.2f}% (ensemble script uses POP_TOL=5%)")

    out = args.out or args.parquet
    merged.to_parquet(out)
    print(f"Wrote {out}")
    print(f"Now run: BC_START_MAP={args.column} BC_RUN_NAME=... "
          "python besag_clifford_vra_opportunity.py")


if __name__ == '__main__':
    main()
