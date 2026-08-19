# -*- coding: utf-8 -*-
"""
rescore_plans.py
================

Re-score an *already generated* Besag-Clifford ensemble from its saved plan
assignments, recording, for every plan:

  * the aggregate opportunity scores O_g, F_g under all three EI weighting
    modes (statewide / equal / district), and
  * the raw per-district win-probability vectors p_d^g,

so that the threshold sweep, the point-estimate-versus-uncertainty-propagation
comparison, and any other scoring ablation can be run *on the existing
ensemble* rather than by regenerating spokes. Nothing here touches the Markov
chain: the plans are fixed, only the score is recomputed, so every ablation
produced from this file is attributable to the scoring choice alone.

Usage (must be run from pipelines/ensemble_generation_layer, because the
ensemble module imports `run_functions` as a top-level module):

    cd pipelines/ensemble_generation_layer
    python rescore_plans.py \
        --plans outputs/bc_plan_assignments_TX_BC_functional.parquet \
        --out   outputs/bc_rescored_TX_BC_functional.csv \
        --workers 30

Optional sanity check against the original run:

    python rescore_plans.py --plans ... --out ... \
        --check-results outputs/bc_opportunity_results_TX_BC_functional.csv

which verifies that the re-scored enacted plan reproduces the O_L / O_B values
reported by the original run (they must match to ~1e-9; a mismatch means the
plan artifact and the scoring code have drifted apart).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing the ensemble module loads the precinct parquet, builds the graph,
# applies the bridge edges, and loads every EI artifact. That is exactly the
# state we need to score a plan, and reusing it guarantees the re-scored
# numbers come from the same code path as the original run.
import besag_clifford_vra_opportunity as bc  # noqa: E402
from gerrychain import GeographicPartition  # noqa: E402

NODES_ORDER = list(bc.graph.nodes())


def _score_one(args):
    """Score a single plan. Returns a flat row dict."""
    plan_name, labels = args
    assignment = {node: int(lab) for node, lab in zip(NODES_ORDER, labels)}
    partition = GeographicPartition(graph=bc.graph,
                                    assignment=assignment,
                                    updaters=bc._build_updaters())
    state_prob, equal_prob, dist_prob = partition["final_elec_model"]

    row = {'plan': plan_name,
           'cut_edges': len(partition["cut_edges"]),
           'county_splits': partition["county_splits"]["count"]}

    for tag, pdict in (('state', state_prob),
                       ('equal', equal_prob),
                       ('dist', dist_prob)):
        if "N/A" in pdict.values():
            for key in ('O_B', 'O_L', 'F_B', 'F_L'):
                row[f'{key}_{tag}'] = float('nan')
            row[f'pL_{tag}'] = row[f'pB_{tag}'] = ''
            continue
        scores = bc.compute_opportunity_scores(
            pdict, bc.M_BAR_B, bc.M_BAR_L, bc.M_BAR_COMBINED)
        for key in ('O_B', 'O_L', 'F_B', 'F_L'):
            row[f'{key}_{tag}'] = scores[key]
        districts = sorted(pdict.keys())
        row[f'pL_{tag}'] = json.dumps(
            [round(float(pdict[d][0]), 6) for d in districts])
        row[f'pB_{tag}'] = json.dumps(
            [round(float(pdict[d][1]), 6) for d in districts])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plans', required=True,
                    help='bc_plan_assignments_<RUN>.parquet from a completed run')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--workers', type=int,
                    default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--limit', type=int, default=None,
                    help='score only the first N plans (smoke test)')
    ap.add_argument('--check-results', default=None,
                    help='bc_opportunity_results_<RUN>.csv to verify against')
    args = ap.parse_args()

    plans = pd.read_parquet(args.plans)
    columns = list(plans.columns)
    if args.limit:
        # Always keep the enacted plan in a smoke test: it is the one whose
        # score we can check against the published numbers.
        columns = (['enacted'] if 'enacted' in columns else []) + \
                  [c for c in columns if c != 'enacted'][:args.limit]
    print(f"Re-scoring {len(columns)} plans from {args.plans} "
          f"({len(plans)} precincts, {args.workers} workers)")

    jobs = [(c, plans[c].values) for c in columns]
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_score_one, jobs)):
                rows.append(row)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(jobs)} plans "
                          f"({time.time()-t0:.0f}s elapsed)")
    else:
        for i, job in enumerate(jobs):
            rows.append(_score_one(job))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(jobs)} plans "
                      f"({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    # Deterministic ordering: enacted, hub, then spokes in index order.
    order = {'enacted': 0, 'hub': 1}
    df['_sort'] = df['plan'].map(lambda p: (order.get(p, 2), p))
    df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} plans, {len(df.columns)} columns) "
          f"in {(time.time()-t0)/60:.1f} min")

    if args.check_results and 'enacted' in set(df['plan']):
        published = pd.read_csv(args.check_results).iloc[0]
        rescored = df[df['plan'] == 'enacted'].iloc[0]
        mode_tag = {'statewide': 'state', 'equal': 'equal',
                    'district': 'dist'}.get(published.get('mode', 'statewide'),
                                            'state')
        print("\nSanity check against the original run "
              f"(mode={published.get('mode')}):")
        ok = True
        for col in ('O_B', 'O_L'):
            a = float(published[f'enacted_{col}'])
            b = float(rescored[f'{col}_{mode_tag}'])
            flag = 'OK' if abs(a - b) < 1e-6 else 'MISMATCH'
            ok &= flag == 'OK'
            print(f"  {col}: published {a:.6f} | re-scored {b:.6f}  [{flag}]")
        if not ok:
            print("  The plan artifact and the current scoring code disagree. "
                  "Do not use the re-scored file until this is resolved.")
            sys.exit(1)


if __name__ == '__main__':
    main()
