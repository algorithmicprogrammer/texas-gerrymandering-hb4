from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_ensemble_plans_from_plan_map(
    plan_map_path: str,
    out_path: str,
    ensemble_id: str,
    cycle: str | int | None = None,
    chamber: str | None = None,
) -> str:
    """
    Create an ensemble_plans.parquet file from a plan_map (plan_district_vtd.parquet).

    Requires plan_map to contain a 'plan_id' column.

    Output columns are designed to match your existing plan schema:
      plan_id, plan_type, cycle, chamber, ensemble_id, generator, seed, constraints_json, created_at
    """
    plan_map_path = str(plan_map_path)
    out_path = str(out_path)

    df = pd.read_parquet(plan_map_path, columns=["plan_id"])
    plan_ids = (
        df["plan_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "plan_id": plan_ids,
            "plan_type": "ENSEMBLE",
            "cycle": cycle,
            "chamber": chamber,
            "ensemble_id": ensemble_id,
            "generator": "recom",
            "seed": None,
            "constraints_json": None,
            "created_at": None,
        }
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble-id", required=True)
    ap.add_argument("--cycle", default=None)
    ap.add_argument("--chamber", default=None)
    args = ap.parse_args()

    build_ensemble_plans_from_plan_map(
        plan_map_path=args.plan_map,
        out_path=args.out,
        ensemble_id=args.ensemble_id,
        cycle=args.cycle,
        chamber=args.chamber,
    )
    print(f"Wrote {args.out}")
