from __future__ import annotations

import argparse

from .db import connect_db
from .schema import create_schema, upsert_opp_defs
from .config import DEFAULT_OPP_DEFS
from .io_load import (
    load_geo_vtd,
    load_election,
    load_election_returns_vtd,
    load_plan,
    load_plan_district_vtd,
)
from .io_export import export_outputs
from .aggregates import build_district_demo_vap, build_district_returns
from .opportunity import build_district_opportunity, build_plan_metrics
from .ensemble_metrics import build_ensemble_distribution, build_plan_vs_ensemble
from .sanity import sanity_checks


STAGES = ["schema", "load", "sanity", "aggregates", "metrics", "ensemble", "ei", "export", "all"]


def main() -> None:
    p = argparse.ArgumentParser(description="Redistricting pipeline (VTD+VAP) with stages")

    p.add_argument("--db", default=":memory:", help="DuckDB path or ':memory:' (default ':memory:')")
    p.add_argument("--stage", default="all", choices=STAGES)

    p.add_argument("--geo-vtd", help="VTD demographics (VAP) parquet/csv")
    p.add_argument("--elections", help="Election metadata parquet/csv")
    p.add_argument("--returns", help="VTD election returns parquet/csv")

    # enacted plan metadata + map
    p.add_argument("--plans", help="Plan metadata parquet/csv (enacted)")
    p.add_argument("--plan-map", help="Assignments (enacted): (plan_id, vtd_geoid, district_id) parquet/csv")

    # ensemble plan metadata + map
    p.add_argument("--ensemble-plans", help="Plan metadata parquet/csv (ensemble)")
    p.add_argument("--ensemble-plan-map", help="Assignments (ensemble): (plan_id, vtd_geoid, district_id) parquet/csv")

    p.add_argument("--ensemble-id", help="Ensemble id (e.g. ENS_TXCD_2024_recom_v1)")
    p.add_argument("--ei-election-id", help="Election_id for EI fit, e.g. TX_SEN_2024_GEN")
    p.add_argument("--ei-run-id", default="EI_RUN_001")

    # Export options
    p.add_argument("--out-dir", help="If provided, export derived tables to this directory.")
    p.add_argument("--export-format", default="parquet", choices=["parquet", "csv"])

    args = p.parse_args()

    con = connect_db(args.db)

    def run_schema():
        create_schema(con)
        upsert_opp_defs(con, list(DEFAULT_OPP_DEFS))

    def run_load():
        if not all([args.geo_vtd, args.elections, args.returns, args.plans, args.plan_map]):
            raise ValueError("For stage=load, provide --geo-vtd --elections --returns --plans --plan-map")

        load_geo_vtd(con, args.geo_vtd)
        load_election(con, args.elections)
        load_election_returns_vtd(con, args.returns)

        # enacted
        load_plan(con, args.plans)
        load_plan_district_vtd(con, args.plan_map)

        # ensemble (optional)
        if args.ensemble_plans:
            load_plan(con, args.ensemble_plans)
        if args.ensemble_plan_map:
            load_plan_district_vtd(con, args.ensemble_plan_map)

    def run_sanity():
        sanity_checks(con)
        print("Sanity checks passed.")

    def run_aggregates():
        build_district_demo_vap(con)
        build_district_returns(con)

    def run_metrics():
        build_district_opportunity(con)
        build_plan_metrics(con)

    def run_ensemble():
        if not args.ensemble_id:
            raise ValueError("For stage=ensemble, provide --ensemble-id")

        # IMPORTANT: pass ensemble_id explicitly
        build_ensemble_distribution(
            con,
            ensemble_id=args.ensemble_id,
            metric_column="n_opportunity_districts",
        )
        build_plan_vs_ensemble(
            con,
            ensemble_id=args.ensemble_id,
            metric_column="n_opportunity_districts",
        )

    def run_ei():
        if not args.ei_election_id or not args.ensemble_id:
            raise ValueError("For stage=ei, provide --ei-election-id and --ensemble-id")
        from .ei.model import fit_hierarchical_ei_vtd

        fit_hierarchical_ei_vtd(
            con=con,
            election_id=args.ei_election_id,
            ei_run_id=args.ei_run_id,
            ensemble_id=args.ensemble_id,
        )

    def run_export():
        if not args.out_dir:
            raise ValueError("For stage=export, provide --out-dir (and optional --export-format).")
        export_outputs(con, args.out_dir, fmt=args.export_format)

    # Execute
    if args.stage == "schema":
        run_schema()
    elif args.stage == "load":
        run_schema()
        run_load()
    elif args.stage == "sanity":
        run_sanity()
    elif args.stage == "aggregates":
        run_aggregates()
    elif args.stage == "metrics":
        run_metrics()
    elif args.stage == "ensemble":
        run_ensemble()
    elif args.stage == "ei":
        run_ei()
    elif args.stage == "export":
        run_export()
    elif args.stage == "all":
        run_schema()
        run_load()
        run_sanity()
        run_aggregates()
        run_metrics()
        run_ensemble()
        run_ei()
        if args.out_dir:
            run_export()

    con.close()
    print(f"Done. stage={args.stage}")


if __name__ == "__main__":
    main()
