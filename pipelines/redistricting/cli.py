from __future__ import annotations

import argparse
import os

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
from .aggregates import build_district_demo_vap, build_district_demo_cvap, build_district_returns
from .opportunity import (
    build_district_opportunity,
    build_plan_metrics,
    build_district_outcome_summary,
    build_plan_opportunity_score,
)
from .ensemble_metrics import (
    build_ensemble_distribution,
    build_plan_vs_ensemble,
    build_ei_plan_vs_ensemble,
)
from .sanity import sanity_checks


STAGES = ["schema", "load", "sanity", "aggregates", "metrics", "ensemble", "ei", "export", "all"]


def _tune_duckdb(con, db_path: str) -> None:
    mem_limit = os.environ.get("DUCKDB_MEMORY_LIMIT", "12GB")
    threads = os.environ.get("DUCKDB_THREADS", "2")
    temp_dir = os.environ.get("DUCKDB_TEMP_DIR", None)

    con.execute("SET preserve_insertion_order=false")
    try:
        con.execute(f"SET threads={int(threads)}")
    except Exception:
        pass
    try:
        con.execute(f"SET memory_limit='{mem_limit}'")
    except Exception:
        pass
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        try:
            con.execute(f"SET temp_directory='{temp_dir}'")
        except Exception:
            pass
    try:
        con.execute("PRAGMA enable_progress_bar=false")
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Redistricting pipeline (CVAP + EI opportunity score) with stages")

    p.add_argument("--db", default=":memory:", help="DuckDB path or ':memory:' (default ':memory:')")
    p.add_argument("--stage", default="all", choices=STAGES)

    p.add_argument("--geo-vtd", help="VTD demographics parquet/csv")
    p.add_argument("--elections", help="Election metadata parquet/csv")
    p.add_argument("--returns", help="VTD election returns parquet/csv")
    p.add_argument("--plans", help="Plan metadata parquet/csv (enacted)")
    p.add_argument("--plan-map", help="Assignments (enacted): (plan_id, vtd_geoid, district_id) parquet/csv")
    p.add_argument("--ensemble-plans", help="Plan metadata parquet/csv (ensemble)")
    p.add_argument("--ensemble-plan-map", help="Assignments (ensemble): (plan_id, vtd_geoid, district_id) parquet/csv")

    p.add_argument("--ensemble-id", help="Ensemble id (e.g. ENS_TXCD_2024_recom_v1)")
    p.add_argument("--ei-election-id", help="Election_id for EI fit, e.g. TX_SEN_2024_GEN")
    p.add_argument("--ei-run-id", default="EI_RUN_001")

    p.add_argument("--ei-draws", type=int, default=2000)
    p.add_argument("--ei-tune", type=int, default=3000)
    p.add_argument("--ei-chains", type=int, default=4)
    p.add_argument("--ei-target-accept", type=float, default=0.97)
    p.add_argument("--ei-max-treedepth", type=int, default=15)
    p.add_argument("--ei-seed", type=int, default=None)

    p.add_argument("--out-dir", help="If provided, export derived tables to this directory.")
    p.add_argument("--export-format", default="parquet", choices=["parquet", "csv"])

    args = p.parse_args()

    con = connect_db(args.db)
    _tune_duckdb(con, args.db)

    def run_schema():
        create_schema(con)
        upsert_opp_defs(con, list(DEFAULT_OPP_DEFS))

    def run_load():
        if not all([args.geo_vtd, args.elections, args.returns, args.plans, args.plan_map]):
            raise ValueError("For stage=load, provide --geo-vtd --elections --returns --plans --plan-map")
        load_geo_vtd(con, args.geo_vtd)
        load_election(con, args.elections)
        load_election_returns_vtd(con, args.returns)
        load_plan(con, args.plans)
        load_plan_district_vtd(con, args.plan_map)
        if args.ensemble_plans and args.ensemble_plan_map:
            load_plan(con, args.ensemble_plans)
            load_plan_district_vtd(con, args.ensemble_plan_map)

    def run_sanity():
        sanity_checks(con)

    def run_aggregates():
        build_district_demo_vap(con)
        build_district_demo_cvap(con)
        build_district_returns(con)

    def run_metrics():
        build_district_opportunity(con)
        build_plan_metrics(con)

    def run_ensemble():
        if not args.ensemble_id:
            raise ValueError("For stage=ensemble, provide --ensemble-id")
        build_ensemble_distribution(
            con,
            ensemble_id=args.ensemble_id,
            metric_columns=["n_opportunity_districts", "mean_group_share"],
        )
        build_plan_vs_ensemble(
            con,
            ensemble_id=args.ensemble_id,
            metric_columns=["n_opportunity_districts", "mean_group_share"],
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
            draws=args.ei_draws,
            tune=args.ei_tune,
            chains=args.ei_chains,
            target_accept=args.ei_target_accept,
            max_treedepth=args.ei_max_treedepth,
            random_seed=args.ei_seed,
        )
        build_district_outcome_summary(con, args.ei_run_id)
        build_plan_opportunity_score(con, args.ei_run_id)
        build_ei_plan_vs_ensemble(con, args.ei_run_id, args.ensemble_id)

    def run_export():
        if args.out_dir:
            export_outputs(con, args.out_dir, fmt=args.export_format)

    if args.stage in ("schema", "all"):
        run_schema()
    if args.stage in ("load", "all"):
        run_load()
    if args.stage in ("sanity", "all"):
        run_sanity()
    if args.stage in ("aggregates", "all"):
        run_aggregates()
    if args.stage in ("metrics", "all"):
        run_metrics()
    if args.stage in ("ensemble", "all"):
        run_ensemble()
    if args.stage in ("ei", "all"):
        run_ei()
    if args.stage in ("export", "all"):
        run_export()

    print(f"Done. stage={args.stage}")


if __name__ == "__main__":
    main()
