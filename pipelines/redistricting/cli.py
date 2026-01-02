from __future__ import annotations

import argparse
from pathlib import Path

from .db import connect_db
from .schema import create_schema, upsert_opp_defs
from .config import DEFAULT_OPP_DEFS
from .io_load import (
    load_geo_vtd,
    load_election,
    load_election_returns_vtd,
    load_plan,
    load_ensemble_plans,
    load_plan_district_vtd,
)
from .sanity import sanity_checks
from .aggregates import build_district_demo_vap, build_district_returns
from .opportunity import build_district_opportunity, build_plan_metrics
from .ensemble_metrics import build_ensemble_distribution, build_plan_vs_ensemble
from .io_export import export_tables

STAGES = ["schema", "load", "sanity", "aggregates", "metrics", "ensemble", "ei", "export", "all"]


def main() -> None:
    p = argparse.ArgumentParser(description="Redistricting pipeline (DuckDB-backed) with optional file export")

    p.add_argument("--db", required=True, help="DuckDB DB file (e.g. data/warehouse/redistricting.duckdb)")
    p.add_argument("--stage", default="all", choices=STAGES)

    # Inputs
    p.add_argument("--geo-vtd", help="VTD demographics/VAP (parquet/csv)")
    p.add_argument("--elections", help="Election metadata (parquet/csv)")
    p.add_argument("--returns", help="VTD election returns (parquet/csv)")
    p.add_argument("--plans", help="Enacted plan metadata (often 1 row) (parquet/csv)")
    p.add_argument("--ensemble-plans", help="Ensemble plan metadata (many plan rows) (parquet/csv)")
    p.add_argument("--plan-map", help="Plan->district->VTD assignments (parquet/csv)")

    # Ensemble + EI
    p.add_argument("--ensemble-id", help="Ensemble identifier (e.g. ENS_TXCD_2024_recom_v1)")
    p.add_argument("--ei-election-id", help="Election_id for EI fit (e.g. TX_SEN_2024_GEN)")
    p.add_argument("--ei-run-id", default="EI_RUN_001")

    # Export
    p.add_argument("--out-dir", help="Directory to export tables")
    p.add_argument("--export-format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--export-tables", help="Comma-separated list of tables to export (optional)")

    args = p.parse_args()
    con = connect_db(args.db)

    def run_schema() -> None:
        create_schema(con)
        upsert_opp_defs(con, list(DEFAULT_OPP_DEFS))

    def run_load() -> None:
        if not all([args.geo_vtd, args.elections, args.returns, args.plans, args.plan_map]):
            raise ValueError(
                "For stage=load (or stage=all), provide: "
                "--geo-vtd, --elections, --returns, --plans, --plan-map "
                "(and optionally --ensemble-plans)"
            )

        load_geo_vtd(con, args.geo_vtd)
        load_election(con, args.elections)
        load_election_returns_vtd(con, args.returns)

        # enacted plan
        load_plan(con, args.plans)

        # optional ensemble plans (only if provided AND exists)
        if args.ensemble_plans:
            ep = Path(args.ensemble_plans)
            if ep.exists():
                load_ensemble_plans(con, str(ep))
            else:
                print(f"[load] Skipping ensemble-plans: file not found: {ep}")

        load_plan_district_vtd(con, args.plan_map)

    def run_sanity() -> None:
        sanity_checks(con)

    def run_aggregates() -> None:
        build_district_demo_vap(con)
        build_district_returns(con)

    def run_metrics() -> None:
        build_district_opportunity(con)
        build_plan_metrics(con)

    def _has_ensemble_context() -> bool:
        """
        Determine if we have an actual ensemble (more than one plan).
        """
        try:
            n_plans = con.execute("SELECT COUNT(*) FROM plan").fetchone()[0]
            n_plan_ids_in_map = con.execute("SELECT COUNT(DISTINCT plan_id) FROM plan_district_vtd").fetchone()[0]
        except Exception:
            return False
        # Need multiple plans in map to treat it as an ensemble distribution
        return (n_plans > 1) and (n_plan_ids_in_map > 1)

    def run_ensemble() -> None:
        if not _has_ensemble_context():
            n_plans = con.execute("SELECT COUNT(*) FROM plan").fetchone()[0]
            n_map = con.execute("SELECT COUNT(DISTINCT plan_id) FROM plan_district_vtd").fetchone()[0]
            print(f"[ensemble] Skipping: no ensemble plan-map available (plan rows={n_plans}, plan_map unique plan_id={n_map}).")
            return

        try:
            build_ensemble_distribution(con, metric_name="n_opportunity_districts", ensemble_id=args.ensemble_id)
            build_plan_vs_ensemble(con, metric_name="n_opportunity_districts", ensemble_id=args.ensemble_id)
        except ValueError as e:
            # Don't hard-fail; still allow exports for enacted plan
            print(f"[ensemble] Skipping due to ValueError: {e}")

    def run_ei() -> None:
        # EI is optional; only run if ensemble exists and user supplied election_id+ensemble_id
        if not args.ei_election_id:
            print("[ei] Skipping: --ei-election-id not provided.")
            return
        if not args.ensemble_id:
            print("[ei] Skipping: --ensemble-id not provided.")
            return
        if not _has_ensemble_context():
            print("[ei] Skipping: no ensemble context (only enacted plan present).")
            return

        from .ei.model import fit_hierarchical_ei_vtd

        fit_hierarchical_ei_vtd(
            con=con,
            election_id=args.ei_election_id,
            ei_run_id=args.ei_run_id,
            ensemble_id=args.ensemble_id,
        )

    def run_export() -> None:
        if not args.out_dir:
            raise ValueError("For stage=export (or export at end of stage=all), provide --out-dir")

        tables = None
        if args.export_tables:
            tables = [t.strip() for t in args.export_tables.split(",") if t.strip()]

        export_tables(con=con, out_dir=args.out_dir, fmt=args.export_format, tables=tables)
        print(f"Exported tables to: {args.out_dir} (format={args.export_format})")

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
