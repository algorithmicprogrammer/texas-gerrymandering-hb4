from __future__ import annotations
import argparse

from .db import connect_db
from .schema import create_schema, upsert_opp_defs
from .config import DEFAULT_OPP_DEFS
from .io_load import (
    load_geo_vtd, load_election, load_election_returns_vtd, load_plan, load_plan_district_vtd
)
from .aggregates import build_district_demo_vap, build_district_returns
from .opportunity import build_district_opportunity, build_plan_metrics
from .ensemble_metrics import build_ensemble_distribution, build_plan_vs_ensemble
from .sanity import sanity_checks

STAGES = ["schema", "load", "sanity", "aggregates", "metrics", "ensemble", "ei", "all"]

def main() -> None:
    p = argparse.ArgumentParser(description="Redistricting pipeline (VTD+VAP) with stages")
    p.add_argument("--db", required=True)

    p.add_argument("--stage", default="all", choices=STAGES)
    p.add_argument("--geo-vtd", help="VTD demographics (VAP) parquet/csv")
    p.add_argument("--elections", help="Election metadata parquet/csv")
    p.add_argument("--returns", help="VTD election returns parquet/csv")
    p.add_argument("--plans", help="Plan metadata parquet/csv")
    p.add_argument("--plan-map", help="Assignments: (plan_id, vtd_geoid, district_id) parquet/csv")

    p.add_argument("--ensemble-id", help="ENS_TXCD_2021_recom_v1")
    p.add_argument("--ei-election-id", help="Election_id for EI fit, e.g. TX_PRES_2020_GEN")
    p.add_argument("--ei-run-id", default="EI_RUN_001")

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
        load_plan(con, args.plans)
        load_plan_district_vtd(con, args.plan_map)

    def run_sanity():
        sanity_checks(con)

    def run_aggregates():
        build_district_demo_vap(con)
        build_district_returns(con)

    def run_metrics():
        build_district_opportunity(con)
        build_plan_metrics(con)

    def run_ensemble():
        build_ensemble_distribution(con, metric_name="n_opportunity_districts")
        build_plan_vs_ensemble(con, metric_name="n_opportunity_districts")

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

    # Execute
    if args.stage == "schema":
        run_schema()
    elif args.stage == "load":
        run_schema(); run_load()
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
    elif args.stage == "all":
        run_schema()
        run_load()
        run_sanity()
        run_aggregates()
        run_metrics()
        run_ensemble()
        run_ei()

    con.close()
    print(f"Done. stage={args.stage}")
