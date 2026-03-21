from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .db import connect_db
from .schema import create_schema, upsert_opp_defs
from .config import DEFAULT_OPP_DEFS
from .io_load import (
    load_geo_vtd,
    load_election,
    load_election_returns_vtd,
    load_candidate_returns_vtd,
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
from .ei.king_vra import run_king_ei_multielection
from .vra import build_vra_benchmark_from_enacted


STAGES = [
    "schema",
    "load",
    "sanity",
    "aggregates",
    "metrics",
    "ensemble",
    "ei",
    "ei-king-multi",
    "export",
    "all",
]


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


def main() -> None:
    p = argparse.ArgumentParser(description="Redistricting pipeline with Brennan-style VRA-aware ReCom support")

    p.add_argument("--db", default=":memory:", help="DuckDB path or ':memory:'")
    p.add_argument("--stage", default="all", choices=STAGES)

    p.add_argument("--geo-vtd", help="VTD demographics parquet/csv")
    p.add_argument("--elections", help="Election metadata parquet/csv")
    p.add_argument("--returns", help="Two-party VTD election returns parquet/csv")
    p.add_argument("--candidate-returns", help="Candidate-level VTD returns parquet/csv for King EI")
    p.add_argument("--plans", help="Plan metadata parquet/csv (enacted)")
    p.add_argument("--plan-map", help="Assignments (enacted): (plan_id, vtd_geoid, district_id)")
    p.add_argument("--ensemble-plans", help="Plan metadata parquet/csv (ensemble)")
    p.add_argument("--ensemble-plan-map", help="Assignments (ensemble): (plan_id, vtd_geoid, district_id)")

    p.add_argument("--ensemble-id", help="Ensemble id")
    p.add_argument("--enacted-plan-id", help="Enacted plan id used for VRA benchmark")
    p.add_argument("--ei-election-id", help="Legacy single-election EI id")
    p.add_argument("--ei-run-id", default="EI_RUN_001")
    p.add_argument("--king-ei-json-out", help="Output JSON artifact for multi-election King EI")
    p.add_argument("--vra-config-json-out", help="Output JSON config consumed by VRA-aware ReCom")
    p.add_argument("--rscript-bin", default="Rscript")
    p.add_argument("--effectiveness-threshold", type=float, default=0.6)

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
        if args.candidate_returns:
            load_candidate_returns_vtd(con, args.candidate_returns)
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

    def run_ei_king_multi():
        if not args.candidate_returns or not args.elections or not args.king_ei_json_out:
            raise ValueError("For stage=ei-king-multi, provide --candidate-returns --elections --king-ei-json-out")
        out_json = run_king_ei_multielection(
            con=con,
            candidate_returns_path=args.candidate_returns,
            elections_path=args.elections,
            output_json=args.king_ei_json_out,
            rscript_bin=args.rscript_bin,
        )
        print(f"[ei-king-multi] wrote {out_json}")
        if args.vra_config_json_out:
            if not args.enacted_plan_id:
                raise ValueError("Provide --enacted-plan-id when writing --vra-config-json-out")
            raw = json.loads(Path(out_json).read_text(encoding="utf-8"))
            pref = {}
            conf = {}
            weights = {}
            elections = []
            groups = []
            for row in raw["group_preferences"]:
                g = row["group_id"]
                e = row["election_id"]
                pref.setdefault(g, {})[e] = row["preferred_candidate_id"]
                conf.setdefault(g, {})[e] = float(row["confidence"])
                weights[e] = float(row.get("election_weight", 1.0))
                if e not in elections:
                    elections.append(e)
                if g not in groups:
                    groups.append(g)

            cfg = {
                "elections": elections,
                "groups": groups,
                "effectiveness_threshold": args.effectiveness_threshold,
                "benchmark": {},
                "election_weights": weights,
                "preferred_candidates": pref,
                "confidence_scores": conf,
                "precinct_support_draws_path": out_json,
                "candidate_returns_path": args.candidate_returns,
                "cvap_columns": {
                    "HCVAP": "cvap_hisp",
                    "BCVAP": "cvap_nh_black",
                    "WCVAP": "cvap_nh_white",
                    "OCVAP": "cvap_other",
                },
                "election_sets": raw.get("election_sets", []),
                "calibration": {"kind": "logistic", "a": 8.0, "b": -4.0},
                "group_preferred_factors": {},
            }
            tmp_cfg_path = Path(args.vra_config_json_out).with_suffix(".tmp.json")
            tmp_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            cfg["benchmark"] = build_vra_benchmark_from_enacted(
                con,
                enacted_plan_id=args.enacted_plan_id,
                vra_config_json=str(tmp_cfg_path),
            )
            Path(args.vra_config_json_out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            tmp_cfg_path.unlink(missing_ok=True)
            print(f"[ei-king-multi] wrote {args.vra_config_json_out}")

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
    if args.stage in ("ei-king-multi", "all"):
        run_ei_king_multi()
    if args.stage in ("export", "all"):
        run_export()

    print(f"Done. stage={args.stage}")
