# pipelines/ensembles/generate_recom_ensemble.py
#
# ReCom ensemble generation with:
#   - geometry repair (make_valid / buffer(0))
#   - reproducible seeding
#   - population constraint (STRICT: never relaxes epsilon)
#   - contiguity constraint
#   - optional compactness constraint via cut-edges cap (fast)
#   - ReCom failure resilience (node_repeats, bounded bipartition_tree, reselection retries)
#   - streaming plan-map parquet output (zstd)
#   - plans metadata parquet output
#   - optional per-plan stats parquet output (cut_edges, max_pop_dev)
#
# PUBLISHABLE FIX (Option A):
#   - Enforce exactly the requested epsilon (never auto-relax).
#   - If enacted plan is invalid under (contiguity OR requested epsilon),
#     do NOT use enacted as initial_state. Instead, seed with a valid plan
#     created by recursive_tree_part on the same graph.

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gerrychain import GeographicPartition, Graph, MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import within_percent_of_ideal_population, contiguous
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree, recursive_tree_part
from gerrychain.updaters import Tally, cut_edges

from redistricting.vra import VRAConfig, DistrictEffectivenessModel, VRAConstraint


@dataclass
class RunConfig:
    ensemble_id: str
    vtds_geo: Path
    enacted_plan_map: Path
    out_plan_map: Path
    out_plans: Path
    out_plan_stats: Optional[Path]

    pop_col: str
    epsilon: float
    n_steps: int
    burnin: int
    thin: int
    seed: int

    vtd_id_col: str
    enacted_vtd_col: str
    enacted_dist_col: str

    flush_every_plans: int
    ignore_geometry_errors: bool

    # ReCom robustness knobs
    node_repeats: int
    reselect_tries: int
    bipartition_max_attempts: int

    # Compactness (fast) via cut-edges cap
    enable_cut_edges_cap: bool
    max_cut_edges_factor: float
    vra_config_json: Optional[Path]


def _read_enacted_map(path: Path, vtd_col: str, dist_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if vtd_col not in df.columns:
        raise ValueError(f"enacted plan-map missing {vtd_col!r}. Columns: {df.columns.tolist()}")

    # tolerate alias
    if dist_col not in df.columns:
        if dist_col == "district_id" and "district" in df.columns:
            dist_col = "district"
        else:
            raise ValueError(f"enacted plan-map missing {dist_col!r}. Columns: {df.columns.tolist()}")

    out = df[[vtd_col, dist_col]].copy()
    out = out.rename(columns={vtd_col: "vtd_geoid", dist_col: "district_id"})
    out["vtd_geoid"] = out["vtd_geoid"].astype(str)
    out["district_id"] = out["district_id"].astype(str)
    return out


def _ensure_outputs(cfg: RunConfig) -> None:
    cfg.out_plan_map.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_plans.parent.mkdir(parents=True, exist_ok=True)
    if cfg.out_plan_stats is not None:
        cfg.out_plan_stats.parent.mkdir(parents=True, exist_ok=True)


def _make_planmap_writer(out_path: Path) -> pq.ParquetWriter:
    schema = pa.schema(
        [
            ("plan_id", pa.string()),
            ("vtd_geoid", pa.string()),
            ("district_id", pa.string()),
        ]
    )
    return pq.ParquetWriter(str(out_path), schema=schema, compression="zstd")


def _write_planmap_rows(writer: pq.ParquetWriter, rows: List[Tuple[str, str, str]]) -> None:
    if not rows:
        return
    plan_ids, vtds, dists = zip(*rows)
    writer.write_table(
        pa.table(
            {
                "plan_id": pa.array(plan_ids, type=pa.string()),
                "vtd_geoid": pa.array(vtds, type=pa.string()),
                "district_id": pa.array(dists, type=pa.string()),
            }
        )
    )


def _repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Repair invalid geometries in-place.
    Prefer GeoSeries.make_valid when available, otherwise fallback to buffer(0).
    """
    invalid_mask = ~gdf.geometry.is_valid
    n_bad = int(invalid_mask.sum())
    if n_bad == 0:
        return gdf

    bad_ids = gdf.loc[invalid_mask].index.astype(str).tolist()[:10]
    print(f"[geom] found {n_bad} invalid geometries (examples: {bad_ids}). Repairing...")

    try:
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask].geometry.make_valid()
    except Exception:
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask].geometry.buffer(0)

    still_bad = int((~gdf.geometry.is_valid).sum())
    if still_bad:
        print(f"[geom] WARNING: {still_bad} geometries still invalid after repair.")
    else:
        print("[geom] repair complete: all geometries valid.")
    return gdf


def _contiguity_failures(graph: Graph, part: GeographicPartition) -> List[Tuple[str, int, int]]:
    """
    Return list of (district_id, n_nodes, n_components) for districts whose induced subgraph is disconnected.
    """
    import networkx as nx

    bad: List[Tuple[str, int, int]] = []
    for dist, nodes in part.parts.items():
        sg = graph.subgraph(nodes)
        n_nodes = int(sg.number_of_nodes())
        if n_nodes == 0:
            bad.append((str(dist), 0, 0))
            continue
        if not nx.is_connected(sg):
            bad.append((str(dist), n_nodes, int(nx.number_connected_components(sg))))
    return bad


def generate_recom_ensemble(cfg: RunConfig) -> None:
    _ensure_outputs(cfg)

    # Reproducibility (gerrychain uses Python random internally)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Read geospatial VTDs (must include geometry + vtd_geoid + pop_col)
    gdf = gpd.read_parquet(cfg.vtds_geo)
    if cfg.vtd_id_col not in gdf.columns:
        raise ValueError(f"{cfg.vtds_geo} missing {cfg.vtd_id_col!r}. Columns: {gdf.columns.tolist()}")
    if "geometry" not in gdf.columns:
        raise ValueError(f"{cfg.vtds_geo} missing geometry column.")
    if cfg.pop_col not in gdf.columns:
        raise ValueError(f"{cfg.vtds_geo} missing pop col {cfg.pop_col!r}. Columns: {gdf.columns.tolist()}")

    gdf = gdf.copy()
    gdf[cfg.vtd_id_col] = gdf[cfg.vtd_id_col].astype(str)
    gdf[cfg.pop_col] = pd.to_numeric(gdf[cfg.pop_col], errors="raise").astype(int)

    # Node ids = vtd_geoid
    gdf = gdf.set_index(cfg.vtd_id_col, drop=False)

    # Repair invalid geometries before graph build
    gdf = _repair_geometries(gdf)

    # Build adjacency graph
    if cfg.ignore_geometry_errors:
        graph = Graph.from_geodataframe(gdf, ignore_errors=True)
    else:
        graph = Graph.from_geodataframe(gdf)

    # Read enacted assignment
    enacted = _read_enacted_map(cfg.enacted_plan_map, cfg.enacted_vtd_col, cfg.enacted_dist_col)
    vtd_to_dist = enacted.set_index("vtd_geoid")["district_id"].to_dict()

    missing = [str(n) for n in graph.nodes if str(n) not in vtd_to_dist]
    if missing:
        raise ValueError(
            f"Enacted plan-map is missing {len(missing)} VTDs relative to the VTD geometry graph. "
            f"Example missing: {missing[:10]}"
        )

    enacted_assignment: Dict[str, str] = {str(n): str(vtd_to_dist[str(n)]) for n in graph.nodes}

    # Attach population to graph nodes
    pop_map = gdf[cfg.pop_col].astype(int).to_dict()
    for n in graph.nodes:
        graph.nodes[n][cfg.pop_col] = int(pop_map[str(n)])

    updaters = {
        "population": Tally(cfg.pop_col, alias="population"),
        "cut_edges": cut_edges,
    }

    enacted_part = GeographicPartition(graph, assignment=enacted_assignment, updaters=updaters)

    # Ideal population
    district_labels = sorted(set(enacted_part.assignment.values()))
    k = len(district_labels)
    total_pop = sum(enacted_part["population"].values())
    ideal = total_pop / k

    # STRICT epsilon: never relax
    eps = float(cfg.epsilon)

    # Compute enacted deviation under requested epsilon (for reporting and initial-state decision)
    pops_enacted = list(enacted_part["population"].values())
    max_dev_enacted = max(abs(p - ideal) / ideal for p in pops_enacted)
    enacted_pop_ok = (max_dev_enacted <= eps)

    if enacted_pop_ok:
        print(
            f"[pop] enacted plan satisfies requested epsilon={eps:.6f} under pop_col={cfg.pop_col!r}. "
            f"Max deviation is {max_dev_enacted:.6f}."
        )
    else:
        print(
            f"[pop] WARNING: enacted plan violates requested epsilon={eps:.6f} under pop_col={cfg.pop_col!r}. "
            f"Max deviation is {max_dev_enacted:.6f}. "
            f"Proceeding WITHOUT relaxing epsilon; will seed from a valid plan."
        )

    pop_constraint = within_percent_of_ideal_population(enacted_part, eps)

    # Optional compactness: cap cut edges relative to enacted plan (reference)
    enacted_cut_edges = len(enacted_part["cut_edges"])
    max_cut_edges = int(enacted_cut_edges * float(cfg.max_cut_edges_factor))

    def cut_edges_cap_constraint(partition: GeographicPartition) -> bool:
        return len(partition["cut_edges"]) <= max_cut_edges

    constraints = [pop_constraint, contiguous]

    if cfg.vra_config_json is not None:
        vra_cfg = VRAConfig.from_json(cfg.vra_config_json)
        cvap_lookup = gdf.reset_index(drop=True)[[cfg.vtd_id_col] + [c for c in vra_cfg.cvap_columns.values() if c in gdf.columns]].copy()
        cvap_lookup = cvap_lookup.rename(columns={cfg.vtd_id_col: "vtd_geoid"})
        evaluator = DistrictEffectivenessModel(vra_cfg)
        constraints.append(VRAConstraint(evaluator, cvap_lookup))
        print(f"[vra] enabled benchmark constraint from {cfg.vra_config_json}")
    if cfg.enable_cut_edges_cap:
        constraints.append(cut_edges_cap_constraint)
        print(
            f"[compactness] enabled cut-edges cap: enacted={enacted_cut_edges}, "
            f"factor={cfg.max_cut_edges_factor:.3f}, max={max_cut_edges}"
        )
    else:
        print("[compactness] cut-edges cap disabled")

    # If enacted plan is not contiguous, log (but do not start from it)
    enacted_contig_fail = _contiguity_failures(graph, enacted_part)
    enacted_contiguous_under_graph = (len(enacted_contig_fail) == 0)

    if not enacted_contiguous_under_graph:
        print("[contiguity] enacted plan is NOT contiguous under this graph representation.")
        print("[contiguity] example failures (district, nodes, components):", enacted_contig_fail[:10])

    # -----------------------------
    # PUBLISHABLE FIX (Option A):
    # Seed from a valid plan if enacted fails contiguity OR fails requested epsilon.
    # -----------------------------
    need_seed = (not enacted_contiguous_under_graph) or (not enacted_pop_ok)

    if need_seed:
        if not enacted_pop_ok:
            print(
                f"[pop] enacted plan violates requested epsilon={eps:.6f} "
                f"(max_dev={max_dev_enacted:.6f}); cannot use enacted as initial_state."
            )

        print("[init] generating a valid seed plan with recursive_tree_part...")

        seed_assignment = recursive_tree_part(
            graph,
            parts=district_labels,
            pop_target=ideal,
            pop_col=cfg.pop_col,
            epsilon=eps,
        )
        initial_part = GeographicPartition(graph, assignment=seed_assignment, updaters=updaters)

        # Verify contiguity
        seed_fail = _contiguity_failures(graph, initial_part)
        if seed_fail:
            raise RuntimeError(
                "Seed plan unexpectedly non-contiguous. This suggests a deeper issue with the graph "
                "or input data. Example failures: " + str(seed_fail[:10])
            )

        # Verify population deviation under requested epsilon
        seed_pops = list(initial_part["population"].values())
        seed_max_dev = max(abs(p - ideal) / ideal for p in seed_pops)
        if seed_max_dev > eps + 1e-12:
            raise RuntimeError(
                f"Seed plan unexpectedly violates requested epsilon={eps:.6f}. "
                f"seed_max_dev={seed_max_dev:.6f}"
            )

        print("[init] seed plan OK (contiguous + population-valid). Proceeding with ReCom chain.")
        initial_state_label = "recursive_tree_part"
    else:
        initial_part = enacted_part
        initial_state_label = "enacted_plan"
        print("[init] enacted plan is valid (contiguous + population-valid). Using enacted as initial_state.")

    # ReCom resilience: bounded bipartition_tree per try + reselection retries
    method = partial(bipartition_tree, max_attempts=int(cfg.bipartition_max_attempts))

    def proposal(partition: GeographicPartition) -> GeographicPartition:
        last_err: Exception | None = None
        for _ in range(int(cfg.reselect_tries)):
            try:
                return recom(
                    partition,
                    pop_col=cfg.pop_col,
                    pop_target=ideal,
                    epsilon=eps,
                    node_repeats=int(cfg.node_repeats),
                    method=method,
                )
            except RuntimeError as e:
                last_err = e
                continue

        raise RuntimeError(
            f"ReCom failed after {cfg.reselect_tries} reselection tries "
            f"(node_repeats={cfg.node_repeats}, bipartition_max_attempts={cfg.bipartition_max_attempts}). "
            f"Last error: {last_err}"
        )

    chain = MarkovChain(
        proposal=proposal,
        constraints=constraints,
        accept=always_accept,
        initial_state=initial_part,
        total_steps=int(cfg.n_steps),
    )

    # Overwrite outputs
    if cfg.out_plan_map.exists():
        cfg.out_plan_map.unlink()
    if cfg.out_plans.exists():
        cfg.out_plans.unlink()
    if cfg.out_plan_stats is not None and cfg.out_plan_stats.exists():
        cfg.out_plan_stats.unlink()

    writer = _make_planmap_writer(cfg.out_plan_map)

    kept = 0
    buffer: List[Tuple[str, str, str]] = []
    plans_meta: List[dict] = []
    plan_stats: List[dict] = []

    def keep(step: int) -> bool:
        return step >= int(cfg.burnin) and ((step - int(cfg.burnin)) % int(cfg.thin)) == 0

    for step_idx, part in enumerate(chain):
        if not keep(step_idx):
            continue

        kept += 1
        plan_id = f"{cfg.ensemble_id}_{kept:06d}"

        # plan-map rows
        for vtd_geoid, dist in part.assignment.items():
            buffer.append((plan_id, str(vtd_geoid), str(dist)))

        # per-plan diagnostics (cheap)
        pops_now = list(part["population"].values())
        max_dev_now = max(abs(p - ideal) / ideal for p in pops_now)
        cut_edges_now = len(part["cut_edges"])

        if cfg.out_plan_stats is not None:
            plan_stats.append(
                dict(
                    plan_id=plan_id,
                    step=int(step_idx),
                    cut_edges=int(cut_edges_now),
                    max_pop_dev=float(max_dev_now),
                )
            )

        constraints_json = {
            "proposal": "recom",
            "pop_col": cfg.pop_col,
            "epsilon_requested": float(cfg.epsilon),
            "epsilon_used": float(eps),  # always equal to epsilon_requested under Option A
            "max_dev_enacted": float(max_dev_enacted),
            "enacted_pop_ok_under_graph": bool(enacted_pop_ok),
            "ideal_pop": float(ideal),
            "n_districts": int(k),
            "burnin": int(cfg.burnin),
            "thin": int(cfg.thin),
            "n_steps": int(cfg.n_steps),
            "seed": int(cfg.seed),
            # initial state provenance
            "initial_state": initial_state_label,
            "enacted_contiguous_under_graph": bool(enacted_contiguous_under_graph),
            "enacted_contiguity_failures_sample": enacted_contig_fail[:10],
            # proposal robustness
            "node_repeats": int(cfg.node_repeats),
            "reselect_tries": int(cfg.reselect_tries),
            "bipartition_max_attempts": int(cfg.bipartition_max_attempts),
            # compactness
            "contiguity": True,
            "enable_cut_edges_cap": bool(cfg.enable_cut_edges_cap),
            "enacted_cut_edges": int(enacted_cut_edges),
            "max_cut_edges_factor": float(cfg.max_cut_edges_factor),
            "max_cut_edges": int(max_cut_edges),
            # diagnostics convenience
            "cut_edges_this_plan": int(cut_edges_now),
            "max_pop_dev_this_plan": float(max_dev_now),
        }

        plans_meta.append(
            dict(
                plan_id=plan_id,
                plan_type="ENSEMBLE",
                cycle=None,
                chamber=None,
                ensemble_id=cfg.ensemble_id,
                generator="recom",
                constraints_json=json.dumps(constraints_json),
                created_at=None,
            )
        )

        if kept % int(cfg.flush_every_plans) == 0:
            _write_planmap_rows(writer, buffer)
            buffer = []
            print(f"[ensemble] kept {kept} plans (flushed)")

    _write_planmap_rows(writer, buffer)
    writer.close()

    pd.DataFrame(plans_meta).to_parquet(cfg.out_plans, index=False)
    if cfg.out_plan_stats is not None:
        pd.DataFrame(plan_stats).to_parquet(cfg.out_plan_stats, index=False)

    print(f"[ensemble] done. kept={kept}")
    print(f"[ensemble] wrote plan-map: {cfg.out_plan_map}")
    print(f"[ensemble] wrote plans meta: {cfg.out_plans}")
    if cfg.out_plan_stats is not None:
        print(f"[ensemble] wrote plan stats: {cfg.out_plan_stats}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate ReCom ensemble from processed geospatial VTDs")

    ap.add_argument("--vtds-geo", required=True, help="Geospatial VTDs parquet with vtd_geoid+geometry+pop_col")
    ap.add_argument("--enacted-plan-map", required=True, help="Enacted plan map parquet (vtd_geoid -> district_id)")
    ap.add_argument("--ensemble-id", required=True)

    ap.add_argument("--out-plan-map", required=True, help="Output parquet plan-map for ensemble")
    ap.add_argument("--out-plans", required=True, help="Output parquet plans metadata for ensemble")
    ap.add_argument(
        "--out-plan-stats",
        default=None,
        help="Optional output parquet per-plan stats (cut_edges, max_pop_dev).",
    )

    ap.add_argument("--pop-col", default="vap_total")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--burnin", type=int, default=500)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20240101)

    ap.add_argument("--vtd-id-col", default="vtd_geoid")
    ap.add_argument("--enacted-vtd-col", default="vtd_geoid")
    ap.add_argument("--enacted-dist-col", default="district_id")

    ap.add_argument("--flush-every-plans", type=int, default=25)
    ap.add_argument(
        "--ignore-geometry-errors",
        action="store_true",
        help="If set, build graph with ignore_errors=True (last resort).",
    )

    # ReCom robustness knobs
    ap.add_argument(
        "--node-repeats",
        type=int,
        default=20,
        help="ReCom node_repeats (higher increases chance of finding valid splits).",
    )
    ap.add_argument(
        "--reselect-tries",
        type=int,
        default=50,
        help="How many times to retry (reselect) if ReCom fails with RuntimeError.",
    )
    ap.add_argument(
        "--bipartition-max-attempts",
        type=int,
        default=2000,
        help="Max attempts inside bipartition_tree per try.",
    )

    # Compactness via cut-edges cap
    ap.add_argument(
        "--enable-cut-edges-cap",
        action="store_true",
        help="Enable compactness constraint: cap cut edges relative to enacted plan.",
    )
    ap.add_argument(
        "--max-cut-edges-factor",
        type=float,
        default=1.15,
        help="If enabled, max cut edges allowed = enacted_cut_edges * this factor.",
    )
    ap.add_argument("--vra-config-json", default=None, help="Optional Brennan-style VRA config JSON")

    return ap


def main() -> None:
    args = build_parser().parse_args()

    cfg = RunConfig(
        ensemble_id=args.ensemble_id,
        vtds_geo=Path(args.vtds_geo),
        enacted_plan_map=Path(args.enacted_plan_map),
        out_plan_map=Path(args.out_plan_map),
        out_plans=Path(args.out_plans),
        out_plan_stats=(Path(args.out_plan_stats) if args.out_plan_stats else None),
        pop_col=args.pop_col,
        epsilon=args.epsilon,
        n_steps=args.n_steps,
        burnin=args.burnin,
        thin=args.thin,
        seed=args.seed,
        vtd_id_col=args.vtd_id_col,
        enacted_vtd_col=args.enacted_vtd_col,
        enacted_dist_col=args.enacted_dist_col,
        flush_every_plans=args.flush_every_plans,
        ignore_geometry_errors=args.ignore_geometry_errors,
        node_repeats=args.node_repeats,
        reselect_tries=args.reselect_tries,
        bipartition_max_attempts=args.bipartition_max_attempts,
        enable_cut_edges_cap=bool(args.enable_cut_edges_cap),
        max_cut_edges_factor=float(args.max_cut_edges_factor),
        vra_config_json=(Path(args.vra_config_json) if args.vra_config_json else None),
    )

    generate_recom_ensemble(cfg)


if __name__ == "__main__":
    main()
