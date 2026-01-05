from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

from gerrychain import GeographicPartition, Graph, MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import within_percent_of_ideal_population
from gerrychain.proposals import recom
from gerrychain.updaters import Tally, cut_edges

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class RunConfig:
    ensemble_id: str
    vtds_geo: Path
    enacted_plan_map: Path
    out_plan_map: Path
    out_plans: Path

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
    Prefer GeoSeries.make_valid when available (newer GeoPandas/Shapely),
    otherwise fallback to buffer(0).
    """
    invalid_mask = ~gdf.geometry.is_valid
    n_bad = int(invalid_mask.sum())
    if n_bad == 0:
        return gdf

    # show a few IDs (index is vtd_geoid in our pipeline)
    bad_ids = gdf.loc[invalid_mask].index.astype(str).tolist()[:10]
    print(f"[geom] found {n_bad} invalid geometries (examples: {bad_ids}). Repairing...")

    try:
        # GeoPandas 0.13+ may expose make_valid on GeoSeries
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask].geometry.make_valid()
    except Exception:
        # classic self-intersection fix
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask].geometry.buffer(0)

    still_bad = int((~gdf.geometry.is_valid).sum())
    if still_bad:
        print(f"[geom] WARNING: {still_bad} geometries still invalid after repair.")
    else:
        print("[geom] repair complete: all geometries valid.")
    return gdf


def generate_recom_ensemble(cfg: RunConfig) -> None:
    _ensure_outputs(cfg)
    np.random.seed(cfg.seed)

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

    # Set node ids = vtd_geoid
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

    assignment: Dict[str, str] = {str(n): str(vtd_to_dist[str(n)]) for n in graph.nodes}

    # Attach population to graph nodes (gerrychain reads node attributes)
    pop_map = gdf[cfg.pop_col].astype(int).to_dict()
    for n in graph.nodes:
        graph.nodes[n][cfg.pop_col] = int(pop_map[str(n)])

    updaters = {"population": Tally(cfg.pop_col, alias="population"), "cut_edges": cut_edges}
    init_part = GeographicPartition(graph, assignment=assignment, updaters=updaters)

    districts = list(set(init_part.assignment.values()))
    k = len(districts)
    total_pop = sum(init_part["population"].values())
    ideal = total_pop / k

    # -----------------------------
    # IMPORTANT FIX:
    # MarkovChain requires initial_state to satisfy constraints.
    # If enacted plan violates epsilon under pop_col, auto-relax epsilon.
    # -----------------------------
    pops = list(init_part["population"].values())
    max_dev = max(abs(p - ideal) / ideal for p in pops)
    eps = float(cfg.epsilon)
    if max_dev > eps:
        eps = float(max_dev) + 1e-6
        print(
            f"[pop] enacted plan violates epsilon={cfg.epsilon:.6f} under pop_col={cfg.pop_col!r}. "
            f"Max deviation is {max_dev:.6f}; relaxing epsilon to {eps:.6f} so the chain can start."
        )

    pop_constraint = within_percent_of_ideal_population(init_part, eps)

    # -----------------------------
    # IMPORTANT FIX:
    # Your gerrychain version's recom expects (partition, ...) as first arg,
    # so proposal must be a function(partition)->new_partition.
    # -----------------------------
    def proposal(partition):
        return recom(
            partition,
            pop_col=cfg.pop_col,
            pop_target=ideal,
            epsilon=eps,
            node_repeats=1,
        )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint],
        accept=always_accept,
        initial_state=init_part,
        total_steps=cfg.n_steps,
    )

    # Overwrite outputs
    if cfg.out_plan_map.exists():
        cfg.out_plan_map.unlink()
    if cfg.out_plans.exists():
        cfg.out_plans.unlink()

    writer = _make_planmap_writer(cfg.out_plan_map)

    kept = 0
    buffer: List[Tuple[str, str, str]] = []
    plans_meta: List[dict] = []

    def keep(step: int) -> bool:
        return step >= cfg.burnin and ((step - cfg.burnin) % cfg.thin) == 0

    for step_idx, part in enumerate(chain):
        if not keep(step_idx):
            continue

        kept += 1
        plan_id = f"{cfg.ensemble_id}_{kept:06d}"

        for vtd_geoid, dist in part.assignment.items():
            buffer.append((plan_id, str(vtd_geoid), str(dist)))

        plans_meta.append(
            dict(
                plan_id=plan_id,
                plan_type="ENSEMBLE",
                cycle=None,
                chamber=None,
                ensemble_id=cfg.ensemble_id,
                generator="recom",
                seed=int(cfg.seed),
                constraints_json=json.dumps(
                    {
                        "proposal": "recom",
                        "epsilon_requested": cfg.epsilon,
                        "epsilon_used": eps,
                        "max_dev_enacted": float(max_dev),
                        "ideal_pop": float(ideal),
                        "n_districts": int(k),
                        "burnin": int(cfg.burnin),
                        "thin": int(cfg.thin),
                        "n_steps": int(cfg.n_steps),
                        "pop_col": cfg.pop_col,
                    }
                ),
                created_at=None,
            )
        )

        if kept % cfg.flush_every_plans == 0:
            _write_planmap_rows(writer, buffer)
            buffer = []
            print(f"[ensemble] kept {kept} plans (flushed)")

    _write_planmap_rows(writer, buffer)
    writer.close()

    pd.DataFrame(plans_meta).to_parquet(cfg.out_plans, index=False)

    print(f"[ensemble] done. kept={kept}")
    print(f"[ensemble] wrote plan-map: {cfg.out_plan_map}")
    print(f"[ensemble] wrote plans meta: {cfg.out_plans}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate ReCom ensemble from processed geospatial VTDs")

    ap.add_argument("--vtds-geo", required=True, help="Geospatial VTDs parquet with vtd_geoid+geometry+pop_col")
    ap.add_argument("--enacted-plan-map", required=True, help="Enacted plan map parquet (vtd_geoid -> district_id)")
    ap.add_argument("--ensemble-id", required=True)

    ap.add_argument("--out-plan-map", required=True, help="Output parquet plan-map for ensemble")
    ap.add_argument("--out-plans", required=True, help="Output parquet plans metadata for ensemble")

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

    return ap


def main() -> None:
    args = build_parser().parse_args()

    cfg = RunConfig(
        ensemble_id=args.ensemble_id,
        vtds_geo=Path(args.vtds_geo),
        enacted_plan_map=Path(args.enacted_plan_map),
        out_plan_map=Path(args.out_plan_map),
        out_plans=Path(args.out_plans),
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
    )

    generate_recom_ensemble(cfg)


if __name__ == "__main__":
    main()

