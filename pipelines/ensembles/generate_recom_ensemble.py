from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Geospatial + GerryChain
import geopandas as gpd
from gerrychain import GeographicPartition, Graph, MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import within_percent_of_ideal_population
from gerrychain.proposals import recom
from gerrychain.updaters import Tally, cut_edges

# Efficient Parquet writing
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class RunConfig:
    ensemble_id: str
    n_steps: int
    burnin: int
    thin: int
    epsilon: float
    seed: int
    pop_col: str
    id_col: str
    enacted_assignment_col: str
    out_plan_map: Path
    out_plans: Path
    flush_every_plans: int


def _read_assignments(enacted_plan_map_path: str, id_col: str) -> pd.DataFrame:
    """
    Reads enacted plan mapping: must contain columns [id_col, district_id] or [id_col, district].
    """
    df = pd.read_parquet(enacted_plan_map_path)
    if id_col not in df.columns:
        raise ValueError(f"enacted plan-map is missing id column {id_col!r}. Columns: {df.columns.tolist()}")

    # tolerate either district_id or district
    if "district_id" in df.columns:
        dist_col = "district_id"
    elif "district" in df.columns:
        dist_col = "district"
    else:
        raise ValueError(
            "enacted plan-map must include a district column named 'district_id' or 'district'. "
            f"Columns: {df.columns.tolist()}"
        )

    out = df[[id_col, dist_col]].copy()
    out = out.rename(columns={dist_col: "district_id"})
    out["district_id"] = out["district_id"].astype(str)
    out[id_col] = out[id_col].astype(str)
    return out


def _attach_population(gdf: gpd.GeoDataFrame, geo_vtd_path: str, id_col: str, pop_col: str) -> gpd.GeoDataFrame:
    """
    Attach population from your tabular geo_vtd.parquet to the geodataframe.
    """
    tab = pd.read_parquet(geo_vtd_path)
    if id_col not in tab.columns:
        raise ValueError(f"{geo_vtd_path} missing id column {id_col!r}. Columns: {tab.columns.tolist()}")
    if pop_col not in tab.columns:
        raise ValueError(f"{geo_vtd_path} missing pop_col {pop_col!r}. Columns: {tab.columns.tolist()}")

    tab = tab[[id_col, pop_col]].copy()
    tab[id_col] = tab[id_col].astype(str)

    gdf = gdf.copy()
    gdf[id_col] = gdf[id_col].astype(str)
    gdf = gdf.merge(tab, on=id_col, how="left")

    if gdf[pop_col].isna().any():
        n_missing = int(gdf[pop_col].isna().sum())
        raise ValueError(
            f"Population merge produced {n_missing} missing {pop_col!r} values. "
            "That means IDs between your VTD geometry file and geo_vtd.parquet do not match."
        )

    # GerryChain expects numeric pop
    gdf[pop_col] = pd.to_numeric(gdf[pop_col], errors="raise").astype(int)
    return gdf


def _build_graph(gdf: gpd.GeoDataFrame, id_col: str) -> Graph:
    """
    Build an adjacency graph from polygons.
    """
    # Ensure index is the node id used by Graph
    gdf = gdf.set_index(id_col, drop=False)
    graph = Graph.from_geodataframe(gdf)
    return graph


def _build_initial_partition(
    graph: Graph,
    gdf: gpd.GeoDataFrame,
    assignments: pd.DataFrame,
    cfg: RunConfig,
) -> Tuple[GeographicPartition, int]:
    """
    Build a GeographicPartition using enacted assignment.
    Returns (partition, num_districts).
    """
    # Assignment dict: node -> district_id
    amap = dict(zip(assignments[cfg.id_col].astype(str), assignments["district_id"].astype(str)))

    # Confirm all nodes have an assignment
    missing = [n for n in graph.nodes if str(n) not in amap]
    if missing:
        raise ValueError(
            f"Enacted assignment missing {len(missing)} graph nodes. "
            f"Example missing ids: {missing[:10]}"
        )

    # Attach population as node attribute
    pop_lookup = dict(zip(gdf[cfg.id_col].astype(str), gdf[cfg.pop_col].astype(int)))
    for n in graph.nodes:
        graph.nodes[n][cfg.pop_col] = int(pop_lookup[str(n)])

    updaters = {
        "population": Tally(cfg.pop_col, alias="population"),
        "cut_edges": cut_edges,
    }

    part = GeographicPartition(graph, assignment=amap, updaters=updaters)
    num_districts = len(set(part.assignment.values()))
    return part, num_districts


def _make_writer(out_path: Path) -> pq.ParquetWriter:
    schema = pa.schema(
        [
            ("plan_id", pa.string()),
            ("vtd_geoid", pa.string()),
            ("district_id", pa.string()),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(str(out_path), schema=schema, compression="zstd")


def _write_plan_map_batch(writer: pq.ParquetWriter, rows: Iterable[Tuple[str, str, str]]) -> None:
    plan_ids, vtds, dists = zip(*rows)
    table = pa.table(
        {
            "plan_id": pa.array(plan_ids, type=pa.string()),
            "vtd_geoid": pa.array(vtds, type=pa.string()),
            "district_id": pa.array(dists, type=pa.string()),
        }
    )
    writer.write_table(table)


def _write_plans_metadata(plans_meta: list[dict], out_plans: Path) -> None:
    out_plans.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plans_meta).to_parquet(out_plans, index=False)


def generate_recom_ensemble(
    vtd_geofile: str,
    geo_vtd_path: str,
    enacted_plan_map_path: str,
    cfg: RunConfig,
) -> None:
    """
    Main routine:
      - Load VTD geometries
      - Merge population
      - Load enacted plan assignment
      - Run ReCom chain
      - Write ensemble_plan_district_vtd.parquet + ensemble_plans.parquet
    """
    np.random.seed(cfg.seed)

    # 1) read geometries (shp, geojson, geoparquet all OK)
    gdf = gpd.read_file(vtd_geofile)
    if cfg.id_col not in gdf.columns:
        raise ValueError(f"{vtd_geofile} missing id column {cfg.id_col!r}. Columns: {gdf.columns.tolist()}")

    # 2) attach population
    gdf = _attach_population(gdf, geo_vtd_path, cfg.id_col, cfg.pop_col)

    # 3) build graph
    graph = _build_graph(gdf, cfg.id_col)

    # 4) enacted assignments
    enacted = _read_assignments(enacted_plan_map_path, cfg.id_col)

    # 5) initial partition
    init_part, k = _build_initial_partition(graph, gdf, enacted, cfg)

    # 6) population constraint target
    total_pop = sum(init_part["population"].values())
    ideal = total_pop / k
    pop_constraint = within_percent_of_ideal_population(init_part, cfg.epsilon)

    # 7) proposal (ReCom)
    proposal = partial(
        recom,
        pop_col=cfg.pop_col,
        pop_target=ideal,
        epsilon=cfg.epsilon,
        node_repeats=1,  # keep simple + stable
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint],
        accept=always_accept,
        initial_state=init_part,
        total_steps=cfg.n_steps,
    )

    # 8) output writers
    # Overwrite output if exists (avoid appending to stale runs)
    if cfg.out_plan_map.exists():
        cfg.out_plan_map.unlink()

    writer = _make_writer(cfg.out_plan_map)

    # 9) iterate chain, write thinned plans
    kept = 0
    plan_rows_buffer: list[Tuple[str, str, str]] = []
    plans_meta: list[dict] = []

    def should_keep(step_idx: int) -> bool:
        if step_idx < cfg.burnin:
            return False
        return ((step_idx - cfg.burnin) % cfg.thin) == 0

    for step_idx, part in enumerate(chain):
        if not should_keep(step_idx):
            continue

        kept += 1
        plan_id = f"{cfg.ensemble_id}_{kept:06d}"

        # assignment is a dict: node_id -> district label
        # Normalize to strings
        for node_id, dist in part.assignment.items():
            plan_rows_buffer.append((plan_id, str(node_id), str(dist)))

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
                        "epsilon": cfg.epsilon,
                        "ideal_pop": ideal,
                        "n_districts": k,
                        "burnin": cfg.burnin,
                        "thin": cfg.thin,
                        "n_steps": cfg.n_steps,
                    }
                ),
                created_at=None,
            )
        )

        # flush in batches
        if kept % cfg.flush_every_plans == 0:
            _write_plan_map_batch(writer, plan_rows_buffer)
            plan_rows_buffer = []

            print(f"[ensemble-gen] kept {kept} plans (wrote batch)")

    # flush tail
    if plan_rows_buffer:
        _write_plan_map_batch(writer, plan_rows_buffer)

    writer.close()

    # Write plans metadata
    _write_plans_metadata(plans_meta, cfg.out_plans)

    print(f"[ensemble-gen] done. kept={kept}")
    print(f"[ensemble-gen] wrote plan-map: {cfg.out_plan_map}")
    print(f"[ensemble-gen] wrote plans meta: {cfg.out_plans}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate ReCom ensemble plan maps (plan_id, vtd_geoid, district_id)")

    ap.add_argument("--vtd-geo", required=True, help="VTD geometry file (shp/geojson/geoparquet)")
    ap.add_argument("--geo-vtd", required=True, help="Tabular VTD data parquet (must include pop_col)")
    ap.add_argument("--enacted-plan-map", required=True, help="Enacted plan map parquet (plan_id,vtd_geoid,district_id)")
    ap.add_argument("--ensemble-id", required=True, help="Ensemble id prefix (e.g., ENS_TXCD_2024_recom_v1)")

    ap.add_argument("--out-plan-map", required=True, help="Output parquet for ensemble plan-map (BIG)")
    ap.add_argument("--out-plans", required=True, help="Output parquet for ensemble plans metadata (SMALL)")

    ap.add_argument("--id-col", default="vtd_geoid", help="VTD id column name shared across inputs")
    ap.add_argument("--pop-col", default="vap_total", help="Population column name (used for constraints)")
    ap.add_argument("--epsilon", type=float, default=0.01, help="Population deviation tolerance (e.g., 0.01 = 1%)")

    ap.add_argument("--n-steps", type=int, default=5000, help="Total chain steps")
    ap.add_argument("--burnin", type=int, default=500, help="Burn-in steps to discard")
    ap.add_argument("--thin", type=int, default=10, help="Keep every `thin` steps after burnin")
    ap.add_argument("--seed", type=int, default=20240101, help="Random seed")

    ap.add_argument("--flush-every-plans", type=int, default=25, help="Flush to parquet every N kept plans")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    cfg = RunConfig(
        ensemble_id=args.ensemble_id,
        n_steps=args.n_steps,
        burnin=args.burnin,
        thin=args.thin,
        epsilon=args.epsilon,
        seed=args.seed,
        pop_col=args.pop_col,
        id_col=args.id_col,
        enacted_assignment_col="district_id",
        out_plan_map=Path(args.out_plan_map),
        out_plans=Path(args.out_plans),
        flush_every_plans=args.flush_every_plans,
    )

    generate_recom_ensemble(
        vtd_geofile=args.vtd_geo,
        geo_vtd_path=args.geo_vtd,
        enacted_plan_map_path=args.enacted_plan_map,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
