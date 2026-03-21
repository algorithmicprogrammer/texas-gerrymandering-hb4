from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Any

import duckdb
import numpy as np
import pandas as pd


@dataclass
class VRAConfig:
    elections: list[str]
    groups: list[str]
    effectiveness_threshold: float
    benchmark: dict[str, int]
    election_weights: dict[str, float]
    preferred_candidates: dict[str, dict[str, str]]
    confidence_scores: dict[str, dict[str, float]]
    precinct_support_draws_path: str
    candidate_returns_path: str
    cvap_columns: dict[str, str]

    @staticmethod
    def from_json(path: str | Path) -> "VRAConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return VRAConfig(**raw)


class DistrictEffectivenessModel:
    """
    Fast-ish district effectiveness evaluator for VRA-aware ReCom.

    The intended artifact is produced by the multi-election King EI step:
      - preferred candidate by group x election
      - confidence score by group x election
      - precinct-level posterior support draws by group x candidate x election

    To keep the Markov chain side pure Python, we consume a precomputed JSON/parquet artifact.
    """

    def __init__(self, cfg: VRAConfig):
        self.cfg = cfg
        self.returns = self._load_candidate_returns(cfg.candidate_returns_path)
        self.draws = self._load_draws(cfg.precinct_support_draws_path)
        self._draw_cache: dict[tuple[str, str], np.ndarray] = {}

    def _load_candidate_returns(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        req = {"election_id", "vtd_geoid", "candidate_id", "votes"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"candidate returns missing columns: {missing}")
        df["vtd_geoid"] = df["vtd_geoid"].astype(str)
        df["candidate_id"] = df["candidate_id"].astype(str)
        df["election_id"] = df["election_id"].astype(str)
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0.0)
        return df

    def _load_draws(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "precinct_draws" in raw:
                df = pd.DataFrame(raw["precinct_draws"])
            else:
                df = pd.DataFrame(raw)
        else:
            df = pd.read_parquet(p)
        if df.empty:
            raise ValueError("No EI draw artifact rows found.")
        return df

    def _get_support_draws(self, election_id: str, vtd_geoid: str, group_id: str, candidate_id: str) -> np.ndarray:
        key = (election_id, vtd_geoid)
        if key not in self._draw_cache:
            sub = self.draws[(self.draws["election_id"] == election_id) & (self.draws["vtd_geoid"].astype(str) == str(vtd_geoid))]
            if sub.empty:
                self._draw_cache[key] = np.zeros((0,))
            else:
                rows = []
                for _, r in sub.iterrows():
                    payload = r.get("json")
                    rows.append(json.loads(payload) if isinstance(payload, str) else payload)
                self._draw_cache[key] = rows
        payloads = self._draw_cache[key]
        if isinstance(payloads, np.ndarray):
            return payloads
        vals = []
        for pl in payloads:
            if group_id in pl and candidate_id in pl[group_id]:
                vals.append(float(pl[group_id][candidate_id]))
        return np.asarray(vals, dtype=float)

    def district_effectiveness(self, assignment: Dict[str, str], cvap_lookup: pd.DataFrame) -> dict[str, dict[str, float]]:
        asn = pd.DataFrame({"vtd_geoid": list(assignment.keys()), "district_id": list(assignment.values())})
        asn["vtd_geoid"] = asn["vtd_geoid"].astype(str)
        out: dict[str, dict[str, float]] = {}

        by_district = asn.groupby("district_id", sort=False)["vtd_geoid"].apply(list).to_dict()
        for district_id, vtds in by_district.items():
            group_scores: dict[str, float] = {}
            for group_id in self.cfg.groups:
                weighted_sum = 0.0
                total_weight = 0.0
                for election_id in self.cfg.elections:
                    candidate_id = self.cfg.preferred_candidates[group_id][election_id]
                    confidence = float(self.cfg.confidence_scores[group_id][election_id])
                    weight = float(self.cfg.election_weights.get(election_id, 1.0))

                    all_draws = []
                    for vtd in vtds:
                        support_draws = self._get_support_draws(election_id, vtd, group_id, candidate_id)
                        if support_draws.size == 0:
                            continue
                        votes = self.returns.loc[
                            (self.returns["election_id"] == election_id)
                            & (self.returns["vtd_geoid"] == vtd)
                            & (self.returns["candidate_id"] == candidate_id),
                            "votes",
                        ]
                        vote_total = float(votes.iloc[0]) if not votes.empty else 0.0
                        all_draws.append(support_draws * vote_total)
                    if not all_draws:
                        continue
                    mat = np.vstack(all_draws)
                    district_draws = mat.sum(axis=0)
                    district_pref_win_prob = float(np.mean(district_draws > 0.0))
                    election_score = confidence * district_pref_win_prob
                    weighted_sum += weight * election_score
                    total_weight += weight
                group_scores[group_id] = weighted_sum / total_weight if total_weight > 0 else 0.0
            out[str(district_id)] = group_scores
        return out

    def plan_effective_counts(self, assignment: Dict[str, str], cvap_lookup: pd.DataFrame) -> dict[str, int]:
        eff = self.district_effectiveness(assignment, cvap_lookup)
        counts = {g: 0 for g in self.cfg.groups}
        for district_scores in eff.values():
            for g, s in district_scores.items():
                if s >= self.cfg.effectiveness_threshold:
                    counts[g] += 1
        return counts


class VRAConstraint:
    def __init__(self, evaluator: DistrictEffectivenessModel, cvap_lookup: pd.DataFrame):
        self.evaluator = evaluator
        self.cvap_lookup = cvap_lookup

    def __call__(self, partition: Any) -> bool:
        assignment = {str(k): str(v) for k, v in partition.assignment.items()}
        counts = self.evaluator.plan_effective_counts(assignment, self.cvap_lookup)
        for group_id, min_required in self.evaluator.cfg.benchmark.items():
            if counts.get(group_id, 0) < int(min_required):
                return False
        return True


def build_vra_benchmark_from_enacted(
    con: duckdb.DuckDBPyConnection,
    enacted_plan_id: str,
    ei_json_path: str,
    effectiveness_threshold: float = 0.5,
    group_ids: Iterable[str] = ("HCVAP", "BCVAP"),
) -> dict[str, int]:
    raw = json.loads(Path(ei_json_path).read_text(encoding="utf-8"))
    pref = pd.DataFrame(raw["group_preferences"])
    groups = list(group_ids)
    plan = con.execute(
        "SELECT vtd_geoid, district_id FROM plan_district_vtd WHERE plan_id = ?",
        [enacted_plan_id],
    ).df()
    if plan.empty:
        raise ValueError(f"No enacted assignment found for plan_id={enacted_plan_id}")
    plan["vtd_geoid"] = plan["vtd_geoid"].astype(str)
    benchmark = {g: 0 for g in groups}
    # conservative placeholder: count majority-CVAP districts by group
    district_demo = con.execute(
        "SELECT * FROM district_demo_cvap WHERE plan_id = ?",
        [enacted_plan_id],
    ).df()
    share_map = {"HCVAP": "share_hisp_cvap", "BCVAP": "share_black_cvap", "WCVAP": "share_white_cvap", "OCVAP": "share_minority_cvap"}
    for g in groups:
        col = share_map[g]
        benchmark[g] = int((district_demo[col] >= effectiveness_threshold).sum())
    return benchmark
