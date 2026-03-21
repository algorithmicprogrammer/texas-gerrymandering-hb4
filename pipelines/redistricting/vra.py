from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd


DEFAULT_CVAP_COLUMNS = {
    "HCVAP": "cvap_hisp",
    "BCVAP": "cvap_nh_black",
    "WCVAP": "cvap_nh_white",
    "OCVAP": "cvap_other",
}


@dataclass
class VRAConfig:
    elections: list[str]
    groups: list[str]
    effectiveness_threshold: float
    benchmark: dict[str, Any]
    election_weights: dict[str, float]
    preferred_candidates: dict[str, dict[str, str]]
    confidence_scores: dict[str, dict[str, float]]
    precinct_support_draws_path: str
    candidate_returns_path: str
    cvap_columns: dict[str, str]
    election_sets: list[dict[str, Any]] | None = None
    calibration: dict[str, Any] | None = None
    group_preferred_factors: dict[str, dict[str, float]] | None = None

    @staticmethod
    def from_json(path: str | Path) -> "VRAConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("election_sets") is None:
            raw["election_sets"] = []
        if raw.get("calibration") is None:
            raw["calibration"] = {"kind": "logistic", "a": 8.0, "b": -4.0}
        if raw.get("group_preferred_factors") is None:
            raw["group_preferred_factors"] = {}
        return VRAConfig(**raw)


class DistrictEffectivenessModel:
    """
    Brennan-style district effectiveness evaluator.

    Key fixes relative to the earlier prototype:
      1. district success is based on whether the preferred candidate actually wins
         the district's aggregated vote totals, not whether inferred support is > 0;
      2. elections are grouped into primary/general or primary/runoff/general sets;
      3. district scores use the same scoring rule for the benchmark and the chain;
      4. scores include a group-control factor c = min(2k, 1), where k is district CVAP share;
      5. scores are optionally calibrated by a logistic transform f(cs).
    """

    def __init__(self, cfg: VRAConfig):
        self.cfg = cfg
        self.returns = self._load_candidate_returns(cfg.candidate_returns_path)
        self.return_lookup = self._build_return_lookup(self.returns)
        self._election_sets = self._normalize_election_sets()
        self._cvap_cols = cfg.cvap_columns or DEFAULT_CVAP_COLUMNS

    def _load_candidate_returns(self, path: str) -> pd.DataFrame:
        p = Path(path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        req = {"election_id", "vtd_geoid", "candidate_id", "votes"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"candidate returns missing columns: {missing}")
        df = df.copy()
        df["election_id"] = df["election_id"].astype(str)
        df["vtd_geoid"] = df["vtd_geoid"].astype(str)
        df["candidate_id"] = df["candidate_id"].astype(str)
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0.0)
        for col in ("stage", "office", "candidate_name", "candidate_race", "party"):
            if col not in df.columns:
                df[col] = None
        if "year" not in df.columns:
            df["year"] = None
        return df

    @staticmethod
    def _build_return_lookup(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (eid, vtd), sub in df.groupby(["election_id", "vtd_geoid"], sort=False):
            out.setdefault(str(eid), {})[str(vtd)] = {
                str(r["candidate_id"]): float(r["votes"])
                for _, r in sub[["candidate_id", "votes"]].iterrows()
            }
        return out

    def _normalize_election_sets(self) -> list[dict[str, Any]]:
        if self.cfg.election_sets:
            sets = []
            for item in self.cfg.election_sets:
                sets.append(
                    {
                        "set_id": str(item["set_id"]),
                        "primary": str(item["primary"]) if item.get("primary") else None,
                        "runoff": str(item["runoff"]) if item.get("runoff") else None,
                        "general": str(item["general"]) if item.get("general") else None,
                        "weight": float(item.get("weight", 1.0)),
                        "year": item.get("year"),
                        "office": item.get("office"),
                    }
                )
            return sets

        meta = (
            self.returns[["election_id", "year", "office", "stage"]]
            .drop_duplicates()
            .copy()
        )
        meta["year"] = meta["year"].fillna(-1)
        meta["office"] = meta["office"].fillna("UNKNOWN")
        meta["stage_norm"] = meta["stage"].fillna("").astype(str).str.lower()
        sets: list[dict[str, Any]] = []
        for (year, office), sub in meta.groupby(["year", "office"], sort=False):
            primary = None
            runoff = None
            general = None
            for _, row in sub.iterrows():
                st = row["stage_norm"]
                eid = str(row["election_id"])
                if "general" in st:
                    general = eid
                elif "runoff" in st:
                    runoff = eid
                elif "primary" in st:
                    primary = eid
                elif primary is None:
                    primary = eid
            chosen = [x for x in [primary, runoff, general] if x is not None]
            if not chosen:
                continue
            weights = [float(self.cfg.election_weights.get(eid, 1.0)) for eid in chosen]
            sets.append(
                {
                    "set_id": f"{year}|{office}",
                    "primary": primary,
                    "runoff": runoff,
                    "general": general,
                    "weight": max(weights) if weights else 1.0,
                    "year": None if year == -1 else int(year),
                    "office": office,
                }
            )
        return sets

    @property
    def election_sets(self) -> list[dict[str, Any]]:
        return self._election_sets

    @staticmethod
    def _district_vote_totals(assignment: Mapping[str, str], return_lookup: dict[str, dict[str, dict[str, float]]], election_id: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        precinct_map = return_lookup.get(str(election_id), {})
        for vtd, district in assignment.items():
            votes = precinct_map.get(str(vtd))
            if not votes:
                continue
            d = out.setdefault(str(district), {})
            for cand, v in votes.items():
                d[cand] = d.get(cand, 0.0) + float(v)
        return out

    @staticmethod
    def _winner(votes: Mapping[str, float]) -> str | None:
        if not votes:
            return None
        items = sorted(votes.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        return str(items[0][0])

    @staticmethod
    def _rank(votes: Mapping[str, float], candidate_id: str) -> int | None:
        if candidate_id not in votes:
            return None
        ordered = sorted(votes.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        for idx, (cand, _) in enumerate(ordered, start=1):
            if str(cand) == str(candidate_id):
                return idx
        return None

    @staticmethod
    def _vote_share(votes: Mapping[str, float], candidate_id: str) -> float:
        total = float(sum(votes.values()))
        if total <= 0:
            return 0.0
        return float(votes.get(str(candidate_id), 0.0)) / total

    def _group_member_factor(self, group_id: str, election_id: str, candidate_id: str) -> float:
        explicit = self.cfg.group_preferred_factors or {}
        if group_id in explicit and election_id in explicit[group_id]:
            return float(explicit[group_id][election_id])

        sub = self.returns[
            (self.returns["election_id"] == str(election_id))
            & (self.returns["candidate_id"] == str(candidate_id))
        ]
        if sub.empty or "candidate_race" not in sub.columns:
            return 1.0
        race = str(sub.iloc[0].get("candidate_race") or "").lower()
        if group_id == "HCVAP" and any(x in race for x in ["latino", "hisp", "hispanic"]):
            return 2.0
        if group_id == "BCVAP" and "black" in race:
            return 2.0
        return 1.0

    def _primary_success(self, district_votes: Mapping[str, float], preferred_candidate: str, runoff_votes: Mapping[str, float] | None) -> bool:
        if not district_votes:
            return False
        preferred_candidate = str(preferred_candidate)
        winner = self._winner(district_votes)
        share = self._vote_share(district_votes, preferred_candidate)
        rank = self._rank(district_votes, preferred_candidate)

        if runoff_votes is None:
            return winner == preferred_candidate
        if share > 0.5:
            return True
        if rank in (1, 2) and self._winner(runoff_votes or {}) == preferred_candidate:
            return True
        return False

    def _set_success(self, district_id: str, district_votes_by_election: Mapping[str, Mapping[str, float]], group_id: str, item: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        primary = item.get("primary")
        runoff = item.get("runoff")
        general = item.get("general")
        # If a set only has one election, treat winning that election as success.
        anchor_election = primary or runoff or general
        if anchor_election is None:
            return 0.0, {"ok": False, "reason": "empty_set"}
        preferred = self.cfg.preferred_candidates.get(group_id, {}).get(str(anchor_election))
        confidence = float(self.cfg.confidence_scores.get(group_id, {}).get(str(anchor_election), 0.0))
        if preferred is None:
            return 0.0, {"ok": False, "reason": "missing_preferred_candidate"}

        factor = self._group_member_factor(group_id, str(anchor_election), str(preferred))
        primary_votes = district_votes_by_election.get(str(primary), {}) if primary else None
        runoff_votes = district_votes_by_election.get(str(runoff), {}) if runoff else None
        general_votes = district_votes_by_election.get(str(general), {}) if general else None

        if primary and general:
            primary_ok = self._primary_success(primary_votes or {}, str(preferred), runoff_votes)
            general_ok = self._winner(general_votes or {}) == str(preferred)
            success = 1.0 if (primary_ok and general_ok) else 0.0
        else:
            target_votes = general_votes or runoff_votes or primary_votes or {}
            success = 1.0 if self._winner(target_votes) == str(preferred) else 0.0

        raw_component = float(item.get("weight", 1.0)) * confidence * factor * success
        detail = {
            "district_id": str(district_id),
            "set_id": str(item["set_id"]),
            "preferred_candidate_id": str(preferred),
            "confidence": confidence,
            "group_member_factor": factor,
            "set_weight": float(item.get("weight", 1.0)),
            "success": success,
        }
        return raw_component, detail

    def _district_cvap_share(self, district_id: str, assignment_df: pd.DataFrame, cvap_lookup: pd.DataFrame, group_id: str) -> float:
        group_col = self._cvap_cols[group_id]
        merged = assignment_df.merge(cvap_lookup[["vtd_geoid", "cvap_total", group_col]], on="vtd_geoid", how="left")
        sub = merged[merged["district_id"] == str(district_id)]
        denom = float(pd.to_numeric(sub["cvap_total"], errors="coerce").fillna(0).sum())
        numer = float(pd.to_numeric(sub[group_col], errors="coerce").fillna(0).sum())
        return 0.0 if denom <= 0 else numer / denom

    def _calibrate(self, x: float) -> float:
        cal = self.cfg.calibration or {"kind": "logistic", "a": 8.0, "b": -4.0}
        if cal.get("kind") != "logistic":
            return max(0.0, min(1.0, x))
        a = float(cal.get("a", 8.0))
        b = float(cal.get("b", -4.0))
        z = a * x + b
        z = max(min(z, 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-z))

    def district_effectiveness(self, assignment: Dict[str, str], cvap_lookup: pd.DataFrame) -> dict[str, dict[str, float]]:
        assignment_df = pd.DataFrame({
            "vtd_geoid": [str(k) for k in assignment.keys()],
            "district_id": [str(v) for v in assignment.values()],
        })
        district_votes_by_election: dict[str, dict[str, dict[str, float]]] = {}
        all_election_ids = [str(eid) for eid in self.cfg.elections]
        for eid in all_election_ids:
            district_votes_by_election[eid] = self._district_vote_totals(assignment, self.return_lookup, eid)

        district_ids = sorted(set(str(v) for v in assignment.values()))
        out: dict[str, dict[str, float]] = {}
        for district_id in district_ids:
            out[district_id] = {}
            for group_id in self.cfg.groups:
                weighted_success = 0.0
                total_weight = 0.0
                for item in self.election_sets:
                    component, detail = self._set_success(district_id, {eid: district_votes_by_election.get(eid, {}).get(district_id, {}) for eid in all_election_ids}, group_id, item)
                    total_weight += float(detail["set_weight"]) * float(detail["confidence"]) * float(detail["group_member_factor"])
                    weighted_success += component
                raw_score = (weighted_success / total_weight) if total_weight > 0 else 0.0
                k = self._district_cvap_share(district_id, assignment_df, cvap_lookup, group_id)
                c = min(2.0 * k, 1.0)
                out[district_id][group_id] = self._calibrate(c * raw_score)
        return out

    def plan_effective_counts(self, assignment: Dict[str, str], cvap_lookup: pd.DataFrame) -> dict[str, int]:
        eff = self.district_effectiveness(assignment, cvap_lookup)
        counts = {g: 0 for g in self.cfg.groups}
        total_any = 0
        both = 0
        for district_scores in eff.values():
            passed = []
            for g, s in district_scores.items():
                if s >= self.cfg.effectiveness_threshold:
                    counts[g] += 1
                    passed.append(g)
            if passed:
                total_any += 1
            if len(passed) >= 2:
                both += 1
        counts["TOTAL_ANY"] = total_any
        counts["BOTH"] = both
        return counts


class VRAConstraint:
    def __init__(self, evaluator: DistrictEffectivenessModel, cvap_lookup: pd.DataFrame):
        self.evaluator = evaluator
        self.cvap_lookup = cvap_lookup

    def __call__(self, partition: Any) -> bool:
        assignment = {str(k): str(v) for k, v in partition.assignment.items()}
        counts = self.evaluator.plan_effective_counts(assignment, self.cvap_lookup)
        benchmark = self.evaluator.cfg.benchmark or {}
        per_group = benchmark.get("per_group", benchmark)
        for group_id, min_required in per_group.items():
            if group_id in {"min_total", "min_any", "min_both"}:
                continue
            if counts.get(group_id, 0) < int(min_required):
                return False
        min_total = benchmark.get("min_total")
        if min_total is None:
            min_total = benchmark.get("min_any")
        if min_total is not None and counts.get("TOTAL_ANY", 0) < int(min_total):
            return False
        min_both = benchmark.get("min_both")
        if min_both is not None and counts.get("BOTH", 0) < int(min_both):
            return False
        return True


def build_vra_benchmark_from_enacted(
    con: duckdb.DuckDBPyConnection,
    enacted_plan_id: str,
    vra_config_json: str,
) -> dict[str, Any]:
    """
    Use the same district scoring rule for the enacted-plan benchmark and the chain.
    """
    cfg = VRAConfig.from_json(vra_config_json)
    evaluator = DistrictEffectivenessModel(cfg)
    plan = con.execute(
        "SELECT vtd_geoid, district_id FROM plan_district_vtd WHERE plan_id = ?",
        [enacted_plan_id],
    ).df()
    if plan.empty:
        raise ValueError(f"No enacted assignment found for plan_id={enacted_plan_id}")
    plan["vtd_geoid"] = plan["vtd_geoid"].astype(str)
    plan["district_id"] = plan["district_id"].astype(str)
    assignment = dict(zip(plan["vtd_geoid"], plan["district_id"]))

    needed_cols = ["vtd_geoid", "cvap_total"] + [cfg.cvap_columns[g] for g in cfg.groups if g in cfg.cvap_columns]
    geo = con.execute(
        f"SELECT {', '.join(needed_cols)} FROM geo_vtd"
    ).df()
    geo["vtd_geoid"] = geo["vtd_geoid"].astype(str)

    counts = evaluator.plan_effective_counts(assignment, geo)
    return {
        "per_group": {g: int(counts.get(g, 0)) for g in cfg.groups},
        "min_total": int(counts.get("TOTAL_ANY", 0)),
        "min_both": int(counts.get("BOTH", 0)),
    }
