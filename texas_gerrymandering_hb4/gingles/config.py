from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Columns:
    vtd_id: str = "vtd_id"
    cd_enacted: str = "cd_enacted"
    pop_total: str = "pop_total"

    dem_share: str = "dem_share"
    rep_share: str = "rep_share"
    dem_share_2p: str = "dem_share_2p"

    # Race proportions (must exist OR be computed upstream)
    race_cols: List[str] = None

    def __post_init__(self):
        if self.race_cols is None:
            object.__setattr__(self, "race_cols", [
                "p_white", "p_black", "p_latino", "p_asian", "p_native", "p_other"
            ])

@dataclass(frozen=True)
class ModelParams:
    draws: int = 1500
    tune: int = 1500
    chains: int = 4
    target_accept: float = 0.92
    random_seed: int = 42

@dataclass(frozen=True)
class AnalysisParams:
    # Gingles-like thresholds
    opportunity_prob_thresh: float = 0.5
    cohesion_thresh: float = 0.6
    pol_gap_delta: float = 0.15

    # Ensemble settings
    n_plans: int = 200
    plan_random_seed: int = 123

    # Minority specification
    minority: str = "black"          # "black" or "latino" etc.
    coalition: bool = False          # if True, default coalition: black+latino

@dataclass(frozen=True)
class Paths:
    raw_input: str = "data/raw/vtd_units_raw.parquet"
    processed_dir: str = "data/processed"
    results_dir: str = "data/results"
