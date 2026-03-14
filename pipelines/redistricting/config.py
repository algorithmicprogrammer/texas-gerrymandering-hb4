from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List

RACE_COLS_VAP: List[str] = [
    "vap_nh_white",
    "vap_nh_black",
    "vap_hisp",
    "vap_nh_asian",
    "vap_nh_native",
    "vap_other",
]

RACE_COLS_CVAP: List[str] = [
    "cvap_nh_white",
    "cvap_nh_black",
    "cvap_hisp",
    "cvap_nh_asian",
    "cvap_nh_native",
    "cvap_other",
]

@dataclass(frozen=True)
class OpportunityDef:
    opp_def_id: str
    group_share_col: str
    threshold: float
    group_label: str  # "BLACK" / "HISP" / "MINORITY"

DEFAULT_OPP_DEFS: Sequence[OpportunityDef] = (
    OpportunityDef("BCVAP50", "share_black_cvap", 0.50, "BLACK"),
    OpportunityDef("HCVAP50", "share_hisp_cvap", 0.50, "HISP"),
    OpportunityDef("MINCVAP50", "share_minority_cvap", 0.50, "MINORITY"),
)
