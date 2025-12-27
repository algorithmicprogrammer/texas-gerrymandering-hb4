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

@dataclass(frozen=True)
class OpportunityDef:
    opp_def_id: str
    group_share_col: str
    threshold: float
    group_label: str  # "BLACK" / "HISP" / "MINORITY"

DEFAULT_OPP_DEFS: Sequence[OpportunityDef] = (
    OpportunityDef("BVAP50", "share_black_vap", 0.50, "BLACK"),
    OpportunityDef("HVAP50", "share_hisp_vap", 0.50, "HISP"),
    OpportunityDef("MINVAP50", "share_minority_vap", 0.50, "MINORITY"),
)
