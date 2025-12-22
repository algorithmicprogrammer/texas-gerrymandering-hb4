from typing import Any, Dict

def build_summary(
    diagnostics: Dict[str, float],
    gingles3: Dict[str, float],
    opp_loss: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "diagnostics": diagnostics,
        "gingles3": gingles3,
        "opportunity_loss": opp_loss,
    }
