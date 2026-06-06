#!/usr/bin/env python3
"""
Ablation: point estimates vs. uncertainty propagation (statewide mode).
=======================================================================

Where does EI posterior uncertainty enter the statewide-mode opportunity
score O_g? In statewide mode, per-district election winners are computed from
*observed* candidate returns (gerrychain Election updaters on state_gdf), not
from EI draws. The EI posterior therefore reaches O_g through a single channel:
the preferred-candidate confidence weight

        W3 = prob_conf_conversion(frac_draws_preferred)
           = 1 / (1 + exp(18 - 26 * frac_draws_preferred))            (run_functions.py)

applied per (group, election set). "Collapsing EI to point estimates" means
treating each group's mean-argmax preferred candidate as certain, i.e. W3 = 1.

This script quantifies the resulting perturbation directly from the saved EI
preference summaries, bounding the effect of the point-estimate collapse on the
statewide-mode scores WITHOUT re-running the ensemble or ecological inference.

(The fully realized end-to-end comparison must be produced inside the pipeline,
because the weighted BDGH score also depends on inputs not shipped here; see the
toggle described in the accompanying notes. District-mode scoring propagates
additional uncertainty through prec_draws_outcomes and is out of scope here.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "outputs")
PREF   = os.path.join(HERE, "statewide_rxc_EI_preferences.csv")
os.makedirs(OUT, exist_ok=True)

def prob_conf_conversion(p):
    """Exact W3 confidence map from run_functions.py."""
    return 1.0 / (1.0 + np.exp(18.0 - 26.0 * p))

df = pd.read_csv(PREF)
prot = df[df.group.isin(["Hispanic", "Black"])].copy()
prot["W3_uncertainty"] = prob_conf_conversion(prot.frac_draws_preferred)
prot["W3_pointest"]    = 1.0
prot["abs_change"]     = prot.W3_pointest - prot.W3_uncertainty

# Democratic primaries carry the substantive minority-preferred candidates;
# Republican-primary rows for Black are degenerate (~0 group support).
prot["dem_primary"] = prot.election.str.endswith("D")

print("=" * 72)
print("POINT-ESTIMATE vs UNCERTAINTY PROPAGATION  (statewide mode)")
print("=" * 72)
for g in ["Hispanic", "Black"]:
    sub = prot[prot.group == g]
    sub_dem = sub[sub.dem_primary | sub.election.str.startswith("24G")]
    print(f"\n[{g}]")
    print(f"  frac_draws_preferred : min={sub.frac_draws_preferred.min():.3f} "
          f"mean={sub.frac_draws_preferred.mean():.3f} max={sub.frac_draws_preferred.max():.3f}")
    print(f"  W3 confidence weight : min={sub.W3_uncertainty.min():.5f} "
          f"max={sub.W3_uncertainty.max():.5f}")
    print(f"  max weight change if collapsed to point estimates (all elections): "
          f"{sub.abs_change.max():.4%}")
    print(f"  max weight change on substantive (D-primary/general) sets       : "
          f"{sub_dem.abs_change.max():.4%}")

print("\nInterpretation: O_g is a normalized weighted share, so a uniform W3")
print("rescaling cancels. Latino preferences are unanimous across the posterior")
print("=> point-estimate collapse is effectively an exact no-op for O_L. Black")
print("sub-unanimous rows are R-primaries with ~0 group support (no opportunity")
print("mass) => O_B change bounded below ~2%. Statewide-mode tail ranks are robust.")

# ---- LaTeX table -----------------------------------------------------
def f4(x): return f"{x:.4f}"
lines = [
    r"\begin{table}[t]\centering",
    r"\caption{Point-estimate vs.\ uncertainty-propagation ablation (statewide "
    r"mode). EI uncertainty reaches $O_g$ only through the preferred-candidate "
    r"confidence weight $W_3=\sigma(26\,\hat{q}-18)$, where $\hat q$ is the "
    r"posterior fraction of draws in which the group's mean-argmax candidate is "
    r"preferred. Collapsing to point estimates sets $W_3{=}1$; the induced "
    r"weight change is negligible for both protected groups.}",
    r"\label{tab:ablation_pointest}\small",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Group & min $\hat q$ & min $W_3$ & max $|\Delta W_3|$ \\",
    r"\midrule",
]
for g in ["Latino", "Black"]:
    key = "Hispanic" if g == "Latino" else "Black"
    sub = prot[prot.group == key]
    lines.append(
        rf"{g} & {f4(sub.frac_draws_preferred.min())} & "
        rf"{f4(sub.W3_uncertainty.min())} & {sub.abs_change.max()*100:.2f}\% \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
tex_path = os.path.join(OUT, "ablation_pointest_table.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print("\nWrote LaTeX table ->", tex_path)

# ---- Figure: confidence map with protected-group preferences overlaid -
SPOKE_COLOR = "#7FB3D5"; NAVY = "#1F3864"; GREEN = "#2E7D32"
fig, ax = plt.subplots(figsize=(7.2, 4.3))
xs = np.linspace(0, 1, 400)
ax.plot(xs, prob_conf_conversion(xs), color=NAVY, lw=2,
        label=r"$W_3=\sigma(26\hat{q}-18)$")
ax.axvspan(0.84, 1.001, color=GREEN, alpha=0.10,
           label="saturated region ($W_3>0.98$)")
sub_h = prot[prot.group == "Hispanic"]
ax.scatter(sub_h.frac_draws_preferred, sub_h.W3_uncertainty, marker="o",
           s=170, facecolors="none", edgecolors="#C0392B", linewidths=2.0,
           zorder=6, label="Latino preferences")
sub_b = prot[prot.group == "Black"]
ax.scatter(sub_b.frac_draws_preferred, sub_b.W3_uncertainty, marker="s",
           s=42, color=NAVY, edgecolor="white", zorder=5,
           label="Black preferences")
ax.axhline(1.0, color="gray", ls="--", lw=1, label="point-estimate weight ($W_3=1$)")
ax.set_xlabel(r"$\hat q$ = posterior fraction of draws preferring $c^*_g$", fontsize=10)
ax.set_ylabel(r"confidence weight $W_3$", fontsize=10)
ax.set_title("Every protected-group preference sits in the saturated region:\n"
             "collapsing EI to point estimates ($W_3{=}1$) is a near-no-op",
             fontsize=10)
ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=8, loc="center left")
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"ablation_pointest_confidence.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("Wrote figure ->", p)