#!/usr/bin/env bash
# ============================================================================
# run_county_sweep.sh -- county-split tolerance sweep for the GeoSim revision.
#
# WHY THIS EXISTS. Table 2 reports spoke-mean county splits of 45.9 against a
# ceiling of int(1.5 * 31) = 46, and spoke-mean cut edges of 2,846 against a
# compactness ceiling of 4,785. The county-split constraint is therefore the
# binding element of the comparison baseline and the compactness constraint is
# inactive -- yet the only constraint sweep in the paper is over compactness.
# This script sweeps the tolerance that actually binds.
#
# DESIGN. Three tolerances x three hub seeds, crossed. The seeds match the
# `replicate` stage, so the existing 1.5x runs at seeds 11/22/33 become the
# loose arm at no extra cost: a 4 x 3 factorial for the price of 9 runs.
# Hub replication at each tolerance is the whole point -- section 5.4.2 shows
# a two-order-of-magnitude p-value spread across hubs at k=10,000, so a
# single-seed tolerance comparison cannot separate constraint effect from hub
# noise. That is exactly the objection you already concede about the
# compactness sweep in 5.5.3; don't reproduce it here.
#
# Run from pipelines/ensemble_generation_layer:
#     bash run_county_sweep.sh pilot     # ~3 min: does the chain still move?
#     bash run_county_sweep.sh sweep     # ~110 min: the 9 production runs
#     bash run_county_sweep.sh sweep-min # ~75 min: drop the 1.25x arm
#     bash run_county_sweep.sh collect   # build the table
# ============================================================================
set -euo pipefail

STAGE="${1:-help}"
WORKERS="${BC_N_WORKERS:-30}"
export BC_N_WORKERS="$WORKERS"
mkdir -p outputs logs

log () { echo "[$(date +%H:%M:%S)] $*"; }

# run_one <run_name> <k> <M> <seed> <county_slack>
run_one () {
  local name="$1" k="$2" m="$3" seed="$4" cs="$5"
  if [[ -f "outputs/bc_run_config_${name}.json" ]]; then
    log "SKIP ${name} (already complete)"; return 0
  fi
  log "START ${name}: k=${k} M=${m} seed=${seed} county_slack=${cs}"
  env BC_COUNTY_SPLITS_SLACK="$cs" \
      BC_RUN_NAME="$name" BC_L_WALK="$k" BC_M_SPOKES="$m" BC_SEED="$seed" \
      python besag_clifford_vra_opportunity.py 2>&1 | tee "logs/${name}.log"
  log "DONE  ${name}"
}

tag () { echo "${1/./p}"; }   # 1.25 -> 1p25

case "$STAGE" in

  # ---- 0. Pilot. The tightest tolerance is the one that can degenerate: at
  #         1.0x the ceiling equals the enacted count (31), so every proposal
  #         that splits one more county self-loops. The enacted plan is still
  #         feasible (31 <= 31) and the test is still exact, but if the hub
  #         accepts zero moves the hub collapses onto the enacted plan and the
  #         ensemble is uninformative.
  #
  #         CHECK IN logs/CS_pilot_1p0.log BEFORE COMMITTING TO THE SWEEP:
  #           "Accepted moves (hub leg)"  -- if 0, the 1.0x arm is degenerate
  #           "accepted-move fraction"    -- the script warns below 0.02
  #         A degenerate arm is still reportable, and arguably interesting
  #         (the tree-weighted measure conditioned on enacted-level county
  #         splitting barely moves), but say so rather than reporting its
  #         rank as if the ensemble had explored anything.
  pilot)
    run_one CS_pilot_1p0  10000 8 11 1.0
    run_one CS_pilot_1p25 10000 8 11 1.25
    log "Compare hub_accepted_moves across the two pilot configs before proceeding."
    ;;

  # ---- 1. Full sweep: 3 tolerances x 3 seeds at k=10,000, M=200.
  #         ~12 min per run on your box => ~110 min total.
  sweep)
    for cs in 1.0 1.1 1.25; do
      for seed in 11 22 33; do
        run_one "TX_k10000_M200_cs$(tag $cs)_seed${seed}" 10000 200 "$seed" "$cs"
      done
    done
    ;;

  # ---- 2. Reduced sweep if the pilot says you're short on time. Keeps the
  #         two arms furthest from the 1.5x default, which is where any real
  #         constraint effect will show up first.
  sweep-min)
    for cs in 1.0 1.1; do
      for seed in 11 22 33; do
        run_one "TX_k10000_M200_cs$(tag $cs)_seed${seed}" 10000 200 "$seed" "$cs"
      done
    done
    ;;

  collect)
    python collect_county_sweep.py --runs 'outputs/bc_run_config_*.json'
    ;;

  *)
    sed -n '3,27p' "$0"
    ;;
esac
