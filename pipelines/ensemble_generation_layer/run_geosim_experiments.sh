#!/usr/bin/env bash
# ============================================================================
# run_geosim_experiments.sh -- experiment matrix for the GeoSim revision.
#
# Run from pipelines/ensemble_generation_layer (the ensemble module imports
# run_functions as a top-level module and writes to ./outputs).
#
#     cd pipelines/ensemble_generation_layer
#     bash run_geosim_experiments.sh pilot     # 10 minutes: calibrate timing
#     bash run_geosim_experiments.sh ksweep    # the headline new experiment
#     bash run_geosim_experiments.sh replicate # hub replication
#     bash run_geosim_experiments.sh tighten   # exact tightened-constraint test
#     bash run_geosim_experiments.sh rescore   # zero-new-chain ablations
#
# Every stage writes outputs/bc_run_config_<RUN>.json recording k, M, the
# seed, the realized acceptance fraction, and how far the spokes travelled --
# the reproducibility gap Reviewer 2 flagged.
#
# TIMING. The submitted run was M=1000 spokes at k=50,000 in 905 min on this
# machine, i.e. roughly 28 CPU-minutes per 50,000-step chain. Run the `pilot`
# stage first and scale from its printed per-spoke time before committing to
# the rest; if the pilot says the 50k stage will not finish in time, drop
# K_LIST to "2000 10000" and report the sweep over those values.
# ============================================================================
set -euo pipefail

STAGE="${1:-help}"
WORKERS="${BC_N_WORKERS:-30}"
export BC_N_WORKERS="$WORKERS"
mkdir -p outputs logs

log () { echo "[$(date +%H:%M:%S)] $*"; }

run_one () {  # run_one <run_name> <k> <M> <seed> [extra env assignments...]
  local name="$1" k="$2" m="$3" seed="$4"; shift 4
  if [[ -f "outputs/bc_run_config_${name}.json" ]]; then
    log "SKIP ${name} (already complete)"; return 0
  fi
  log "START ${name}: k=${k} M=${m} seed=${seed} ${*}"
  env "$@" \
      BC_RUN_NAME="$name" BC_L_WALK="$k" BC_M_SPOKES="$m" BC_SEED="$seed" \
      python besag_clifford_vra_opportunity.py 2>&1 | tee "logs/${name}.log"
  log "DONE  ${name}"
}

case "$STAGE" in

  # ---- 0. Ten-minute calibration run. Confirms the patched script runs and
  #         prints a per-spoke wall time to budget the rest of the matrix.
  pilot)
    run_one TX_pilot_k2000 2000 8 20260812
    log "Per-spoke time is in logs/TX_pilot_k2000.log; scale before continuing."
    ;;

  # ---- 1. Chain-length sweep. THE headline new experiment: exactness holds
  #         at every k, so this shows the rank-1 Latino result is not an
  #         artifact of one arbitrarily chosen amount of simulated movement.
  #         M=200 throughout so the resolution floor (1/201) is common across
  #         rows; the submitted M=1000, k=50,000 run is reported alongside.
  ksweep)
    for k in ${K_LIST:-2000 10000 50000}; do
      run_one "TX_k${k}_M200" "$k" 200 20260813
    done
    ;;

  # ---- 2. Hub replication. Different master seeds give different hub states
  #         and independent spoke ensembles; the rank should be stable.
  #         Reviewer 1 asked for exactly this.
  replicate)
    for seed in 11 22 33; do
      run_one "TX_k10000_M200_seed${seed}" 10000 200 "$seed"
    done
    ;;

  # ---- 3. Exact tightened-constraint test. The submitted paper filtered the
  #         existing ensemble, which is descriptive only. Regenerating the
  #         ensemble under the tighter tolerance makes the test exact for that
  #         constrained measure. Cheap because exactness does not depend on k.
  tighten)
    for slack in 1.2 1.0; do
      tag="${slack/./p}"
      run_one "TX_k10000_M200_comp${tag}" 10000 200 20260814 \
              BC_COMPACTNESS_SLACK="$slack"
    done
    ;;

  # ---- 4. Second enacted plan (stretch goal). Requires a CD2021 column in
  #         the precinct parquet (see add_second_plan.py) AND its own
  #         contiguity repair: bridge edges are plan-specific, so the enacted
  #         plan for this run must induce connected districts before the
  #         sampler will start.
  secondplan)
    run_one TX_PLANC2193_k10000_M200 10000 200 20260814 BC_START_MAP=CD2021
    ;;

  # ---- 5. Zero-new-chain ablations over the ensemble already generated for
  #         the submission: threshold sweep, uncertainty propagation, and
  #         locality diagnostics.
  rescore)
    RUN="${RUN:-TX_BC_functional}"
    python rescore_plans.py \
      --plans "outputs/bc_plan_assignments_${RUN}.parquet" \
      --out   "outputs/bc_rescored_${RUN}.csv" \
      --workers "$WORKERS" \
      --check-results "outputs/bc_opportunity_results_${RUN}.csv"
    python analyze_ablations.py threshold --scores "outputs/bc_rescored_${RUN}.csv"
    python analyze_ablations.py modes     --scores "outputs/bc_rescored_${RUN}.csv"
    python analyze_ablations.py locality  --scores "outputs/bc_rescored_${RUN}.csv" \
      --plans "outputs/bc_plan_assignments_${RUN}.parquet"
    ;;

  # ---- 6. Collect every completed run into the configuration table.
  collect)
    python analyze_ablations.py ksweep --runs 'outputs/bc_run_config_*.json'
    ;;

  *)
    sed -n '3,25p' "$0"
    ;;
esac
