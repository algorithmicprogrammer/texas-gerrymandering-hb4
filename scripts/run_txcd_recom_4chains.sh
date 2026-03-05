#!/usr/bin/env bash
set -euo pipefail

# ---- Inputs (your current paths) ----
VTDS_GEO="data/processed/geospatial/vtds.parquet"
ENACTED_PLAN_MAP="data/processed/plan_district_vtd.parquet"

# ---- Output directory ----
OUTDIR="data/processed/ensembles/ENS_TXCD_2024_recom_v1"
mkdir -p "${OUTDIR}"

# ---- Shared parameters ----
POP_COL="total_pop"
EPSILON="0.01"

N_STEPS="60000"
BURNIN="5000"
THIN="5"

NODE_REPEATS="25"
RESELECT_TRIES="75"
BIPARTITION_MAX_ATTEMPTS="3000"

ENABLE_CUT_EDGES_CAP="--enable-cut-edges-cap"
MAX_CUT_EDGES_FACTOR="1.15"

# ---- Chain definitions ----
# Feel free to change seeds or add more chains.
declare -a CHAIN_IDS=("c1" "c2" "c3" "c4")
declare -a SEEDS=("101" "202" "303" "404")

# ---- Manifest ----
MANIFEST="${OUTDIR}/manifest.json"

cat > "${MANIFEST}" <<EOF
{
  "ensemble_base_id": "ENS_TXCD_2024_recom_v1",
  "vtds_geo": "${VTDS_GEO}",
  "enacted_plan_map": "${ENACTED_PLAN_MAP}",
  "pop_col": "${POP_COL}",
  "epsilon": ${EPSILON},
  "n_steps": ${N_STEPS},
  "burnin": ${BURNIN},
  "thin": ${THIN},
  "node_repeats": ${NODE_REPEATS},
  "reselect_tries": ${RESELECT_TRIES},
  "bipartition_max_attempts": ${BIPARTITION_MAX_ATTEMPTS},
  "compactness": {
    "type": "cut_edges_cap",
    "enabled": true,
    "max_cut_edges_factor": ${MAX_CUT_EDGES_FACTOR}
  },
  "chains": [
EOF

for i in "${!CHAIN_IDS[@]}"; do
  CHAIN="${CHAIN_IDS[$i]}"
  SEED="${SEEDS[$i]}"

  ENSEMBLE_ID="ENS_TXCD_2024_recom_v1_${CHAIN}"

  OUT_PLAN_MAP="${OUTDIR}/${ENSEMBLE_ID}_planmap.parquet"
  OUT_PLANS="${OUTDIR}/${ENSEMBLE_ID}_plans.parquet"
  OUT_PLAN_STATS="${OUTDIR}/${ENSEMBLE_ID}_planstats.parquet"
  OUT_LOG="${OUTDIR}/${ENSEMBLE_ID}.log"

  echo "=== Running ${ENSEMBLE_ID} (seed=${SEED}) ==="
  echo "Log: ${OUT_LOG}"

  # Run and tee logs
  python -m pipelines.ensembles.generate_recom_ensemble \
    --vtds-geo "${VTDS_GEO}" \
    --enacted-plan-map "${ENACTED_PLAN_MAP}" \
    --ensemble-id "${ENSEMBLE_ID}" \
    --out-plan-map "${OUT_PLAN_MAP}" \
    --out-plans "${OUT_PLANS}" \
    --out-plan-stats "${OUT_PLAN_STATS}" \
    --pop-col "${POP_COL}" \
    --epsilon "${EPSILON}" \
    --n-steps "${N_STEPS}" \
    --burnin "${BURNIN}" \
    --thin "${THIN}" \
    --seed "${SEED}" \
    --node-repeats "${NODE_REPEATS}" \
    --reselect-tries "${RESELECT_TRIES}" \
    --bipartition-max-attempts "${BIPARTITION_MAX_ATTEMPTS}" \
    ${ENABLE_CUT_EDGES_CAP} \
    --max-cut-edges-factor "${MAX_CUT_EDGES_FACTOR}" \
    2>&1 | tee "${OUT_LOG}"

  # Append chain info to manifest
  cat >> "${MANIFEST}" <<EOF
    {
      "chain_id": "${CHAIN}",
      "ensemble_id": "${ENSEMBLE_ID}",
      "seed": ${SEED},
      "out_plan_map": "${OUT_PLAN_MAP}",
      "out_plans": "${OUT_PLANS}",
      "out_plan_stats": "${OUT_PLAN_STATS}",
      "log": "${OUT_LOG}"
    }$( [[ "$i" -lt "$((${#CHAIN_IDS[@]} - 1))" ]] && echo "," || echo "" )
EOF
done

cat >> "${MANIFEST}" <<EOF
  ]
}
EOF

echo
echo "=== Done ==="
echo "Manifest: ${MANIFEST}"
echo "Outputs in: ${OUTDIR}"
