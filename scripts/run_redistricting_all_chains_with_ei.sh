#!/usr/bin/env bash
set -euo pipefail

BASE="data/processed/ensembles/ENS_TXCD_2024_recom_v1"

# Shared inputs
GEO_VTD="data/processed/geo_vtd.parquet"
ELECTIONS="data/processed/elections.parquet"
RETURNS="data/processed/election_returns_vtd.parquet"
PLANS="data/processed/plans.parquet"
PLAN_MAP="data/processed/plan_district_vtd.parquet"

# EI settings
EI_ELECTION_ID="TX_SEN_2024_GEN"
EI_DRAWS="2000"
EI_TUNE="3000"
EI_CHAINS="4"
EI_TARGET_ACCEPT="0.97"
EI_MAX_TREEDEPTH="15"

# Chain IDs + seeds (match what you used when generating ensembles)
declare -a CHAINS=("c1" "c2" "c3" "c4")
declare -a SEEDS=("101" "202" "303" "404")

for i in "${!CHAINS[@]}"; do
  c="${CHAINS[$i]}"
  seed="${SEEDS[$i]}"

  ENSEMBLE_ID="ENS_TXCD_2024_recom_v1_${c}"
  DB="${BASE}/${c}.duckdb"

  ENSEMBLE_PLANS="${BASE}/${ENSEMBLE_ID}_plans.parquet"
  ENSEMBLE_PLANMAP="${BASE}/${ENSEMBLE_ID}_planmap.parquet"

  OUTDIR="${BASE}/derived_${c}"
  mkdir -p "${OUTDIR}"

  echo "============================================================"
  echo "Running redistricting pipeline for ${ENSEMBLE_ID}"
  echo "DB: ${DB}"
  echo "OUT: ${OUTDIR}"
  echo "============================================================"

  # (Optional) clean slate if rerunning
  rm -f "${DB}"

  # SCHEMA (REQUIRED before load)
  python -m pipelines.redistricting.cli \
    --db "${DB}" \
    --stage schema

  # LOAD
  python -m pipelines.redistricting.cli \
    --db "${DB}" \
    --stage load \
    --geo-vtd "${GEO_VTD}" \
    --elections "${ELECTIONS}" \
    --returns "${RETURNS}" \
    --plans "${PLANS}" \
    --plan-map "${PLAN_MAP}" \
    --ensemble-plans "${ENSEMBLE_PLANS}" \
    --ensemble-plan-map "${ENSEMBLE_PLANMAP}"

  # SANITY / AGGREGATES / METRICS / ENSEMBLE
  python -m pipelines.redistricting.cli --db "${DB}" --stage sanity
  python -m pipelines.redistricting.cli --db "${DB}" --stage aggregates
  python -m pipelines.redistricting.cli --db "${DB}" --stage metrics

  python -m pipelines.redistricting.cli \
    --db "${DB}" \
    --stage ensemble \
    --ensemble-id "${ENSEMBLE_ID}"

  # EI (run after metrics)
  python -m pipelines.redistricting.cli \
    --db "${DB}" \
    --stage ei \
    --ensemble-id "${ENSEMBLE_ID}" \
    --ei-election-id "${EI_ELECTION_ID}" \
    --ei-run-id "EI_${EI_ELECTION_ID}_${c}" \
    --ei-draws "${EI_DRAWS}" \
    --ei-tune "${EI_TUNE}" \
    --ei-chains "${EI_CHAINS}" \
    --ei-target-accept "${EI_TARGET_ACCEPT}" \
    --ei-max-treedepth "${EI_MAX_TREEDEPTH}" \
    --ei-seed "${seed}"

  # EXPORT
  python -m pipelines.redistricting.cli \
    --db "${DB}" \
    --stage export \
    --out-dir "${OUTDIR}" \
    --export-format parquet

  echo "✅ Done ${ENSEMBLE_ID}"
done

echo "✅ All chains complete."