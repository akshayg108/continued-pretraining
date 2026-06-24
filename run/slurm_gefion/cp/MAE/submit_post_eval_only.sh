#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/max_manifest.csv"
RUNNER="${SCRIPT_DIR}/run_post_eval_only.sh"
AGGREGATOR="${SCRIPT_DIR}/aggregate_post_eval_only.sh"
OUTPUT_ROOT="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/MAE"
COMBINED_DIR="${OUTPUT_ROOT}/combined"
INCLUDE_POOL_STRATEGY="cls"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --include-pool-strategy) INCLUDE_POOL_STRATEGY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${COMBINED_DIR}" "/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"

if [ ! -f "${MANIFEST}" ]; then
    echo "Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

MANIFEST_ROWS="$(python3 - "${MANIFEST}" "${INCLUDE_POOL_STRATEGY}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
include_pool_strategy = sys.argv[2]
with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))
if include_pool_strategy != "all":
    rows = [row for row in rows if row["pool_strategy"] == include_pool_strategy]
print(len(rows))
PY
)"

if [ "${MANIFEST_ROWS}" -le 0 ]; then
    echo "Manifest has no runnable rows for pool_strategy=${INCLUDE_POOL_STRATEGY}: ${MANIFEST}" >&2
    exit 1
fi

SEEDS_PER_RUN=3
TASK_COUNT=$((MANIFEST_ROWS * SEEDS_PER_RUN))
ARRAY_MAX=$((TASK_COUNT - 1))
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "Submitting MAE MAX post eval-only jobs"
echo "Manifest: ${MANIFEST}"
echo "Included pool strategy: ${INCLUDE_POOL_STRATEGY}"
echo "Selected manifest rows: ${MANIFEST_ROWS}"
echo "Array tasks: ${TASK_COUNT} (${MANIFEST_ROWS} runs x ${SEEDS_PER_RUN} seeds; max 6 concurrent)"
echo ""

ARRAY_JOB_ID="$(
    sbatch --parsable --array=0-${ARRAY_MAX}%6 \
        "${RUNNER}" \
        --manifest "${MANIFEST}" \
        --include-pool-strategy "${INCLUDE_POOL_STRATEGY}"
)"

STATUS_CSV="${COMBINED_DIR}/mae_max_eval_fix_status_${ARRAY_JOB_ID}.csv"
COMBINED_CSV="${COMBINED_DIR}/mae_max_eval_fix_combined_results_${ARRAY_JOB_ID}.csv"
JOB_FILE="${COMBINED_DIR}/mae_max_eval_fix_jobs_${ARRAY_JOB_ID}_${TIMESTAMP}.env"

{
    echo "workflow=mae_max_eval_fix"
    echo "submitted_at=${TIMESTAMP}"
    echo "manifest=${MANIFEST}"
    echo "included_pool_strategy=${INCLUDE_POOL_STRATEGY}"
    echo "manifest_rows=${MANIFEST_ROWS}"
    echo "seeds_per_run=${SEEDS_PER_RUN}"
    echo "task_count=${TASK_COUNT}"
    echo "array_job_id=${ARRAY_JOB_ID}"
    echo "STATUS_CSV=${STATUS_CSV}"
    echo "COMBINED_CSV=${COMBINED_CSV}"
} > "${JOB_FILE}"

AGG_JOB_ID="$(
    sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} \
        "${AGGREGATOR}" \
        --manifest "${MANIFEST}" \
        --job-file "${JOB_FILE}" \
        --eval-pool-strategy "${INCLUDE_POOL_STRATEGY}"
)"
{
    echo "aggregation_job_id=${AGG_JOB_ID}"
    echo "job_file=${JOB_FILE}"
} >> "${JOB_FILE}"

echo "Submitted array job: ${ARRAY_JOB_ID}"
echo "Submitted aggregation job: ${AGG_JOB_ID} (dependency: afterany:${ARRAY_JOB_ID})"
echo "Job ID file: ${JOB_FILE}"
