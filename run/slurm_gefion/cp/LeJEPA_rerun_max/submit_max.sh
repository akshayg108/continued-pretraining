#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/max_manifest.csv"
RUNNER="${SCRIPT_DIR}/run_max.sh"
AGGREGATOR="${SCRIPT_DIR}/aggregate_max.sh"
OUTPUT_ROOT="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA_rerun_max"
COMBINED_DIR="${OUTPUT_ROOT}/combined"

mkdir -p "${COMBINED_DIR}" "/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"

if [ ! -f "${MANIFEST}" ]; then
    echo "Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

MANIFEST_ROWS="$(python3 - "${MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"

if [ "${MANIFEST_ROWS}" -le 0 ]; then
    echo "Manifest has no runnable rows: ${MANIFEST}" >&2
    exit 1
fi

SEEDS_PER_RUN=3
TASK_COUNT=$((MANIFEST_ROWS * SEEDS_PER_RUN))
ARRAY_MAX=$((TASK_COUNT - 1))
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "Submitting LeJEPA rerun MAX jobs"
echo "Manifest: ${MANIFEST}"
echo "Manifest rows: ${MANIFEST_ROWS}"
echo "Array tasks: ${TASK_COUNT} (${MANIFEST_ROWS} runs x ${SEEDS_PER_RUN} seeds; max 12 concurrent)"
echo ""

ARRAY_JOB_ID="$(sbatch --parsable --array=0-${ARRAY_MAX}%12 "${RUNNER}" --manifest "${MANIFEST}")"

STATUS_CSV="${COMBINED_DIR}/lejepa_rerun_max_status_${ARRAY_JOB_ID}.csv"
COMBINED_CSV="${COMBINED_DIR}/lejepa_rerun_max_combined_results_${ARRAY_JOB_ID}.csv"
JOB_FILE="${COMBINED_DIR}/lejepa_rerun_max_jobs_${ARRAY_JOB_ID}_${TIMESTAMP}.env"

{
    echo "workflow=lejepa_rerun_max"
    echo "submitted_at=${TIMESTAMP}"
    echo "manifest=${MANIFEST}"
    echo "manifest_rows=${MANIFEST_ROWS}"
    echo "seeds_per_run=${SEEDS_PER_RUN}"
    echo "task_count=${TASK_COUNT}"
    echo "array_job_id=${ARRAY_JOB_ID}"
    echo "STATUS_CSV=${STATUS_CSV}"
    echo "COMBINED_CSV=${COMBINED_CSV}"
} > "${JOB_FILE}"

AGG_JOB_ID="$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} "${AGGREGATOR}" --manifest "${MANIFEST}" --job-file "${JOB_FILE}")"
{
    echo "aggregation_job_id=${AGG_JOB_ID}"
    echo "job_file=${JOB_FILE}"
} >> "${JOB_FILE}"

echo "Submitted array job: ${ARRAY_JOB_ID}"
echo "Submitted aggregation job: ${AGG_JOB_ID} (dependency: afterany:${ARRAY_JOB_ID})"
echo "Job ID file: ${JOB_FILE}"
