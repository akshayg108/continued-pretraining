#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/remaining_manifest.csv"
RUNNER="${SCRIPT_DIR}/run_remaining.sh"
AGGREGATOR="${SCRIPT_DIR}/aggregate_remaining.sh"
OUTPUT_ROOT="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA"
COMBINED_DIR="${OUTPUT_ROOT}/combined"
MISSING_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --missing-only) MISSING_ONLY=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

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

RUNNER_MANIFEST="${MANIFEST}"
RUNNER_ROWS="${MANIFEST_ROWS}"
SEEDS_PER_RUN=3
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [ "${MISSING_ONLY}" -eq 1 ]; then
    RUNNER_MANIFEST="${COMBINED_DIR}/lejepa_remaining_missing_manifest_${TIMESTAMP}.csv"
    RUNNER_ROWS="$(
        python3 - "${MANIFEST}" "${RUNNER_MANIFEST}" <<'PY'
import csv
import json
import sys
from pathlib import Path

source_manifest = Path(sys.argv[1])
target_manifest = Path(sys.argv[2])
seeds = [42, 43, 44]
output_root = Path("/dcai/projects/iu_0092/projects/cp/outputs/logs")

with source_manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

fieldnames = rows[0].keys() if rows else []
missing_rows = []

for row in rows:
    if row["group"] == "random":
        log_dir = output_root / "cp" / "LeJEPA" / "random" / row["display_name"]
    else:
        log_dir = output_root / "cp" / "LeJEPA" / "pretrained" / row["display_name"] / row["backbone_tag"]

    complete = True
    for seed in seeds:
        result_path = log_dir / f"{row['backbone_tag']}_{row['dataset']}_n{row['n_samples']}_seed{seed}.json"
        if not result_path.exists():
            complete = False
            break

    if not complete:
        missing_rows.append(row)

with target_manifest.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(missing_rows)

print(len(missing_rows))
PY
    )"

    if [ "${RUNNER_ROWS}" -le 0 ]; then
        echo "No missing rows found in manifest; nothing to resubmit." >&2
        echo "You can aggregate/check the full manifest with:" >&2
        echo "  sbatch ${AGGREGATOR} --manifest ${MANIFEST}" >&2
        exit 1
    fi
fi

TASK_COUNT=$((RUNNER_ROWS * SEEDS_PER_RUN))
ARRAY_MAX=$((TASK_COUNT - 1))

echo "Submitting remaining LeJEPA jobs"
echo "Manifest: ${MANIFEST}"
echo "Manifest rows: ${MANIFEST_ROWS}"
if [ "${MISSING_ONLY}" -eq 1 ]; then
    echo "Runner manifest: ${RUNNER_MANIFEST}"
    echo "Missing rows to run: ${RUNNER_ROWS}"
fi
echo "Array tasks: ${TASK_COUNT} (${RUNNER_ROWS} runs x ${SEEDS_PER_RUN} seeds)"
echo ""

ARRAY_JOB_ID="$(sbatch --parsable --array=0-${ARRAY_MAX} "${RUNNER}" --manifest "${RUNNER_MANIFEST}")"

STATUS_CSV="${COMBINED_DIR}/lejepa_remaining_status_${ARRAY_JOB_ID}.csv"
COMBINED_CSV="${COMBINED_DIR}/lejepa_remaining_combined_results_${ARRAY_JOB_ID}.csv"
JOB_FILE="${COMBINED_DIR}/lejepa_remaining_jobs_${ARRAY_JOB_ID}_${TIMESTAMP}.env"

{
    echo "workflow=remaining"
    echo "submitted_at=${TIMESTAMP}"
    echo "mode=$([ "${MISSING_ONLY}" -eq 1 ] && echo missing_only || echo full)"
    echo "manifest=${MANIFEST}"
    echo "runner_manifest=${RUNNER_MANIFEST}"
    echo "manifest_rows=${MANIFEST_ROWS}"
    echo "runner_manifest_rows=${RUNNER_ROWS}"
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
echo ""
echo "Check later with:"
echo "  bash ${SCRIPT_DIR}/check_all_max_results.sh ${JOB_FILE}"
