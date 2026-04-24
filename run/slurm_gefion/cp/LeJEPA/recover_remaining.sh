#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/dcai/projects/iu_0092/projects/cp/continued-pretraining"
COMBINED_DIR="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA/combined"
SUBMITTER="${SCRIPT_DIR}/submit_remaining.sh"

JOB_FILE=""
ARCHIVE_CHECKPOINTS=0
DRY_RUN=0
NO_SUBMIT=0

usage() {
    cat <<EOF
Usage: bash ${0} [options]

Cancel the previous manifest-driven LeJEPA remaining-run array, preserve JSON
outputs, and re-submit with the current code.

Options:
  --job-file PATH          Previous lejepa_remaining_jobs_*.env file.
                           Defaults to the newest file in ${COMBINED_DIR}.
  --archive-checkpoints    Archive repo-local ./checkpoints after old jobs are gone.
  --dry-run                Print actions without running scancel/mv/submit.
  --no-submit              Cancel/archive only; do not submit a replacement job.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-file) JOB_FILE="$2"; shift 2 ;;
        --archive-checkpoints) ARCHIVE_CHECKPOINTS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-submit) NO_SUBMIT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "${JOB_FILE}" ]; then
    JOB_FILE="$(ls -t "${COMBINED_DIR}"/lejepa_remaining_jobs_*.env 2>/dev/null | head -n 1 || true)"
fi

if [ -z "${JOB_FILE}" ] || [ ! -f "${JOB_FILE}" ]; then
    echo "Could not find previous remaining job file." >&2
    echo "Pass it explicitly with --job-file PATH." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${JOB_FILE}"

if [ "${workflow:-}" != "remaining" ]; then
    echo "Job file is not a remaining-run workflow file: ${JOB_FILE}" >&2
    exit 1
fi

if [ -z "${array_job_id:-}" ] || [ -z "${aggregation_job_id:-}" ]; then
    echo "Job file is missing array_job_id or aggregation_job_id: ${JOB_FILE}" >&2
    exit 1
fi

run_or_echo() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[dry-run] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "Recovering remaining LeJEPA runs"
echo "  Previous job file: ${JOB_FILE}"
echo "  Previous array job: ${array_job_id}"
echo "  Previous aggregation job: ${aggregation_job_id}"
echo "  Preserve JSON outputs: yes"
echo ""

if command -v squeue >/dev/null 2>&1; then
    echo "Current old-job queue state:"
    squeue -j "${array_job_id},${aggregation_job_id}" || true
    echo ""
fi

if ! command -v scancel >/dev/null 2>&1 && [ "${DRY_RUN}" -eq 0 ]; then
    echo "scancel is required but was not found on PATH" >&2
    exit 1
fi

echo "Cancelling previous remaining array and aggregation jobs..."
if [ "${DRY_RUN}" -eq 1 ]; then
    run_or_echo scancel "${array_job_id}" "${aggregation_job_id}"
else
    scancel "${array_job_id}" "${aggregation_job_id}" || true
fi
echo ""

if [ "${ARCHIVE_CHECKPOINTS}" -eq 1 ]; then
    can_archive=1
    if command -v squeue >/dev/null 2>&1 && [ "${DRY_RUN}" -eq 0 ]; then
        if squeue -h -j "${array_job_id},${aggregation_job_id}" | grep -q .; then
            can_archive=0
            echo "Old jobs are still visible in squeue; skipping checkpoint archive for safety."
            echo "Run this script again with --archive-checkpoints once they disappear."
            echo ""
        fi
    fi

    if [ "${can_archive}" -eq 1 ]; then
        old_ckpt_dir="${REPO_ROOT}/checkpoints"
        if [ -e "${old_ckpt_dir}" ]; then
            archive_dir="${REPO_ROOT}/checkpoints_old_sft_collision_$(date +%Y%m%d_%H%M%S)"
            echo "Archiving repo-local checkpoints:"
            echo "  ${old_ckpt_dir}"
            echo "  -> ${archive_dir}"
            run_or_echo mv "${old_ckpt_dir}" "${archive_dir}"
            echo ""
        else
            echo "No repo-local checkpoints directory found at ${old_ckpt_dir}; nothing to archive."
            echo ""
        fi
    fi
fi

if [ "${NO_SUBMIT}" -eq 1 ]; then
    echo "Skipping re-submit because --no-submit was provided."
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1 && [ "${DRY_RUN}" -eq 0 ]; then
    echo "sbatch is required but was not found on PATH" >&2
    exit 1
fi

echo "Submitting replacement remaining-run array..."
run_or_echo bash "${SUBMITTER}"
