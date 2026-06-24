#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARRAY_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/mae_run_max.sh")
echo "Submitted array job: ${ARRAY_JOB_ID}"

AGG_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} "${SCRIPT_DIR}/mae_aggregate_max.sh")
echo "Submitted aggregation job: ${AGG_JOB_ID} (dependency: afterany:${ARRAY_JOB_ID})"
