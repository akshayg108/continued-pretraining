#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
"$CP_PYTHON" "$REPO/run/precp.py" list
sbatch --chdir="$REPO" --export=ALL \
    --output="$CP_ROOT/outputs/slurm-log/precp-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/precp-%A_%a.err" \
    "$@" "$REPO/run/slurm/precp.sh"
