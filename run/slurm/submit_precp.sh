#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
if [[ ! -d "$CP_ROOT/data/imagenet_val" ]]; then
    printf 'Missing ImageNet validation cache: %s\n' "$CP_ROOT/data/imagenet_val" >&2
    exit 1
fi
"$CP_PYTHON" "$REPO/run/precp.py" list
REFERENCE=$(sbatch --parsable --chdir="$REPO" --export=ALL \
    --output="$CP_ROOT/outputs/slurm-log/precp-reference-%j.out" \
    --error="$CP_ROOT/outputs/slurm-log/precp-reference-%j.err" \
    "$REPO/run/slurm/precp_reference.sh")
REFERENCE=${REFERENCE%%;*}
printf 'Reference preparation job: %s\n' "$REFERENCE"
sbatch --chdir="$REPO" --export=ALL --dependency="afterok:$REFERENCE" \
    --output="$CP_ROOT/outputs/slurm-log/precp-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/precp-%A_%a.err" \
    "$@" "$REPO/run/slurm/precp.sh"
