#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
    printf 'Usage: %s (configure SOURCE_TASKS, SOURCE_STEPS, or SBATCH_* through the environment)\n' "$0" >&2
    exit 2
fi

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
export IMAGENET_TRAIN_DIR="${IMAGENET_TRAIN_DIR:-$CP_ROOT/data/imagenet/train}"
export SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-$CP_ROOT/outputs/source_coverage_v1}"
if [[ ! -d "$IMAGENET_TRAIN_DIR" && ! -f "${IMAGENET_TRAIN_ARCHIVE:-}" ]]; then
    printf 'Provide the full ImageNet training folder or IMAGENET_TRAIN_ARCHIVE: %s\n' "$IMAGENET_TRAIN_DIR" >&2
    exit 1
fi

PREPARE=$(sbatch --parsable --chdir="$REPO" --export=ALL \
    --output="$CP_ROOT/outputs/slurm-log/source-prepare-%j.out" \
    --error="$CP_ROOT/outputs/slurm-log/source-prepare-%j.err" \
    "$REPO/run/slurm/source_pretrain.sh" prepare)
PREPARE=${PREPARE%%;*}
printf 'Source preparation job: %s\n' "$PREPARE"
sbatch --chdir="$REPO" --export=ALL --dependency="afterok:$PREPARE" \
    --array="${SOURCE_TASKS:-0-1%2}" \
    --output="$CP_ROOT/outputs/slurm-log/source-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/source-%A_%a.err" \
    "$REPO/run/slurm/source_pretrain.sh" train
