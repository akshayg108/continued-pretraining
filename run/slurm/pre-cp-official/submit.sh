#!/bin/bash
set -euo pipefail

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: bash submit.sh [--dry-run]" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }

REPO_ROOT="${PRECP_NORM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTPUT_BASE="${PRECP_NORM_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${PRECP_NORM_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
LOG_DIR="$OUTPUT_BASE/slurm-log"
SBATCH=(sbatch --parsable --job-name=precp-native --array=0-3%4
    --chdir="$REPO_ROOT" --partition=nvidia --account=civil
    --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1
    --cpus-per-task=8 --mem=96G --time=96:00:00
    --output="$LOG_DIR/precp-native-%A_%a.out" --error="$LOG_DIR/precp-native-%A_%a.err"
    --export="ALL,PRECP_NORM_REPO_ROOT=$REPO_ROOT,PRECP_NORM_OUTPUT_BASE=$OUTPUT_BASE,PRECP_NORM_CACHE_DIR=$CACHE_DIR"
    "$REPO_ROOT/run/slurm/pre-cp-official/array.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 1 ]; then
    for ID in 0 1 2 3; do
        PRECP_NORM_REPO_ROOT="$REPO_ROOT" PRECP_NORM_OUTPUT_BASE="$OUTPUT_BASE" \
            PRECP_NORM_CACHE_DIR="$CACHE_DIR" SLURM_ARRAY_TASK_ID="$ID" \
            bash "$REPO_ROOT/run/slurm/pre-cp-official/array.sh" --dry-run
    done
else
    mkdir -p "$LOG_DIR"
    "${SBATCH[@]}"
fi
