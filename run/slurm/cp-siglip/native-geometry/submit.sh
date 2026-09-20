#!/bin/bash
set -euo pipefail

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: bash submit.sh [--dry-run]" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }
REPO_ROOT="${SIGLIP_GEOMETRY_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
OUTPUT_BASE="${SIGLIP_GEOMETRY_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${SIGLIP_GEOMETRY_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
IMAGENET_DIR="${SIGLIP_GEOMETRY_IMAGENET_DIR:-$CACHE_DIR/imagenet_val}"
LOG_DIR="$OUTPUT_BASE/slurm-log"
SBATCH=(sbatch --parsable --job-name=siglip-geometry --partition=nvidia --account=civil
    --chdir="$REPO_ROOT" --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1
    --cpus-per-task=8 --mem=64G --time=24:00:00
    --output="$LOG_DIR/siglip-geometry-%j.out" --error="$LOG_DIR/siglip-geometry-%j.err"
    --export="ALL,SIGLIP_GEOMETRY_REPO_ROOT=$REPO_ROOT,SIGLIP_GEOMETRY_OUTPUT_BASE=$OUTPUT_BASE,SIGLIP_GEOMETRY_CACHE_DIR=$CACHE_DIR,SIGLIP_GEOMETRY_IMAGENET_DIR=$IMAGENET_DIR"
    "$REPO_ROOT/run/slurm/cp-siglip/native-geometry/run.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 1 ]; then
    SIGLIP_GEOMETRY_REPO_ROOT="$REPO_ROOT" bash "$REPO_ROOT/run/slurm/cp-siglip/native-geometry/run.sh" --dry-run
else
    [ -d "$IMAGENET_DIR" ] || { echo "Missing ImageNet validation cache: $IMAGENET_DIR" >&2; exit 4; }
    mkdir -p "$LOG_DIR"
    "${SBATCH[@]}"
fi
