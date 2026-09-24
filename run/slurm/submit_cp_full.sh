#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
V100_CONCURRENCY="${CP_V100_CONCURRENCY:-12}"
A100_CONCURRENCY="${CP_A100_CONCURRENCY:-12}"
for limit in "$V100_CONCURRENCY" "$A100_CONCURRENCY"; do
    if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
        printf 'CP_V100_CONCURRENCY and CP_A100_CONCURRENCY must be positive integers.\n' >&2
        exit 2
    fi
done

"$CP_PYTHON" "$REPO/run/cp_full.py" check --root "$CP_ROOT"
"$CP_PYTHON" "$REPO/run/cp_full.py" list
V100_TASKS=$("$CP_PYTHON" "$REPO/run/cp_full.py" array --gpu v100)
A100_TASKS=$("$CP_PYTHON" "$REPO/run/cp_full.py" array --gpu a100)
for tasks in "$V100_TASKS" "$A100_TASKS"; do
    if [[ ! "$tasks" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        printf 'The CP runner returned an invalid or empty Slurm task list: %s\n' "$tasks" >&2
        exit 1
    fi
done

for gpu in v100 a100; do
    if [[ "$gpu" == v100 ]]; then
        tasks="$V100_TASKS"
        limit="$V100_CONCURRENCY"
    else
        tasks="$A100_TASKS"
        limit="$A100_CONCURRENCY"
    fi
    job=$(sbatch --parsable "$@" --chdir="$REPO" --export=ALL \
        --gres="gpu:$gpu:1" --array="$tasks%$limit" \
        --output="$CP_ROOT/outputs/slurm-log/cp-full-$gpu-%A_%a.out" \
        --error="$CP_ROOT/outputs/slurm-log/cp-full-$gpu-%A_%a.err" \
        "$REPO/run/slurm/cp_full.sh")
    printf '%s CP array: %s\n' "$gpu" "${job%%;*}"
done
