#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
CONCURRENCY="${CP_CONCURRENCY:-10}"
if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    printf 'CP_CONCURRENCY must be a positive integer.\n' >&2
    exit 2
fi

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

common=("$@" --chdir="$REPO" --export=ALL --nodes=1 --ntasks=1)
v100_job=$(sbatch --parsable "${common[@]}" \
    --gres=gpu:v100:1 --array="$V100_TASKS%$CONCURRENCY" \
    --output="$CP_ROOT/outputs/slurm-log/cp-full-v100-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/cp-full-v100-%A_%a.err" \
    "$REPO/run/slurm/cp_full.sh")
v100_job="${v100_job%%;*}"
printf 'V100 CP array: %s (concurrency=%s)\n' "$v100_job" "$CONCURRENCY"

a100_job=$(sbatch --parsable "${common[@]}" \
    --gres=gpu:a100:1 --array="$A100_TASKS%$CONCURRENCY" \
    --dependency="afterany:$v100_job" \
    --output="$CP_ROOT/outputs/slurm-log/cp-full-a100-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/cp-full-a100-%A_%a.err" \
    "$REPO/run/slurm/cp_full.sh")
printf 'A100 CP array: %s (concurrency=%s; after V100 array %s ends)\n' \
    "${a100_job%%;*}" "$CONCURRENCY" "$v100_job"
