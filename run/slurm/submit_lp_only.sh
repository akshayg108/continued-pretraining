#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$CP_ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

v100=$("$CP_PYTHON" "$REPO/run/lp_only.py" array --phase post --root "$CP_ROOT" --gpu v100)
a100=$("$CP_PYTHON" "$REPO/run/lp_only.py" array --phase post --root "$CP_ROOT" --gpu a100)
phases=(pre post post)
gpus=(v100 v100 a100)
tasks=(0-22 "$v100" "$a100")
previous=""

# Ordinary arrays run sequentially so all LP reruns share a ten-job cap.
for index in 0 1 2; do
    [[ -n "${tasks[$index]}" ]] || continue
    phase="${phases[$index]}"
    gpu="${gpus[$index]}"
    dependency=()
    if [[ -n "$previous" ]]; then
        dependency=(--dependency="afterany:$previous")
    fi
    job=$(sbatch --parsable --chdir="$REPO" --export=ALL \
        --job-name="lp-$phase" --gres="gpu:$gpu:1" \
        --array="${tasks[$index]}%10" ${dependency[@]+"${dependency[@]}"} \
        --output="$CP_ROOT/outputs/slurm-log/lp-$phase-%A_%a.out" \
        --error="$CP_ROOT/outputs/slurm-log/lp-$phase-%A_%a.err" \
        "$REPO/run/slurm/lp_only.sh" --phase "$phase" "$@")
    previous="${job%%;*}"
    printf 'LP-only %s array: %s (%s, concurrency=10)\n' "$phase" "$previous" "$gpu"
done
