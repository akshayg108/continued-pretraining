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

# Keep all tasks held until their GPU requests are assigned within one array.
job=$(sbatch --parsable "$@" --hold --chdir="$REPO" --export=ALL \
    --gres=gpu:v100:1 --array="$V100_TASKS,$A100_TASKS%$CONCURRENCY" \
    --output="$CP_ROOT/outputs/slurm-log/cp-full-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/cp-full-%A_%a.err" \
    "$REPO/run/slurm/cp_full.sh")
job="${job%%;*}"
printf 'CP array: %s (held; shared concurrency=%s)\n' "$job" "$CONCURRENCY"

if ! scontrol update "JobId=${job}_[${A100_TASKS}]" Gres=gpu:a100:1; then
    printf 'GPU assignment failed; array %s remains held. Do not release it manually.\n' "$job" >&2
    exit 1
fi
if ! squeue --array --noheader --jobs="$job" --Format='ArrayTaskID:12,tres-per-node:80' |
    awk -v v100="$V100_TASKS" -v a100="$A100_TASKS" '
        BEGIN {
            n = split(v100, ids, ",")
            for (i = 1; i <= n; i++) expected[ids[i]] = "gpu:v100:1"
            n = split(a100, ids, ",")
            for (i = 1; i <= n; i++) expected[ids[i]] = "gpu:a100:1"
        }
        {
            request = $2
            sub(/^gres[:\/]/, "", request)
            if (NF != 2 || !($1 in expected) || expected[$1] != request) bad = 1
            delete expected[$1]
        }
        END {
            for (id in expected) bad = 1
            exit bad
        }
    '; then
    printf 'GPU verification failed; array %s remains held. Do not release it manually.\n' "$job" >&2
    exit 1
fi
if ! scontrol release "$job"; then
    printf 'Could not release array %s; inspect its state before retrying.\n' "$job" >&2
    exit 1
fi
printf 'CP array: %s (released; V100 + A100 running limit=%s)\n' "$job" "$CONCURRENCY"
