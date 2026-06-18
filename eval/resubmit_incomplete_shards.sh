#!/bin/bash
# Resubmit every sweep shard that is NEITHER ~complete NOR currently running, so the sweep
# converges despite the cluster (uid 0 / admin) periodically cancelling long GPU jobs.
# Safe to run repeatedly: the sweep resumes each shard from its CSV; complete shards are skipped
# and running shards are excluded (avoids two tasks writing the same postcp_sweep_<i>.csv).
#
#   bash eval/resubmit_incomplete_shards.sh          # uses 226 rows as "complete"
#   bash eval/resubmit_incomplete_shards.sh 226      # override the per-shard target row count
#
# N stays 12 (the array max index + 1), so resumed shards keep their original 12-way slice.
cd "$(dirname "$0")/.."
TARGET="${1:-226}"
NSHARDS_TOTAL=12

run=" $(squeue -h -u "$USER" -o '%K' | tr '\n' ' ') "   # array task ids currently queued/running
todo=""
for i in $(seq 0 $((NSHARDS_TOTAL - 1))); do
    f="eval/outputs/postcp_sweep_${i}.csv"
    n=$([ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0)
    case "$run" in *" $i "*) tag="(running, skip)";; *) tag="";; esac
    if [ "$n" -lt "$TARGET" ] && [ -z "$tag" ]; then todo="${todo}${i},"; tag="-> RESUBMIT"; fi
    printf "  shard %2d: %4d rows %s\n" "$i" "$n" "$tag"
done
todo=${todo%,}

if [ -n "$todo" ]; then
    echo "resubmitting incomplete & not-running shards: ${todo}"
    sbatch --array="${todo}" eval/run_postcp_sweep.sh
else
    echo "nothing to resubmit — all shards complete or currently running."
fi
