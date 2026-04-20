#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SUBMIT_LOG_PATH="${1:-${REPO_ROOT}/submit_log}"
OUTPUT_ROOT="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA"
COMBINED_DIR="${OUTPUT_ROOT}/combined"
STATUS_CSV="${COMBINED_DIR}/lejepa_max_status.csv"
COMBINED_CSV="${COMBINED_DIR}/lejepa_max_combined_results.csv"

EXPECTED_SUBMITTERS=(
    "random/OctMNIST/submit_max.sh"
    "random/PathMNIST/submit_max.sh"
    "random/Food101/submit_max.sh"
    "pretrained/OctMNIST/dinov3_submit_max.sh"
    "pretrained/OctMNIST/mae_submit_max.sh"
    "pretrained/OctMNIST/clip_submit_max.sh"
    "pretrained/PathMNIST/dinov3_submit_max.sh"
    "pretrained/PathMNIST/mae_submit_max.sh"
    "pretrained/Food101/dinov3_submit_max.sh"
)

mkdir -p "${COMBINED_DIR}"

if ! command -v sacct >/dev/null 2>&1; then
    echo "sacct is required but was not found on PATH" >&2
    exit 1
fi

if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is required but was not found on PATH" >&2
    exit 1
fi

declare -a submitters=()
declare -a array_jobs=()
declare -a agg_jobs=()

resolve_array_job_from_agg() {
    local agg_job_id="$1"
    local job_info dep

    job_info="$(scontrol show job "${agg_job_id}" 2>/dev/null || true)"
    if [ -z "${job_info}" ]; then
        echo "$((agg_job_id - 1))"
        return
    fi

    dep="$(printf '%s\n' "${job_info}" | tr ' ' '\n' | grep '^Dependency=' | head -n 1 | cut -d= -f2-)"
    if [[ "${dep}" =~ afterany:([0-9_]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        # Fallback: submit_all_max.sh submits each array job immediately before
        # its aggregation job, so agg_job_id = array_job_id + 1.
        echo "$((agg_job_id - 1))"
    fi
}

if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    if [ "$#" -ne "${#EXPECTED_SUBMITTERS[@]}" ]; then
        echo "Expected ${#EXPECTED_SUBMITTERS[@]} aggregation job IDs, got $#." >&2
        exit 1
    fi

    AGG_ID_ARGS=("$@")
    for i in "${!EXPECTED_SUBMITTERS[@]}"; do
        submitters+=("${EXPECTED_SUBMITTERS[$i]}")
        agg_jobs+=("${AGG_ID_ARGS[$i]}")
    done

    for agg_job_id in "${agg_jobs[@]}"; do
        array_job_id="$(resolve_array_job_from_agg "${agg_job_id}")"
        if [ -z "${array_job_id}" ] || ! [[ "${array_job_id}" =~ ^[0-9]+$ ]]; then
            echo "Could not resolve array dependency for aggregation job ${agg_job_id}" >&2
            exit 1
        fi
        array_jobs+=("${array_job_id}")
    done
else
    if [ ! -f "${SUBMIT_LOG_PATH}" ]; then
        echo "submit log not found: ${SUBMIT_LOG_PATH}" >&2
        exit 1
    fi

    current_submitter=""
    while IFS= read -r line; do
        if [[ "${line}" =~ ^Submitting:\ (.+)$ ]]; then
            current_submitter="${BASH_REMATCH[1]}"
        elif [[ "${line}" =~ ^Submitted\ array\ job:\ ([0-9]+)$ ]]; then
            submitters+=("${current_submitter}")
            array_jobs+=("${BASH_REMATCH[1]}")
        elif [[ "${line}" =~ ^Submitted\ aggregation\ job:\ ([0-9]+) ]]; then
            agg_jobs+=("${BASH_REMATCH[1]}")
        fi
    done < "${SUBMIT_LOG_PATH}"
fi

if [ "${#submitters[@]}" -eq 0 ]; then
    echo "No submitted jobs found." >&2
    exit 1
fi

if [ "${#submitters[@]}" -ne "${#array_jobs[@]}" ] || [ "${#submitters[@]}" -ne "${#agg_jobs[@]}" ]; then
    echo "submit log is incomplete or malformed: counts do not match" >&2
    exit 1
fi

csv_path_for_submitter() {
    case "$1" in
        random/OctMNIST/submit_max.sh)
            echo "${OUTPUT_ROOT}/random/OctMNIST/SCRATCH_lejepa_cp_results.csv"
            ;;
        random/PathMNIST/submit_max.sh)
            echo "${OUTPUT_ROOT}/random/PathMNIST/SCRATCH_lejepa_cp_results.csv"
            ;;
        random/Food101/submit_max.sh)
            echo "${OUTPUT_ROOT}/random/Food101/SCRATCH_lejepa_cp_results.csv"
            ;;
        pretrained/OctMNIST/dinov3_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/OctMNIST/DINOv3/all/DINOv3_lejepa_cp_results.csv"
            ;;
        pretrained/OctMNIST/mae_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/OctMNIST/MAE/all/MAE_lejepa_cp_results.csv"
            ;;
        pretrained/OctMNIST/clip_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/OctMNIST/CLIP/all/CLIP_lejepa_cp_results.csv"
            ;;
        pretrained/PathMNIST/dinov3_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/PathMNIST/DINOv3/all/DINOv3_lejepa_cp_results.csv"
            ;;
        pretrained/PathMNIST/mae_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/PathMNIST/MAE/all/MAE_lejepa_cp_results.csv"
            ;;
        pretrained/Food101/dinov3_submit_max.sh)
            echo "${OUTPUT_ROOT}/pretrained/Food101/DINOv3/all/DINOv3_lejepa_cp_results.csv"
            ;;
        *)
            echo "unknown submitter: $1" >&2
            return 1
            ;;
    esac
}

check_array_job() {
    local array_job_id="$1"
    local output
    output="$(sacct -n -P -j "${array_job_id}" --format=JobIDRaw,JobName,State,ExitCode 2>/dev/null || true)"

    if [ -z "${output}" ]; then
        echo "UNKNOWN|0|0|0|"
        return
    fi

    local ok=0
    local bad=0
    local running=0
    local pending=0
    local bad_rows=()

    while IFS='|' read -r jobidraw jobname state exitcode; do
        [ -z "${jobidraw}" ] && continue
        if [[ "${jobidraw}" != *_* ]]; then
            continue
        fi

        case "${state}" in
            COMPLETED)
                ok=$((ok + 1))
                ;;
            PENDING|CONFIGURING|RUNNING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
                if [[ "${state}" == "PENDING" ]]; then
                    pending=$((pending + 1))
                else
                    running=$((running + 1))
                fi
                ;;
            *)
                bad=$((bad + 1))
                bad_rows+=("${jobidraw}:${state}:${exitcode}")
                ;;
        esac
    done <<< "${output}"

    local status="SUCCESS"
    if [ "${bad}" -gt 0 ]; then
        status="FAILED"
    elif [ "${running}" -gt 0 ] || [ "${pending}" -gt 0 ]; then
        status="INCOMPLETE"
    fi

    local details
    details="$(IFS=';'; echo "${bad_rows[*]-}")"
    echo "${status}|${ok}|${bad}|$((running + pending))|${details}"
}

check_single_job() {
    local job_id="$1"
    local row
    row="$(sacct -n -P -j "${job_id}" --format=JobIDRaw,State,ExitCode 2>/dev/null | head -n 1 || true)"
    if [ -z "${row}" ]; then
        echo "UNKNOWN|"
        return
    fi

    local _jobid state exitcode
    IFS='|' read -r _jobid state exitcode <<< "${row}"
    echo "${state}|${exitcode}"
}

{
    echo "submitter,array_job_id,array_status,seed_tasks_completed,seed_tasks_failed,seed_tasks_running_or_pending,array_failure_details,aggregation_job_id,aggregation_state,aggregation_exit_code,csv_path,csv_exists"

    overall_ok=1
    combined_inputs=()

    for i in "${!submitters[@]}"; do
        submitter="${submitters[$i]}"
        array_job_id="${array_jobs[$i]}"
        agg_job_id="${agg_jobs[$i]}"
        csv_path="$(csv_path_for_submitter "${submitter}")"

        IFS='|' read -r array_status seed_ok seed_bad seed_running array_details <<< "$(check_array_job "${array_job_id}")"
        IFS='|' read -r agg_state agg_exit_code <<< "$(check_single_job "${agg_job_id}")"

        csv_exists="no"
        if [ -f "${csv_path}" ]; then
            csv_exists="yes"
        fi

        if [ "${array_status}" != "SUCCESS" ] || [ "${agg_state}" != "COMPLETED" ]; then
            overall_ok=0
        fi

        if [ "${csv_exists}" = "yes" ]; then
            combined_inputs+=("${submitter}|${array_job_id}|${agg_job_id}|${csv_path}")
        fi

        printf '%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s\n' \
            "${submitter}" \
            "${array_job_id}" \
            "${array_status}" \
            "${seed_ok}" \
            "${seed_bad}" \
            "${seed_running}" \
            "${array_details}" \
            "${agg_job_id}" \
            "${agg_state}" \
            "${agg_exit_code}" \
            "${csv_path}" \
            "${csv_exists}"
    done
} > "${STATUS_CSV}"

python3 - "${COMBINED_CSV}" "${combined_inputs[@]}" <<'PY'
import csv
import sys
from pathlib import Path

combined_csv = Path(sys.argv[1])
entries = sys.argv[2:]

fieldnames = [
    "submitter",
    "array_job_id",
    "aggregation_job_id",
    "backbone",
    "dataset",
    "n_samples",
    "model_size",
    "run",
    "pre_knn_f1",
    "pre_knn_f1_std",
    "pre_linear_f1",
    "pre_linear_f1_std",
    "post_knn_f1",
    "post_knn_f1_std",
    "post_linear_f1",
    "post_linear_f1_std",
    "post_sft_f1",
    "post_sft_f1_std",
]

rows = []
for entry in entries:
    submitter, array_job_id, agg_job_id, csv_path = entry.split("|", 3)
    path = Path(csv_path)
    if not path.exists():
        continue

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_out = {k: "" for k in fieldnames}
            row_out["submitter"] = submitter
            row_out["array_job_id"] = array_job_id
            row_out["aggregation_job_id"] = agg_job_id
            for key in fieldnames:
                if key in row:
                    row_out[key] = row[key]
            rows.append(row_out)

with combined_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "Status report written to: ${STATUS_CSV}"
echo "Combined results written to: ${COMBINED_CSV}"
echo ""

python3 - "${STATUS_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

status_csv = Path(sys.argv[1])

bad_rows = []
with status_csv.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (
            row["array_status"] != "SUCCESS"
            or row["aggregation_state"] != "COMPLETED"
            or row["csv_exists"] != "yes"
        ):
            bad_rows.append(row)

if not bad_rows:
    print("All tracked runs completed successfully and produced CSV outputs.")
    raise SystemExit(0)

print("Some runs are incomplete, failed, or missing CSV outputs:")
for row in bad_rows:
    print(
        f"- {row['submitter']}: "
        f"array_job={row['array_job_id']} array_status={row['array_status']} "
        f"failed_tasks={row['seed_tasks_failed']} running_or_pending={row['seed_tasks_running_or_pending']} "
        f"agg_job={row['aggregation_job_id']} agg_state={row['aggregation_state']} "
        f"csv_exists={row['csv_exists']}"
    )
    if row["array_failure_details"]:
        print(f"  details: {row['array_failure_details']}")

raise SystemExit(2)
PY
