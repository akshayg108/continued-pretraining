#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUBMITTERS=(
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

echo "Submitting all LeJEPA MAX jobs from ${SCRIPT_DIR}"
echo ""

for relative_path in "${SUBMITTERS[@]}"; do
    submitter="${SCRIPT_DIR}/${relative_path}"
    echo "============================================================"
    echo "Submitting: ${relative_path}"
    echo "============================================================"
    bash "${submitter}"
    echo ""
done

echo "All LeJEPA MAX submissions have been queued."
