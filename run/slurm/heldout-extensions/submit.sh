#!/bin/bash
set -euo pipefail

REPO_ROOT="${HELDOUT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${HELDOUT_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
case "$PY" in /*) ;; *) echo "HELDOUT_PYTHON must be an absolute executable path" >&2; exit 2 ;; esac
[ -x "$PY" ] || { echo "HELDOUT_PYTHON must be an absolute executable path: $PY" >&2; exit 2; }
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
exec "$PY" -m eval.heldout_extensions.submit "$@"
