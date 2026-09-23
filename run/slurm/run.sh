#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

ROOT="${CP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd -- "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -u "$ROOT/run/data_cache.py" run "$@"
