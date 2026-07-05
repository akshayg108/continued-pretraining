#!/bin/bash
#SBATCH --job-name=test3-expb
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/test3-expb-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/test3-expb-%j.err

# ============================================================
# Test 3 — fix Exp B's broken LeJEPA reference, then re-aggregate sa_lp_recovery.
# (eval/rest/test3_rerun_exp_b.py --run-expb). It drops the stale LeJEPA rows from
# eval/outputs/exp_b/{cars196,food101,octmnist}.csv, re-runs eval/run_exp_b.py
# --methods LeJEPA on the now-fixed checkpoints, and re-aggregates
# eval/outputs/sa_lp_recovery.csv from ALL exp_b/*.csv.
#
# --mem=256G because food101 SA-LP at MAX holds full patch-token tensors
# (196x the [cls] vector) — matches run/slurm/exp-b/run_exp_b.sh.
# Only 3 datasets are affected (cars196, food101, octmnist) -> stage those 3.
# ============================================================

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}   Node: ${SLURM_NODELIST}   Start: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi

# ============================================================
# Paths
# ============================================================
DATA_ROOT="/scratch/gs4133/zhd/CP/data"
PROC_DIR="${DATA_ROOT}/stable_datasets/processed"
CKPT_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp"

# test3's affected LeJEPA datasets (from rerun_geometry.csv ∩ Exp B datasets)
TEST3_DATASETS=(cars196 food101 octmnist)
TEST3_SUBPATHS=(cars196 food101 med_mnist/octmnist-size=224)

# back up the current recovery table before it is overwritten
cp -f eval/outputs/sa_lp_recovery.csv eval/outputs/sa_lp_recovery.bak.csv 2>/dev/null || true

# ============================================================
# Stage the 3 processed caches to node-local fast storage.
# run_exp_b.py takes a single --cache-dir and appends stable_datasets/processed/<subpath>,
# so we stage into LOCAL_CACHE/stable_datasets/processed/<subpath> and pass --cache-dir LOCAL_CACHE
# via test3's --cache-dir (which it forwards to run_exp_b.py).
# ============================================================
CACHE_DIR="${DATA_ROOT}"
STAGE_ROOT=""
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    avail_kb=$(df -Pk "${root}" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "${avail_kb}" ] && [ "${avail_kb}" -ge $((150*1024*1024)) ]; then STAGE_ROOT="${root}"; break; fi
done

if [ -n "${STAGE_ROOT}" ]; then
    LOCAL_CACHE="${STAGE_ROOT}/cpdata-${SLURM_JOB_ID}"
    trap 'rm -rf "${LOCAL_CACHE}"' EXIT
    for idx in "${!TEST3_DATASETS[@]}"; do
        sub="${TEST3_SUBPATHS[$idx]}"
        src="${PROC_DIR}/${sub}"
        dst="${LOCAL_CACHE}/stable_datasets/processed/${sub}"
        if [ -d "${src}" ]; then
            echo "===== staging ${src} -> ${dst} ====="
            mkdir -p "$(dirname "${dst}")" && rsync -a "${src}/" "${dst}/"
        else
            echo "WARN: processed cache missing: ${src}"
        fi
    done
    CACHE_DIR="${LOCAL_CACHE}"
    echo "===== node-local CACHE_DIR=${CACHE_DIR} ====="
else
    echo "WARN: no node-local root with >=150G free; using /scratch CACHE_DIR=${CACHE_DIR}."
fi

# ============================================================
# Run: drop stale LeJEPA rows -> re-run Exp B (LeJEPA) on the 3 datasets -> re-aggregate.
# ============================================================
echo "=========================================="
echo "Test 3: re-run Exp B LeJEPA reference for {cars196, food101, octmnist}"
echo "  ckpt-root: ${CKPT_ROOT}"
echo "  cache-dir: ${CACHE_DIR}"
echo "=========================================="

python -u eval/rest/test3_rerun_exp_b.py --run-expb \
    --csv eval/outputs/rerun_geometry.csv \
    --exp-b-dir eval/outputs/exp_b \
    --ckpt-root "${CKPT_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --out eval/outputs/sa_lp_recovery.csv \
    2>&1

echo ""
echo "=========================================="
echo "Done. Send me: eval/outputs/sa_lp_recovery.csv"
echo "  (backup of the old table is at eval/outputs/sa_lp_recovery.bak.csv)"
echo "End Time: $(date)"
echo "=========================================="
