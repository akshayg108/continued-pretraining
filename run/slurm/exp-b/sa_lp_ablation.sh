#!/bin/bash
#SBATCH --job-name=sa-ablate
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/sa-ablate-%x-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/sa-ablate-%x-%j.err

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
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
# Ablation config — SA-LP recipe debug on a MAE-CP+MAE checkpoint.
# Change DATASET / NSAMPLES / SEED together if you switch datasets.
# ============================================================
DATA_DIR="/scratch/gs4133/zhd/CP/data"
DATASET="cars196"
NSAMPLES=8144                 # MAX for cars196
SEED=42
PROCESSED_SUBPATH="cars196"   # stable_datasets/processed/<this>
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts/cp/MAE/pretrained/Cars196/MAE"
NUM_WORKERS=8

# ============================================================
# Stage this dataset's processed cache to node-local fast storage.
# HF download_and_prepare() checks the PROCESSED cache only; when it is
# complete the loader mmaps it directly and never reads downloads/.
# ============================================================
SRC_PROC="${DATA_DIR}/stable_datasets/processed/${PROCESSED_SUBPATH}"
if [ -d "${SRC_PROC}" ]; then
    NEED_KB=$(( $(du -sk "${SRC_PROC}" | awk '{print $1}') + 5*1024*1024 ))   # dataset size + 5G headroom
    STAGE_ROOT=""
    for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
        [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
        avail_kb=$(df -Pk "${root}" 2>/dev/null | awk 'NR==2{print $4}')
        if [ -n "${avail_kb}" ] && [ "${avail_kb}" -ge "${NEED_KB}" ]; then STAGE_ROOT="${root}"; break; fi
    done
    if [ -n "${STAGE_ROOT}" ]; then
        LOCAL_CACHE="${STAGE_ROOT}/cpdata-${SLURM_JOB_ID}"
        DEST_PROC="${LOCAL_CACHE}/stable_datasets/processed/${PROCESSED_SUBPATH}"
        echo "===== staging ${SRC_PROC} -> ${DEST_PROC} (root=${STAGE_ROOT}) ====="
        mkdir -p "$(dirname "${DEST_PROC}")"
        rsync -a "${SRC_PROC}/" "${DEST_PROC}/"
        trap 'rm -rf "${LOCAL_CACHE}"' EXIT          # free node-local copy on job exit
        DATA_DIR="${LOCAL_CACHE}"
        echo "===== DATA_DIR -> ${DATA_DIR} (node-local; reads now fast) ====="
    else
        echo "WARN: no node-local root with >= $((NEED_KB/1024/1024))G free (tried \$TMPDIR /tmpdata /dev/shm); using /scratch."
    fi
else
    echo "WARN: processed cache not found at ${SRC_PROC}; using /scratch DATA_DIR=${DATA_DIR}."
fi

echo "=========================================="
echo "SA-LP ablation: ${DATASET} (MAE-CP+MAE) n=${NSAMPLES} seed=${SEED}"
echo "  ckpt-dir: ${CKPT_DIR}"
echo "  cache-dir: ${DATA_DIR}"
echo "=========================================="

python -u eval/ablate_sa_lp.py \
    --ckpt-dir "${CKPT_DIR}" \
    --dataset "${DATASET}" \
    --n-samples "${NSAMPLES}" \
    --seed "${SEED}" \
    --cache-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    2>&1

echo "End Time: $(date)"
