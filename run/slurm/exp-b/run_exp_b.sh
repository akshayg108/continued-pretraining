#!/bin/bash
#SBATCH --job-name=exp-b
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/exp-b-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/exp-b-%A_%a.err

# Exp B (Selective-Aggregation LP) — one array task per dataset; stages that dataset node-local.
# Each task runs {MAE,LeJEPA} x {DINOv3,CLIP,MAE} x <its dataset> x MAX x available seeds.

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}   Start: $(date)"
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
# Per-dataset config (indexed by array task id)
# ============================================================
# indices 0-6 keep the original ordering (octmnist=1); 7-14 are the newly-added datasets
DATASETS=(food101 octmnist plant_village organamnist galaxy10 fgvc_aircraft cars196 breastmnist cub200 dermamnist dtd eurosat flowers102 oxford_pet pathmnist)
SUBPATHS=(food101 med_mnist/octmnist-size=224 plant_village med_mnist/organamnist-size=224 galaxy10 fgvc_aircraft cars196 med_mnist/breastmnist-size=224 cub200 med_mnist/dermamnist-size=224 dtd eurosat flowers102 oxford_pet med_mnist/pathmnist-size=224)
i=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$i]}
PROCESSED_SUBPATH=${SUBPATHS[$i]}

DATA_DIR="/scratch/gs4133/zhd/CP/data"
CKPT_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp"
OUT="eval/outputs/exp_b/${DATASET}.csv"
NUM_WORKERS=8

# ============================================================
# Stage this dataset's processed cache to node-local fast storage.
# HF download_and_prepare() checks the PROCESSED cache only; when it is complete the loader
# mmaps it directly and never reads downloads/. (food101's cache is large; if /tmpdata is too
# small the size-check falls back to /scratch.)
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
        echo "WARN: no node-local root with >= $((NEED_KB/1024/1024))G free; using /scratch."
    fi
else
    echo "WARN: processed cache not found at ${SRC_PROC}; using /scratch DATA_DIR=${DATA_DIR}."
fi

echo "=========================================="
echo "Exp B SA-LP: dataset=${DATASET}  (MAE/LeJEPA x DINOv3/CLIP/MAE x MAX x seeds)"
echo "  ckpt-root: ${CKPT_ROOT}"
echo "  cache-dir: ${DATA_DIR}"
echo "  out:       ${OUT}"
echo "=========================================="

python -u eval/run_exp_b.py \
    --ckpt-root "${CKPT_ROOT}" \
    --cache-dir "${DATA_DIR}" \
    --datasets "${DATASET}" \
    --out "${OUT}" \
    --num-workers "${NUM_WORKERS}" \
    2>&1

echo "End Time: $(date)"
