#!/bin/bash
#SBATCH --job-name=int1-features
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/int1-features-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/int1-features-%A_%a.err

# ============================================================
# INT1 — frozen-feature dump for Intervention-1 (counterfactual surgery, INT1_PREREG.md):
# bank + query features per (encoder, dataset) saved as .npz. 4 encoders x 1 dataset per
# array task (eval/new_direction/int1_features_dump.py). ZERO ckpts, public timm weights.
#
# After the whole array finishes: no concat — rsync the whole npz dir back:
#   rsync -av <cluster>:.../eval/outputs/int1_features/ local:.../eval/outputs/int1_features/
# (60 files, ~1.3 GB). Then everything runs locally (int1_run.py + int1_verdict.py).
# Queue-friendly tip:  sbatch --array=0-14%5 run/slurm/new_direction/int1_features.sh
# ============================================================

echo "=========================================="
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env
set -euo pipefail   # rank-aware rerun hardening (Codex follow-up 2026-07-16)

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Codex round-2: hard-fail if CUDA is absent — a silent CPU dump wastes the GPU
# allocation and runs 20-50x slower
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node'; print('torch:', torch.__version__, 'cuda: True')"
nvidia-smi

DATASETS=(food101 octmnist plant_village organamnist galaxy10 fgvc_aircraft cars196 breastmnist cub200 dermamnist dtd eurosat flowers102 oxford_pet pathmnist)
SUBPATHS=(food101 med_mnist/octmnist-size=224 plant_village med_mnist/organamnist-size=224 galaxy10 fgvc_aircraft cars196 med_mnist/breastmnist-size=224 cub200 med_mnist/dermamnist-size=224 dtd eurosat flowers102 oxford_pet med_mnist/pathmnist-size=224)
i=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$i]}
PROCESSED_SUBPATH=${SUBPATHS[$i]}

DATA_ROOT="/scratch/gs4133/zhd/CP/data"
DL_DIR="${DATA_ROOT}/stable_datasets/downloads"
PROC_DIR="${DATA_ROOT}/stable_datasets/processed"
OUT_DIR="eval/outputs/int1_features"
mkdir -p "${OUT_DIR}"

# ============================================================
# Stage this dataset's processed cache node-local
# ============================================================
SRC_PROC="${PROC_DIR}/${PROCESSED_SUBPATH}"
if [ -d "${SRC_PROC}" ]; then
    NEED_KB=$(( $(du -sk "${SRC_PROC}" | awk '{print $1}') + 5*1024*1024 ))
    STAGE_ROOT=""
    for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
        [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
        avail_kb=$(df -Pk "${root}" 2>/dev/null | awk 'NR==2{print $4}')
        if [ -n "${avail_kb}" ] && [ "${avail_kb}" -ge "${NEED_KB}" ]; then STAGE_ROOT="${root}"; break; fi
    done
    if [ -n "${STAGE_ROOT}" ]; then
        LOCAL_CACHE="${STAGE_ROOT}/cpdata-${SLURM_JOB_ID}"
        trap 'rm -rf "${LOCAL_CACHE}"' EXIT
        DEST_PROC="${LOCAL_CACHE}/stable_datasets/processed/${PROCESSED_SUBPATH}"
        echo "===== staging ${SRC_PROC} -> ${DEST_PROC} ====="
        mkdir -p "$(dirname "${DEST_PROC}")" && rsync -a "${SRC_PROC}/" "${DEST_PROC}/"
        PROC_DIR="${LOCAL_CACHE}/stable_datasets/processed"
        DL_DIR="${LOCAL_CACHE}/stable_datasets/downloads"; mkdir -p "${DL_DIR}"
        echo "===== node-local: PROC_DIR=${PROC_DIR} ====="
    else
        echo "WARN: no node-local root with enough space; using /scratch."
    fi
else
    echo "WARN: processed cache not found at ${SRC_PROC}; using /scratch."
fi

echo "=========================================="
echo "INT1 features dump: dataset=${DATASET}"
echo "=========================================="

python -u eval/new_direction/int1_features_dump.py \
    --datasets "${DATASET}" \
    --device cuda \
    --download-dir  "${DL_DIR}" \
    --processed-dir "${PROC_DIR}" \
    --outdir "eval/outputs/int1_features" \
    2>&1

echo ""
echo "npz cells written under ${OUT_DIR}/ for dataset ${DATASET}"
echo "End Time: $(date)"
