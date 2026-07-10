#!/bin/bash
#SBATCH --job-name=expI-layer-post
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/expI-layer-post-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/expI-layer-post-%A_%a.err

# ============================================================
# Exp I (post-CP side) — per-layer kNN + uniformity for MAX checkpoints
# (eval/F4_gate/layerwise_postcp.py). One array task per dataset.
# Methods: LeJEPA/SimCLR/DIET on {DINOv3, CLIP, MAE} (main grid) + LeJEPA/SimCLR on
# SigLIP (cp-siglip root; auto-skipped with a warning if the root is absent).
# No ImageNet needed. Checkpoints are read from /scratch (many small reads — that is
# the dominant IO and cannot be staged cheaply); the dataset IS staged node-local.
#
# After the whole array finishes, concat shards on the login node:
#   python -c "import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('eval/outputs/layerwise_postcp_shards/*.csv'))]).to_csv('eval/outputs/layerwise_postcp.csv', index=False)"
# and send me eval/outputs/layerwise_postcp.csv (+ layerwise_pre.csv from the pre job).
# Queue-friendly tip:  sbatch --array=0-14%5 run/slurm/eval/exp_i_layerwise_post.sh
# ============================================================

echo "=========================================="
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi

# ============================================================
# Per-dataset config (indexed by array task id) — same order as exp-b
# ============================================================
DATASETS=(food101 octmnist plant_village organamnist galaxy10 fgvc_aircraft cars196 breastmnist cub200 dermamnist dtd eurosat flowers102 oxford_pet pathmnist)
SUBPATHS=(food101 med_mnist/octmnist-size=224 plant_village med_mnist/organamnist-size=224 galaxy10 fgvc_aircraft cars196 med_mnist/breastmnist-size=224 cub200 med_mnist/dermamnist-size=224 dtd eurosat flowers102 oxford_pet med_mnist/pathmnist-size=224)
i=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$i]}
PROCESSED_SUBPATH=${SUBPATHS[$i]}

DATA_ROOT="/scratch/gs4133/zhd/CP/data"
DL_DIR="${DATA_ROOT}/stable_datasets/downloads"
PROC_DIR="${DATA_ROOT}/stable_datasets/processed"
CKPT_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp"
SIGLIP_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip"
OUT_DIR="eval/outputs/layerwise_postcp_shards"
mkdir -p "${OUT_DIR}"

# ============================================================
# Stage this dataset's processed cache node-local (no ImageNet for Exp I)
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

# ============================================================
# Run: all MAX ckpts of this dataset (LeJEPA/SimCLR/DIET x D3/CLIP/MAE + SigLIP grid)
# ============================================================
echo "=========================================="
echo "Exp I layerwise_postcp: dataset=${DATASET}"
echo "=========================================="

python -u eval/F4_gate/layerwise_postcp.py \
    --datasets "${DATASET}" \
    --ckpt-root "${CKPT_ROOT}" \
    --siglip-root "${SIGLIP_ROOT}" \
    --download-dir  "${DL_DIR}" \
    --processed-dir "${PROC_DIR}" \
    --out "${OUT_DIR}/${DATASET}.csv" \
    2>&1

echo ""
echo "Shard written: ${OUT_DIR}/${DATASET}.csv"
echo "End Time: $(date)"
