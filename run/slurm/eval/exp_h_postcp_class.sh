#!/bin/bash
#SBATCH --job-name=expH-postclass
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
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/expH-postclass-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/expH-postclass-%A_%a.err

# ============================================================
# Exp H — post-CP class-anchored geometry at MAX (eval/adjudicate/postcp_class_sweep.py).
# One array task per dataset: processes that dataset's MAX checkpoints
# (4 methods x 3 encoders x seeds, ~36 ckpts), computing within/between-class
# spread + cdnv on the production pooled readout. Feeds P-D (does CP-induced
# WITHIN-class spreading mediate the DeltaFT side of the reversal?).
# No ImageNet needed. Reads ckpts only — safe to run while post-cp evaluations
# of other jobs are still in flight (they never rewrite ckpts).
# NOTE: run AFTER the retrained ckpts are in place (test1 all-PASS = they are).
#
# After the whole array finishes, concat shards on the login node:
#   python -c "import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('eval/outputs/postcp_class_shards/*.csv'))]).to_csv('eval/outputs/postcp_class_max.csv', index=False)"
# and send me eval/outputs/postcp_class_max.csv.
# Queue-friendly tip:  sbatch --array=0-14%5 run/slurm/eval/exp_h_postcp_class.sh
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
OUT_DIR="eval/outputs/postcp_class_shards"
mkdir -p "${OUT_DIR}"

# ============================================================
# Stage this dataset's processed cache node-local (no ImageNet needed)
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
# Run: all MAX ckpts of this dataset (4 methods x 3 encoders x seeds)
# ============================================================
echo "=========================================="
echo "Exp H postcp_class_sweep: dataset=${DATASET}  ckpt-root=${CKPT_ROOT}"
echo "=========================================="

python -u eval/adjudicate/postcp_class_sweep.py \
    --ckpt-root "${CKPT_ROOT}" \
    --datasets "${DATASET}" \
    --download-dir  "${DL_DIR}" \
    --processed-dir "${PROC_DIR}" \
    --out "${OUT_DIR}/${DATASET}.csv" \
    2>&1

echo ""
echo "Shard written: ${OUT_DIR}/${DATASET}.csv"
echo "End Time: $(date)"
