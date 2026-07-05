#!/bin/bash
#SBATCH --job-name=test2-geo
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/test2-geo-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/test2-geo-%j.err

# ============================================================
# Test 2 — recompute post-CP geometry for the 59 re-trained checkpoints
# (eval/rest/test2_geometry.py). Single V100 job: loops over all 59 ckpts in
# eval/outputs/rerun_geometry.csv, re-embeds each dataset (<=5000) + ImageNet-val,
# and writes eval/outputs/rest_geometry.csv (l2_norm_cv / uniformity_t2 /
# neighbor_overlap_k50 + d_cv / d_unif / d_overlap vs pre-CP geometry_15.csv).
#
# The 59 ckpts touch only 4 datasets (cars196, food101, octmnist, pathmnist);
# we node-stage those + ImageNet-val so the 59x ImageNet re-embeddings are fast.
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
# Paths (user's cluster layout)
# ============================================================
DATA_ROOT="/scratch/gs4133/zhd/CP/data"
DL_DIR="${DATA_ROOT}/stable_datasets/downloads"
PROC_DIR="${DATA_ROOT}/stable_datasets/processed"
IMAGENET_DIR="${DATA_ROOT}/imagenet_val"
OUT="eval/outputs/rest_geometry.csv"

# datasets the 59 rerun ckpts span, + their processed-cache subpaths
TEST2_DATASETS=(cars196 food101 octmnist pathmnist)
TEST2_SUBPATHS=(cars196 food101 med_mnist/octmnist-size=224 med_mnist/pathmnist-size=224)

# ============================================================
# Stage the 4 processed caches + ImageNet-val to node-local fast storage.
# (Same idiom as run/slurm/cp-siglip/*: the HF loader mmaps a complete processed
# cache and never reads downloads/; /scratch random reads starve the GPU.)
# ============================================================
STAGE_ROOT=""
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    avail_kb=$(df -Pk "${root}" 2>/dev/null | awk 'NR==2{print $4}')
    # need room for the 4 datasets + imagenet + headroom (~120G is plenty; food101/pathmnist are the big ones)
    if [ -n "${avail_kb}" ] && [ "${avail_kb}" -ge $((120*1024*1024)) ]; then STAGE_ROOT="${root}"; break; fi
done

if [ -n "${STAGE_ROOT}" ]; then
    LOCAL_CACHE="${STAGE_ROOT}/cpdata-${SLURM_JOB_ID}"
    trap 'rm -rf "${LOCAL_CACHE}"' EXIT
    # per-dataset processed caches
    for idx in "${!TEST2_DATASETS[@]}"; do
        sub="${TEST2_SUBPATHS[$idx]}"
        src="${PROC_DIR}/${sub}"
        dst="${LOCAL_CACHE}/stable_datasets/processed/${sub}"
        if [ -d "${src}" ]; then
            echo "===== staging ${src} -> ${dst} ====="
            mkdir -p "$(dirname "${dst}")" && rsync -a "${src}/" "${dst}/"
        else
            echo "WARN: processed cache missing: ${src} (will fall back to /scratch for ${TEST2_DATASETS[$idx]})"
        fi
    done
    # ImageNet-val (hit 59x — re-embedded per ckpt/encoder)
    if [ -d "${IMAGENET_DIR}" ]; then
        echo "===== staging ${IMAGENET_DIR} -> ${LOCAL_CACHE}/imagenet_val ====="
        mkdir -p "${LOCAL_CACHE}/imagenet_val" && rsync -a "${IMAGENET_DIR}/" "${LOCAL_CACHE}/imagenet_val/"
        IMAGENET_DIR="${LOCAL_CACHE}/imagenet_val"
    fi
    PROC_DIR="${LOCAL_CACHE}/stable_datasets/processed"
    DL_DIR="${LOCAL_CACHE}/stable_datasets/downloads"; mkdir -p "${DL_DIR}"
    echo "===== node-local: PROC_DIR=${PROC_DIR}  IMAGENET_DIR=${IMAGENET_DIR} ====="
else
    echo "WARN: no node-local root with >=120G free; using /scratch (slower)."
fi

# ============================================================
# Run
# ============================================================
echo "=========================================="
echo "Test 2: post-CP geometry for 59 rerun ckpts -> ${OUT}"
echo "  download-dir : ${DL_DIR}"
echo "  processed-dir: ${PROC_DIR}"
echo "  imagenet-dir : ${IMAGENET_DIR}"
echo "=========================================="

python -u eval/rest/test2_geometry.py \
    --csv eval/outputs/rerun_geometry.csv \
    --download-dir  "${DL_DIR}" \
    --processed-dir "${PROC_DIR}" \
    --imagenet-dir  "${IMAGENET_DIR}" \
    --geometry15 eval/outputs/geometry_15.csv \
    --out "${OUT}" \
    2>&1

echo ""
echo "=========================================="
echo "Done. Send me: eval/outputs/rest_geometry.csv"
echo "End Time: $(date)"
echo "=========================================="
