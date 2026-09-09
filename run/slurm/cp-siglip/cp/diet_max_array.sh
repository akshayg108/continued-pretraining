#!/bin/bash
#SBATCH --job-name=sg-diet
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --array=0-14%12
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/sg-diet-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/sg-diet-%A_%a.err
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
if [ "${DRY_RUN}" -eq 0 ] && [ "${SIGLIP_DIET_SKIP_ENV_SETUP:-0}" != 1 ]; then
 module load miniconda/3-4.11.0
 source "$(conda info --base)/etc/profile.d/conda.sh"
 conda activate env
fi
set -euo pipefail
if [ -n "${SIGLIP_DIET_REPO_ROOT:-}" ]; then
 REPO_ROOT="${SIGLIP_DIET_REPO_ROOT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
 REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gs4133/zhd/CP/continued-pretraining}"
 [ -f "${REPO_ROOT}/continued_pretraining.py" ] || REPO_ROOT="${REPO_ROOT}/continued-pretraining"
else
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
[ -f "${REPO_ROOT}/continued_pretraining.py" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
PROTOCOL="${REPO_ROOT}/eval/F5_decision_score/siglip_diet_protocol.py"
OUT_ROOT="${SIGLIP_DIET_OUT_ROOT:-/scratch/gs4133/zhd/CP/outputs}"
DATA_ROOT="${SIGLIP_DIET_DATA_DIR:-/scratch/gs4133/zhd/CP/data}"
PY="${PYTHON:-python3}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
[ "${TASK_ID}" -ge 0 ] && [ "${TASK_ID}" -lt 15 ] || { echo "invalid dataset task ${TASK_ID}" >&2; exit 2; }
eval "$("${PY}" "${PROTOCOL}" --cell "$((TASK_ID*3))" --format env)"
SEEDS=(42 43 44); MODEL_ID="vit_base_patch16_siglip_224.v2_webli"
CKPT_DIR="${OUT_ROOT}/ckpts/cp-siglip/cp/DIET/${DISPLAY_NAME}/SigLIP"
LOG_DIR="${OUT_ROOT}/logs/cp-siglip/cp/DIET/${DISPLAY_NAME}/SigLIP"
FT_OUTDIR="${OUT_ROOT}/full_ft_v1"; DATA_DIR="${DATA_ROOT}"
LOCAL_CACHE=""
WORK_DIR=""
build_cp_command() {
 local s=$1 r=$2
 CP_CMD=("${PY}" -u continued_pretraining.py --cp-method diet --dataset "${DATASET}" --backbone "${MODEL_ID}" --n-samples "${N_SAMPLES}" --epochs 150 --batch-size 32 --lr 1e-4 --weight-decay 0.05 --freeze-epochs 15 --num-trained-blocks 2 --knn-k 20 --num-workers 8 --label-smoothing 0.3 --mixup-alpha 1.0 --cutmix-alpha 1.0 --mixup-cutmix-prob 0.0 --mixup-cutmix-switch-prob 0.5 --pool-strategy map --accumulate-grad-batches 1 --checkpoint-dir "${CKPT_DIR}" --cache-dir "${DATA_DIR}" --project "diet-cp-siglip-${DATASET}" --run-name "SigLIP_${DATASET}_n${N_SAMPLES}_blk2_s${s}" --seed "${s}" --resume --skip-baseline --results-json "${r}")
}
render_command() { printf '%q ' "$@"; }
complete() { ${PY} "${PROTOCOL}" --check-result "$1" --cell "$2" --ckpt "$3" >/dev/null 2>&1; }
if [ "${DRY_RUN}" -eq 1 ]; then
 "${PY}" "${PROTOCOL}" --verify-prereg >/dev/null
 R="${LOG_DIR}/SigLIP_${DATASET}_n${N_SAMPLES}_seed42.json"; C="${CKPT_DIR}/cp/${DATASET}_${MODEL_ID}_n${N_SAMPLES}_s42.ckpt"; complete "${R}" "$((TASK_ID*3))" "${C}" && SKIP=yes || SKIP=no
 build_cp_command 42 "${R}"; echo "DRY-RUN task_id=${TASK_ID}"; echo "DRY-RUN dataset=${DATASET}"; echo "DRY-RUN display=${DISPLAY_NAME}"; echo "DRY-RUN seeds=42 43 44"; echo "DRY-RUN n_samples=${N_SAMPLES}"; echo "DRY-RUN processed_subpath=${PROCESSED_SUBPATH}"; echo "DRY-RUN skip=${SKIP}"; echo "DRY-RUN ft_outdir=${FT_OUTDIR}"; echo "DRY-RUN command=$(render_command "${CP_CMD[@]}")"; build_cp_command 43 "${LOG_DIR}/SigLIP_${DATASET}_n${N_SAMPLES}_seed43.json"; echo "DRY-RUN command_seed43=$(render_command "${CP_CMD[@]}")"; build_cp_command 44 "${LOG_DIR}/SigLIP_${DATASET}_n${N_SAMPLES}_seed44.json"; echo "DRY-RUN command_seed44=$(render_command "${CP_CMD[@]}")"; echo "DRY-RUN ft_command=${PY} eval/full_ft/run.py --manifest <private-manifest> --task-id ${TASK_ID} --cache-dir ${DATA_ROOT} --outdir ${FT_OUTDIR} --device cuda --seeds 42"; exit 0
fi
cd "${REPO_ROOT}"; mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${FT_OUTDIR}"
SRC="${DATA_ROOT}/stable_datasets/processed/${PROCESSED_SUBPATH}"; [ -d "${SRC}" ] || { echo "missing processed cache: ${SRC}" >&2; exit 4; }
"${PY}" "${PROTOCOL}" --verify-prereg
"${PY}" -c "import torch,timm; assert torch.cuda.is_available(); timm.create_model('${MODEL_ID}', pretrained=False, num_classes=0)"
nvidia-smi
NEED=$(( $(du -sk "${SRC}" | awk '{print $1}') + 5*1024*1024 ))
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
 [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue; AVAIL=$(df -Pk "${root}" | awk 'NR==2{print $4}')
 if [ "${AVAIL:-0}" -ge "${NEED}" ]; then LOCAL_CACHE=$(mktemp -d "${root}/siglip-diet-${SLURM_JOB_ID:-local}-${TASK_ID}.XXXXXX"); trap 'rm -rf -- "${LOCAL_CACHE:-}" "${WORK_DIR:-}"' EXIT; mkdir -p "${LOCAL_CACHE}/stable_datasets/processed/$(dirname "${PROCESSED_SUBPATH}")"; rsync -a "${SRC}/" "${LOCAL_CACHE}/stable_datasets/processed/${PROCESSED_SUBPATH}/"; DATA_DIR="${LOCAL_CACHE}"; break; fi
done
[ "${DATA_DIR}" != "${DATA_ROOT}" ] || echo "WARN: staging unavailable; using shared cache ${SRC}"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/siglip-ft-work-${SLURM_JOB_ID:-local}-${TASK_ID}.XXXXXX")
trap 'rm -rf -- "${LOCAL_CACHE:-}" "${WORK_DIR:-}"' EXIT
MANIFEST="${WORK_DIR}/manifest.json"; ${PY} eval/full_ft/manifest.py --repo-root "${REPO_ROOT}" --methods DIET --encoders SigLIP --budgets MAX --phases post --checkpoint-root "${OUT_ROOT}/ckpts" --output "${MANIFEST}"
FINAL_STATUS=0
for i in 0 1 2; do
 SEED=${SEEDS[$i]}; CELL=$((TASK_ID*3+i)); RESULT="${LOG_DIR}/SigLIP_${DATASET}_n${N_SAMPLES}_seed${SEED}.json"; CKPT="${CKPT_DIR}/cp/${DATASET}_${MODEL_ID}_n${N_SAMPLES}_s${SEED}.ckpt"
 if ! complete "${RESULT}" "${CELL}" "${CKPT}"; then build_cp_command "${SEED}" "${RESULT}"; printf '[CP] '; render_command "${CP_CMD[@]}"; printf '\n'; if ! "${CP_CMD[@]}" || ! complete "${RESULT}" "${CELL}" "${CKPT}"; then FINAL_STATUS=1; continue; fi; fi
 if ! "${PY}" eval/full_ft/run.py --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${DATA_DIR}" --outdir "${FT_OUTDIR}" --device cuda --seeds "${SEED}"; then FINAL_STATUS=1; fi
done
exit "${FINAL_STATUS}"
