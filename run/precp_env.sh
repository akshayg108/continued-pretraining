#!/usr/bin/env bash

export CP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CP_ROOT="${CP_ROOT:-$(dirname -- "$CP_REPO_ROOT")}"
export CP_PYTHON="$CP_ROOT/env/bin/python3"
export STABLE_DATASETS_CACHE_DIR="$CP_ROOT/data/stable_datasets"
export XDG_CACHE_HOME="$CP_ROOT/data/.cache"
export HF_HOME="$CP_ROOT/data/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TORCH_HOME="$CP_ROOT/data/torch"
export CUDA_CACHE_PATH="$CP_ROOT/data/cuda-cache"
export TORCHINDUCTOR_CACHE_DIR="$CP_ROOT/data/torchinductor"
export TRITON_CACHE_DIR="$CP_ROOT/data/triton"
export PIP_CACHE_DIR="$CP_ROOT/data/pip-cache"
export CONDA_PKGS_DIRS="$CP_ROOT/data/conda-pkgs"
export TMPDIR="$CP_ROOT/data/tmp"
export WANDB_DIR="$CP_ROOT/outputs/wandb"
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CP_REPO_ROOT"

mkdir -p "$STABLE_DATASETS_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" \
    "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR" "$WANDB_DIR" \
    "$CP_ROOT/outputs/slurm-log" "$CP_ROOT/outputs/environment"
chmod 700 "$HF_HOME"
