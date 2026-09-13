"""CPU-only checks for the isolated ViT-L LeJEPA microbatch retry."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from eval.vitl_completion.protocol import cp_recipe


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/slurm/cp-L/cp/lejepa_64x4.sh"


def run_script(tmp_path, *args, extra_env=None, script=SCRIPT):
    env = dict(os.environ, PYTHON=sys.executable,
               VITL_LEJEPA_64X4_REPO_ROOT=str(ROOT),
               VITL_LEJEPA_64X4_OUTPUT_BASE=str(tmp_path / "outputs"),
               VITL_LEJEPA_64X4_CACHE_DIR=str(tmp_path / "shared"))
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        env.pop(name, None)
    env.update(extra_env or {})
    return subprocess.run(["bash", str(script), *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def commands(output):
    return [shlex.split(line)[1:] for line in output.splitlines() if line.startswith("COMMAND ")]


def test_dry_run_previews_six_isolated_cp_seeds_without_writing(tmp_path):
    result = run_script(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = commands(result.stdout)
    assert len(rows) == 6
    pairs = []
    for command in rows:
        args = dict(zip(command[2::2], command[3::2]))
        pairs.append((args["--dataset"], int(args["--seed"])))
        n = {"plant_village": 43596, "organamnist": 34561}[args["--dataset"]]
        recipe = cp_recipe("LeJEPA", n)
        recipe.update(batch_size=64, accumulate_grad_batches=4)
        for key, value in recipe.items():
            assert float(args["--" + key.replace("_", "-")]) == float(value)
        assert args["--cp-method"] == "lejepa" and args["--n-samples"] == str(n)
        assert args["--backbone"] == "vit_large_patch16_dinov3.lvd1689m"
        assert args["--pool-strategy"] == "cls"
        assert "/vitl_lejepa_64x4_v1/" in args["--results-json"]
        assert args["--project"] == "vitl_lejepa_64x4_v1"
        for flag in ("--resume", "--random-init", "--post-cp-sft", "--pre-cp-sft", "--skip-baseline"):
            assert flag not in command
    assert pairs == [(d, s) for d in ("plant_village", "organamnist") for s in (42, 43, 44)]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("task_id,dataset,seed", [(0, "plant_village", 42), (3, "organamnist", 42), (5, "organamnist", 44)])
def test_spooled_array_selects_exactly_one_seed(tmp_path, task_id, dataset, seed):
    assert SCRIPT.is_file()
    spool = tmp_path / "slurm_script"
    spool.write_text(SCRIPT.read_text())
    result = run_script(tmp_path, "--dry-run", script=spool, extra_env={
        "SLURM_JOB_ID": "100", "SLURM_ARRAY_JOB_ID": "90", "SLURM_ARRAY_TASK_ID": str(task_id)})
    assert result.returncode == 0, result.stdout + result.stderr
    rows = commands(result.stdout)
    assert len(rows) == 1
    args = dict(zip(rows[0][2::2], rows[0][3::2]))
    assert args["--dataset"] == dataset and args["--seed"] == str(seed)
    assert "/vitl_lejepa_64x4_v1/90/" in args["--results-json"]


@pytest.mark.parametrize("task_id", ["-1", "6", "hello", "1+1"])
def test_invalid_task_id_is_rejected_before_execution(tmp_path, task_id):
    result = run_script(tmp_path, "--dry-run", extra_env={"SLURM_ARRAY_TASK_ID": task_id})
    assert result.returncode == 2
    assert not commands(result.stdout)


def test_fixed_resources_and_no_direct_training_on_login_node(tmp_path):
    assert SCRIPT.is_file()
    source = SCRIPT.read_text()
    for setting in ("#SBATCH --array=0-5%6", "#SBATCH --gres=gpu:a100:1",
                    "#SBATCH --constraint=80g", "#SBATCH --time=96:00:00",
                    "#SBATCH --mem=96G", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1"):
        assert setting in source
    result = run_script(tmp_path)
    assert result.returncode == 2
    assert "sbatch" in result.stderr
    result = run_script(tmp_path, "--batch-size", "32")
    assert result.returncode == 2


@pytest.mark.parametrize("exit_code", [0, 7])
def test_staging_arithmetic_and_cleanup_preserve_shared_data(tmp_path, exit_code):
    source = tmp_path / "shared/stable_datasets/processed/plant_village"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"source data")
    stage = tmp_path / "stage"
    stage.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "training.json"
    fake_python = bindir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == '-c': sys.exit(0)\n"
        "if args[0] == '-': os.execv(sys.executable, [sys.executable, *args])\n"
        "opts = dict(zip(args[1::2], args[2::2]))\n"
        "cache = pathlib.Path(opts['--cache-dir'])\n"
        "assert (cache / 'stable_datasets/processed/plant_village/sample.bin').read_bytes() == b'source data'\n"
        "pathlib.Path(os.environ['TRAINING_MARKER']).write_text(json.dumps(opts))\n"
        f"sys.exit({exit_code})\n")
    fake_python.chmod(0o755)
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 0\n")
    smi.chmod(0o755)
    result = run_script(tmp_path, extra_env={
        "PYTHON": str(fake_python), "PATH": f"{bindir}:{os.environ['PATH']}",
        "TMPDIR": str(stage), "TRAINING_MARKER": str(marker),
        "VITL_LEJEPA_64X4_SKIP_ENV_SETUP": "1", "SLURM_JOB_ID": "100",
        "SLURM_ARRAY_JOB_ID": "90", "SLURM_ARRAY_TASK_ID": "0"})
    assert result.returncode == exit_code, result.stdout + result.stderr
    args = json.loads(marker.read_text())
    private = Path(args["--cache-dir"])
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"source data"
    assert args["--batch-size"] == "64" and args["--accumulate-grad-batches"] == "4"
