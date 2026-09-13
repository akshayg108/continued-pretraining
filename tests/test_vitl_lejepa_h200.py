"""CPU-only checks for dataset-level ViT-L LeJEPA jobs on H200."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import pytest

from eval.vitl_completion.protocol import cp_recipe


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/slurm/cp-L/cp/lejepa_h200.sh"


def run_script(tmp_path, *args, extra_env=None, script=SCRIPT):
    env = dict(os.environ, PYTHON=sys.executable,
               VITL_LEJEPA_H200_REPO_ROOT=str(ROOT),
               VITL_LEJEPA_H200_OUTPUT_BASE=str(tmp_path / "outputs"),
               VITL_LEJEPA_H200_CACHE_DIR=str(tmp_path / "shared"))
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        env.pop(name, None)
    env.update(extra_env or {})
    return subprocess.run(["bash", str(script), *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def commands(output):
    return [shlex.split(line)[1:] for line in output.splitlines() if line.startswith("COMMAND ")]


def test_dry_run_previews_two_datasets_with_three_seeds_without_writing(tmp_path):
    result = run_script(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = commands(result.stdout)
    assert len(rows) == 6
    pairs = []
    for command in rows:
        args = dict(zip(command[2::2], command[3::2]))
        pairs.append((args["--dataset"], int(args["--seed"])))
        n = {"plant_village": 43596, "organamnist": 34561}[args["--dataset"]]
        for key, value in cp_recipe("LeJEPA", n).items():
            assert float(args["--" + key.replace("_", "-")]) == float(value)
        assert args["--cp-method"] == "lejepa" and args["--n-samples"] == str(n)
        assert args["--backbone"] == "vit_large_patch16_dinov3.lvd1689m"
        assert args["--pool-strategy"] == "cls"
        assert "/vitl_lejepa_h200_v1/" in args["--results-json"]
        assert args["--project"] == "vitl_lejepa_h200_v1"
        assert args["--checkpoint-dir"].endswith(f"/{args['--dataset']}/seed{args['--seed']}")
        for flag in ("--resume", "--random-init", "--post-cp-sft", "--pre-cp-sft", "--skip-baseline"):
            assert flag not in command
    assert pairs == [(d, s) for d in ("plant_village", "organamnist") for s in (42, 43, 44)]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("task_id,dataset", [(0, "plant_village"), (1, "organamnist")])
def test_spooled_array_selects_three_seeds_for_one_dataset(tmp_path, task_id, dataset):
    assert SCRIPT.is_file()
    spool = tmp_path / "slurm_script"
    spool.write_text(SCRIPT.read_text())
    result = run_script(tmp_path, "--dry-run", script=spool, extra_env={
        "SLURM_JOB_ID": "100", "SLURM_ARRAY_JOB_ID": "90", "SLURM_ARRAY_TASK_ID": str(task_id)})
    assert result.returncode == 0, result.stdout + result.stderr
    rows = commands(result.stdout)
    assert len(rows) == 3
    for seed, command in zip((42, 43, 44), rows):
        args = dict(zip(command[2::2], command[3::2]))
        assert args["--dataset"] == dataset and args["--seed"] == str(seed)
        assert "/vitl_lejepa_h200_v1/90/" in args["--results-json"]


@pytest.mark.parametrize("task_id", ["-1", "2", "hello", "1+1"])
def test_invalid_task_id_is_rejected_before_execution(tmp_path, task_id):
    result = run_script(tmp_path, "--dry-run", extra_env={"SLURM_ARRAY_TASK_ID": task_id})
    assert result.returncode == 2
    assert not commands(result.stdout)


def test_fixed_h200_resources_and_no_direct_training_on_login_node(tmp_path):
    assert SCRIPT.is_file()
    source = SCRIPT.read_text()
    for setting in ("#SBATCH --array=0-1%2", "#SBATCH --gres=gpu:h200:1",
                    "#SBATCH --account=civil", "#SBATCH --partition=nvidia",
                    "#SBATCH --qos=nvidia", "#SBATCH --time=96:00:00",
                    "#SBATCH --mem=96G", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1"):
        assert setting in source
    for obsolete in ("--constraint=80g", "require_a100_80gb", "--exclude=cn253,cn259"):
        assert obsolete not in source
    result = run_script(tmp_path)
    assert result.returncode == 2 and "sbatch" in result.stderr
    result = run_script(tmp_path, "--batch-size", "64")
    assert result.returncode == 2


@pytest.mark.parametrize("name,memory_gib,count,accepted", [
    ("NVIDIA H200", 140, 1, True),
    ("NVIDIA H200 NVL", 140, 1, True),
    ("NVIDIA H100 80GB HBM3", 80, 1, False),
    ("NVIDIA H200 MIG", 140, 1, False),
    ("NVIDIA H200", 18, 1, False),
    ("NVIDIA H200", 140, 2, False),
    ("NVIDIA H200", 140, 0, False),
])
def test_runtime_gpu_guard_requires_one_full_h200(monkeypatch, name, memory_gib, count, accepted):
    assert SCRIPT.is_file()
    code = SCRIPT.read_text().split("<<'GPU_CHECK'\n", 1)[1].split("\nGPU_CHECK", 1)[0]
    props = types.SimpleNamespace(name=name, total_memory=memory_gib * 1024**3)
    cuda = types.SimpleNamespace(is_available=lambda: count > 0,
                                 device_count=lambda: count,
                                 get_device_properties=lambda index: props)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    if accepted:
        exec(code, {})
    else:
        with pytest.raises(SystemExit):
            exec(code, {})


@pytest.mark.parametrize("failed_seed", [0, 43])
def test_sequential_execution_staging_and_failure_cleanup(tmp_path, failed_seed):
    source = tmp_path / "shared/stable_datasets/processed/plant_village"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"source data")
    stage = tmp_path / "stage"
    stage.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "training.jsonl"
    fake_python = bindir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['-']: sys.exit(0)\n"
        "if args[0] == '-': os.execv(sys.executable, [sys.executable, *args])\n"
        "opts = dict(zip(args[1::2], args[2::2]))\n"
        "cache = pathlib.Path(opts['--cache-dir'])\n"
        "assert (cache / 'stable_datasets/processed/plant_village/sample.bin').read_bytes() == b'source data'\n"
        "with pathlib.Path(os.environ['TRAINING_MARKER']).open('a') as handle:\n"
        "    handle.write(json.dumps(opts) + '\\n')\n"
        f"if int(opts['--seed']) == {failed_seed}: sys.exit(7)\n"
        "pathlib.Path(opts['--results-json']).write_text(json.dumps({'seed': int(opts['--seed'])}))\n")
    fake_python.chmod(0o755)
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 0\n")
    smi.chmod(0o755)
    env = {
        "PYTHON": str(fake_python), "PATH": f"{bindir}:{os.environ['PATH']}",
        "TMPDIR": str(stage), "TRAINING_MARKER": str(marker),
        "VITL_LEJEPA_H200_SKIP_ENV_SETUP": "1", "SLURM_JOB_ID": "100",
        "SLURM_ARRAY_JOB_ID": "90", "SLURM_ARRAY_TASK_ID": "0"}
    result = run_script(tmp_path, extra_env=env)
    assert result.returncode == (1 if failed_seed else 0), result.stdout + result.stderr
    rows = [json.loads(line) for line in marker.read_text().splitlines()]
    assert [int(row['--seed']) for row in rows] == [42, 43, 44]
    assert len({row['--cache-dir'] for row in rows}) == 1
    for args in rows:
        private = Path(args["--cache-dir"])
        assert private.parent == stage and not private.exists()
        assert args["--batch-size"] == "128" and args["--accumulate-grad-batches"] == "2"
        path = tmp_path / f"outputs/vitl_lejepa_h200_v1/90/commands/LeJEPA/plant_village/seed{args['--seed']}.json"
        record = json.loads(path.read_text())
        assert record["protocol"] == "vitl_lejepa_h200_v1"
        assert record["initialization"] == "public_pretrained"
        assert record["entrypoint_sha256"] == hashlib.sha256((ROOT / "continued_pretraining.py").read_bytes()).hexdigest()
    assert (source / "sample.bin").read_bytes() == b"source data"
    assert not list(stage.iterdir())
    if failed_seed:
        assert "FAIL LeJEPA plant_village seed=43" in result.stderr
        assert "SUCCESS LeJEPA plant_village seed=44" in result.stdout
    repeated = run_script(tmp_path, extra_env=env)
    assert repeated.returncode != 0
    assert len(marker.read_text().splitlines()) == 3
