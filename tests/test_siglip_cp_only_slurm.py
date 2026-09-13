"""CPU-only checks for the one-seed-per-job SigLIP restart launcher."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from eval.full_ft import run as ft
from eval.siglip_mainrule import cp_only as cp
from eval.siglip_mainrule import protocol as original
from eval.siglip_mainrule import run as legacy


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "run/slurm/cp-siglip/cp-only/submit.sh"
ARRAY = ROOT / "run/slurm/cp-siglip/cp-only/array.sh"


def run_submit(tmp_path, *args, env=None):
    merged = dict(os.environ, SIGLIP_CP_ONLY_OUTPUT_BASE=str(tmp_path / "outputs"),
                  SIGLIP_CP_ONLY_LOG_DIR=str(tmp_path / "logs"))
    if env:
        merged.update(env)
    return subprocess.run(["bash", str(SUBMIT), *args], cwd=ROOT, env=merged,
                          capture_output=True, text=True)


def mark_complete(base, indices):
    doc = cp.build_manifest(base)
    for index in indices:
        entry = doc["tasks"][index]
        task, seed = entry["source_task"], entry["seed"]
        paths = legacy.seed_paths({"output_root": doc["source_output_root"]}, task, seed)
        row = dict(dataset=task["dataset"], n_samples=task["n_samples"],
                   backbone=task["model_id"], method="lejepa", seed=seed, epochs=150,
                   random_init=False, no_cp=False, post_knn_f1=.6, post_linear_f1=.7,
                   post_knn_acc=.6, post_linear_acc=.7)
        ft.atomic_json(paths["cp_result"], row)
        ft.atomic_json(paths["receipt"], dict(
            schema_version=1, protocol=original.PROTOCOL, status="cp_complete", seed=seed,
            cp_recipe=task["cp_recipe"], task_sha256=ft.digest_json(task),
            implementation_sha256=doc["training_implementation_sha256"],
            cp_result_sha256=ft.file_sha256(paths["cp_result"])))


def test_cp_only_slurm_entrypoints_exist():
    assert SUBMIT.is_file() and ARRAY.is_file()


def test_dry_run_selects_four_missing_seeds_and_prints_no_ft_commands(tmp_path):
    mark_complete(tmp_path / "outputs", [0, 1, 3, 4, 6])
    result = run_submit(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    submit = next(line for line in result.stdout.splitlines() if line.startswith("SUBMIT "))
    assert "--array=2,5,7,8%12" in shlex.split(submit)
    assert "4 jobs, one CP seed per job, 0 FT fits" in result.stdout
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert [(r["task_id"], r["seed"]) for r in rows] == [(2, 44), (5, 44), (7, 43), (8, 44)]
    for row in rows:
        assert row["stage"] == "cp" and "--resume" not in row["command"]
        assert "--post-cp-sft" not in row["command"]
    assert not (tmp_path / "outputs" / cp.PROTOCOL).exists()
    assert "full_ft/run.py" not in result.stdout


def test_submit_launches_one_array_with_one_a100_80gb_per_seed(tmp_path):
    mark_complete(tmp_path / "outputs", [0, 1, 3, 4, 6])
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "open(os.environ['CALLS'], 'a').write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('1234')\n")
    sbatch.chmod(0o755)
    result = run_submit(tmp_path, "--concurrency", "4", env={
        "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 1
    command = commands[0]
    for argument in ("--array=2,5,7,8%4", "--gres=gpu:a100:1", "--constraint=80g",
                     "--time=96:00:00", "--mem=96G", "--cpus-per-task=8"):
        assert argument in command
    assert command[-1] == str(ARRAY)
    assert "SIGLIP_CP_ONLY_MANIFEST=" in next(a for a in command if a.startswith("--export="))


def test_all_cp_complete_means_no_submission(tmp_path):
    mark_complete(tmp_path / "outputs", range(9))
    result = run_submit(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nothing to submit" in result.stdout
    assert "SUBMIT sbatch" not in result.stdout


@pytest.mark.parametrize("args", [("--concurrency", "0"), ("--concurrency", "13"),
                                  ("--concurrency", "no"), ("--resume",),
                                  ("--methods", "DIET")])
def test_submit_rejects_invalid_or_recipe_changing_arguments(tmp_path, args):
    result = run_submit(tmp_path, *args)
    assert result.returncode == 2
    assert "SUBMIT" not in result.stdout


def test_corrupt_source_receipt_aborts_before_submission(tmp_path):
    base = tmp_path / "outputs"
    mark_complete(base, [0])
    result = base / original.PROTOCOL / "cp_results/LeJEPA/octmnist/seed42.json"
    result.write_text("{}")
    command = run_submit(tmp_path, "--dry-run")
    assert command.returncode != 0
    assert "SUBMIT" not in command.stdout


def test_spooled_array_dry_run_preserves_one_seed_per_job(tmp_path):
    manifest = tmp_path / "manifest.json"
    ft.atomic_json(manifest, cp.build_manifest(tmp_path / "outputs"))
    spool = tmp_path / "slurm_script"
    spool.write_text(ARRAY.read_text())
    env = dict(os.environ, SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=str(ROOT),
               SLURM_ARRAY_TASK_ID="7", SIGLIP_CP_ONLY_MANIFEST=str(manifest),
               SIGLIP_CP_ONLY_CACHE_DIR="/shared/cache")
    result = subprocess.run(["bash", str(spool), "--dry-run"], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["seed"] == 43
    assert rows[0]["command"][rows[0]["command"].index("--dataset") + 1] == "food101"
    assert not (tmp_path / "outputs" / cp.PROTOCOL).exists()


def test_array_stages_and_cleans_a_private_dataset_copy(tmp_path):
    source = tmp_path / "shared/stable_datasets/processed/example/data"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"sample")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "runner.txt"
    fake_python = bindir / "python"
    fake_python.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "if [ \"${1:-}\" = - ]; then echo example/data; exit 0; fi\n"
        "if [ \"${1:-}\" = -c ]; then exit 0; fi\n"
        "cache=''\nwhile [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = --cache-dir ]; then cache=$2; shift 2; else shift; fi\n"
        "done\n"
        "test -f \"${cache}/stable_datasets/processed/example/data/sample.bin\"\n"
        "printf '%s\\n' \"${cache}\" > \"${RUNNER_MARKER}\"\n")
    fake_python.chmod(0o755)
    nvidia_smi = bindir / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/bash\nexit 0\n")
    nvidia_smi.chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
               PYTHON=str(fake_python), RUNNER_MARKER=str(marker), TMPDIR=str(stage),
               SIGLIP_CP_ONLY_SKIP_ENV_SETUP="1", SIGLIP_CP_ONLY_REPO_ROOT=str(ROOT),
               SIGLIP_CP_ONLY_MANIFEST=str(manifest), SLURM_ARRAY_TASK_ID="2",
               SIGLIP_CP_ONLY_CACHE_DIR=str(tmp_path / "shared"))
    result = subprocess.run(["bash", str(ARRAY)], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    private = Path(marker.read_text().strip())
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"sample"


def test_array_has_fixed_resources_and_no_seed_loop():
    text = ARRAY.read_text()
    for fragment in ("#SBATCH --gres=gpu:a100:1", "#SBATCH --constraint=80g",
                     "#SBATCH --time=96:00:00", "module load miniconda/3-4.11.0",
                     "conda activate env", "mktemp -d", "trap 'rm -rf --",
                     "df -Pk", "du -sk", "processed_subpath", "nvidia-smi"):
        assert fragment in text
    assert "for seed" not in text and "full_ft/run.py" not in text
