"""CPU-only submission and node-staging tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "run/slurm/cp-L/completion/submit.sh"
ARRAY = ROOT / "run/slurm/cp-L/completion/array.sh"


def run_submit(tmp_path, *args, env=None):
    merged = dict(os.environ, PYTHON=sys.executable,
                  VITL_COMPLETION_OUTPUT_BASE=str(tmp_path / "outputs"),
                  VITL_COMPLETION_LOG_DIR=str(tmp_path / "logs"))
    if env:
        merged.update(env)
    return subprocess.run(["bash", str(SUBMIT), *args], cwd=ROOT, env=merged,
                          capture_output=True, text=True)


@pytest.mark.parametrize("datasets,tasks", [((), 24), (("breastmnist",), 3),
                                          (("food101", "flowers102"), 6)])
def test_dry_run_selected_counts_resources_and_no_training(tmp_path, datasets, tasks):
    args = ["--datasets", *datasets] if datasets else []
    result = run_submit(tmp_path, *args, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"--array=0-{tasks - 1}%12" in result.stdout
    for flag in ("--gres=gpu:a100:1", "--constraint=80g", "--mem=96G", "--time=96:00:00"):
        assert flag in result.stdout
    assert f"{tasks} tasks, {tasks * 3} CP fits, 0 FT fits" in result.stdout
    manifests = list((tmp_path / "outputs/vitl_completion_manifests").glob("*.json"))
    assert len(manifests) == 1
    doc = json.loads(manifests[0].read_text())
    assert len(doc["tasks"]) == tasks
    if datasets:
        assert {t["dataset"] for t in doc["tasks"]} == set(datasets)
    assert not Path(doc["output_root"]).exists()
    commands = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert [c["seed"] for c in commands] == [42, 43, 44]
    assert all(c["stage"] == "cp" for c in commands)


@pytest.mark.parametrize("args", [("--datasets",), ("--datasets", "dtd"),
                                 ("--datasets", "typo"), ("--datasets", "food101", "food101"),
                                 ("--concurrency", "0"), ("--concurrency", "13"),
                                 ("--concurrency", "abc"), ("--batch-size", "256"),
                                 ("--gres", "gpu:v100:1")])
def test_invalid_selection_and_resource_overrides_never_submit(tmp_path, args):
    result = run_submit(tmp_path, *args)
    assert result.returncode == 2
    assert "SUBMIT" not in result.stdout


def test_submission_calls_sbatch_once_with_all_selected_tasks(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('1234')\n")
    sbatch.chmod(0o755)
    result = run_submit(tmp_path, "--datasets", "breastmnist", "flowers102",
                        "--concurrency", "4", env={
                            "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 1
    assert "--array=0-5%4" in commands[0]
    assert "--gres=gpu:a100:1" in commands[0] and "--constraint=80g" in commands[0]
    assert commands[0][-1] == str(ARRAY)


def test_array_has_fixed_gpu_and_private_cache():
    source = ARRAY.read_text()
    for fragment in ("#SBATCH --gres=gpu:a100:1", "#SBATCH --constraint=80g",
                     "#SBATCH --time=96:00:00", "module load miniconda/3-4.11.0",
                     "conda activate env", "set -euo pipefail", "mktemp -d", "du -sk",
                     "df -Pk", "rsync -a", "nvidia-smi", "require_a100_80gb"):
        assert fragment in source
    assert "--post-cp-sft" not in source
    assert "--pre-cp-sft" not in source


def test_spooled_array_resolves_repo_and_dry_runs_three_seeds(tmp_path):
    result = run_submit(tmp_path, "--datasets", "flowers102", "--dry-run")
    assert result.returncode == 0, result.stderr
    manifest = next((tmp_path / "outputs/vitl_completion_manifests").glob("*.json"))
    spool = tmp_path / "slurm_script"
    spool.write_text(ARRAY.read_text())
    env = dict(os.environ, PYTHON=sys.executable, SLURM_JOB_ID="42",
               SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID="0",
               VITL_COMPLETION_MANIFEST=str(manifest),
               VITL_COMPLETION_CACHE_DIR="/shared/cache")
    result = subprocess.run(["bash", str(spool), "--dry-run"], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE source=/shared/cache/stable_datasets/processed/flowers102" in result.stdout
    commands = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert [c["seed"] for c in commands] == [42, 43, 44]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_array_stages_and_cleans_private_cache_even_after_failure(tmp_path, exit_code):
    cache = tmp_path / "shared"
    source = cache / "stable_datasets/processed/example/data"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"sample")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "runner.txt"
    fake_python = bindir / "python"
    fake_python.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "if [ \"${1:-}\" = - ]; then echo example/data; exit 0; fi\n"
        "if [ \"${1:-}\" = -c ]; then exit 0; fi\n"
        "cache=''\nwhile [ $# -gt 0 ]; do\n"
        "if [ \"$1\" = --cache-dir ]; then cache=$2; shift 2; else shift; fi\ndone\n"
        "test -f \"${cache}/stable_datasets/processed/example/data/sample.bin\"\n"
        "printf '%s\\n' \"${cache}\" > \"${RUNNER_MARKER}\"\n"
        f"exit {exit_code}\n")
    fake_python.chmod(0o755)
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 0\n")
    smi.chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
               PYTHON=str(fake_python), RUNNER_MARKER=str(marker), TMPDIR=str(stage),
               VITL_COMPLETION_SKIP_ENV_SETUP="1", VITL_COMPLETION_REPO_ROOT=str(ROOT),
               VITL_COMPLETION_MANIFEST=str(tmp_path / "unused.json"),
               SLURM_ARRAY_TASK_ID="0", VITL_COMPLETION_CACHE_DIR=str(cache))
    result = subprocess.run(["bash", str(ARRAY)], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == exit_code, result.stdout + result.stderr
    private = Path(marker.read_text().strip())
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"sample"
