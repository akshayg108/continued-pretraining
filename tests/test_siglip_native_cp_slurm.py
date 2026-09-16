"""CPU-only checks of submission, spooled scripts, and dataset staging."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from eval.full_ft import run as artifacts
from eval import siglip_native_cp as cp


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "run/slurm/cp-siglip/native-norm/submit.sh"
ARRAY = SUBMIT.with_name("array.sh")


def run_submit(tmp_path, *args, env=None):
    assert SUBMIT.is_file()
    merged = dict(os.environ, SIGLIP_NATIVE_OUTPUT_BASE=str(tmp_path / "outputs"),
                  SIGLIP_NATIVE_LOG_DIR=str(tmp_path / "logs"))
    merged.update(env or {})
    return subprocess.run(["bash", str(SUBMIT), *args], cwd=ROOT, env=merged,
                          capture_output=True, text=True)


def test_full_dry_run_covers_all_jobs_without_gpu_or_ft(tmp_path):
    result = run_submit(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "63 jobs, 135 CP fits, 0 FT fits" in result.stdout
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 135
    assert len({r["task_id"] for r in rows}) == 63
    fits = set()
    for row in rows:
        command = row["command"]
        fits.add((command[command.index("--cp-method") + 1],
                  command[command.index("--dataset") + 1], row["seed"]))
        assert command[command.index("--normalization-mode") + 1] == "pretrained"
        assert "--skip-baseline" in command
        assert "--resume" not in command and "--post-cp-sft" not in command
        assert row["stage"] == "cp"
    assert len(fits) == 135
    assert not (tmp_path / "outputs" / cp.PROTOCOL).exists()
    assert "full_ft/run.py" not in result.stdout
    submissions = [shlex.split(line)[1:] for line in result.stdout.splitlines() if line.startswith("SUBMIT ")]
    assert len(submissions) == 3
    caps, assigned = [], set()
    for command, (gpu, count) in zip(submissions, [("v100", 24), ("a100", 30), ("a100_80gb", 9)]):
        gpu_type = "v100" if gpu == "v100" else "a100"
        assert f"--gres=gpu:{gpu_type}:1" in command
        assert ("--constraint=80g" in command) == (gpu == "a100_80gb")
        selection = next(a.split("=", 1)[1] for a in command if a.startswith("--array="))
        indices, cap = selection.split("%")
        ids = set(map(int, indices.split(",")))
        assert len(ids) == count and not ids & assigned
        assigned.update(ids)
        caps.append(int(cap))
    assert assigned == set(range(63)) and sum(caps) == 12


def test_filtered_submit_splits_gpu_profiles_without_leaking_80gb_constraint(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "open(os.environ['CALLS'], 'a').write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('1234')\n")
    sbatch.chmod(0o755)
    result = run_submit(tmp_path, "--datasets", "food101", "dtd", "--methods", "LeJEPA",
                        "--concurrency", "4", env={"PATH": f"{bindir}:{os.environ['PATH']}",
                                                   "CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 jobs, 6 CP fits, 0 FT fits" in result.stdout
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2
    v100, a100_80gb = commands
    assert "--array=3%1" in v100 and "--gres=gpu:v100:1" in v100
    assert not any(a.startswith("--constraint=") for a in v100)
    assert "--array=0,1,2%3" in a100_80gb and "--constraint=80g" in a100_80gb
    assert "--gres=gpu:a100:1" in a100_80gb
    for command in commands:
        for flag in ("--time=96:00:00", "--mem=96G", "--cpus-per-task=8"):
            assert flag in command
        assert command[-1] == str(ARRAY)


def test_concurrency_one_chains_arrays_afterany_even_when_a_seed_fails(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "path = Path(os.environ['CALLS'])\n"
        "with path.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(str(2000 + len(path.read_text().splitlines()))+';hpc')\n")
    sbatch.chmod(0o755)
    result = run_submit(tmp_path, "--concurrency", "1", env={
        "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 3
    assert not any(a.startswith("--dependency=") for a in commands[0])
    assert "--dependency=afterany:2001" in commands[1]
    assert "--dependency=afterany:2002" in commands[2]
    assert all(next(a for a in c if a.startswith("--array=")).endswith("%1") for c in commands)


@pytest.mark.parametrize("args", [("--concurrency", "0"), ("--concurrency", "13"),
                                  ("--concurrency", "no"), ("--resume",),
                                  ("--methods",), ("--datasets",),
                                  ("--methods", "MAE"), ("--datasets", "invalid")])
def test_invalid_submit_arguments_fail_before_sbatch(tmp_path, args):
    result = run_submit(tmp_path, *args)
    assert result.returncode != 0
    assert "SUBMIT" not in result.stdout


def test_spooled_array_keeps_single_seed_big_jobs(tmp_path):
    manifest = tmp_path / "manifest.json"
    artifacts.atomic_json(manifest, cp.build_manifest(tmp_path / "outputs", datasets=["food101"], methods=["SimCLR"]))
    spool = tmp_path / "slurm_script"
    spool.write_text(ARRAY.read_text())
    env = dict(os.environ, SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=str(ROOT),
               SLURM_ARRAY_TASK_ID="1", SIGLIP_NATIVE_MANIFEST=str(manifest),
               SIGLIP_NATIVE_CACHE_DIR="/shared/cache")
    result = subprocess.run(["bash", str(spool), "--dry-run"], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["seed"] == 43
    command = rows[0]["command"]
    assert command[command.index("--dataset") + 1] == "food101"
    assert not (tmp_path / "outputs" / cp.PROTOCOL).exists()


def test_array_stages_and_cleans_private_data_copy(tmp_path):
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
               SIGLIP_NATIVE_SKIP_ENV_SETUP="1", SIGLIP_NATIVE_REPO_ROOT=str(ROOT),
               SIGLIP_NATIVE_MANIFEST=str(manifest), SLURM_ARRAY_TASK_ID="2",
               SIGLIP_NATIVE_CACHE_DIR=str(tmp_path / "shared"))
    result = subprocess.run(["bash", str(ARRAY)], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    private = Path(marker.read_text().strip())
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"sample"


def test_all_complete_means_no_submission(tmp_path):
    doc = cp.build_manifest(tmp_path / "outputs", datasets=["dtd"], methods=["DIET"])
    task = doc["tasks"][0]
    for seed in task["seeds"]:
        paths = cp.result_paths(doc, task, seed)
        checkpoint = (Path(doc["output_root"]) / "attempts/DIET/dtd" / f"seed{seed}/test/cp.ckpt")
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"native")
        row = dict(dataset="dtd", n_samples=1880, backbone=task["model_id"], method="diet",
                   seed=seed, epochs=150, random_init=False, no_cp=False,
                   normalization_mode="pretrained", normalization=cp.NORMALIZATION,
                   cp_config=dict(task["cp_recipe"], pool_strategy="map", skip_baseline=True,
                                  skip_final_eval=False, resume=False, pre_cp_sft=False, post_cp_sft=False),
                   post_knn_f1=.6, post_linear_f1=.7, post_knn_acc=.6, post_linear_acc=.7)
        artifacts.atomic_json(paths["cp_result"], row)
        artifacts.atomic_json(paths["receipt"], dict(cp._identity(doc, task, seed), status="cp_complete",
                             checkpoint_path=str(checkpoint), checkpoint_sha256=artifacts.file_sha256(checkpoint),
                             cp_result_sha256=artifacts.file_sha256(paths["cp_result"])))
    result = run_submit(tmp_path, "--datasets", "dtd", "--methods", "DIET")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nothing to submit" in result.stdout
    assert not any(line.startswith("SUBMIT ") for line in result.stdout.splitlines())


def test_shell_syntax_and_resources():
    for path in (SUBMIT, ARRAY):
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    text = ARRAY.read_text()
    assert "#SBATCH --gres=" not in text and "#SBATCH --constraint=" not in text
    assert "require_gpu(tasks[task_id][\"gpu_profile\"])" in text
    for fragment in ("#SBATCH --time=96:00:00", "conda activate env", "mktemp -d",
                     "trap 'rm -rf --", "du -sk", "df -Pk", "nvidia-smi"):
        assert fragment in text
