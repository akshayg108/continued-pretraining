"""Shell-contract tests for the held-out continued-pretraining launcher."""

import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "run/slurm/heldout-cp"
SUBMIT = DIRECTORY / "submit.sh"
PREPARE = DIRECTORY / "prepare.sh"
ARRAY = DIRECTORY / "array.sh"


def fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "heldout-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:3] == ['-m', 'eval.heldout_cp', 'plan']:\n"
        "    expected = os.environ.get('EXPECTED_PLAN_CWD')\n"
        "    if expected and (os.getcwd() != expected or os.environ.get('PYTHONPATH', '').split(':')[0] != expected):\n"
        "        raise SystemExit('plan must run from repo with repo first on PYTHONPATH')\n"
        "    target = pathlib.Path(args[args.index('--manifest') + 1])\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text(json.dumps({'cp_jobs': 48, 'fits': 144}))\n"
        "print(json.dumps({'argv': args, 'executable': sys.executable}))\n"
    )
    executable.chmod(0o755)
    return executable


def run_submit(tmp_path: Path, *args: str, extra_env=None, cwd=ROOT):
    python = fake_python(tmp_path)
    env = dict(
        os.environ,
        HELDOUT_PYTHON=str(python),
        HELDOUT_REPO_ROOT=str(ROOT),
        HELDOUT_CACHE_DIR=str(tmp_path / "cache"),
        HELDOUT_OUTPUT_BASE=str(tmp_path / "outputs"),
        HELDOUT_LOG_DIR=str(tmp_path / "logs"),
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(SUBMIT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, python


def fake_scontrol(tmp_path, bindir):
    calls = tmp_path / "scontrol.jsonl"
    executable = bindir / "scontrol"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"with open({str(calls)!r}, 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_DEPENDENCY_TASK') in sys.argv[1:]:\n"
        "    raise SystemExit(1)\n"
    )
    executable.chmod(0o755)
    return calls


def test_entrypoints_exist_and_have_valid_shell_syntax():
    for script in (SUBMIT, PREPARE, ARRAY):
        assert script.is_file()
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_dry_run_creates_manifest_and_prints_both_arrays_without_sbatch(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "sbatch-called"
    sbatch = bindir / "sbatch"
    sbatch.write_text(f"#!/bin/bash\ntouch {shlex.quote(str(marker))}\nexit 99\n")
    sbatch.chmod(0o755)

    result, python = run_submit(
        tmp_path,
        "--concurrency",
        "7",
        "--dry-run",
        extra_env={"PATH": f"{bindir}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    manifests = list((tmp_path / "outputs" / "heldout_cp_manifests").glob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text()) == {"cp_jobs": 48, "fits": 144}
    assert not (manifests[0].stat().st_mode & stat.S_IWUSR)
    lines = [
        shlex.split(line.removeprefix("SUBMIT "))
        for line in result.stdout.splitlines()
        if line.startswith("SUBMIT ")
    ]
    assert len(lines) == 2
    assert "--array=0-7%7" in lines[0]
    assert "--array=0-47%7" in lines[1]
    assert "--hold" in lines[1]
    assert not any(a.startswith("--dependency=") for a in lines[1])
    controls = [
        shlex.split(line.removeprefix("CONTROL "))
        for line in result.stdout.splitlines()
        if line.startswith("CONTROL ")
    ]
    assert len(controls) == 49
    assert controls[6] == [
        "scontrol",
        "update",
        "JobId=DRY_RUN_CP_6",
        "Dependency=afterok:DRY_RUN_PREP_2",
    ]
    assert controls[-1] == ["scontrol", "release", "DRY_RUN_CP"]
    exports = [next(a for a in line if a.startswith("--export=")) for line in lines]
    for export in exports:
        assert f"HELDOUT_PYTHON={python}" in export
        assert f"HELDOUT_MANIFEST={manifests[0]}" in export
        assert f"HELDOUT_REPO_ROOT={ROOT}" in export
        assert f"HELDOUT_CACHE_DIR={tmp_path / 'cache'}" in export


def test_submit_uses_v100_resources_and_afterok_dependency(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['SBATCH_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print('81234;greene' if '--array=0-7%12' in sys.argv else '81235;greene')\n"
    )
    sbatch.chmod(0o755)
    controls_path = fake_scontrol(tmp_path, bindir)
    result, _ = run_submit(
        tmp_path,
        extra_env={
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "SBATCH_CALLS": str(calls),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2
    for command, array, script in zip(
        commands, ("--array=0-7%12", "--array=0-47%12"), (PREPARE, ARRAY)
    ):
        for argument in (
            array,
            "--partition=nvidia",
            "--qos=nvidia",
            "--account=civil",
            "--gres=gpu:v100:1",
            "--cpus-per-task=8",
            "--mem=96G",
            "--time=96:00:00",
            f"--chdir={ROOT}",
        ):
            assert argument in command
        assert command[-1] == str(script)
    assert "--hold" in commands[1]
    controls = [json.loads(line) for line in controls_path.read_text().splitlines()]
    assert len(controls) == 49
    from eval.heldout_cp.protocol import build_manifest, DATASETS

    for task, command in zip(build_manifest(tmp_path)["tasks"], controls[:-1]):
        dataset_id = DATASETS.index(task["dataset"])
        assert command == [
            "update",
            f"JobId=81235_{task['task_id']}",
            f"Dependency=afterok:81234_{dataset_id}",
        ]
    assert controls[-1] == ["release", "81235"]
    assert "Submitted preparation job 81234" in result.stdout


def test_submit_plans_from_repo_with_repo_first_on_pythonpath(tmp_path):
    result, _ = run_submit(
        tmp_path,
        "--dry-run",
        cwd=tmp_path,
        extra_env={"EXPECTED_PLAN_CWD": str(ROOT)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cp_submission_failure_still_reports_preparation_job_id(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "if [[ \" $* \" == *' --array=0-7%12 '* ]]; then echo '91234;greene'; exit 0; fi\n"
        "echo 'CP submission failed' >&2\n"
        "exit 1\n"
    )
    sbatch.chmod(0o755)
    fake_scontrol(tmp_path, bindir)
    result, _ = run_submit(
        tmp_path,
        extra_env={"PATH": f"{bindir}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0
    assert "Submitted preparation job 91234" in result.stdout


def test_dependency_update_failure_never_releases_cp_array(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\nif [[ \" $* \" == *' --array=0-7%12 '* ]]; then echo 81234; else echo 81235; fi\n"
    )
    sbatch.chmod(0o755)
    controls_path = fake_scontrol(tmp_path, bindir)
    result, _ = run_submit(
        tmp_path,
        extra_env={
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "FAIL_DEPENDENCY_TASK": "JobId=81235_6",
        },
    )
    assert result.returncode != 0
    controls = [json.loads(line) for line in controls_path.read_text().splitlines()]
    assert controls[-1][1] == "JobId=81235_6"
    assert not any(c[0] == "release" for c in controls)
    assert "remains held" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--concurrency", "0"),
        ("--concurrency", "13"),
        ("--concurrency", "abc"),
        ("--resume",),
        ("--dry-run", "extra"),
    ],
)
def test_submit_rejects_invalid_arguments_before_plan_or_sbatch(tmp_path, args):
    result, _ = run_submit(tmp_path, *args)
    assert result.returncode == 2
    assert "SUBMIT" not in result.stdout
    assert not (tmp_path / "outputs" / "heldout_cp_manifests").exists()


@pytest.mark.parametrize(
    "script,task_id,verb,id_flag",
    [
        (PREPARE, "3", "prepare", "--dataset-id"),
        (ARRAY, "19", "run", "--task-id"),
    ],
)
def test_worker_dry_run_uses_pinned_python_and_fixed_module_cli(
    tmp_path, script, task_id, verb, id_flag
):
    python = fake_python(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    env = dict(
        os.environ,
        HELDOUT_PYTHON=str(python),
        HELDOUT_REPO_ROOT=str(ROOT),
        HELDOUT_CACHE_DIR=str(tmp_path / "shared-cache"),
        HELDOUT_MANIFEST=str(manifest),
        SLURM_ARRAY_TASK_ID=task_id,
        SLURM_CPUS_PER_TASK="8",
    )
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"python={python}" in result.stdout
    row = json.loads(result.stdout.splitlines()[-1])
    argv = row["argv"]
    assert argv[:3] == ["-m", "eval.heldout_cp", verb]
    assert argv[argv.index(id_flag) + 1] == task_id
    assert argv[argv.index("--manifest") + 1] == str(manifest)
    assert argv[argv.index("--cache-dir") + 1] == str(tmp_path / "shared-cache")
    assert argv[argv.index("--num-workers") + 1] == "8"
    assert argv[-1] == "--dry-run"


def test_workers_pin_threads_pythonpath_and_avoid_environment_or_cache_staging():
    for script in (PREPARE, ARRAY):
        text = script.read_text()
        for fragment in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONUNBUFFERED",
            'PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        ):
            assert fragment in text
        for forbidden in (
            "conda activate",
            "conda run",
            "module load",
            "rsync",
            "mktemp",
            "stable_datasets/processed",
        ):
            assert forbidden not in text


def test_worker_rejects_non_absolute_or_non_executable_python(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    base_env = dict(
        os.environ,
        HELDOUT_REPO_ROOT=str(ROOT),
        HELDOUT_CACHE_DIR=str(tmp_path / "cache"),
        HELDOUT_MANIFEST=str(manifest),
        SLURM_ARRAY_TASK_ID="0",
    )
    for python in ("python3", str(tmp_path / "missing-python")):
        result = subprocess.run(
            ["bash", str(PREPARE), "--dry-run"],
            cwd=ROOT,
            env=dict(base_env, HELDOUT_PYTHON=python),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "absolute executable" in result.stderr
