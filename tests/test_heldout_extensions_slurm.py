"""Slurm grouping and dependency safety for encoder extensions."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_heldout_extensions import source_manifest, make_doc
from eval.heldout_extensions import protocol as p

DIRECTORY = p.ROOT / "run/slurm/heldout-extensions"


def launcher():
    assert (p.ROOT / "eval/heldout_extensions/submit.py").is_file(), "Extension submitter is missing"
    return importlib.import_module("eval.heldout_extensions.submit")


def test_resource_groups_and_encoder_target_dependencies(tmp_path, source_manifest, monkeypatch):
    launch = launcher()
    doc = make_doc(tmp_path, source_manifest)
    commands = []
    numbers = iter(("100;cluster", "101", "102", "103"))
    def execute(command, **kwargs):
        commands.append(command)
        return SimpleResult(next(numbers) if command[0] == "sbatch" else "")
    monkeypatch.setattr(launch.subprocess, "run", execute)
    launch.submit(doc, tmp_path / "manifest.json", cache_dir=tmp_path / "cache",
                  log_dir=tmp_path / "logs", python=sys.executable, concurrency=12, dry_run=False)
    batch = [c for c in commands if c[0] == "sbatch"]
    assert len(batch) == 4
    assert sum("--gres=gpu:a100:1" in c for c in batch) == 2
    assert sum("--gres=gpu:v100:1" in c for c in batch) == 2
    assert all(not any("constraint" in a or "80g" in a for a in c) for c in batch)
    jobs, preps, tasks = {}, {}, {}
    for job, command in zip((100, 101, 102, 103), batch):
        array = next(a for a in command if a.startswith("--array=")).split("=", 1)[1]
        ids, throttle = array.split("%")
        assert throttle == "12"
        indices = [int(i) for i in ids.split(",")]
        is_prep = command[-1] == "prepare"
        for index in indices:
            row = doc["preparations" if is_prep else "tasks"][index]
            assert f"--gres=gpu:{row['gpu']}:1" in command
            (preps if is_prep else tasks)[index] = job
        if not is_prep:
            assert "--hold" in command
        jobs[job] = command
    assert len(preps) == 16 and len(tasks) == 48
    controls = [c for c in commands if c[:2] == ["scontrol", "update"]]
    assert len(controls) == 48
    for task, command in zip(doc["tasks"], controls):
        index = task["task_id"]
        prep_id = task["preparation_id"]
        assert command == ["scontrol", "update", f"JobId={tasks[index]}_{index}",
                           f"Dependency=afterok:{preps[prep_id]}_{prep_id}"]
    releases = [c for c in commands if c[:2] == ["scontrol", "release"]]
    assert len(releases) == 2
    assert commands.index(releases[0]) > commands.index(controls[-1])


class SimpleResult:
    def __init__(self, stdout):
        self.stdout = stdout


def test_dependency_failure_keeps_both_cp_arrays_held(tmp_path, source_manifest, monkeypatch):
    launch = launcher()
    doc = make_doc(tmp_path, source_manifest)
    commands = []
    numbers = iter(("100", "101", "102", "103"))
    def execute(command, **kwargs):
        commands.append(command)
        if command[:2] == ["scontrol", "update"]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleResult(next(numbers))
    monkeypatch.setattr(launch.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        launch.submit(doc, tmp_path / "manifest.json", cache_dir=tmp_path,
                      log_dir=tmp_path / "logs", python=sys.executable, concurrency=12, dry_run=False)
    assert not any(c[:2] == ["scontrol", "release"] for c in commands)


def test_dry_run_never_invokes_scheduler(tmp_path, source_manifest, monkeypatch, capsys):
    launch = launcher()
    doc = make_doc(tmp_path, source_manifest)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("scheduler called"))
    launch.submit(doc, tmp_path / "manifest.json", cache_dir=tmp_path,
                  log_dir=tmp_path / "logs", python=sys.executable, concurrency=12, dry_run=True)
    output = capsys.readouterr().out
    assert output.count("SUBMIT sbatch ") == 4
    assert output.count("CONTROL scontrol update ") == 48


def test_workers_and_submit_have_valid_shell_syntax():
    for filename in ("submit.sh", "worker.sh"):
        path = DIRECTORY / filename
        assert path.is_file(), f"Missing {filename}"
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage,index", [("prepare", 8), ("run", 0)])
def test_worker_uses_extension_namespace_and_python(tmp_path, source_manifest, stage, index):
    path = DIRECTORY / "worker.sh"
    assert path.is_file(), "Extension worker is missing"
    manifest = tmp_path / "manifest.json"
    p.atomic_json(manifest, make_doc(tmp_path, source_manifest))
    env = dict(os.environ, HELDOUT_PYTHON=sys.executable, HELDOUT_REPO_ROOT=str(p.ROOT),
               HELDOUT_EXT_MANIFEST=str(manifest), SLURM_ARRAY_TASK_ID=str(index),
               HELDOUT_CACHE_DIR=str(tmp_path / "cache"), SLURM_CPUS_PER_TASK="8")
    result = subprocess.run(["bash", str(path), stage, "--dry-run"], env=env, cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "gpu=a100" in result.stdout
    if stage == "run":
        assert "pool=map" in result.stdout and "seed=44" in result.stdout
