"""A100 retries must preserve the frozen recipe, baselines, and other jobs."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from eval import heldout_metric_roundoff as numerics
from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/slurm/heldout-cp/lejepa_a100.sh"


def recovery():
    assert (ROOT / "eval/heldout_a100_retry.py").is_file(), "A100 recovery is missing"
    return importlib.import_module("eval.heldout_a100_retry")


def fake_environment(monkeypatch, *, name="NVIDIA A100-PCIE-40GB", memory_gib=40,
                     available=True, count=1, missing_reader=None):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        get_device_name=lambda index: name,
        get_device_properties=lambda index: SimpleNamespace(total_memory=memory_gib * 2**30),
    )
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True),
    )
    torch.precision_calls = []
    torch.set_float32_matmul_precision = torch.precision_calls.append
    datasets = ModuleType("stable_datasets")
    datasets.images = SimpleNamespace(**{
        name: object() for name in (
            "MedMNIST", "AID", "RESISC45", "StanfordDogs", "JenaFlowers30", "Flavia", "IP102"
        ) if name != missing_reader
    })
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "stable_datasets", datasets)
    return torch


def prepared(tmp_path):
    doc = p.build_manifest(tmp_path)
    manifest = tmp_path / "manifest.json"
    p.atomic_json(manifest, doc)
    for dataset in p.DATASETS:
        for encoder in p.ENCODER_ORDER:
            p.atomic_json(p.geometry_path(doc, encoder, dataset), dict(
                p.identity(doc, encoder, dataset), status="complete",
                n_geometry=1000, uniformity_t2=-1.0,
            ))
            for seed in p.SEEDS:
                row = dict(
                    p.identity(doc, encoder, dataset, seed), status="complete",
                    train_indices=list(range(1000)), data={"n_train_actual": 1000},
                    **{f"pre_{metric}": 0.5 for metric in p.METRICS},
                )
                if dataset == "jena_flowers30":
                    row["evaluation_numerics"] = dict(
                        policy=numerics.POLICY,
                        wrapper_sha256=p.file_sha256(Path(numerics.__file__)),
                        base_implementation_sha256=doc["implementation_sha256"],
                    )
                p.atomic_json(p.pre_path(doc, encoder, dataset, seed), row)
        p.freeze_predictions(doc, dataset)
    return doc, manifest


def test_original_runtime_still_rejects_a100(monkeypatch):
    fake_environment(monkeypatch)
    with pytest.raises(RuntimeError, match="Expected a V100"):
        runtime.check_environment()


@pytest.mark.parametrize("name,memory_gib", [
    ("NVIDIA A100-PCIE-40GB", 40), ("NVIDIA A100-SXM4-80GB", 80),
])
def test_retry_accepts_both_a100_capacities_and_preserves_precision(
    tmp_path, monkeypatch, name, memory_gib
):
    module = recovery()
    torch = fake_environment(monkeypatch, name=name, memory_gib=memory_gib)
    doc = p.build_manifest(tmp_path)
    original_check = runtime.check_environment
    with module.a100_allocation(doc, doc["tasks"][21], 42):
        assert runtime.check_environment() == name
        assert torch.precision_calls == ["highest"]
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    assert runtime.check_environment is original_check


@pytest.mark.parametrize("kwargs,match", [
    ({"name": "Tesla V100-PCIE-32GB"}, "A100"),
    ({"name": "NVIDIA H100"}, "A100"),
    ({"available": False}, "exactly one"),
    ({"count": 2}, "exactly one"),
    ({"missing_reader": "IP102"}, "IP102"),
])
def test_retry_does_not_bypass_other_environment_checks(tmp_path, monkeypatch, kwargs, match):
    module = recovery()
    fake_environment(monkeypatch, **kwargs)
    doc = p.build_manifest(tmp_path)
    original_check, original_write = runtime.check_environment, p.atomic_json
    with pytest.raises(RuntimeError, match=match):
        with module.a100_allocation(doc, doc["tasks"][21], 42):
            runtime.check_environment()
    assert runtime.check_environment is original_check
    assert p.atomic_json is original_write


def test_hardware_and_roundoff_audits_compose_without_changing_baselines(tmp_path, monkeypatch):
    module = recovery()
    doc, manifest = prepared(tmp_path)
    task, seed = doc["tasks"][15], 42
    snapshots = {path: path.read_bytes() for path in p.root_path(doc).rglob("*.json")}
    manifest_bytes = manifest.read_bytes()
    raw = dict.fromkeys(p.METRICS, 1.0000001192092896)
    fake_environment(monkeypatch)
    monkeypatch.setattr(runtime, "evaluate", lambda *args: raw)
    original_check, original_write, original_evaluate = (
        runtime.check_environment, p.atomic_json, runtime.evaluate
    )
    with module.a100_allocation(doc, task, seed):
        gpu = runtime.check_environment()
        baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
        record = dict(
            baseline, method=task["method"], recipe=task["recipe"], gpu=gpu,
            no_ft=True, initialization="public_pretrained",
            pre_sha256=p.file_sha256(p.pre_path(doc, task["encoder"], task["dataset"], seed)),
            predictions_sha256=p.file_sha256(p.predictions_path(doc, task["dataset"])),
        )
        attempt = tmp_path / "attempt.json"
        p.atomic_json(attempt, dict(record, status="running"))
        with numerics.roundoff_evaluation(doc, "post"):
            args = runtime.evaluation_args(task["encoder"], task["dataset"], seed, tmp_path, 0)
            scores = runtime.evaluate(None, None, None, args, baseline["train_indices"])
            row = dict(record, **{f"post_{key}": value for key, value in scores.items()})
            p.validate_result(doc, task, seed, row)
            target = p.result_path(doc, task, seed)
            p.atomic_json(target, row)
    saved = json.loads(target.read_text())
    audit = saved["resource_override"]
    assert audit["policy"] == module.POLICY
    assert audit["planned_gpu"] == "v100"
    assert audit["requested_gpu"] == "a100"
    assert audit["actual_gpu"] == saved["gpu"] == "NVIDIA A100-PCIE-40GB"
    assert audit["gpu_total_memory_bytes"] == 40 * 2**30
    assert audit["wrapper_sha256"] == p.file_sha256(Path(module.__file__))
    assert audit["base_implementation_sha256"] == doc["implementation_sha256"]
    assert json.loads(attempt.read_text())["resource_override"] == audit
    assert saved["evaluation_numerics"]["raw_scores"] == raw
    assert all(saved[f"post_{metric}"] == 1.0 for metric in p.METRICS)
    assert "resource_override" not in record
    assert runtime.check_environment is original_check
    assert runtime.evaluate is original_evaluate
    assert p.atomic_json is original_write
    assert manifest.read_bytes() == manifest_bytes
    assert all(path.read_bytes() == content for path, content in snapshots.items())
    assert p.implementation_sha256() == doc["implementation_sha256"]
    from eval.heldout_cp.collect import collect
    assert collect(doc, tmp_path / "report") == dict(
        expected=144, verified=1, missing=143, invalid=0
    )


@pytest.mark.parametrize("task_id", [1, 2, 16, 17, 40, 41])
def test_non_lejepa_tasks_cannot_use_retry(tmp_path, task_id):
    module = recovery()
    doc = p.build_manifest(tmp_path)
    with pytest.raises(ValueError, match="LeJEPA"):
        with module.a100_allocation(doc, doc["tasks"][task_id], 42):
            pytest.fail("Other methods must not enter the A100 recovery context")


def test_preflight_checks_all_prepared_datasets_without_changing_predictions(tmp_path, capsys):
    module = recovery()
    doc, manifest = prepared(tmp_path)
    snapshots = {path: path.read_bytes() for path in p.root_path(doc).rglob("*.json")}
    module.main(["preflight", "--manifest", str(manifest)])
    assert "16 LeJEPA jobs" in capsys.readouterr().out
    assert all(path.read_bytes() == content for path, content in snapshots.items())
    p.predictions_path(doc, "jena_flowers30").unlink()
    with pytest.raises(ValueError, match="Missing frozen predictions"):
        module.main(["preflight", "--manifest", str(manifest)])
    assert not p.predictions_path(doc, "jena_flowers30").exists()


def test_preflight_rejects_changed_numerical_wrapper(tmp_path):
    module = recovery()
    doc, manifest = prepared(tmp_path)
    path = p.pre_path(doc, "DINOv3", "jena_flowers30", 42)
    row = json.loads(path.read_text())
    row["evaluation_numerics"]["wrapper_sha256"] = "changed"
    p.atomic_json(path, row)
    with pytest.raises(ValueError):
        module.main(["preflight", "--manifest", str(manifest)])


def test_cpu_dry_run_keeps_manifest_and_reports_actual_requested_gpu(tmp_path):
    recovery()
    manifest = tmp_path / "manifest.json"
    doc = p.build_manifest(tmp_path)
    p.atomic_json(manifest, doc)
    command = [sys.executable, "-m", "eval.heldout_a100_retry", "fit",
               "--manifest", str(manifest), "--task-id", "21", "--seed", "42", "--dry-run"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gpu=a100" in result.stdout
    assert "LeJEPA ip102" in result.stdout
    assert p.load_manifest(manifest) == doc
    command[command.index("--task-id") + 1] = "22"
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "LeJEPA" in result.stderr


def test_fit_delegates_to_existing_roundoff_recovery(tmp_path, monkeypatch):
    module = recovery()
    doc, manifest = prepared(tmp_path)
    calls = []
    fake_environment(monkeypatch)

    def fit(argv):
        assert runtime.check_environment() == "NVIDIA A100-PCIE-40GB"
        calls.append(argv)

    monkeypatch.setattr(numerics, "main", fit)
    args = ["fit", "--manifest", str(manifest), "--task-id", "15", "--seed", "43",
            "--num-workers", "2", "--cache-dir", str(tmp_path)]
    module.main(args)
    assert calls == [args]
    assert p.load_manifest(manifest) == doc


@pytest.mark.parametrize("fail_first", [False, True])
def test_shell_runs_only_requested_task_and_serial_seeds(tmp_path, fail_first):
    assert SCRIPT.is_file(), "A100 batch script is missing"
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:a100:1" in source
    assert "--constraint" not in source
    assert "scancel" not in source
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    python, calls = tmp_path / "python", tmp_path / "calls.jsonl"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_FIRST') == '1':\n"
        "    raise SystemExit(7)\n"
    )
    python.chmod(0o755)
    result = subprocess.run(["bash", str(SCRIPT)], env=dict(
        os.environ, HELDOUT_PYTHON=str(python), HELDOUT_REPO_ROOT=str(ROOT),
        HELDOUT_MANIFEST=str(tmp_path / "manifest.json"), SLURM_ARRAY_TASK_ID="21",
        CALLS=str(calls), FAIL_FIRST=str(int(fail_first)),
    ), capture_output=True, text=True)
    assert result.returncode == (7 if fail_first else 0), result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == (1 if fail_first else 3)
    for seed, command in zip(p.SEEDS, commands):
        assert command[:3] == ["-m", "eval.heldout_a100_retry", "fit"]
        assert command[command.index("--task-id") + 1] == "21"
        assert command[command.index("--seed") + 1] == str(seed)
