"""Numerical recovery must not change the frozen CP implementation or artifacts."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/slurm/heldout-cp/roundoff.sh"
SERVER_PERFECT_F1 = 1.0000001192092896


def recovery():
    assert (ROOT / "eval/heldout_metric_roundoff.py").is_file()
    return importlib.import_module("eval.heldout_metric_roundoff")


@pytest.mark.parametrize("metric", p.METRICS)
def test_reproduced_float32_overshoot_is_clipped_without_mutating_input(metric):
    module = recovery()
    raw = dict.fromkeys(p.METRICS, 0.5)
    raw[metric] = SERVER_PERFECT_F1
    with pytest.raises(ValueError):
        p.check_metrics({f"pre_{key}": value for key, value in raw.items()}, "pre")
    scores = module.canonical_scores(raw)
    assert raw[metric] == SERVER_PERFECT_F1
    assert scores[metric] == 1.0
    assert all(scores[key] == raw[key] for key in p.METRICS if key != metric)
    p.check_metrics({f"pre_{key}": value for key, value in scores.items()}, "pre")


@pytest.mark.parametrize(
    "value", [None, True, "1.0", float("nan"), float("inf"), -0.001, 1.0001]
)
def test_real_invalid_scores_remain_errors(value):
    module = recovery()
    raw = dict.fromkeys(p.METRICS, 0.5)
    raw["knn_f1"] = value
    with pytest.raises(ValueError, match="knn_f1"):
        module.canonical_scores(raw)


def test_missing_metric_is_not_silently_filled():
    with pytest.raises(ValueError, match="linear_acc"):
        recovery().canonical_scores(dict(knn_f1=0.5, knn_acc=0.5, linear_f1=0.5))


@pytest.mark.parametrize(
    "value, expected", [(0.0, 0.0), (1.0, 1.0), (0.9999999, 0.9999999), (-1e-7, 0.0)]
)
def test_only_out_of_range_values_are_changed(value, expected):
    scores = recovery().canonical_scores(dict.fromkeys(p.METRICS, value))
    assert scores == dict.fromkeys(p.METRICS, expected)


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_recovery_preserves_manifest_and_records_raw_metrics(tmp_path, monkeypatch, phase):
    module = recovery()
    doc = p.build_manifest(tmp_path)
    manifest = tmp_path / "manifest.json"
    p.atomic_json(manifest, doc)
    original_manifest = manifest.read_bytes()
    original_check = p.check_metrics
    original_writer = p.atomic_json
    raw = dict.fromkeys(p.METRICS, 0.5)
    raw["knn_f1"] = SERVER_PERFECT_F1
    calls = []

    def evaluate(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(runtime, "evaluate", evaluate)
    args = runtime.evaluation_args("DINOv3", "jena_flowers30", 42, tmp_path, 0)
    target = tmp_path / "scores.json"
    with module.roundoff_evaluation(doc, phase):
        scores = runtime.evaluate("model", "device", {}, args, list(range(1000)))
        assert scores["knn_f1"] == 1.0
        row = dict(
            p.identity(doc, "DINOv3", "jena_flowers30", 42),
            status="complete",
            **{f"{phase}_{key}": value for key, value in scores.items()},
        )
        p.check_metrics(row, phase)
        p.atomic_json(target, row)
        assert "evaluation_numerics" not in row
        assert p.load_manifest(manifest) == doc
    saved = json.loads(target.read_text())
    assert saved[f"{phase}_knn_f1"] == 1.0
    audit = saved["evaluation_numerics"]
    assert audit["raw_scores"] == raw
    assert audit["phase"] == phase
    assert audit["wrapper_sha256"] == p.file_sha256(Path(module.__file__))
    assert audit["base_implementation_sha256"] == doc["implementation_sha256"]
    assert len(calls) == 1
    assert runtime.evaluate is evaluate
    assert p.atomic_json is original_writer
    assert p.check_metrics is original_check
    assert manifest.read_bytes() == original_manifest
    assert p.implementation_sha256() == doc["implementation_sha256"]


def test_failed_recovery_restores_runtime_and_does_not_publish(tmp_path, monkeypatch):
    module = recovery()
    original_writer = p.atomic_json
    raw = dict.fromkeys(p.METRICS, 0.5)
    raw["knn_f1"] = float("nan")
    evaluate = lambda *args: raw
    monkeypatch.setattr(runtime, "evaluate", evaluate)
    args = runtime.evaluation_args("DINOv3", "jena_flowers30", 42, tmp_path, 0)
    with pytest.raises(ValueError, match="knn_f1"):
        with module.roundoff_evaluation(p.build_manifest(tmp_path), "pre"):
            runtime.evaluate(None, None, None, args, [])
            p.atomic_json(tmp_path / "should-not-exist.json", raw)
    assert runtime.evaluate is evaluate
    assert p.atomic_json is original_writer
    assert not (tmp_path / "should-not-exist.json").exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("stage,task_id,n_calls", [("prepare", 5, 1), ("run", 15, 3)])
def test_batch_recovery_uses_new_entrypoint_for_all_seeds(
    tmp_path, stage, task_id, n_calls, dry_run
):
    assert SCRIPT.is_file()
    fake_python = tmp_path / "python"
    calls = tmp_path / "calls.jsonl"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    env = dict(
        os.environ,
        HELDOUT_PYTHON=str(fake_python),
        HELDOUT_REPO_ROOT=str(ROOT),
        HELDOUT_MANIFEST=str(tmp_path / "manifest.json"),
        SLURM_ARRAY_TASK_ID=str(task_id),
        CALLS=str(calls),
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), stage, *(["--dry-run"] if dry_run else [])],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == n_calls
    for index, command in enumerate(commands):
        assert command[:3] == [
            "-m",
            "eval.heldout_metric_roundoff",
            "prepare" if stage == "prepare" else "fit",
        ]
        assert ("--dry-run" in command) is dry_run
        if stage == "run":
            assert command[command.index("--seed") + 1] == str(42 + index)


def test_cpu_cli_checks_original_manifest_without_loading_gpu_code(tmp_path):
    recovery()
    manifest = tmp_path / "manifest.json"
    doc = p.build_manifest(tmp_path)
    p.atomic_json(manifest, doc)
    command = [
        sys.executable,
        "-m", "eval.heldout_metric_roundoff", "prepare",
        "--manifest", str(manifest),
        "--dataset-id", "5", "--dry-run",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "PREPARE jena_flowers30" in result.stdout
    doc["tasks"][15]["recipe"]["lr"] = 0.01
    p.atomic_json(manifest, doc)
    failed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert failed.returncode != 0
    assert "manifest differs" in failed.stderr


def test_published_recovery_results_work_with_original_collector(tmp_path, monkeypatch):
    from eval.heldout_cp.collect import collect

    module = recovery()
    doc = p.build_manifest(tmp_path)
    dataset = "jena_flowers30"
    monkeypatch.setattr(
        runtime, "evaluate", lambda *args: dict.fromkeys(p.METRICS, SERVER_PERFECT_F1)
    )
    indices = list(range(1000))
    with module.roundoff_evaluation(doc, "pre"):
        for encoder in p.ENCODER_ORDER:
            p.atomic_json(
                p.geometry_path(doc, encoder, dataset),
                dict(
                    p.identity(doc, encoder, dataset),
                    status="complete", n_geometry=1000, uniformity_t2=-1.0,
                ),
            )
            for seed in p.SEEDS:
                args = runtime.evaluation_args(encoder, dataset, seed, tmp_path, 0)
                scores = runtime.evaluate(None, None, None, args, indices)
                row = dict(
                    p.identity(doc, encoder, dataset, seed),
                    status="complete", data={"n_train_actual": 1000}, train_indices=indices,
                    **{f"pre_{key}": value for key, value in scores.items()},
                )
                p.atomic_json(p.pre_path(doc, encoder, dataset, seed), row)
    predictions = p.freeze_predictions(doc, dataset)
    frozen_bytes = predictions.read_bytes()
    with module.roundoff_evaluation(doc, "post"):
        for task in doc["tasks"]:
            if task["dataset"] != dataset:
                continue
            for seed in p.SEEDS:
                baseline = p.validate_pre(doc, task["encoder"], dataset, seed)
                args = runtime.evaluation_args(task["encoder"], dataset, seed, tmp_path, 0)
                scores = runtime.evaluate(None, None, None, args, indices)
                row = dict(
                    baseline, method=task["method"], recipe=task["recipe"], no_ft=True,
                    initialization="public_pretrained",
                    pre_sha256=p.file_sha256(p.pre_path(doc, task["encoder"], dataset, seed)),
                    predictions_sha256=p.file_sha256(predictions),
                    **{f"post_{key}": value for key, value in scores.items()},
                )
                p.validate_result(doc, task, seed, row)
                p.atomic_json(p.result_path(doc, task, seed), row)
    counts = collect(doc, tmp_path / "report")
    assert counts == dict(expected=144, verified=18, missing=126, invalid=0)
    assert predictions.read_bytes() == frozen_bytes
    assert p.implementation_sha256() == doc["implementation_sha256"]
