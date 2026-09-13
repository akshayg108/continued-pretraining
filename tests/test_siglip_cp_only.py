"""CP-only recovery must preserve completed results and restart missing seeds."""
import builtins
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from eval.full_ft import run as ft
from eval.siglip_mainrule import protocol as original
from eval.siglip_mainrule import run as legacy


MODULE = "eval.siglip_mainrule.cp_only"


def test_cp_only_entrypoint_exists():
    assert importlib.util.find_spec(MODULE) is not None


@pytest.fixture
def cp(monkeypatch):
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(legacy, "implementation_sha256", lambda: "training-hash")
    return module


@pytest.fixture
def manifest(cp, tmp_path):
    path = tmp_path / "manifest.json"
    ft.atomic_json(path, cp.build_manifest(tmp_path / "outputs"))
    return path


def result_row(task, seed):
    return dict(dataset=task["dataset"], n_samples=task["n_samples"],
                backbone=task["model_id"], method=task["method"].lower(),
                seed=seed, epochs=150, random_init=False, no_cp=False,
                post_knn_f1=.61, post_linear_f1=.72,
                post_knn_acc=.63, post_linear_acc=.74)


def mark_source_complete(doc, entry):
    task, seed = entry["source_task"], entry["seed"]
    paths = legacy.seed_paths({"output_root": doc["source_output_root"]}, task, seed)
    ft.atomic_json(paths["cp_result"], result_row(task, seed))
    ft.atomic_json(paths["receipt"], dict(
        schema_version=1, protocol=original.PROTOCOL, status="cp_complete",
        task_sha256=ft.digest_json(task), seed=seed, cp_recipe=task["cp_recipe"],
        implementation_sha256="training-hash",
        cp_result_sha256=ft.file_sha256(paths["cp_result"])))
    return paths


def test_exact_three_datasets_one_seed_per_task_and_unchanged_recipe(cp, manifest):
    doc = cp.load_manifest(manifest)
    entries = doc["tasks"]
    assert len(entries) == 9
    assert [(e["source_task"]["dataset"], e["seed"]) for e in entries] == [
        (dataset, seed) for dataset in ("octmnist", "pathmnist", "food101")
        for seed in (42, 43, 44)]
    source = original.build_manifest(Path(doc["output_root"]).parent)
    for entry in entries:
        task = entry["source_task"]
        assert task == source["tasks"][task["task_id"]]
        assert task["method"] == "LeJEPA"
        assert task["cp_recipe"]["batch_size"] == 64
        assert task["cp_recipe"]["accumulate_grad_batches"] == 4
        assert task["cp_recipe"]["num_trained_blocks"] == -1
    assert doc["output_root"] != doc["source_output_root"]


def test_plan_selects_only_four_missing_cp_seeds_without_requiring_ft(cp, tmp_path):
    base = tmp_path / "outputs"
    doc = cp.build_manifest(base)
    missing = {("octmnist", 44), ("pathmnist", 44), ("food101", 43), ("food101", 44)}
    for entry in doc["tasks"]:
        if (entry["source_task"]["dataset"], entry["seed"]) not in missing:
            mark_source_complete(doc, entry)
    path = tmp_path / "selection.json"
    assert cp.write_plan(base, path) == [2, 5, 7, 8]
    assert cp.load_manifest(path)["selected_task_ids"] == [2, 5, 7, 8]
    assert not Path(doc["output_root"]).exists()
    with pytest.raises(FileExistsError):
        cp.write_plan(base, path)


def test_dry_run_has_one_cp_command_no_ft_no_resume_and_no_torch(
        cp, manifest, tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__
    def reject_torch(name, *args, **kwargs):
        assert name != "torch", "dry-run imported torch"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_torch)
    assert cp.run_task(manifest, 2, cache_dir=tmp_path, dry_run=True) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 1 and rows[0]["seed"] == 44 and rows[0]["stage"] == "cp"
    command = rows[0]["command"]
    assert "--skip-baseline" in command
    assert not {"--resume", "--pre-cp-sft", "--post-cp-sft", "--skip-final-eval"} & set(command)
    assert "eval/full_ft/run.py" not in " ".join(command)
    assert command[command.index("--project") + 1] == cp.PROTOCOL
    assert not Path(cp.load_manifest(manifest)["output_root"]).exists()


def test_finished_source_cp_is_skipped_without_gpu_or_ft(cp, manifest, monkeypatch, tmp_path):
    doc = cp.load_manifest(manifest)
    paths = mark_source_complete(doc, doc["tasks"][0])
    before = {key: paths[key].read_bytes() for key in ("cp_result", "receipt")}
    monkeypatch.setattr(cp, "require_gpu", lambda: pytest.fail("GPU was requested"))
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: pytest.fail("training was launched"))
    assert cp.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert {key: paths[key].read_bytes() for key in before} == before


def fake_training(cp, monkeypatch, *, fail=False, bad_metric=False):
    commands = []
    monkeypatch.setattr(cp, "require_gpu", lambda: None)
    def execute(command, **kwargs):
        commands.append(command)
        assert "--cp-method" in command and "--resume" not in command
        assert "--post-cp-sft" not in command
        def value(flag):
            return command[command.index(flag) + 1]
        dataset, model, seed = value("--dataset"), value("--backbone"), int(value("--seed"))
        ckpt = Path(value("--checkpoint-dir")) / "cp" / (
            f"{dataset}_{model}_n{value('--n-samples')}_s{seed}.ckpt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"new-weights")
        if fail:
            raise subprocess.CalledProcessError(1, command)
        row = dict(dataset=dataset, n_samples=int(value("--n-samples")), backbone=model,
                   method=value("--cp-method"), seed=seed, epochs=150,
                   random_init=False, no_cp=False, post_knn_f1=.6, post_linear_f1=.7,
                   post_knn_acc=.6, post_linear_acc=.7)
        if bad_metric:
            row["post_linear_f1"] = 2.0
        result = Path(value("--results-json"))
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(row, indent=2))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(cp.subprocess, "run", execute)
    return commands


def test_partial_source_is_untouched_and_one_seed_restarts_from_public_weights(
        cp, manifest, monkeypatch, tmp_path):
    doc = cp.load_manifest(manifest)
    entry = doc["tasks"][2]
    old = legacy.seed_paths({"output_root": doc["source_output_root"]}, entry["source_task"], 44)
    old["checkpoint"].parent.mkdir(parents=True)
    old["checkpoint"].write_bytes(b"partial-old-weights")
    ft.atomic_json(old["receipt"], {"status": "cp_pending"})
    receipt_before = old["receipt"].read_bytes()
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(manifest, 2, cache_dir=tmp_path) == 0
    assert len(calls) == 1 and calls[0][calls[0].index("--seed") + 1] == "44"
    assert old["checkpoint"].read_bytes() == b"partial-old-weights"
    assert old["receipt"].read_bytes() == receipt_before
    paths = cp.result_paths(doc, entry)
    record = json.loads(paths["receipt"].read_text())
    assert record["status"] == "cp_complete" and record["initialization"] == "public_pretrained"
    assert record["training_implementation_sha256"] == "training-hash"
    assert record["cp_result_sha256"] == ft.file_sha256(paths["cp_result"])
    calls.clear()
    assert cp.run_task(manifest, 2, cache_dir=tmp_path) == 0
    assert calls == []


def test_failed_restart_uses_a_fresh_attempt_next_time_without_resume(
        cp, manifest, monkeypatch, tmp_path):
    calls = fake_training(cp, monkeypatch, fail=True)
    with pytest.raises(subprocess.CalledProcessError):
        cp.run_task(manifest, 7, cache_dir=tmp_path)
    first_command = calls[0]
    first_dir = Path(first_command[first_command.index("--checkpoint-dir") + 1])
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(manifest, 7, cache_dir=tmp_path) == 0
    second_dir = Path(calls[0][calls[0].index("--checkpoint-dir") + 1])
    assert first_dir != second_dir and first_dir.exists() and second_dir.exists()


def test_completed_restart_is_excluded_from_later_plans(cp, manifest, monkeypatch, tmp_path):
    fake_training(cp, monkeypatch)
    assert cp.run_task(manifest, 8, cache_dir=tmp_path) == 0
    doc = cp.load_manifest(manifest)
    selected = cp.write_plan(Path(doc["output_root"]).parent, tmp_path / "later.json")
    assert 8 not in selected and len(selected) == 8


def test_source_becomes_complete_after_submission_is_skipped(cp, manifest, monkeypatch, tmp_path):
    doc = cp.load_manifest(manifest)
    mark_source_complete(doc, doc["tasks"][8])
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: pytest.fail("duplicate CP"))
    assert cp.run_task(manifest, 8, cache_dir=tmp_path) == 0


def test_live_original_seed_lock_prevents_duplicate_training(cp, manifest, monkeypatch, tmp_path):
    doc = cp.load_manifest(manifest)
    entry = doc["tasks"][2]
    old = legacy.seed_paths({"output_root": doc["source_output_root"]}, entry["source_task"], 44)
    calls = fake_training(cp, monkeypatch)
    with ft.seed_lock(old["receipt"]):
        with pytest.raises(RuntimeError, match="running"):
            cp.run_task(manifest, 2, cache_dir=tmp_path)
    assert calls == []


@pytest.mark.parametrize("change", ["result", "recipe", "code"])
def test_corrupt_completed_source_is_not_silently_skipped(cp, manifest, change):
    doc = cp.load_manifest(manifest)
    entry = doc["tasks"][0]
    paths = mark_source_complete(doc, entry)
    if change == "result":
        row = json.loads(paths["cp_result"].read_text())
        row["post_knn_f1"] = .99
        ft.atomic_json(paths["cp_result"], row)
    else:
        row = json.loads(paths["receipt"].read_text())
        if change == "recipe":
            row["cp_recipe"]["num_trained_blocks"] = 2
        else:
            row["implementation_sha256"] = "different-code"
        ft.atomic_json(paths["receipt"], row)
    with pytest.raises(ValueError):
        cp.completed_result(doc, entry)


def test_invalid_final_metrics_are_never_marked_complete(cp, manifest, monkeypatch, tmp_path):
    fake_training(cp, monkeypatch, bad_metric=True)
    with pytest.raises(ValueError):
        cp.run_task(manifest, 2, cache_dir=tmp_path)
    doc = cp.load_manifest(manifest)
    paths = cp.result_paths(doc, doc["tasks"][2])
    assert json.loads(paths["receipt"].read_text())["status"] == "cp_pending"
    assert not paths["cp_result"].exists()


def test_no_gpu_means_no_attempt_or_output(cp, manifest, monkeypatch, tmp_path):
    def no_gpu():
        raise RuntimeError("CUDA required")
    monkeypatch.setattr(cp, "require_gpu", no_gpu)
    with pytest.raises(RuntimeError, match="CUDA"):
        cp.run_task(manifest, 0, cache_dir=tmp_path)
    assert not Path(cp.load_manifest(manifest)["output_root"]).exists()


@pytest.mark.parametrize("change", ["dataset", "seed", "recipe", "selection", "root"])
def test_manifest_rejects_changes_outside_the_frozen_nine_seed_panel(cp, manifest, change):
    doc = json.loads(manifest.read_text())
    if change == "dataset":
        doc["tasks"][0]["source_task"]["dataset"] = "dtd"
    elif change == "seed":
        doc["tasks"][0]["seed"] = 45
    elif change == "recipe":
        doc["tasks"][0]["source_task"]["cp_recipe"]["batch_size"] = 32
    elif change == "selection":
        doc["selected_task_ids"] = [0, 0]
    else:
        doc["output_root"] = doc["source_output_root"]
    ft.atomic_json(manifest, doc)
    with pytest.raises(ValueError):
        cp.load_manifest(manifest)
