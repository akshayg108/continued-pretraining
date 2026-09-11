"""Contracts for the dataset-selectable, CP-only ViT-L completion pass."""
from collections import Counter
import builtins
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from eval.full_ft.manifest import DATASET_META, SEEDS
from eval.full_ft.run import atomic_json
from eval.vitl_completion import protocol as p
from eval.vitl_completion import run as r


MISSING = {"breastmnist", "octmnist", "organamnist", "pathmnist",
           "plant_village", "food101", "flowers102", "oxford_pet"}
DEPTHS = {"breastmnist": 2, "octmnist": -1, "organamnist": 6,
          "pathmnist": -1, "plant_village": 6, "food101": -1,
          "flowers102": 2, "oxford_pet": 2}


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, p.build_manifest(tmp_path / "outputs"))
    return path


def test_exact_grid_native_model_sizes_seeds_and_new_paths(manifest):
    doc = p.load_manifest(manifest)
    tasks = doc["tasks"]
    assert Counter(t["method"] for t in tasks) == {"LeJEPA": 8, "SimCLR": 8, "DIET": 8}
    assert {t["dataset"] for t in tasks} == MISSING
    assert [t["task_id"] for t in tasks] == list(range(24))
    assert [t["n_samples"] for t in tasks] == sorted(
        [t["n_samples"] for t in tasks], reverse=True)
    for task in tasks:
        assert task["model_id"] == "vit_large_patch16_dinov3.lvd1689m"
        assert task["pool"] == "cls" and task["size"] == "MAX"
        assert task["n_samples"] == DATASET_META[task["dataset"]][2]
        assert task["processed_subpath"] == DATASET_META[task["dataset"]][1]
        assert set(task["checkpoints"]) == {"42", "43", "44"}
        for seed in SEEDS:
            checkpoint = Path(task["checkpoints"][str(seed)])
            assert Path(doc["output_root"]) in checkpoint.parents
            assert "cp-L" not in checkpoint.parts


def test_recipes_preserve_existing_vitl_batches_and_mainrule_depth(manifest):
    for task in p.load_manifest(manifest)["tasks"]:
        recipe = task["cp_recipe"]
        assert recipe["num_trained_blocks"] == DEPTHS[task["dataset"]]
        assert recipe["epochs"] == 150
        assert recipe["freeze_epochs"] == recipe["warmup_epochs"] == 15
        assert recipe["lr"] == 1e-4 and recipe["weight_decay"] == .05
        if task["method"] == "DIET":
            assert recipe["batch_size"] == 32 and recipe["accumulate_grad_batches"] == 1
            assert recipe["label_smoothing"] == .3
            assert recipe["mixup_cutmix_prob"] == 0.0
        else:
            assert recipe["batch_size"] == 128 and recipe["accumulate_grad_batches"] == 2
            assert recipe["proj_dim"] == 128 and recipe["hidden_dim"] == 2048
            if task["method"] == "LeJEPA":
                assert recipe["n_views"] == 8 and recipe["lamb"] == .02
            else:
                assert recipe["temperature"] == .5


@pytest.mark.parametrize("n,depth", [(9999, 2), (10000, 4), (25000, 4),
                                   (25001, 6), (50000, 6), (50001, -1)])
def test_mainrule_boundaries(n, depth):
    assert p.trained_blocks(n) == depth


@pytest.mark.parametrize("datasets", [[], ["dtd"], ["unknown"],
                                     ["food101", "food101"], "food101"])
def test_reject_invalid_or_already_completed_dataset_selections(tmp_path, datasets):
    with pytest.raises(ValueError):
        p.build_manifest(tmp_path, datasets=datasets)


def test_selection_order_does_not_change_manifest_and_paths(tmp_path):
    first = p.build_manifest(tmp_path, datasets=["breastmnist", "food101"])
    second = p.build_manifest(tmp_path, datasets=["food101", "breastmnist"])
    assert first == second
    assert len(first["tasks"]) == 6
    full = p.build_manifest(tmp_path)
    for task in first["tasks"]:
        other = next(t for t in full["tasks"] if
                     (t["dataset"], t["method"]) == (task["dataset"], task["method"]))
        assert r.task_identity(task) == r.task_identity(other)
        assert r.seed_paths(first, task, 42) == r.seed_paths(full, other, 42)


@pytest.mark.parametrize("change", ["missing", "duplicate", "recipe", "path", "namespace", "float"])
def test_manifest_rejects_mutated_tasks(manifest, change):
    doc = json.loads(manifest.read_text())
    if change == "missing":
        doc["tasks"].pop()
    elif change == "duplicate":
        doc["tasks"][0] = copy.deepcopy(doc["tasks"][1])
    elif change == "recipe":
        doc["tasks"][0]["cp_recipe"]["num_trained_blocks"] = 2
    elif change == "path":
        doc["tasks"][0]["checkpoints"]["42"] = "/old/checkpoint.ckpt"
    elif change == "namespace":
        doc["output_root"] = str(Path(doc["output_root"]).parent / "cp-L")
    else:
        doc["tasks"][0]["cp_recipe"]["epochs"] = 150.0
    atomic_json(manifest, doc)
    with pytest.raises(ValueError):
        p.load_manifest(manifest)


def test_dry_run_all_commands_are_cp_only_with_paired_frozen_evaluation(
        manifest, tmp_path, capsys, monkeypatch):
    real_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("dry-run imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    doc = p.load_manifest(manifest)
    for task in doc["tasks"]:
        assert r.run_task(manifest, task["task_id"], cache_dir=tmp_path, dry_run=True) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 72
    for index, row in enumerate(rows):
        assert row["stage"] == "cp" and row["seed"] == SEEDS[index % 3]
        command = row["command"]
        assert "--cp-method" in command
        assert not set(command) & {"--pre-cp-sft", "--post-cp-sft", "--no-cp",
                                   "--skip-baseline", "--skip-final-eval", "--resume"}
        assert not any("full_ft/run.py" in arg for arg in command)
        assert command[command.index("--pool-strategy") + 1] == "cls"
    assert not Path(doc["output_root"]).exists()


@pytest.mark.parametrize("name,gib,available,count,valid", [
    ("NVIDIA A100-SXM4-80GB", 79.15, True, 1, True),
    ("NVIDIA A100 80GB PCIe", 79.15, True, 1, True),
    ("NVIDIA A100-SXM4-40GB", 39.5, True, 1, False),
    ("NVIDIA H100 80GB HBM3", 79.15, True, 1, False),
    ("NVIDIA A100-SXM4-80GB", 10, True, 1, False),
    ("NVIDIA A100-SXM4-80GB", 79.15, False, 0, False),
    ("NVIDIA A100-SXM4-80GB", 79.15, True, 2, False),
])
def test_gpu_gate(name, gib, available, count, valid, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count,
        get_device_properties=lambda index: SimpleNamespace(name=name, total_memory=int(gib * 1024**3)))))
    if valid:
        r.require_a100_80gb()
    else:
        with pytest.raises(RuntimeError, match="A100|CUDA"):
            r.require_a100_80gb()


def cp_result(task, seed):
    row = dict(dataset=task["dataset"], n_samples=task["n_samples"],
               backbone=task["model_id"], method=task["method"].lower(),
               seed=seed, epochs=150, random_init=False, no_cp=False)
    row.update({f"{phase}_{metric}": .6 for phase in ("pre", "post")
                for metric in ("knn_f1", "linear_f1", "knn_acc", "linear_acc")})
    return row


def fake_training(monkeypatch, manifest, task_id=0, fail_seed=None):
    doc = p.load_manifest(manifest)
    task = doc["tasks"][task_id]
    calls = []
    monkeypatch.setattr(r, "require_a100_80gb", lambda: None)
    monkeypatch.setattr(r, "implementation_sha256", lambda: "a" * 64)
    monkeypatch.setattr(r, "software_versions", lambda: {"test": "1"})

    def execute(command, **kwargs):
        assert "--cp-method" in command
        seed = int(command[command.index("--seed") + 1])
        calls.append((seed, command))
        paths = r.seed_paths(doc, task, seed)
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        paths["checkpoint"].write_bytes(f"weights-{seed}".encode())
        if seed == fail_seed:
            raise subprocess.CalledProcessError(1, command)
        atomic_json(paths["cp_result"], cp_result(task, seed))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(r.subprocess, "run", execute)
    return doc, task, calls


def test_wrong_gpu_rejected_before_training_or_outputs(manifest, tmp_path, monkeypatch):
    doc = p.load_manifest(manifest)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA"):
        r.run_task(manifest, 0, cache_dir=tmp_path)
    assert not Path(doc["output_root"]).exists()


def test_three_sequential_seeds_and_complete_resume(monkeypatch, manifest, tmp_path):
    doc, task, calls = fake_training(monkeypatch, manifest)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert [s for s, _ in calls] == list(SEEDS)
    for seed in SEEDS:
        receipt = json.loads(r.seed_paths(doc, task, seed)["receipt"].read_text())
        assert receipt["status"] == "cp_complete"
        assert len(receipt["checkpoint_sha256"]) == 64
    calls.clear()
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert calls == []


def test_resume_after_selection_changes_task_index(monkeypatch, manifest, tmp_path):
    doc = p.load_manifest(manifest)
    old_id = next(t["task_id"] for t in doc["tasks"] if
                  t["dataset"] == "breastmnist" and t["method"] == "DIET")
    _doc, _task, calls = fake_training(monkeypatch, manifest, old_id)
    assert r.run_task(manifest, old_id, cache_dir=tmp_path) == 0
    subset = tmp_path / "subset.json"
    atomic_json(subset, p.build_manifest(tmp_path / "outputs", datasets=["breastmnist"]))
    calls.clear()
    assert old_id != 0
    assert r.run_task(subset, 0, cache_dir=tmp_path) == 0
    assert calls == []


def test_failed_seed_does_not_block_other_seeds_and_resumes_checkpoint(
        monkeypatch, manifest, tmp_path):
    doc, task, calls = fake_training(monkeypatch, manifest, fail_seed=42)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    assert [s for s, _ in calls] == list(SEEDS)
    assert json.loads(r.seed_paths(doc, task, 42)["receipt"].read_text())["status"] == "cp_pending"
    _doc, _task, calls = fake_training(monkeypatch, manifest)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert [s for s, _ in calls] == [42]
    assert "--resume" in calls[0][1]


@pytest.mark.parametrize("change", ["unbound", "weights", "result", "code", "receipt"])
def test_incompatible_artifacts_fail_closed(monkeypatch, manifest, tmp_path, change):
    doc, task, calls = fake_training(monkeypatch, manifest)
    if change == "unbound":
        paths = r.seed_paths(doc, task, 42)
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        paths["checkpoint"].write_bytes(b"unbound")
    else:
        assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
        paths = r.seed_paths(doc, task, 42)
        if change == "weights":
            paths["checkpoint"].write_bytes(b"changed")
        elif change == "result":
            row = json.loads(paths["cp_result"].read_text())
            row["pre_knn_f1"] = .4
            atomic_json(paths["cp_result"], row)
        elif change == "code":
            monkeypatch.setattr(r, "implementation_sha256", lambda: "b" * 64)
        else:
            row = json.loads(paths["receipt"].read_text())
            row["status"] = "unknown"
            atomic_json(paths["receipt"], row)
    calls.clear()
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    assert 42 not in [s for s, _ in calls]


@pytest.mark.parametrize("key,value", [("pre_knn_f1", None), ("post_linear_f1", float("nan")),
                                      ("seed", 42.0), ("random_init", 0)])
def test_result_validation_checks_baselines_and_strict_identity(manifest, key, value):
    task = p.load_manifest(manifest)["tasks"][0]
    row = cp_result(task, 42)
    row[key] = value
    with pytest.raises(ValueError):
        r.validate_cp_result(row, task, 42)
