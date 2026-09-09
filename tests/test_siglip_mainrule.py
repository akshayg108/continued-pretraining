"""Contract tests for the isolated SigLIP main-grid-unfreezing pass."""
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
from eval.full_ft import run as ft
from eval.siglip_mainrule import protocol as p
from eval.siglip_mainrule import run as r


AFFECTED = {"galaxy10", "eurosat", "organamnist", "plant_village",
            "food101", "pathmnist", "octmnist"}


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    ft.atomic_json(path, p.build_manifest(tmp_path / "outputs"))
    return path


def test_exact_29_tasks_three_seeds_and_native_pool(manifest):
    doc = p.load_manifest(manifest)
    tasks = doc["tasks"]
    assert Counter(t["method"] for t in tasks) == {"LeJEPA": 7, "SimCLR": 7, "DIET": 15}
    for method in ("LeJEPA", "SimCLR"):
        assert {t["dataset"] for t in tasks if t["method"] == method} == AFFECTED
    assert {t["dataset"] for t in tasks if t["method"] == "DIET"} == set(DATASET_META)
    assert len(ft.load_tasks(manifest)) == 29
    assert [t["n_samples"] for t in tasks] == sorted([t["n_samples"] for t in tasks], reverse=True)
    for task in tasks:
        assert task["pool"] == "map" and "v2_webli" in task["model_id"]
        assert task["budget"] == "MAX" and task["phase"] == "post"
        assert set(task["checkpoints"]) == {"42", "43", "44"}
        for seed in SEEDS:
            assert len(task["checkpoints"][str(seed)]) == 1
            ckpt = Path(task["checkpoints"][str(seed)][0])
            assert ckpt.is_relative_to(Path(doc["output_root"]))
            assert "cp-siglip" not in ckpt.parts


@pytest.mark.parametrize("n,depth", [(9999, 2), (10000, 4), (25000, 4),
                                   (25001, 6), (50000, 6), (50001, -1)])
def test_main_grid_depth_boundaries(n, depth):
    assert p.trained_blocks(n) == depth


def test_exact_depths_and_recipes(manifest):
    expected = {"galaxy10": 4, "eurosat": 4, "organamnist": 6,
                "plant_village": 6, "food101": -1, "pathmnist": -1, "octmnist": -1}
    for task in p.load_manifest(manifest)["tasks"]:
        recipe = task["cp_recipe"]
        depth = recipe["num_trained_blocks"]
        assert depth == expected.get(task["dataset"], 2)
        assert recipe["epochs"] == 150
        assert recipe["freeze_epochs"] == recipe["warmup_epochs"] == 15
        assert recipe["lr"] == 1e-4 and recipe["weight_decay"] == .05
        if task["method"] == "DIET":
            assert recipe["batch_size"] == 32 and recipe["accumulate_grad_batches"] == 1
            assert recipe["mixup_cutmix_prob"] == 0.0
            assert recipe["label_smoothing"] == .3
        else:
            assert recipe["batch_size"] * recipe["accumulate_grad_batches"] == 256
            if task["method"] == "SimCLR":
                assert recipe["batch_size"] == 256 and recipe["temperature"] == .5
            else:
                assert recipe["accumulate_grad_batches"] == {2: 1, 4: 2, 6: 2, -1: 4}[depth]
                assert recipe["n_views"] == 8 and recipe["lamb"] == .02


@pytest.mark.parametrize("change", ["missing", "duplicate", "old_depth", "old_path", "namespace", "float_int"])
def test_manifest_rejects_edited_grid_or_recipe(manifest, change):
    doc = json.loads(manifest.read_text())
    if change == "missing":
        doc["tasks"].pop()
    elif change == "duplicate":
        doc["tasks"][0] = copy.deepcopy(doc["tasks"][1])
    elif change == "old_depth":
        doc["tasks"][0]["cp_recipe"]["num_trained_blocks"] = 2
    elif change == "old_path":
        doc["tasks"][0]["checkpoints"]["42"] = ["/old/cp-siglip/checkpoint.ckpt"]
    elif change == "namespace":
        doc["output_root"] = str(Path(doc["output_root"]).parent / "old")
    else:
        doc["tasks"][0]["cp_recipe"]["epochs"] = 150.0
    ft.atomic_json(manifest, doc)
    with pytest.raises(ValueError):
        p.load_manifest(manifest)


def test_dry_run_commands_no_training_artifacts(manifest, tmp_path, capsys):
    doc = p.load_manifest(manifest)
    for task in doc["tasks"]:
        assert r.run_task(manifest, task["task_id"], cache_dir=tmp_path, dry_run=True) == 0
    lines = capsys.readouterr().out.splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 29 * 3 * 2
    for cp, full_ft in zip(rows[::2], rows[1::2]):
        assert cp["stage"] == "cp" and full_ft["stage"] == "full_ft"
        assert cp["seed"] == full_ft["seed"]
        cmd = cp["command"]
        assert "--skip-baseline" in cmd
        assert "--pre-cp-sft" not in cmd and "--post-cp-sft" not in cmd
        assert "--resume" not in cmd and "--skip-final-eval" not in cmd
        assert cmd[cmd.index("--pool-strategy") + 1] == "map"
        assert full_ft["command"][full_ft["command"].index("--device") + 1] == "cuda"
        assert full_ft["command"][full_ft["command"].index("--seeds") + 1] == str(cp["seed"])
    assert not Path(doc["output_root"]).exists()


def test_dry_run_does_not_import_torch(monkeypatch, manifest, tmp_path):
    real_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("dry-run imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    assert r.run_task(manifest, 0, cache_dir=tmp_path, dry_run=True) == 0


def test_real_run_requires_cuda_before_artifacts_or_subprocess(
        monkeypatch, manifest, tmp_path):
    doc = p.load_manifest(manifest)
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(r.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(RuntimeError, match="CUDA"):
        r.run_task(manifest, 0, cache_dir=tmp_path)

    assert calls == []
    assert not Path(doc["output_root"]).exists()


def cp_result(task, seed):
    return dict(dataset=task["dataset"], n_samples=task["n_samples"],
                backbone=task["model_id"], method=task["method"].lower(),
                seed=seed, epochs=150, random_init=False, no_cp=False,
                post_knn_f1=.6, post_linear_f1=.7, post_knn_acc=.6, post_linear_acc=.7)


def fake_ft_result(doc, task, seed):
    identity = {k: task[k] for k in ft.IDENTITY_FIELDS}
    row = dict(identity, schema_version=1, status="success", sft_protocol="full_ft_v1",
               seed=seed, sft_acc=.7, sft_f1=.7, sft_auroc=.8,
               sft_total_params=100, sft_trainable_params=100,
               n_train_actual=task["n_samples"], n_test=10,
               task_sha256=ft.digest_json(identity), implementation_sha256="a" * 64,
               train_indices_sha256="b" * 64, test_labels_sha256="c" * 64,
               checkpoint_sha256=ft.file_sha256(task["checkpoints"][str(seed)][0]))
    ft.atomic_json(ft.result_path(Path(doc["output_root"]) / "full_ft", task, seed), row)


def fake_processes(monkeypatch, manifest, *, fail_cp=None, partial_ft=False):
    doc = p.load_manifest(manifest)
    task = doc["tasks"][0]
    calls = []
    monkeypatch.setattr(r, "require_cuda", lambda: None)
    def execute(command, **kwargs):
        is_cp = "--cp-method" in command
        seed = int(command[command.index("--seed" if is_cp else "--seeds") + 1])
        calls.append(("cp" if is_cp else "ft", seed, command))
        if is_cp:
            if seed == fail_cp:
                raise subprocess.CalledProcessError(1, command)
            paths = r.seed_paths(doc, task, seed)
            paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            paths["checkpoint"].write_bytes(f"weights-{seed}".encode())
            ft.atomic_json(paths["cp_result"], cp_result(task, seed))
        else:
            fake_ft_result(doc, task, seed)
            if partial_ft:
                path = ft.result_path(Path(doc["output_root"]) / "full_ft", task, seed)
                row = json.loads(path.read_text())
                row["sft_trainable_params"] = 50
                ft.atomic_json(path, row)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(r.subprocess, "run", execute)
    return doc, task, calls


def test_three_seeds_cp_then_full_ft_and_cp_resume(monkeypatch, manifest, tmp_path):
    doc, task, calls = fake_processes(monkeypatch, manifest)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert [(stage, seed) for stage, seed, _ in calls] == [
        (stage, seed) for seed in SEEDS for stage in ("cp", "ft")]
    for seed in SEEDS:
        receipt = json.loads(r.seed_paths(doc, task, seed)["receipt"].read_text())
        assert receipt["status"] == "cp_complete"
        assert receipt["cp_recipe"]["num_trained_blocks"] == -1
    calls.clear()
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    assert [(stage, seed) for stage, seed, _ in calls] == [("ft", seed) for seed in SEEDS]


def test_cp_failure_never_runs_ft_for_that_seed(monkeypatch, manifest, tmp_path):
    _doc, _task, calls = fake_processes(monkeypatch, manifest, fail_cp=42)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    assert ("ft", 42) not in [(stage, seed) for stage, seed, _ in calls]
    assert ("cp", 43) in [(stage, seed) for stage, seed, _ in calls]


def test_ft_gate_rejects_partially_frozen_result(monkeypatch, manifest, tmp_path):
    fake_processes(monkeypatch, manifest, partial_ft=True)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1


def test_unbound_existing_checkpoint_rejected(monkeypatch, manifest, tmp_path):
    doc, task, calls = fake_processes(monkeypatch, manifest)
    paths = r.seed_paths(doc, task, 42)
    paths["checkpoint"].parent.mkdir(parents=True)
    paths["checkpoint"].write_bytes(b"old-two-block-weights")
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    assert not any(seed == 42 for _, seed, _ in calls)
    assert paths["checkpoint"].read_bytes() == b"old-two-block-weights"


def test_resume_refuses_changed_checkpoint_or_receipt(monkeypatch, manifest, tmp_path):
    doc, task, calls = fake_processes(monkeypatch, manifest)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    r.seed_paths(doc, task, 42)["checkpoint"].write_bytes(b"modified")
    receipt = r.seed_paths(doc, task, 43)["receipt"]
    row = json.loads(receipt.read_text())
    row["cp_recipe"]["num_trained_blocks"] = 2
    ft.atomic_json(receipt, row)
    calls.clear()
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    assert [(stage, seed) for stage, seed, _ in calls] == [("ft", 44)]


def test_incomplete_same_recipe_cp_can_resume(monkeypatch, manifest, tmp_path):
    doc, task, _calls = fake_processes(monkeypatch, manifest, fail_cp=42)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 1
    paths = r.seed_paths(doc, task, 42)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    paths["checkpoint"].write_bytes(b"partial-training")
    _doc, _task, calls = fake_processes(monkeypatch, manifest)
    assert r.run_task(manifest, 0, cache_dir=tmp_path) == 0
    cp = [cmd for stage, seed, cmd in calls if stage == "cp" and seed == 42]
    assert len(cp) == 1 and "--resume" in cp[0]


@pytest.mark.parametrize("bad", [float("nan"), None, 2.0])
def test_cp_result_must_have_finite_valid_readouts(manifest, bad):
    task = p.load_manifest(manifest)["tasks"][0]
    row = cp_result(task, 42)
    row["post_linear_f1"] = bad
    with pytest.raises(ValueError):
        r.validate_cp_result(row, task, 42)
