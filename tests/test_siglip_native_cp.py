"""CPU-only protocol checks for the complete native-normalization SigLIP rerun."""
import builtins
from collections import Counter
import csv
import importlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from eval.full_ft import run as ft
from eval.siglip_mainrule.protocol import cp_recipe


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cp(monkeypatch):
    assert (ROOT / "eval/siglip_native_cp.py").is_file(), "Native CP runner is missing"
    module = importlib.import_module("eval.siglip_native_cp")
    monkeypatch.setattr(module, "implementation_sha256", lambda: "native-code-hash")
    return module


def manifest(cp, tmp_path, **kwargs):
    path = tmp_path / "manifest.json"
    ft.atomic_json(path, cp.build_manifest(tmp_path / "outputs", **kwargs))
    return path


def test_all_63_jobs_cover_135_distinct_fits_and_preserve_recipes(cp, tmp_path):
    doc = cp.build_manifest(tmp_path)
    assert doc["protocol"] == "siglip_native_cp_v1"
    assert len(doc["tasks"]) == 63
    fits = set()
    large, small = [], []
    for i, task in enumerate(doc["tasks"]):
        assert task["task_id"] == i
        assert task["cp_recipe"] == cp_recipe(task["method"], task["n_samples"])
        assert task["budget"] == "MAX" and task["pool"] == "map"
        if task["dataset"] in {"octmnist", "pathmnist", "food101"}:
            assert len(task["seeds"]) == 1
            large.append(task)
        else:
            assert task["seeds"] == [42, 43, 44]
            small.append(task)
        for seed in task["seeds"]:
            key = (task["method"], task["dataset"], seed)
            assert key not in fits
            fits.add(key)
    assert len(large) == 27 and len(small) == 36 and len(fits) == 135
    assert Counter(t["gpu_profile"] for t in doc["tasks"]) == {
        "v100": 24, "a100": 30, "a100_80gb": 9}


@pytest.mark.parametrize("method", ["DIET", "SimCLR", "LeJEPA"])
@pytest.mark.parametrize("blocks", [2, 4, 6, -1])
def test_gpu_routing_uses_method_and_trainable_depth(cp, method, blocks):
    expected = "v100" if blocks == 2 else (
        "a100_80gb" if method == "LeJEPA" and blocks == -1 else "a100")
    assert cp.gpu_profile(method, blocks) == expected


@pytest.mark.parametrize("concurrency", [1, 2, 3, 4, 12])
def test_submission_groups_respect_total_concurrency(cp, tmp_path, concurrency):
    doc = cp.build_manifest(tmp_path)
    groups = cp.submission_groups(doc, concurrency)
    assert [(g["gpu_profile"], len(g["task_ids"])) for g in groups] == [
        ("v100", 24), ("a100", 30), ("a100_80gb", 9)]
    assert sorted(i for g in groups for i in g["task_ids"]) == list(range(63))
    assert all(0 < g["concurrency"] <= min(concurrency, len(g["task_ids"])) for g in groups)
    if concurrency >= 3:
        assert sum(g["concurrency"] for g in groups) == concurrency
        assert not any(g["serial"] for g in groups)
    else:
        assert all(g["serial"] for g in groups)


def test_submission_groups_only_include_selected_tasks(cp, tmp_path):
    doc = cp.build_manifest(tmp_path)
    task = next(t for t in doc["tasks"] if t["gpu_profile"] == "v100")
    doc["selected_task_ids"] = [task["task_id"]]
    groups = cp.submission_groups(doc, 12)
    assert len(groups) == 1 and groups[0]["concurrency"] == 1
    assert groups[0]["task_ids"] == doc["selected_task_ids"]
    doc["selected_task_ids"] = []
    assert cp.submission_groups(doc, 12) == []


def test_filters_preserve_three_seed_coverage(cp, tmp_path):
    doc = cp.build_manifest(tmp_path, datasets=["food101", "dtd"], methods=["LeJEPA"])
    assert len(doc["tasks"]) == 4
    assert sum(len(t["seeds"]) for t in doc["tasks"]) == 6
    for kwargs in ({"datasets": ["invalid"]}, {"methods": ["MAE"]},
                   {"datasets": []}, {"methods": ["DIET", "DIET"]}):
        with pytest.raises(ValueError):
            cp.build_manifest(tmp_path, **kwargs)


@pytest.mark.parametrize("change", ["norm", "seed", "recipe", "code", "root", "gpu"])
def test_manifest_is_immutable(cp, tmp_path, change):
    path = manifest(cp, tmp_path)
    doc = json.loads(path.read_text())
    if change == "norm":
        doc["normalization"]["mean"] = [.485, .456, .406]
    elif change == "seed":
        doc["tasks"][0]["seeds"] = [45]
    elif change == "recipe":
        doc["tasks"][0]["cp_recipe"]["batch_size"] = 1
    elif change == "code":
        doc["implementation_sha256"] = "old-code"
    elif change == "gpu":
        doc["tasks"][0]["gpu_profile"] = "h200"
    else:
        doc["output_root"] = str(tmp_path / "siglip_mainrule_v1")
    ft.atomic_json(path, doc)
    with pytest.raises(ValueError):
        cp.load_manifest(path)


def test_dry_run_never_loads_cuda_and_only_runs_cp_and_post_eval(cp, tmp_path, monkeypatch, capsys):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["SimCLR"])
    original_import = builtins.__import__
    def reject(name, *a, **kw):
        assert name != "torch", "dry-run imported torch"
        return original_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", reject)
    assert cp.run_task(path, 0, cache_dir=tmp_path, dry_run=True) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [r["seed"] for r in rows] == [42, 43, 44]
    for row in rows:
        command = row["command"]
        assert command[command.index("--normalization-mode") + 1] == "pretrained"
        assert "--skip-baseline" in command
        assert not {"--resume", "--no-cp", "--random-init", "--pre-cp-sft", "--post-cp-sft", "--skip-final-eval"} & set(command)
        assert "siglip_native_cp_v1" in command[command.index("--checkpoint-dir") + 1]
    assert not (tmp_path / "outputs/siglip_native_cp_v1").exists()


def fake_training(cp, monkeypatch, fail_seeds=(), bad_norm=False):
    calls = []
    monkeypatch.setattr(cp, "require_gpu", lambda profile: "test GPU " + profile)
    def execute(command, **kwargs):
        calls.append(command)
        def value(flag):
            return command[command.index(flag) + 1]
        seed = int(value("--seed"))
        assert "--resume" not in command
        dataset, model = value("--dataset"), value("--backbone")
        ckpt = Path(value("--checkpoint-dir")) / "cp" / f"{dataset}_{model}_n{value('--n-samples')}_s{seed}.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(f"native weights {seed}".encode())
        if seed in fail_seeds:
            raise subprocess.CalledProcessError(1, command)
        method = value("--cp-method")
        name = {"diet": "DIET", "lejepa": "LeJEPA", "simclr": "SimCLR"}[method]
        config = cp_recipe(name, int(value("--n-samples")))
        config.update(pool_strategy="map", skip_baseline=True, skip_final_eval=False,
                      resume=False, pre_cp_sft=False, post_cp_sft=False)
        row = dict(dataset=dataset, n_samples=int(value("--n-samples")),
                   backbone=model, method=method, seed=seed, epochs=150,
                   no_cp=False, random_init=False, normalization_mode="pretrained",
                   normalization={"mean": [.1 if bad_norm else .5] * 3, "std": [.5] * 3},
                   cp_config=config, post_knn_f1=.6, post_linear_f1=.7,
                   post_knn_acc=.6, post_linear_acc=.7)
        ft.atomic_json(value("--results-json"), row)
    monkeypatch.setattr(cp.subprocess, "run", execute)
    return calls


def test_seeds_are_sequential_and_completed_new_results_are_skipped(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["LeJEPA"])
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0
    assert [int(c[c.index("--seed") + 1]) for c in calls] == [42, 43, 44]
    calls.clear()
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0
    assert calls == []
    doc = cp.load_manifest(path)
    p = cp.result_paths(doc, doc["tasks"][0], 42)
    record = json.loads(p["receipt"].read_text())
    assert record["status"] == "cp_complete" and record["initialization"] == "public_pretrained"
    assert record["cp_result_sha256"] == ft.file_sha256(p["cp_result"])


def test_failed_seed_restarts_fresh_and_does_not_prevent_other_seeds(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["LeJEPA"])
    calls = fake_training(cp, monkeypatch, fail_seeds={43})
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    assert len(calls) == 3
    first = Path(calls[1][calls[1].index("--checkpoint-dir") + 1])
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0
    assert len(calls) == 1 and calls[0][calls[0].index("--seed") + 1] == "43"
    second = Path(calls[0][calls[0].index("--checkpoint-dir") + 1])
    assert first != second and first.exists() and second.exists()


def test_old_artifacts_are_never_reused_or_modified(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["DIET"])
    old = tmp_path / "outputs/siglip_mainrule_v1/cp_results/DIET/dtd/seed42.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"old": true}')
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0 and len(calls) == 3
    assert old.read_text() == '{"old": true}'


def test_incorrect_normalization_is_not_published(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["DIET"])
    fake_training(cp, monkeypatch, bad_norm=True)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    doc = cp.load_manifest(path)
    for seed in (42, 43, 44):
        p = cp.result_paths(doc, doc["tasks"][0], seed)
        assert not p["cp_result"].exists()
        assert json.loads(p["receipt"].read_text())["status"] == "cp_failed"


def test_filtered_later_plan_recognizes_same_completed_fit(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["DIET"])
    fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0
    all_doc = cp.build_manifest(tmp_path / "outputs")
    task = next(t for t in all_doc["tasks"] if t["dataset"] == "dtd" and t["method"] == "DIET")
    assert task["task_id"] != 0
    assert cp.completed_result(all_doc, task, 42) is not None


def test_changed_completed_result_is_rejected(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["DIET"])
    fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 0
    doc = cp.load_manifest(path)
    p = cp.result_paths(doc, doc["tasks"][0], 42)
    row = json.loads(p["cp_result"].read_text())
    row["post_knn_f1"] = .99
    ft.atomic_json(p["cp_result"], row)
    with pytest.raises(ValueError, match="SHA256"):
        cp.completed_result(doc, doc["tasks"][0], 42)


def test_collect_keeps_missing_status_separate_from_available_means(cp, tmp_path, monkeypatch, capsys):
    path = manifest(cp, tmp_path, datasets=["dtd"], methods=["DIET"])
    fake_training(cp, monkeypatch, fail_seeds={43})
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    capsys.readouterr()
    assert cp.collect_results(tmp_path / "outputs", datasets=["dtd"], methods=["DIET"]) == 0
    output = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(output.out)))
    assert [(r["seed"], r["status"]) for r in rows] == [
        ("42", "VERIFIED"), ("43", "CP_FAILED"), ("44", "VERIFIED"),
        ("MEAN", "n=2/3"), ("SD", "n=2/3")]
    assert rows[1]["post_knn_f1"] == ""
    assert rows[3]["post_knn_f1"] == "0.60000000"
    assert "2/3" in output.err and "Verified native CP results" not in output.out


def test_collect_reports_all_requested_seeds_before_any_runs(cp, tmp_path, capsys):
    assert cp.collect_results(tmp_path) == 0
    output = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(output.out)))
    assert len(rows) == 135 and {r["status"] for r in rows} == {"MISSING"}
    assert "0/135" in output.err
    assert not (tmp_path / cp.PROTOCOL).exists()


def test_active_fit_lock_prevents_duplicate_training(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["food101"], methods=["DIET"])
    doc = cp.load_manifest(path)
    receipt = cp.result_paths(doc, doc["tasks"][0], 42)["receipt"]
    calls = fake_training(cp, monkeypatch)
    with ft.seed_lock(receipt):
        assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    assert calls == [] and not receipt.exists()


def test_incompatible_failed_receipt_cannot_be_retried(cp, tmp_path, monkeypatch):
    path = manifest(cp, tmp_path, datasets=["food101"], methods=["DIET"])
    fake_training(cp, monkeypatch, fail_seeds={42})
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    doc = cp.load_manifest(path)
    receipt = cp.result_paths(doc, doc["tasks"][0], 42)["receipt"]
    record = json.loads(receipt.read_text())
    record["normalization"] = {"mean": [.1] * 3, "std": [.5] * 3}
    ft.atomic_json(receipt, record)
    calls = fake_training(cp, monkeypatch)
    assert cp.run_task(path, 0, cache_dir=tmp_path) == 1
    assert calls == []


@pytest.mark.parametrize("field,value", [("post_knn_f1", float("nan")),
                                        ("post_linear_f1", 1.2),
                                        ("post_knn_acc", None)])
def test_bad_metrics_are_rejected_but_zero_scores_are_valid(cp, tmp_path, field, value):
    task = cp.build_manifest(tmp_path)["tasks"][0]
    row = dict(dataset=task["dataset"], n_samples=task["n_samples"], backbone=task["model_id"],
               method=task["method"].lower(), seed=42, epochs=150, random_init=False, no_cp=False,
               normalization_mode="pretrained", normalization=cp.NORMALIZATION,
               cp_config=dict(task["cp_recipe"], pool_strategy="map", skip_baseline=True,
                              skip_final_eval=False, resume=False, pre_cp_sft=False, post_cp_sft=False),
               **{m: 0.0 for m in cp.METRICS})
    cp.validate_result(row, task, 42)
    row[field] = value
    with pytest.raises(ValueError):
        cp.validate_result(row, task, 42)


@pytest.mark.parametrize("profile,name,gb,count", [
    ("a100_80gb", "A100", 40, 1), ("a100_80gb", "A100 MIG", 80, 1),
    ("a100", "Tesla V100", 32, 1), ("v100", "A100", 80, 1),
    ("v100", "V100", 32, 2), ("a100", "A100", 80, 0),
    ("a100", "A100 MIG", 40, 1)])
def test_gpu_guard_rejects_incompatible_allocations(monkeypatch, profile, name, gb, count):
    module = importlib.import_module("eval.siglip_native_cp")
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: count > 0, device_count=lambda: count,
        get_device_properties=lambda _: SimpleNamespace(name=name, total_memory=gb * 1024**3)))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    with pytest.raises(RuntimeError):
        module.require_gpu(profile)


@pytest.mark.parametrize("profile,name,gb", [
    ("v100", "Tesla V100-PCIE-16GB", 16), ("v100", "Tesla V100-PCIE-32GB", 32),
    ("a100", "NVIDIA A100-SXM4-40GB", 40), ("a100", "NVIDIA A100 80GB PCIe", 80),
    ("a100_80gb", "NVIDIA A100 80GB PCIe", 80)])
def test_gpu_guard_accepts_requested_profile(monkeypatch, profile, name, gb):
    module = importlib.import_module("eval.siglip_native_cp")
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name=name, total_memory=gb * 1024**3)))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert module.require_gpu(profile) == name
