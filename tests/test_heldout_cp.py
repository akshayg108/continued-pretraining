"""Protocol and collection checks without a GPU or dataset downloads."""

import importlib
import json
from pathlib import Path

import pytest


def protocol():
    path = Path(__file__).resolve().parents[1] / "eval/heldout_cp/protocol.py"
    assert path.is_file(), "Held-out CP protocol is missing"
    return importlib.import_module("eval.heldout_cp.protocol")


def test_complete_factorial_with_three_serial_seeds(tmp_path):
    p = protocol()
    doc = p.build_manifest(tmp_path)
    assert len(doc["tasks"]) == 48
    assert len(doc["datasets"]) == 8
    identities = set()
    for task in doc["tasks"]:
        assert task["seeds"] == [42, 43, 44]
        assert task["n_samples"] == 1000
        assert task["gpu"] == "v100"
        assert task["recipe"]["num_trained_blocks"] == 2
        assert task["recipe"]["epochs"] == 150
        for seed in task["seeds"]:
            identities.add((task["encoder"], task["method"], task["dataset"], seed))
    assert len(identities) == 144
    assert {t["encoder"] for t in doc["tasks"]} == {"DINOv3", "CLIP"}
    assert {t["method"] for t in doc["tasks"]} == {"LeJEPA", "DIET", "SimCLR"}
    assert doc["no_ft"] is True


def test_official_normalization_and_small_budget_recipe(tmp_path):
    p = protocol()
    tasks = p.build_manifest(tmp_path)["tasks"]
    for task in tasks:
        norm = task["normalization"]
        if task["encoder"] == "DINOv3":
            assert norm == {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        else:
            assert norm == {
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
            }
        assert task["recipe"]["batch_size"] == (32 if task["method"] == "DIET" else 256)
        assert task["recipe"]["accumulate_grad_batches"] == 1
        assert task["recipe"]["freeze_epochs"] == 15


def test_manifest_rejects_silent_recipe_edit(tmp_path):
    p = protocol()
    doc = p.build_manifest(tmp_path)
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(doc))
    assert p.load_manifest(file) == doc
    doc["tasks"][0]["recipe"]["lr"] = 1e-3
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest"):
        p.load_manifest(file)


def test_predictions_cannot_freeze_with_missing_preparations(tmp_path):
    p = protocol()
    with pytest.raises((ValueError, FileNotFoundError)):
        p.freeze_predictions(p.build_manifest(tmp_path), "aid")
    assert not (tmp_path / p.PROTOCOL / "predictions/aid.json").exists()


def test_summary_uses_real_paired_seeds_and_sample_sd():
    p = protocol()
    rows = [
        dict(pre_knn_f1=0.3, post_knn_f1=0.5, pre_linear_f1=0.4, post_linear_f1=0.7),
        dict(pre_knn_f1=0.4, post_knn_f1=0.7, pre_linear_f1=0.6, post_linear_f1=0.8),
    ]
    result = p.summarize(rows)
    assert result["n_seeds"] == 2
    assert result["delta_knn_f1_mean"] == pytest.approx(0.25)
    assert result["delta_knn_f1_sd"] == pytest.approx(0.1 / 2**0.5)
    assert result["delta_linear_f1_mean"] == pytest.approx(0.25)
    assert p.summarize(rows[:1])["delta_knn_f1_sd"] is None
    assert p.summarize([])["n_seeds"] == 0


def prepared_fixture(tmp_path, datasets=None):
    p = protocol()
    doc = p.build_manifest(tmp_path)
    for encoder in p.ENCODER_ORDER:
        for dataset in p.DATASETS if datasets is None else datasets:
            index = p.DATASETS.index(dataset)
            for seed in p.SEEDS:
                row = dict(
                    p.identity(doc, encoder, dataset, seed),
                    status="complete",
                    train_indices=list(range(1000)),
                    data={"n_train_actual": 1000},
                    **{f"pre_{m}": 0.5 for m in p.METRICS},
                )
                p.atomic_json(p.pre_path(doc, encoder, dataset, seed), row)
            row = dict(
                p.identity(doc, encoder, dataset),
                status="complete",
                n_geometry=1000,
                uniformity_t2=-0.1 * (index + 1),
            )
            p.atomic_json(p.geometry_path(doc, encoder, dataset), row)
    return p, doc


def test_prediction_scores_are_frozen_before_training_and_cannot_be_replaced(tmp_path):
    p, doc = prepared_fixture(tmp_path)
    path = p.freeze_predictions(doc, "aid")
    original = path.read_bytes()
    assert json.loads(original)["initial_uniformity"]["CLIP"] == pytest.approx(-0.3)
    assert p.freeze_predictions(doc, "aid").read_bytes() == original
    source = p.geometry_path(doc, "CLIP", "aid")
    changed = json.loads(source.read_text())
    changed["uniformity_t2"] = -0.01
    p.atomic_json(source, changed)
    with pytest.raises(ValueError, match="mismatch"):
        p.freeze_predictions(doc, "aid")
    assert path.read_bytes() == original


def test_no_new_prediction_record_after_a_training_attempt(tmp_path):
    p, doc = prepared_fixture(tmp_path)
    attempt = p.root_path(doc) / "attempts/DINOv3/LeJEPA/bloodmnist/seed42/attempt1"
    attempt.mkdir(parents=True)
    with pytest.raises(ValueError, match="after CP"):
        p.freeze_predictions(doc, "bloodmnist")


def test_ready_dataset_does_not_wait_for_unprepared_datasets(tmp_path):
    p, doc = prepared_fixture(tmp_path, datasets=("aid",))
    other_attempt = (
        p.root_path(doc) / "attempts/DINOv3/LeJEPA/bloodmnist/seed42/attempt1"
    )
    other_attempt.mkdir(parents=True)
    path = p.freeze_predictions(doc, "aid")
    assert path == p.predictions_path(doc, "aid")
    assert path.is_file()
    with pytest.raises(FileNotFoundError):
        p.freeze_predictions(doc, "flavia")


def test_duplicate_training_indices_are_not_a_completed_baseline(tmp_path):
    p, doc = prepared_fixture(tmp_path)
    path = p.pre_path(doc, "CLIP", "aid", 42)
    row = json.loads(path.read_text())
    row["train_indices"] = [0] * 1000
    p.atomic_json(path, row)
    with pytest.raises(ValueError, match="1000-image"):
        p.validate_pre(doc, "CLIP", "aid", 42)


def test_cpu_dry_run_lists_all_three_fresh_runs_without_importing_training(tmp_path):
    import subprocess
    import sys

    p = protocol()
    path = tmp_path / "manifest.json"
    p.atomic_json(path, p.build_manifest(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.heldout_cp",
            "run",
            "--manifest",
            str(path),
            "--task-id",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=p.ROOT,
    )
    assert result.returncode == 0, result.stderr
    for seed in (42, 43, 44):
        assert f"seed={seed}" in result.stdout
    assert "torch" not in result.stderr


def test_bad_task_id_is_rejected_in_dry_run(tmp_path):
    import subprocess
    import sys

    p = protocol()
    path = tmp_path / "manifest.json"
    p.atomic_json(path, p.build_manifest(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.heldout_cp",
            "run",
            "--manifest",
            str(path),
            "--task-id",
            "48",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=p.ROOT,
    )
    assert result.returncode != 0
    assert "task" in result.stderr.lower()


def completed_fixture(tmp_path, datasets=None):
    p, doc = prepared_fixture(tmp_path, datasets)
    for task in doc["tasks"]:
        if datasets is not None and task["dataset"] not in datasets:
            continue
        predictions = p.freeze_predictions(doc, task["dataset"])
        for seed in task["seeds"]:
            baseline_path = p.pre_path(doc, task["encoder"], task["dataset"], seed)
            baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
            gain = (8 - p.DATASETS.index(task["dataset"])) * 0.01
            row = dict(
                baseline,
                method=task["method"],
                recipe=task["recipe"],
                no_ft=True,
                initialization="public_pretrained",
                pre_sha256=p.file_sha256(baseline_path),
                predictions_sha256=p.file_sha256(predictions),
                **{f"post_{m}": 0.5 + gain for m in p.METRICS},
            )
            p.atomic_json(p.result_path(doc, task, seed), row)
    return p, doc


def test_collection_uses_dataset_means_not_seed_pseudoreplicates(tmp_path):
    import csv
    from eval.heldout_cp.collect import collect

    _, doc = completed_fixture(tmp_path)
    out = tmp_path / "report"
    result = collect(doc, out)
    assert result["verified"] == 144
    with (out / "summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 48
    assert all(r["n_seeds"] == "3" for r in summaries)
    with (out / "correlations.csv").open() as handle:
        correlations = list(csv.DictReader(handle))
    assert len(correlations) == 16
    assert all(r["n_datasets"] == "8" for r in correlations)
    assert all(float(r["spearman_rho"]) == pytest.approx(1.0) for r in correlations)


def test_incomplete_results_are_not_imputed_or_given_a_full_correlation(tmp_path):
    import csv
    from eval.heldout_cp.collect import collect

    p, doc = completed_fixture(tmp_path)
    p.result_path(doc, doc["tasks"][0], 42).unlink()
    result = collect(doc, tmp_path / "report")
    assert result["verified"] == 143
    assert result["missing"] == 1
    with (tmp_path / "report/correlations.csv").open() as handle:
        correlations = list(csv.DictReader(handle))
    incomplete = [
        r
        for r in correlations
        if r["encoder"] == "DINOv3" and r["method"] in ("LeJEPA", "MEAN_METHODS")
    ]
    assert len(incomplete) == 4
    assert all(
        r["status"] == "INCOMPLETE" and not r["spearman_rho"] for r in incomplete
    )


def test_invalid_results_are_distinct_from_missing_results(tmp_path):
    from eval.heldout_cp.collect import collect

    p, doc = completed_fixture(tmp_path)
    path = p.result_path(doc, doc["tasks"][0], 42)
    row = json.loads(path.read_text())
    row["post_knn_f1"] = float("nan")
    path.write_text(json.dumps(row))
    result = collect(doc, tmp_path / "report")
    assert result["verified"] == 143
    assert result["invalid"] == 1
    assert result["missing"] == 0


def test_collect_ready_dataset_results_while_others_are_unprepared(tmp_path):
    from eval.heldout_cp.collect import collect

    _, doc = completed_fixture(tmp_path, datasets=("aid",))
    result = collect(doc, tmp_path / "report")
    assert result == dict(expected=144, verified=18, missing=126, invalid=0)


def test_run_task_only_checks_its_dataset_preparations(tmp_path, monkeypatch):
    from eval.heldout_cp.run import run_task
    import subprocess

    p, doc = prepared_fixture(tmp_path, datasets=("aid",))
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    task = next(t for t in doc["tasks"] if t["dataset"] == "aid")
    run_task(tmp_path / "manifest.json", doc, task, cache_dir=tmp_path, num_workers=0)
    assert [c[c.index("--seed") + 1] for c in calls] == ["42", "43", "44"]


def test_prediction_publication_waits_for_another_array_worker(tmp_path):
    import threading
    import time

    p, doc = prepared_fixture(tmp_path)
    results, errors = [], []

    def worker():
        try:
            results.append(p.freeze_predictions(doc, "aid"))
        except Exception as exc:
            errors.append(exc)

    with p.seed_lock(p.predictions_path(doc, "aid")):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        time.sleep(0.1)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1 and results[0].is_file()
