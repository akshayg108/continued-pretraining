"""Held-out encoder-extension contracts without downloads or GPU allocation."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from eval.heldout_cp import protocol as original


def extension():
    path = original.ROOT / "eval/heldout_extensions/protocol.py"
    assert path.is_file(), "Encoder-extension protocol is missing"
    return importlib.import_module("eval.heldout_extensions.protocol")


@pytest.fixture
def source_manifest(tmp_path):
    doc = original.build_manifest(tmp_path / "original")
    for dataset in original.DATASETS:
        for encoder in original.ENCODER_ORDER:
            for seed in original.SEEDS:
                original.atomic_json(original.pre_path(doc, encoder, dataset, seed), dict(
                    original.identity(doc, encoder, dataset, seed), status="complete",
                    train_indices=list(range(1000)), data={"n_train_actual": 1000},
                    **{f"pre_{metric}": .5 for metric in original.METRICS},
                ))
            original.atomic_json(original.geometry_path(doc, encoder, dataset), dict(
                original.identity(doc, encoder, dataset), status="complete",
                n_geometry=1200, uniformity_t2=-1., geometry_indices=list(range(1200)),
            ))
        original.freeze_predictions(doc, dataset)
    path = tmp_path / "source.json"
    original.atomic_json(path, doc)
    return path


def make_doc(tmp_path, source_manifest):
    return extension().build_manifest(tmp_path / "new", source_manifest)


def test_grid_gpu_readout_and_native_normalization(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    assert len(doc["tasks"]) == 48 and len(doc["preparations"]) == 16
    assert doc["no_ft"] is True
    assert {t["encoder"] for t in doc["tasks"]} == {"SigLIP", "DINOv3L"}
    assert sum(t["gpu"] == "a100" for t in doc["tasks"]) == 32
    assert sum(t["gpu"] == "v100" for t in doc["tasks"]) == 16
    for task in doc["tasks"]:
        assert task["seeds"] == [42, 43, 44] and task["n_samples"] == 1000
        assert task["recipe"]["epochs"] == 150
        assert task["recipe"]["freeze_epochs"] == 15
        assert task["recipe"]["num_trained_blocks"] == 2
        expected_recipe = original.cp_recipe(task["method"], 1000)
        if task["encoder"] == "DINOv3L" and task["method"] in {"LeJEPA", "SimCLR"}:
            expected_recipe.update(batch_size=128, accumulate_grad_batches=2)
        assert task["recipe"] == expected_recipe
        prep = doc["preparations"][task["preparation_id"]]
        assert (prep["encoder"], prep["dataset"]) == (task["encoder"], task["dataset"])
        if task["encoder"] == "SigLIP":
            assert task["model_id"] == "vit_base_patch16_siglip_224.v2_webli"
            assert task["pool"] == "map" and task["embed_dim"] == 768
            assert task["normalization"] == {"mean": [.5]*3, "std": [.5]*3}
            assert task["gpu"] == ("a100" if task["method"] == "LeJEPA" else "v100")
        else:
            assert task["model_id"] == "vit_large_patch16_dinov3.lvd1689m"
            assert task["pool"] == "cls" and task["embed_dim"] == 1024
            assert task["normalization"] == original.EXPECTED_NORMALIZATIONS["DINOv3"]
            assert task["gpu"] == prep["gpu"] == "a100"
            recipe = task["recipe"]
            assert recipe["batch_size"] == (32 if task["method"] == "DIET" else 128)
            assert recipe["accumulate_grad_batches"] == (1 if task["method"] == "DIET" else 2)
    assert original.implementation_sha256() == "3d181c4845e2d0e0278cc508e312ff12ce5f402ada8e27edb569fadcc95e3ade"


def test_source_baselines_are_pinned_and_must_match(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    row = p.source_baseline(doc, "aid", 43)
    assert row["train_indices"] == list(range(1000))
    source = original.load_manifest(source_manifest)
    path = original.pre_path(source, "CLIP", "aid", 43)
    changed = json.loads(path.read_text())
    changed["train_indices"][-1] = 1001
    original.atomic_json(path, changed)
    with pytest.raises(ValueError):
        p.source_baseline(doc, "aid", 43)
    with pytest.raises(ValueError):
        make_doc(tmp_path, source_manifest)


def test_manifest_rejects_recipe_and_source_edits(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    path = tmp_path / "manifest.json"
    p.atomic_json(path, doc)
    assert p.load_manifest(path) == doc
    doc["tasks"][0]["pool"] = "cls"
    p.atomic_json(path, doc)
    with pytest.raises(ValueError):
        p.load_manifest(path)


def prepare_artifacts(doc, encoder, dataset):
    p = extension()
    for seed in p.SEEDS:
        source = p.source_baseline(doc, dataset, seed)
        p.atomic_json(p.pre_path(doc, encoder, dataset, seed), dict(
            p.identity(doc, encoder, dataset, seed), status="complete",
            train_indices=source["train_indices"], data=source["data"],
            software={"version": "test"}, pretrained_weights_sha256="weights",
            **{f"pre_{m}": .5 for m in p.METRICS},
        ))
    p.atomic_json(p.geometry_path(doc, encoder, dataset), dict(
        p.identity(doc, encoder, dataset), status="complete", n_geometry=1200,
        geometry_indices=list(range(1200)), uniformity_t2=-1.,
    ))


def test_predictions_are_encoder_target_local_and_immutable(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    prepare_artifacts(doc, "SigLIP", "aid")
    path = p.freeze_predictions(doc, "SigLIP", "aid")
    content = path.read_bytes()
    assert p.freeze_predictions(doc, "SigLIP", "aid").read_bytes() == content
    with pytest.raises(FileNotFoundError):
        p.freeze_predictions(doc, "DINOv3L", "aid")
    geometry = p.geometry_path(doc, "SigLIP", "aid")
    row = json.loads(geometry.read_text())
    row["uniformity_t2"] = -.2
    p.atomic_json(geometry, row)
    with pytest.raises(ValueError):
        p.freeze_predictions(doc, "SigLIP", "aid")
    assert path.read_bytes() == content


def test_predictions_cannot_be_created_after_cp(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    prepare_artifacts(doc, "DINOv3L", "aid")
    (p.root_path(doc) / "attempts/DINOv3L/LeJEPA/aid/seed42/old").mkdir(parents=True)
    with pytest.raises(ValueError, match="after CP"):
        p.freeze_predictions(doc, "DINOv3L", "aid")


def test_new_baseline_cannot_change_training_subset(tmp_path, source_manifest):
    p = extension()
    doc = make_doc(tmp_path, source_manifest)
    prepare_artifacts(doc, "SigLIP", "aid")
    path = p.pre_path(doc, "SigLIP", "aid", 42)
    row = json.loads(path.read_text())
    row["train_indices"][-1] = 1001
    p.atomic_json(path, row)
    with pytest.raises(ValueError):
        p.validate_pre(doc, "SigLIP", "aid", 42)


def test_cpu_plan_and_dry_run(tmp_path, source_manifest):
    p = extension()
    path = tmp_path / "manifest.json"
    for args in (
        ["plan", "--source-manifest", str(source_manifest), "--output-base", str(tmp_path)],
        ["prepare", "--preparation-id", "8", "--dry-run"],
        ["run", "--task-id", "0", "--dry-run"],
    ):
        result = subprocess.run([sys.executable, "-m", "eval.heldout_extensions", *args,
                                 "--manifest", str(path)], cwd=p.ROOT, text=True,
                                capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
    assert "seed=42" in result.stdout and "seed=44" in result.stdout
    assert "gpu=a100" in result.stdout and "pool=map" in result.stdout


def test_preparation_reuses_source_indices_and_skips_verified_baselines(tmp_path, source_manifest, monkeypatch):
    p = extension()
    assert (p.ROOT / "eval/heldout_extensions/prepare.py").is_file(), "Extension preparation is missing"
    preparation = importlib.import_module("eval.heldout_extensions.prepare")
    rt = importlib.import_module("eval.heldout_extensions.runtime")
    doc = make_doc(tmp_path, source_manifest)
    task = doc["preparations"][0]
    pl, torch = ModuleType("lightning"), ModuleType("torch")
    pl.seed_everything = lambda *a, **kw: None
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "lightning", pl)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(rt, "check_environment", lambda gpu: "Tesla V100")
    monkeypatch.setattr(rt, "software", lambda: {"version": "test"})
    monkeypatch.setattr(rt, "dataset_record", lambda d, c, i: {"n_train_actual": 1000})
    monkeypatch.setattr(rt, "load_model", lambda *a: (object(), "cuda:0", {}, "weights"))
    calls = []
    def geometry(model, device, config, args, indices):
        assert indices == list(range(1200))
        return dict(n_geometry=1200, geometry_indices=indices, uniformity_t2=-1.)
    def evaluate(model, device, config, args, indices):
        assert indices == list(range(1000))
        calls.append(args.seed)
        return {m: .8 for m in p.METRICS}, {"raw_scores": {m: .8 for m in p.METRICS}}
    monkeypatch.setattr(rt, "initial_geometry", geometry)
    monkeypatch.setattr(rt, "evaluate", evaluate)
    preparation.prepare(doc, task, cache_dir=tmp_path, num_workers=0)
    assert calls == [42, 43, 44]
    assert p.predictions_path(doc, "SigLIP", task["dataset"]).is_file()
    preparation.prepare(doc, task, cache_dir=tmp_path, num_workers=0)
    assert calls == [42, 43, 44]
    baseline = p.validate_pre(doc, "SigLIP", task["dataset"], 42)
    assert baseline["evaluation_numerics"]["raw_scores"]["knn_f1"] == .8
    monkeypatch.setattr(rt, "dataset_record", lambda *a: {"n_train_actual": 999})
    with pytest.raises(ValueError, match="data"):
        preparation.prepare(doc, task, cache_dir=tmp_path, num_workers=0)


def test_collection_reports_only_verified_paired_results(tmp_path, source_manifest):
    p = extension()
    assert (p.ROOT / "eval/heldout_extensions/collect.py").is_file(), "Extension collector is missing"
    collect = importlib.import_module("eval.heldout_extensions.collect").collect
    doc = make_doc(tmp_path, source_manifest)
    task = doc["tasks"][0]
    prepare_artifacts(doc, task["encoder"], task["dataset"])
    prediction = p.freeze_predictions(doc, task["encoder"], task["dataset"])
    for seed, score in ((42, .6), (43, .8)):
        baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
        row = dict(p.identity(doc, task["encoder"], task["dataset"], seed),
                   status="complete", method=task["method"], recipe=task["recipe"], no_ft=True,
                   initialization="public_pretrained", data=baseline["data"],
                   software=baseline["software"], pretrained_weights_sha256="weights", gpu="NVIDIA A100",
                   pre_sha256=p.file_sha256(p.pre_path(doc, task["encoder"], task["dataset"], seed)),
                   predictions_sha256=p.file_sha256(prediction),
                   **{f"pre_{m}": .5 for m in p.METRICS}, **{f"post_{m}": score for m in p.METRICS})
        p.atomic_json(p.result_path(doc, task, seed), row)
    counts = collect(doc, tmp_path / "reports")
    assert counts == dict(expected=144, verified=2, missing=142, invalid=0)
    import csv
    with (tmp_path / "reports/summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["n_seeds"] == "2" and rows[0]["status"] == "INCOMPLETE"
    assert float(rows[0]["delta_knn_f1_mean"]) == pytest.approx(.2)
    assert float(rows[0]["delta_knn_f1_sd"]) == pytest.approx(.2 / 2**.5)
    with (tmp_path / "reports/correlations.csv").open() as handle:
        assert all(r["spearman_rho"] == "" for r in csv.DictReader(handle))
    changed = json.loads(prediction.read_text())
    changed["initial_uniformity"] = 0.
    p.atomic_json(prediction, changed)
    counts = collect(doc, tmp_path / "reports")
    assert counts["verified"] == 0 and counts["invalid"] == 2


@pytest.mark.parametrize("task_id", [0, 1, 2, 24, 25, 26])
def test_fit_publishes_paired_scores_with_correct_setup_and_skips_completed(
    tmp_path, source_manifest, monkeypatch, task_id
):
    p = extension()
    runner = importlib.import_module("eval.heldout_extensions.run")
    rt = importlib.import_module("eval.heldout_extensions.runtime")
    doc = make_doc(tmp_path, source_manifest)
    task = doc["tasks"][task_id]
    prepare_artifacts(doc, task["encoder"], task["dataset"])
    p.freeze_predictions(doc, task["encoder"], task["dataset"])
    pl, loggers = ModuleType("lightning"), ModuleType("lightning.pytorch.loggers")
    pl.seed_everything = lambda *a, **kw: None
    experiment = SimpleNamespace(summary={}, finish=lambda: None)
    loggers.WandbLogger = lambda **kw: SimpleNamespace(experiment=experiment)
    monkeypatch.setitem(sys.modules, "lightning", pl)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", loggers)
    data, cp = ModuleType("stable_cp.data"), ModuleType("continued_pretraining")
    calls = []
    def transforms(cfg, **kw):
        assert cfg["normalization"] == task["normalization"]
        assert kw["n_views"] == {"LeJEPA": 8, "DIET": 1, "SimCLR": 2}[task["method"]]
        return "aug", "clean"
    data.create_transforms = transforms
    data.create_train_datamodule = lambda *a, indices: ("datamodule", indices)
    cp.create_optim_config = lambda args, warmup: "optim"
    def train(module, data, args, config, dim, frozen, logger, checkpoint, *, method):
        assert dim == task["embed_dim"] and frozen == 15
        assert args.pool_strategy == task["pool"]
        assert args.accumulate_grad_batches == task["recipe"]["accumulate_grad_batches"]
        assert method == task["method"].lower()
        calls.append("train")
        path = Path(checkpoint)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"checkpoint")
    cp.run_training = train
    monkeypatch.setitem(sys.modules, "stable_cp.data", data)
    monkeypatch.setitem(sys.modules, "continued_pretraining", cp)
    name = task["method"].lower()
    implementation = ModuleType(f"stable_cp.methods.{name}.{name}_cp")
    def setup(model, dim, optim, *a, **kw):
        assert dim == task["embed_dim"] and kw["pool_strategy"] == task["pool"]
        assert kw["num_samples"] == 1000
        calls.append("setup")
        return object()
    setattr(implementation, f"setup_{name}", setup)
    implementation.build_sigreg_loss = lambda args: "sigreg"
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    monkeypatch.setattr(rt, "check_environment", lambda profile: f"NVIDIA {profile.upper()}")
    monkeypatch.setattr(rt, "software", lambda: {"version": "test"})
    monkeypatch.setattr(rt, "dataset_record", lambda *a: {"n_train_actual": 1000})
    monkeypatch.setattr(rt, "load_model", lambda *a: (object(), "cuda:0", {"normalization": task["normalization"]}, "weights"))
    def evaluate(model, device, cfg, args, indices):
        assert indices == list(range(1000))
        assert args.pool_strategy == task["pool"] and args.embed_dim == task["embed_dim"]
        return {m: .7 for m in p.METRICS}, {"raw_scores": {m: .7 for m in p.METRICS}}
    monkeypatch.setattr(rt, "evaluate", evaluate)
    runner.fit_seed(doc, task, 42, cache_dir=tmp_path, num_workers=0)
    row = json.loads(p.result_path(doc, task, 42).read_text())
    assert p.validate_result(doc, task, 42, row) == row
    assert row["pre_knn_f1"] == .5 and row["post_knn_f1"] == .7
    assert row["gpu"] == f"NVIDIA {task['gpu'].upper()}"
    runner.fit_seed(doc, task, 42, cache_dir=tmp_path, num_workers=0)
    assert calls == ["setup", "train"]
