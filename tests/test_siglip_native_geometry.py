"""Protocol checks for the native-normalization figure geometry supplement."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "run/slurm/cp-siglip/native-geometry"


def runner():
    assert (ROOT / "eval/siglip_native_geometry.py").exists(), "Geometry runner is missing"
    return importlib.import_module("eval.siglip_native_geometry")


def test_plan_covers_all_targets_and_seed_dependent_galaxy_splits():
    module = runner()
    tasks = module.build_plan()
    assert len(tasks) == 17
    assert len({t["dataset"] for t in tasks}) == 15
    assert len({(t["dataset"], t["seed"]) for t in tasks}) == 17
    for task in tasks:
        assert task["n_samples"] == module.DATASET_META[task["dataset"]][2]
        assert task["seed"] in ((42, 43, 44) if task["dataset"] == "galaxy10" else (42,))
    expected = module.load_expected_uniformity()
    assert set(expected) == {(t["dataset"], t["seed"]) for t in tasks}
    assert expected[("food101", 42)] == pytest.approx(-1.45527317)
    assert expected[("galaxy10", 43)] == pytest.approx(-.23977574)


def test_geometry_bank_uses_the_historical_stratified_sample():
    from sklearn.model_selection import train_test_split
    module = runner()
    labels = np.repeat(np.arange(10), 601)
    expected, _ = train_test_split(np.arange(len(labels)), train_size=5000,
                                   stratify=labels, random_state=42)
    assert np.array_equal(module.bank_indices(labels), np.sort(expected))
    assert np.array_equal(module.bank_indices(np.arange(20)), np.arange(20))


def test_feature_validation_rejects_wrong_shape_nonfinite_and_zero_norms():
    module = runner()
    features = np.ones((4, 768), dtype=np.float32)
    module.check_features(features, np.arange(4), 4)
    for bad in (features[:, :2], np.zeros_like(features), features * np.nan):
        with pytest.raises(ValueError):
            module.check_features(bad, np.arange(4), 4)


def test_native_uniformity_must_reproduce_the_completed_baseline():
    module = runner()
    module.check_uniformity("dtd", 42, -1.53213280)
    with pytest.raises(ValueError, match="baseline"):
        module.check_uniformity("dtd", 42, -.25)
    with pytest.raises(ValueError):
        module.check_uniformity("galaxy10", 45, -.25)


def test_npz_publication_preserves_raw_lengths_and_never_overwrites(tmp_path):
    module = runner()
    path = tmp_path / "raw.npz"
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    module.write_new_npz(path, bank_X=features, bank_y=np.arange(4))
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["bank_X"], features)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_new_npz(path, bank_X=features * 2)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_submit_dry_run_is_one_v100_job_with_no_training(tmp_path):
    script = SCRIPTS / "submit.sh"
    assert script.is_file(), "Submission script is missing"
    result = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True,
                            text=True, env=dict(os.environ, PYTHON=sys.executable,
                                              SIGLIP_GEOMETRY_OUTPUT_BASE=str(tmp_path / "outputs")))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--gres=gpu:v100:1" in result.stdout
    assert "--array" not in result.stdout
    assert "17 target splits" in result.stdout
    assert "no_cp=True no_ft=True no_lp=True" in result.stdout
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("name", ["submit.sh", "run.sh"])
def test_shell_syntax_and_unknown_arguments(name):
    script = SCRIPTS / name
    assert script.is_file()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(["bash", str(script), "--invalid"], capture_output=True, text=True)
    assert result.returncode == 2


def test_export_refuses_incomplete_runs(tmp_path):
    module = runner()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.export_results(tmp_path)
    assert not (tmp_path / "geometry.csv").exists()


def test_geometry_matches_existing_metrics_and_dense_uniformity(monkeypatch):
    module = runner()
    # Only the registry import needs a stand-in; every metric below runs for real.
    if importlib.util.find_spec("stable_datasets") is None:
        names = ("MedMNIST", "Galaxy10Decal", "Food101", "FGVCAircraft", "EuroSAT",
                 "PlantVillage", "DTD", "Cars196", "CUB200", "Flowers102", "OxfordPet")
        monkeypatch.setitem(sys.modules, "stable_datasets", SimpleNamespace(
            images=SimpleNamespace(**{name: object for name in names})))
    metrics = importlib.import_module("eval.utils.geometry_metrics")
    rng = np.random.RandomState(10)
    features = rng.normal(size=(80, 768)).astype(np.float32)
    reference = rng.normal(size=(91, 768)).astype(np.float32)
    values, indices = module.geometry_values(features, np.arange(80) % 4, reference, device="cpu")
    assert np.array_equal(indices, np.arange(80))
    assert values["uniformity_t2"] == pytest.approx(metrics.wang_isola_uniformity(features), abs=2e-6)
    for key, expected in metrics.mmd_rbf_components(features, reference).items():
        assert values[key] == pytest.approx(expected, abs=1e-7)
    assert values["neighbor_overlap_k50"] == metrics.neighbor_overlap(features, reference, k=50)
    assert values["neighbor_overlap_k20"] == metrics.neighbor_overlap(features, reference, k=20)
    norms = np.linalg.norm(features, axis=1)
    assert values["l2_norm_mean"] == pytest.approx(norms.mean())
    assert values["l2_norm_cv"] == pytest.approx(norms.std() / norms.mean())


def make_complete_export(module, outdir):
    from eval.precp_official_norm import write_new_json

    common = dict(protocol=module.PROTOCOL, status="complete", encoder="SigLIP",
                  backbone=module.MODEL_ID, pool_strategy="map", initialization="public_pretrained",
                  no_cp=True, no_ft=True, no_lp=True, feature_precision="float32",
                  normalization={"mean": [.5] * 3, "std": [.5] * 3}, weights_sha256="synthetic-weights")
    reference_path = outdir / "features/imagenet.npz"
    module.write_new_npz(reference_path, bank_X=np.ones((5000, 768), dtype=np.float32),
                          bank_y=np.arange(5000) % 1000)
    reference_hash = module.file_hash(reference_path)
    write_new_json(outdir / "imagenet.json", dict(common, dataset="imagenet", n_samples=5000,
                   feature_file="features/imagenet.npz", feature_file_sha256=reference_hash))
    for task in module.build_plan():
        n_bank = min(task["n_samples"], 5000)
        name = f"features/{task['dataset']}__seed{task['seed']}.npz"
        features = np.ones((n_bank, 768), dtype=np.float32)
        module.write_new_npz(outdir / name, bank_X=features, bank_y=np.arange(n_bank),
                              full_norms=np.full(task["n_samples"], np.sqrt(768), dtype=np.float32))
        values = {key: 0. for key in module.FIELDS}
        values.update(l2_norm_mean=float(np.sqrt(768)), full_l2_norm_mean=float(np.sqrt(768)),
                      uniformity_t2=module.load_expected_uniformity()[(task["dataset"], task["seed"])],
                      uniformity_t2_subset=-1., mmd_gamma=1.)
        record = dict(common, **task, **values, n_bank=n_bank, reference_feature_sha256=reference_hash,
                      feature_file=name, feature_file_sha256=module.file_hash(outdir / name))
        write_new_json(outdir / "results" / f"{task['dataset']}__seed{task['seed']}.json", record)


def test_export_averages_only_matching_native_extractions_and_checks_files(tmp_path):
    module = runner()
    make_complete_export(module, tmp_path)
    rows = module.export_results(tmp_path)
    assert len(rows) == 15
    galaxy = next(row for row in rows if row["dataset"] == "galaxy10")
    assert galaxy["n_seeds"] == 3
    assert galaxy["uniformity_t2"] == pytest.approx(np.mean([-.24035185, -.23977574, -.24034768]))
    path = tmp_path / "results/food101__seed42.json"
    record = json.loads(path.read_text())
    record["normalization"] = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="normalization"):
        module.export_results(tmp_path)
    record["normalization"] = {"mean": [.5] * 3, "std": [.5] * 3}
    path.write_text(json.dumps(record))
    (tmp_path / record["feature_file"]).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checksum"):
        module.export_results(tmp_path)


def test_extraction_uses_one_public_model_and_native_transform_for_reference_and_target(tmp_path, monkeypatch):
    import torch
    module = runner()
    events = []
    normalization = {"mean": [.5] * 3, "std": [.5] * 3}
    task = dict(dataset="breastmnist", seed=42, n_samples=8)
    monkeypatch.setattr(module, "DATASET_META", {"breastmnist": ("BreastMNIST", "breast", 8)})
    monkeypatch.setattr(module, "build_plan", lambda: [task])
    monkeypatch.setattr(module, "load_expected_uniformity", lambda: {("breastmnist", 42): -.34638163})
    monkeypatch.setattr(module, "check_gpu", lambda: "Tesla V100")
    monkeypatch.setattr(module, "_weights_hash", lambda model: "frozen-public-weights")
    monkeypatch.setattr(module, "_provenance", lambda: {"code_sha256": {}})
    cache = tmp_path / "cache"
    (cache / "stable_datasets/processed/breast").mkdir(parents=True)
    imagenet = tmp_path / "imagenet_val"
    imagenet.mkdir()
    model = SimpleNamespace(pretrained_cfg=normalization,
                            requires_grad_=lambda flag: events.append(("requires_grad", flag)),
                            eval=lambda: events.append("eval"), to=lambda device: None)

    def load_backbone(args, img_size, pretrained):
        events.append(("model", args.backbone, img_size, pretrained))
        return model, torch.device("cuda")

    class BaseTrain:
        hf_dataset = SimpleNamespace(_fingerprint="fixed-train")

        def __len__(self):
            return 8

    clean = SimpleNamespace(dataset=SimpleNamespace(dataset=BaseTrain()))

    def eval_data(args, cfg, cache_dir):
        assert cfg["normalization"] == normalization
        assert args.seed == 42 and args.n_samples == 8
        events.append("clean-native-transform")
        return "native-eval", object(), object(), clean, list(range(8))

    monkeypatch.setitem(sys.modules, "lightning", SimpleNamespace(seed_everything=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "continued_pretraining", SimpleNamespace(
        _create_shared_eval_data=eval_data, get_dataset_config=lambda name: {"num_classes": 2},
        load_backbone=load_backbone))

    def transforms(cfg, n_views, strong_aug):
        assert cfg["normalization"] == normalization
        assert cfg["input_size"] == 224
        return "unused-train", "native-reference-transform"

    monkeypatch.setitem(sys.modules, "stable_cp.data", SimpleNamespace(create_transforms=transforms))

    def reference_loader(path, transform, workers):
        assert transform == "native-reference-transform"
        return "imagenet-loader", np.arange(5000), {"n_samples": 5000}

    monkeypatch.setattr(module, "load_reference", reference_loader)

    def extract(backbone, loader, device, pool_strategy):
        assert backbone is model and pool_strategy == "map"
        n = 5000 if loader == "imagenet-loader" else 8
        events.append(("extract", n))
        return np.ones((n, 768), dtype=np.float32), np.arange(n) % 2

    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", SimpleNamespace(extract_features=extract))
    values = dict.fromkeys(module.FIELDS, 0.)
    values.update(uniformity_t2=-.34638163, uniformity_t2_subset=-.34638163,
                  l2_norm_mean=float(np.sqrt(768)), full_l2_norm_mean=float(np.sqrt(768)), mmd_gamma=1.)
    monkeypatch.setattr(module, "geometry_values", lambda *args, **kwargs: (values, np.arange(8)))
    output = tmp_path / "result"
    module.run(cache, imagenet, output, num_workers=0)
    assert events.count(("model", module.MODEL_ID, 224, True)) == 1
    assert ("requires_grad", False) in events
    assert [e for e in events if isinstance(e, tuple) and e[0] == "extract"] == [("extract", 5000), ("extract", 8)]
    saved = json.loads((output / "results/breastmnist__seed42.json").read_text())
    assert saved["normalization"] == normalization
    assert saved["weights_sha256"] == "frozen-public-weights"
    assert saved["no_cp"] and saved["no_ft"] and saved["no_lp"]
    assert (output / "geometry.csv").exists()
