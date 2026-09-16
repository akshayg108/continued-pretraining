"""CPU tests for the isolated frozen checkpoint audit."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from eval.full_ft.run import file_sha256
from eval.siglip_mainrule.protocol import cp_recipe

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "eval/siglip_food101_postcheck.py"
LAUNCH = ROOT / "run/slurm/cp-siglip/post-cp/submit_food101.sh"
ARRAY = ROOT / "run/slurm/cp-siglip/post-cp/food101_v100.sh"
MODEL = "vit_base_patch16_siglip_224.v2_webli"


def audit_module():
    assert MODULE.is_file(), "Food-101 checkpoint audit is missing"
    return importlib.import_module("eval.siglip_food101_postcheck")


def artifacts(base, method="LeJEPA", seed=42, source="mainrule", score=.1):
    if source == "mainrule":
        root = base / "siglip_mainrule_v1"
        ckpt_dir = root / "checkpoints" / method / "food101" / "cp"
    else:
        root = base / "siglip_lejepa_recheck_v1" / "17950124"
        ckpt_dir = root / "checkpoints" / method / "food101" / f"seed{seed}" / "cp"
    ckpt = ckpt_dir / f"food101_{MODEL}_n75750_s{seed}.ckpt"
    result = root / "cp_results" / method / "food101" / f"seed{seed}.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(f"{source}-{method}-{seed}".encode())
    row = dict(dataset="food101", n_samples=75750, backbone=MODEL,
               method=method.lower(), seed=seed, epochs=150, random_init=False, no_cp=False,
               post_knn_f1=score, post_linear_f1=score, post_knn_acc=score,
               post_linear_acc=score)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(row))
    if source == "mainrule":
        receipt = root / "provenance" / method / "food101" / f"seed{seed}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(dict(
            protocol="siglip_mainrule_v1", status="cp_complete", seed=seed,
            cp_recipe=cp_recipe(method, 75750), cp_result_sha256=file_sha256(result),
            checkpoint_sha256=file_sha256(ckpt))))
    return ckpt, result


def test_plan_keeps_low_scores_and_lists_missing_inputs(tmp_path):
    mod = audit_module()
    ckpt, result = artifacts(tmp_path, seed=43, score=.00019421)
    manifest = mod.build_manifest(tmp_path)
    assert len(manifest["tasks"]) == 1
    assert len(manifest["missing"]) == 8
    task = manifest["tasks"][0]
    assert task["seed"] == 43 and task["method"] == "LeJEPA"
    assert task["checkpoint"] == str(ckpt)
    assert task["cp_result_sha256"] == file_sha256(result)
    assert task["checkpoint_sha256"] == file_sha256(ckpt)
    assert task["input_source"] == "mainrule"


def test_recheck_is_explicit_and_never_replaces_mainrule(tmp_path):
    mod = audit_module()
    old, _ = artifacts(tmp_path, seed=43, score=.0002)
    new, _ = artifacts(tmp_path, seed=43, source="recheck", score=.8)
    first = mod.build_manifest(tmp_path)
    second = mod.build_manifest(tmp_path, source="recheck")
    assert first["tasks"][0]["checkpoint"] == str(old)
    assert second["tasks"][0]["checkpoint"] == str(new)
    assert second["tasks"][0]["input_source"] == "recheck-17950124"
    assert len(second["missing"]) == 3


@pytest.mark.parametrize("mutation", ["checkpoint", "result", "receipt"])
def test_modified_inputs_are_rejected(tmp_path, mutation):
    mod = audit_module()
    ckpt, result = artifacts(tmp_path)
    task = mod.build_manifest(tmp_path)["tasks"][0]
    if mutation == "checkpoint":
        ckpt.write_bytes(b"other weights")
    elif mutation == "result":
        result.write_text(result.read_text() + " ")
    else:
        Path(task["receipt"]).write_text("{}")
    with pytest.raises(ValueError, match="changed|mismatch|receipt"):
        mod.validate_inputs(task)


def test_result_without_checkpoint_is_not_a_public_weight_fallback(tmp_path):
    mod = audit_module()
    ckpt, _ = artifacts(tmp_path)
    ckpt.unlink()
    manifest = mod.build_manifest(tmp_path)
    assert not manifest["tasks"] and len(manifest["missing"]) == 9


def test_invalid_existing_result_is_not_silently_excluded(tmp_path):
    mod = audit_module()
    _, result = artifacts(tmp_path)
    row = json.loads(result.read_text())
    row["dataset"] = "octmnist"
    result.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        mod.build_manifest(tmp_path)


def test_shell_syntax_and_dry_run_never_submits(tmp_path):
    audit_module()
    artifacts(tmp_path)
    for script in (LAUNCH, ARRAY):
        check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
    env = dict(os.environ, SIGLIP_POST_OUTPUT_BASE=str(tmp_path),
               PYTHON=sys.executable, SIGLIP_POST_CACHE_DIR=str(tmp_path / "cache"))
    run = subprocess.run(["bash", str(LAUNCH), "--dry-run"], env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "--gres=gpu:v100:1" in run.stdout
    assert "--array=0%9" in run.stdout
    assert "normalization=official" in run.stdout
    assert "cp_training=false ft=false" in run.stdout
    assert "--constraint=80g" not in run.stdout
    assert len(list((tmp_path / "siglip_food101_postcheck_manifests").glob("*.json"))) == 1
    assert not (tmp_path / "siglip_food101_postcheck_official_norm_v1").exists()


@pytest.mark.parametrize("failure", [None, "load", "normalization", "features", "changed"])
def test_frozen_evaluation_uses_cp_weights_and_official_normalization(tmp_path, monkeypatch,
                                                                    failure):
    mod = audit_module()
    ckpt, _ = artifacts(tmp_path)
    task = mod.build_manifest(tmp_path)["tasks"][0]
    events = []
    model = SimpleNamespace(pretrained_cfg={"mean": (.5, .5, .5), "std": (.5, .5, .5)})
    if failure == "normalization":
        model.pretrained_cfg["mean"] = (.485, .456, .406)
    model.requires_grad_ = lambda flag: events.append(("freeze", flag))
    model.eval = lambda: events.append("eval")
    model.to = lambda device: events.append(("to", device.type))
    config = dict(input_size=224, normalization={"mean": [.485, .456, .406],
                  "std": [.229, .224, .225]}, splits=["train", "test", "test"])
    labels = {"lp": np.repeat(np.arange(101), 750),
              "knn": np.repeat(np.arange(101), 750), "test": np.repeat(np.arange(101), 250)}
    features = {k: np.broadcast_to(np.ones((1, 768), dtype=np.float32), (len(v), 768))
                for k, v in labels.items()}
    if failure == "features":
        features["test"] = np.broadcast_to(np.full((1, 768), np.nan), (25250, 768))
    loaders = {k: SimpleNamespace(dataset=SimpleNamespace(dataset=SimpleNamespace(transform=k)))
               for k in labels}
    loaders["test"].dataset = range(25250)

    def load(args, *, img_size, pretrained):
        assert not pretrained and args.backbone == MODEL and img_size == 224
        events.append("architecture_only")
        return model, SimpleNamespace(type="cuda")

    def strict(backbone, path):
        assert backbone is model and Path(path) == ckpt
        if failure == "load":
            raise ValueError("Incomplete backbone state")
        events.append("strict_cp_load")
        return dict(prefix="backbone.", n_tensors=200)

    def create_loaders(args, cfg, cache):
        assert cfg["normalization"] == {"mean": [.5] * 3, "std": [.5] * 3}
        assert args.seed == 42 and args.n_samples == 75750 and args.batch_size == 64
        return "eval_tf", loaders["test"], loaders["lp"], loaders["knn"], range(75750)

    def extract(backbone, loader, device, *, pool_strategy, verbose):
        assert backbone is model and pool_strategy == "map"
        key = next(k for k, v in loaders.items() if v is loader)
        events.append("extract_" + key)
        return features[key], labels[key]

    def knn(train, train_y, test, test_y, *, k):
        assert train is features["knn"] and test is features["test"] and k == 20
        events.append("knn")
        return dict(knn_f1=.81, knn_acc=.82)

    def lp(train, train_y, test, test_y, **kw):
        assert train is features["lp"] and test is features["test"]
        assert kw["lr"] == 1e-3 and kw["min_epochs"] == 150
        assert kw["min_steps"] == 10000 and kw["batch_size"] == 512
        events.append("lp")
        if failure == "changed":
            ckpt.write_bytes(b"changed during evaluation")
        return dict(linear_pytorch_f1=.86, linear_pytorch_acc=.87)

    monkeypatch.setitem(sys.modules, "lightning", SimpleNamespace(
        seed_everything=lambda seed, workers: events.append(("seed", seed))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: "Tesla V100-PCIE-32GB")))
    monkeypatch.setitem(sys.modules, "continued_pretraining", SimpleNamespace(
        load_backbone=load, get_dataset_config=lambda name: config,
        _create_shared_eval_data=create_loaders))
    monkeypatch.setitem(sys.modules, "eval.full_ft.checkpoint", SimpleNamespace(
        discard_native_head=lambda backbone: events.append("discard_head"),
        load_backbone_state=strict))
    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", SimpleNamespace(
        extract_features=extract, knn_evaluate=knn, linear_probe_pytorch_evaluate=lp))
    if failure:
        with pytest.raises(ValueError):
            mod.run_task(task, cache_dir=tmp_path / "cache", outdir=tmp_path / "audit")
        assert not list((tmp_path / "audit").rglob("*.json"))
        if failure in {"load", "normalization"}:
            assert not any(isinstance(e, str) and e.startswith("extract_") for e in events)
        return
    output = mod.run_task(task, cache_dir=tmp_path / "cache", outdir=tmp_path / "audit")
    row = json.loads(output.read_text())
    assert row["status"] == "complete" and row["initialization"] == "cp_checkpoint"
    assert row["cp_training_performed"] is False and row["no_ft"] is True
    assert row["post_knn_f1"] == .81 and row["post_linear_f1"] == .86
    assert row["checkpoint_sha256"] == file_sha256(ckpt)
    assert row["original_post_metrics"]["post_knn_f1"] == .1
    assert row["input_source"] == "mainrule" and row["seed"] == 42
    assert row["normalization"] == {"mean": [.5] * 3, "std": [.5] * 3}
    assert config["normalization"]["mean"][0] == .485
    assert events == [("seed", 42), "architecture_only", "discard_head", "strict_cp_load",
                      ("freeze", False), "eval", ("to", "cuda"), "extract_lp",
                      "extract_test", "extract_knn", "knn", "lp"]
    with pytest.raises(FileExistsError):
        mod.run_task(task, cache_dir=tmp_path / "cache", outdir=tmp_path / "audit")


@pytest.mark.parametrize("arguments", [
    ["--concurrency", "0"], ["--source", "unknown"], ["--methods"],
    ["--source", "recheck", "--methods", "SimCLR"], ["--seeds", "42", "42"],
])
def test_invalid_submission_options_do_not_submit(tmp_path, arguments):
    audit_module()
    env = dict(os.environ, SIGLIP_POST_OUTPUT_BASE=str(tmp_path), PYTHON=sys.executable)
    run = subprocess.run(["bash", str(LAUNCH), "--dry-run", *arguments], env=env,
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert "SUBMIT" not in run.stdout


def test_recheck_dry_run_with_explicit_seed_selection(tmp_path):
    audit_module()
    artifacts(tmp_path, source="recheck", seed=44)
    artifacts(tmp_path, source="recheck", seed=45)
    env = dict(os.environ, SIGLIP_POST_OUTPUT_BASE=str(tmp_path), PYTHON=sys.executable)
    run = subprocess.run(["bash", str(LAUNCH), "--source", "recheck", "--seeds", "44",
                          "--concurrency", "2", "--dry-run"], env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "--array=0%2" in run.stdout and "seed=44" in run.stdout
    assert "seed=45" not in run.stdout and "recheck-17950124" in run.stdout


def test_staging_failure_preserves_checkpoint_and_shared_cache(tmp_path):
    mod = audit_module()
    ckpt, _ = artifacts(tmp_path)
    manifest = tmp_path / "manifest.json"
    mod.write_new_json(manifest, mod.build_manifest(tmp_path))
    source = tmp_path / "shared/stable_datasets/processed/food101"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"keep shared data")
    stage = tmp_path / "stage"
    stage.mkdir()
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\nimport os,pathlib,sys\n"
        "if sys.argv[1] == '-c': sys.exit(0)\n"
        "cache = pathlib.Path(sys.argv[sys.argv.index('--cache-dir') + 1])\n"
        "assert (cache / 'stable_datasets/processed/food101/sample.bin').read_bytes() == b'keep shared data'\n"
        "pathlib.Path(os.environ['STAGING_MARKER']).write_text(str(cache))\n"
        "sys.exit(9)\n")
    fake.chmod(0o755)
    marker = tmp_path / "marker.txt"
    env = dict(os.environ, SIGLIP_POST_SKIP_ENV_SETUP="1", SIGLIP_POST_REPO_ROOT=str(ROOT),
               SIGLIP_POST_MANIFEST=str(manifest), SIGLIP_POST_CACHE_DIR=str(tmp_path / "shared"),
               SIGLIP_POST_OUTPUT_BASE=str(tmp_path / "audit"), SLURM_JOB_ID="123",
               SLURM_ARRAY_JOB_ID="122", SLURM_ARRAY_TASK_ID="0", PYTHON=str(fake),
               TMPDIR=str(stage), STAGING_MARKER=str(marker))
    run = subprocess.run(["bash", str(ARRAY)], env=env, capture_output=True, text=True)
    assert run.returncode == 9, run.stdout + run.stderr
    private = Path(marker.read_text())
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"keep shared data"
    assert ckpt.read_bytes() == b"mainrule-LeJEPA-42"
    assert not list((tmp_path / "audit").rglob("*.json"))
