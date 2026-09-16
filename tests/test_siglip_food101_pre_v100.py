"""CPU-only checks for the isolated Food-101 pretrained baseline audit."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/slurm/cp-siglip/pre-cp/food101_v100.sh"


def script_text():
    assert SCRIPT.is_file(), "The isolated V100 baseline launcher is missing."
    return SCRIPT.read_text()


def test_script_syntax_and_resources():
    text = script_text()
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for line in ("#SBATCH --gres=gpu:v100:1", "#SBATCH --array=0-2%3",
                 "#SBATCH --account=civil", "#SBATCH --partition=nvidia"):
        assert line in text
    assert "--constraint=80g" not in text


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("task_id,seed", [(0, 42), (1, 43), (2, 44)])
def test_dry_run_selects_one_seed_without_loading_gpu_environment(tmp_path, task_id, seed,
                                                                official):
    script_text()
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task_id),
               SIGLIP_PRE_OUTPUT_BASE=str(tmp_path / "outputs"))
    command = ["bash", str(SCRIPT), "--dry-run"]
    if official:
        command.append("--official-normalization")
    result = subprocess.run(command, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"dataset=food101 seed={seed}" in result.stdout
    assert "initialization=public_pretrained" in result.stdout
    assert "evaluators=knn,pytorch_lp" in result.stdout
    assert "gpu=v100" in result.stdout
    mode = "official" if official else "dataset"
    protocol = "siglip_food101_precheck_official_norm_v1" if official else "siglip_food101_precheck_v1"
    assert f"normalization={mode}" in result.stdout
    assert f"/{protocol}/" in result.stdout
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("task_id", ["-1", "3", "not-an-index"])
def test_invalid_task_id_is_rejected(task_id):
    script_text()
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                            env=dict(os.environ, SLURM_ARRAY_TASK_ID=task_id),
                            capture_output=True, text=True)
    assert result.returncode == 2


def test_unknown_option_is_rejected_even_with_dry_run():
    result = subprocess.run(["bash", str(SCRIPT), "--official-normalization", "--typo", "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 2


def test_python_uses_shared_loaders_and_only_frozen_evaluators():
    text = script_text()
    source = text.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    named = {node.func.id: node for node in calls if isinstance(node.func, ast.Name)}
    assert {"_create_shared_eval_data", "load_backbone", "knn_evaluate",
            "linear_probe_pytorch_evaluate"} <= named.keys()
    assert not {"run_training", "sft_evaluate", "zero_shot_eval", "kmeans_evaluate",
                "linear_probe_evaluate"} & named.keys()
    load = named["load_backbone"]
    assert any(k.arg == "pretrained" and k.value.value is True for k in load.keywords)
    assert len([n for n in calls if isinstance(n.func, ast.Name)
                and n.func.id == "extract_features"]) == 3
    lp = named["linear_probe_pytorch_evaluate"]
    settings = {k.arg: k.value.value for k in lp.keywords if isinstance(k.value, ast.Constant)}
    assert settings["min_epochs"] == 150
    assert settings["min_steps"] == 10000
    assert settings["batch_size"] == 512
    assert settings["lr"] == 1e-3
    assert 'pool_strategy="map"' in source
    assert 'n_samples=75750' in source
    assert 'seed_everything(args.seed, workers=True)' in source
    assert '.requires_grad_(False)' in source
    assert 'open("x"' in source


def test_dataset_staging_uses_arithmetic_and_private_cache():
    text = script_text()
    assert 'NEED_KB=$((SOURCE_KB + 5 * 1024 * 1024))' in text
    assert 'mktemp -d' in text
    assert 'rsync -a' in text
    assert "trap 'rm -rf -- \"$LOCAL_CACHE\"' EXIT" in text
    assert 'Insufficient node-local storage' in text


@pytest.mark.parametrize("mode,pretrained_norm,invalid", [
    ("dataset", {}, False),
    ("official", {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}, False),
    ("official", {}, True),
    ("official", {"mean": (0.5, 0.5, 0.5), "std": (0.229, 0.224, 0.225)}, True),
])
def test_inline_evaluation_dispatch_and_json_schema(tmp_path, monkeypatch, mode,
                                                   pretrained_norm, invalid):
    source = script_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    events = []
    model = SimpleNamespace(pretrained_cfg={"test": "public-weights", **pretrained_norm})
    dataset_norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ds_cfg = dict(input_size=224, normalization=dataset_norm, splits=["train", "test", "test"])
    expected_norm = ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
                     if mode == "official" else dataset_norm)
    protocol = ("siglip_food101_precheck_official_norm_v1" if mode == "official"
                else "siglip_food101_precheck_v1")
    model.requires_grad_ = lambda value: events.append(("requires_grad", value))
    model.eval = lambda: events.append("eval")
    model.to = lambda device: events.append(("device", device.type))
    labels = {
        "lp": np.repeat(np.arange(101), 750),
        "knn": np.repeat(np.arange(101), 750),
        "test": np.repeat(np.arange(101), 250),
    }
    loaders = {
        "lp": SimpleNamespace(dataset=SimpleNamespace(dataset=SimpleNamespace(transform="train_tf"))),
        "knn": SimpleNamespace(), "test": SimpleNamespace(dataset=range(25250)),
    }
    features = {key: np.broadcast_to(np.ones((1, 768), dtype=np.float32), (len(value), 768))
                for key, value in labels.items()}

    def load(args, *, img_size, pretrained):
        assert args.backbone == "vit_base_patch16_siglip_224.v2_webli"
        assert args.seed == 43 and args.pool_strategy == "map"
        assert pretrained and img_size == 224
        events.append("load_public")
        return model, SimpleNamespace(type="cuda")

    def create_loaders(args, config, cache_dir):
        assert config["normalization"] == expected_norm
        assert config["input_size"] == ds_cfg["input_size"]
        assert config["splits"] == ds_cfg["splits"]
        return "eval_tf", loaders["test"], loaders["lp"], loaders["knn"], range(75750)

    def extract(backbone, loader, device, *, pool_strategy, verbose):
        assert backbone is model and pool_strategy == "map"
        key = next(key for key, value in loaders.items() if loader is value)
        events.append(f"extract_{key}")
        return features[key], labels[key]

    def knn(train, train_labels, test, test_labels, *, k):
        assert train is features["knn"] and test is features["test"] and k == 20
        events.append("knn")
        return dict(knn_f1=.34, knn_acc=.35)

    def lp(train, train_labels, test, test_labels, **kwargs):
        assert train is features["lp"] and test is features["test"]
        assert kwargs["batch_size"] == 512 and kwargs["min_epochs"] == 150
        events.append("lp")
        return dict(linear_pytorch_f1=.48, linear_pytorch_acc=.49)

    monkeypatch.setitem(sys.modules, "lightning", SimpleNamespace(
        seed_everything=lambda seed, workers: events.append(("seed", seed, workers))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(get_device_name=lambda index: "Tesla V100")))
    monkeypatch.setitem(sys.modules, "continued_pretraining", SimpleNamespace(
        load_backbone=load,
        get_dataset_config=lambda name: ds_cfg,
        _create_shared_eval_data=create_loaders))
    monkeypatch.setitem(sys.modules, "stable_cp", SimpleNamespace(
        __file__=str(ROOT / "stable_cp/__init__.py")))
    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", SimpleNamespace(
        extract_features=extract, knn_evaluate=knn, linear_probe_pytorch_evaluate=lp))
    monkeypatch.setattr(sys, "argv", ["-", str(tmp_path / "cache"), str(tmp_path), "43",
                                     mode, protocol])
    monkeypatch.chdir(ROOT)
    if invalid:
        with pytest.raises(ValueError, match="normalization"):
            exec(compile(source, str(SCRIPT), "exec"), {"__name__": "__main__"})
        assert not (tmp_path / "seed43.json").exists()
        assert "extract_lp" not in events
        return
    exec(compile(source, str(SCRIPT), "exec"), {"__name__": "__main__"})
    row = json.loads((tmp_path / "seed43.json").read_text())
    assert row["no_cp"] and row["no_ft"] and row["status"] == "complete"
    assert row["pre_knn_f1"] == .34 and row["pre_linear_f1"] == .48
    assert row["n_samples"] == 75750 and row["n_test"] == 25250
    assert row["code_sha256"] and row["train_indices_sha256"]
    assert row["protocol"] == protocol
    assert row["normalization_mode"] == mode
    assert row["normalization"] == expected_norm
    assert row["dataset_normalization"] == dataset_norm
    assert ds_cfg["normalization"] == dataset_norm
    assert events == [("seed", 43, True), "load_public", ("requires_grad", False),
                      "eval", ("device", "cuda"), "extract_lp", "extract_test",
                      "extract_knn", "knn", "lp"]


def test_staging_cleans_only_private_copy_even_when_evaluation_fails(tmp_path):
    script_text()
    source = tmp_path / "shared/stable_datasets/processed/food101"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"sample")
    stage = tmp_path / "stage"
    stage.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport os,pathlib,sys\n"
        "if sys.argv[1] == '-c': sys.exit(0)\n"
        "cache = pathlib.Path(sys.argv[2])\n"
        "assert (cache / 'stable_datasets/processed/food101/sample.bin').read_bytes() == b'sample'\n"
        "pathlib.Path(os.environ['STAGING_MARKER']).write_text(str(cache))\n"
        "sys.exit(9)\n")
    fake_python.chmod(0o755)
    marker = tmp_path / "marker.txt"
    env = dict(os.environ, SIGLIP_PRE_SKIP_ENV_SETUP="1", SIGLIP_PRE_REPO_ROOT=str(ROOT),
               SIGLIP_PRE_CACHE_DIR=str(tmp_path / "shared"),
               SIGLIP_PRE_OUTPUT_BASE=str(tmp_path / "outputs"),
               SLURM_JOB_ID="123", SLURM_ARRAY_JOB_ID="122", SLURM_ARRAY_TASK_ID="0",
               PYTHON=str(fake_python), TMPDIR=str(stage), STAGING_MARKER=str(marker))
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 9, result.stdout + result.stderr
    private = Path(marker.read_text())
    assert private.parent == stage and not private.exists()
    assert (source / "sample.bin").read_bytes() == b"sample"
    assert not list((tmp_path / "outputs").rglob("*.json"))
