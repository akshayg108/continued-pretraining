import json
import sys
from types import SimpleNamespace

import pytest
import torch

import continued_pretraining as cp


@pytest.mark.parametrize("method", ["diet", "lejepa", "simclr", "mae"])
@pytest.mark.parametrize("pre_only", [False, True])
@pytest.mark.parametrize("with_ft", [False, True])
def test_pipeline_uses_native_normalization_and_same_backbone(
    monkeypatch, tmp_path, method, pre_only, with_ft
):
    calls = []
    backbone = SimpleNamespace(
        num_features=4,
        generation=0,
        pretrained_cfg={"mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25)},
    )
    indices = [2, 4, 6, 8]
    augmented, clean, test = object(), object(), object()
    expected_norm = {"mean": [0.5] * 3, "std": [0.25] * 3}
    experiment = SimpleNamespace(log=lambda *a, **kw: None, summary={}, finish=lambda: None)
    monkeypatch.setattr(cp, "WandbLogger", lambda **kw: SimpleNamespace(experiment=experiment))
    monkeypatch.setattr(cp, "load_backbone", lambda *a, **kw: (backbone, torch.device("cpu")))
    monkeypatch.setattr(
        cp, "get_dataset_config", lambda name: {"input_size": 224, "num_classes": 2}
    )

    def shared(args, config, data_dir):
        assert config["normalization"] == expected_norm
        return object(), test, augmented, clean, indices

    def data(args, config, data_dir, selected, method_config):
        assert config["normalization"] == expected_norm
        assert selected is indices
        return object(), method_config["n_views"]

    def setup(model, dim, optim, *extra, **kwargs):
        assert model is backbone and dim == 4
        assert kwargs["num_samples"] == len(indices)
        return SimpleNamespace(backbone=model)

    def evaluate(model, train_loader, test_loader, device, **kwargs):
        calls.append("pre" if model.generation == 0 else "post")
        assert model is backbone
        assert train_loader is augmented and test_loader is test
        assert kwargs["knn_train_loader"] is clean
        assert kwargs["pool_strategy"] == "map"
        return {
            "knn_f1": 0.5 + model.generation * 0.1,
            "knn_acc": 0.5,
            "linear_pytorch_f1": 0.6,
            "linear_pytorch_acc": 0.6,
        }

    def train(module, data, args, config, dim, freeze, logger, checkpoint):
        calls.append("cp")
        assert module.backbone is backbone
        assert method in str(checkpoint)
        backbone.generation += 1

    monkeypatch.setattr(cp, "_create_shared_eval_data", shared)
    monkeypatch.setattr(cp, "_create_cp_data", data)
    monkeypatch.setattr(cp, "_get_methods", lambda: {method: {"setup": setup, "n_views": 2}})
    monkeypatch.setattr(cp, "zero_shot_eval", evaluate)
    monkeypatch.setattr(cp, "run_training", train)

    def ft_data(args, config, data_dir, transform, selected):
        assert config["normalization"] == expected_norm
        assert selected is indices
        return object()

    def ft(model, *a, prefix, **kwargs):
        assert model is backbone
        calls.append(prefix)
        return {
            f"{prefix}_f1": 0.75,
            f"{prefix}_acc": 0.75,
            f"{prefix}_auroc": 0.8,
            "sft_protocol": "full_ft_v1",
            "sft_trainable_params": 8,
            "sft_total_params": 8,
        }

    monkeypatch.setattr(cp, "_create_sft_data", ft_data)
    monkeypatch.setattr(cp, "sft_evaluate", ft)
    output = tmp_path / "result.json"
    argv = [
        "continued_pretraining.py",
        "--dataset",
        "food101",
        "--backbone",
        "vit_base_patch16_siglip_224.v2_webli",
        "--cp-method",
        method,
        "--n-samples",
        "4",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--cache-dir",
        str(tmp_path / "data"),
        "--results-json",
        str(output),
    ]
    if pre_only:
        argv.append("--no-cp")
    if with_ft:
        argv.append("--pre-cp-sft")
        if not pre_only:
            argv.append("--post-cp-sft")
    monkeypatch.setattr(sys, "argv", argv)
    cp.main()
    expected = ["pre"] + (["pre_sft"] if with_ft else [])
    if not pre_only:
        expected += ["cp", "post"] + (["post_sft"] if with_ft else [])
    assert calls == expected
    result = json.loads(output.read_text())
    assert result["normalization"] == expected_norm
    assert result["pre_knn_f1"] == 0.5
    assert ("post_knn_f1" in result) is not pre_only
    assert not any("sa_lp" in key or "random_init" in key for key in result)
    assert ("pre_sft_f1" in result) == with_ft
    if with_ft:
        assert result["pre_sft_protocol"] == "full_ft_v1"
        assert result["pre_sft_f1"] == 0.75


def test_training_schedule_uses_optimizer_steps():
    args = SimpleNamespace(
        n_samples=1000,
        batch_size=128,
        accumulate_grad_batches=2,
        epochs=150,
        lr=1e-4,
        weight_decay=0.05,
    )
    config = cp.create_optim_config(args, 15)
    assert config["scheduler"]["max_steps"] == 600
    assert config["scheduler"]["warmup_steps"] == 60
