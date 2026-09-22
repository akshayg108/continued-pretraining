"""Execution contracts with synthetic features; no downloads or GPU required."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime
from eval.heldout_cp.run import training_args


def test_all_training_arguments_match_locked_recipe(tmp_path):
    for task in p.build_manifest(tmp_path)["tasks"]:
        args = training_args(task, 43, tmp_path, 4, tmp_path / "attempt")
        for key, value in task["recipe"].items():
            assert getattr(args, key) == value
        assert args.n_samples == 1000
        assert args.num_trained_blocks == 2
        assert args.seed == 43
        assert args.pool_strategy == "cls"
        assert args.normalization_mode == "pretrained"
        assert not args.resume and not args.random_init
        assert not args.pre_cp_sft and not args.post_cp_sft


def test_evaluation_restores_device_and_reuses_exact_indices(monkeypatch):
    seeds, calls = [], []
    pl = ModuleType("lightning")
    pl.seed_everything = lambda seed, **kw: seeds.append(seed)
    monkeypatch.setitem(sys.modules, "lightning", pl)
    data = ModuleType("stable_cp.data")
    config = {"normalization": p.EXPECTED_NORMALIZATIONS["CLIP"]}
    indices = list(range(1000))

    def transforms(cfg, **kwargs):
        assert cfg == config
        assert kwargs == {"n_views": 1, "strong_aug": False}
        return "augmented", "clean"

    def loaders(args, cfg, train_tf, test_tf, cache_dir, **kwargs):
        assert kwargs["indices"] is indices
        assert cfg is config and test_tf == "clean"
        calls.append(train_tf)
        return "test", train_tf, indices

    data.create_transforms = transforms
    data.create_eval_loaders = loaders
    monkeypatch.setitem(sys.modules, "stable_cp.data", data)
    evaluation = ModuleType("stable_cp.evaluation.zero_shot_eval")

    def extract(model, loader, device, **kwargs):
        assert model.device == device == "cuda:0"
        assert kwargs == {"pool_strategy": "cls"}
        size = 17 if loader == "test" else 1000
        return np.ones((size, 768)), np.arange(size) % 8

    def knn(train_x, train_y, test_x, test_y, **kwargs):
        assert kwargs == {"k": 20} and len(train_x) == 1000
        assert len(test_x) == 17
        return dict(knn_f1=0.6, knn_acc=0.7)

    def lp(train_x, train_y, test_x, test_y, **kwargs):
        assert kwargs == dict(
            device="cuda:0", lr=1e-3, min_epochs=150, min_steps=10000, batch_size=512
        )
        assert len(train_x) == 1000 and len(test_x) == 17
        return dict(linear_pytorch_f1=0.8, linear_pytorch_acc=0.9)

    evaluation.extract_features = extract
    evaluation.knn_evaluate = knn
    evaluation.linear_probe_pytorch_evaluate = lp
    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", evaluation)

    class Model:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

    args = runtime.evaluation_args("CLIP", "aid", 43, "/cache", 0)
    scores = runtime.evaluate(Model(), "cuda:0", config, args, indices)
    assert scores == dict(knn_f1=0.6, knn_acc=0.7, linear_f1=0.8, linear_acc=0.9)
    assert calls == ["augmented", "clean"]
    assert seeds == [43, 43]


@pytest.mark.parametrize("encoder", p.ENCODER_ORDER)
def test_model_loading_requires_native_mean_std(monkeypatch, encoder):
    from eval import precp_official_norm

    module = ModuleType("continued_pretraining")
    native = p.EXPECTED_NORMALIZATIONS[encoder]
    model = SimpleNamespace(pretrained_cfg=native, num_features=768)
    model.to = lambda device: model

    def load(args, img_size, pretrained):
        assert pretrained is True and img_size == 224
        return model, "cuda:0"

    module.load_backbone = load
    module.get_dataset_config = lambda dataset: {
        "normalization": "incorrect_dataset_default"
    }
    module.configure_normalization = lambda config, model, mode: {
        "normalization": native
    }
    monkeypatch.setitem(sys.modules, "continued_pretraining", module)
    monkeypatch.setattr(
        precp_official_norm, "_weights_hash", lambda model: "public-weights"
    )
    args = runtime.evaluation_args(encoder, "bloodmnist", 42, "/cache", 0)
    _, _, config, weights = runtime.load_model(encoder, args)
    assert config["normalization"] == native and weights == "public-weights"
    model.pretrained_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        runtime.load_model(encoder, args)


def test_shard_fingerprint_detects_image_change_with_identical_metadata(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    metadata = source_dir / "_metadata.json"
    shard = source_dir / "shard-000.arrow"
    metadata.write_text('{"num_rows": 1, "shard_filenames": ["shard-000.arrow"]}')
    shard.write_bytes(b"red image bytes")
    source = SimpleNamespace(_shard_dir=source_dir, _shard_paths=[shard])
    original = runtime.source_fingerprint(source, tmp_path)
    sha256 = runtime.file_sha256
    reads = []

    def track(path):
        reads.append(path)
        return sha256(path)

    monkeypatch.setattr(runtime, "file_sha256", track)
    assert runtime.source_fingerprint(source, tmp_path) == original
    assert not reads
    shard.write_bytes(b"new image bytes")
    changed = runtime.source_fingerprint(source, tmp_path)
    assert changed != original
    assert shard.resolve() in reads
