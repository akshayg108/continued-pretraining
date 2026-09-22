"""Model-specific execution contracts with synthetic features."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from eval.heldout_cp import protocol as original
from eval.heldout_extensions import protocol as p


def runtime():
    assert (p.ROOT / "eval/heldout_extensions/runtime.py").is_file(), "Extension runtime is missing"
    return importlib.import_module("eval.heldout_extensions.runtime")


@pytest.mark.parametrize("encoder", p.ENCODER_ORDER)
def test_loader_checks_native_normalization_and_feature_dimension(monkeypatch, encoder):
    rt = runtime()
    from eval import precp_official_norm
    spec = p.ENCODERS[encoder]
    module = ModuleType("continued_pretraining")
    model = SimpleNamespace(pretrained_cfg=spec["normalization"], num_features=spec["embed_dim"])
    model.to = lambda device: model

    def load(args, *, img_size, pretrained):
        assert args.backbone == spec["model_id"]
        assert args.pool_strategy == spec["pool"]
        assert img_size == 224 and pretrained
        return model, "cuda:0"

    module.load_backbone = load
    module.get_dataset_config = lambda dataset: {}
    module.configure_normalization = lambda config, model, mode: dict(normalization=spec["normalization"])
    monkeypatch.setitem(sys.modules, "continued_pretraining", module)
    monkeypatch.setattr(precp_official_norm, "_weights_hash", lambda model: "weights")
    args = rt.evaluation_args(encoder, "aid", 42, "/cache", 0)
    assert rt.load_model(encoder, args)[3] == "weights"
    model.num_features = 100
    with pytest.raises(ValueError, match="dimension"):
        rt.load_model(encoder, args)
    model.num_features = spec["embed_dim"]
    model.pretrained_cfg = dict(mean=[0]*3, std=[1]*3)
    with pytest.raises(ValueError, match="normalization"):
        rt.load_model(encoder, args)


@pytest.mark.parametrize("encoder", p.ENCODER_ORDER)
def test_evaluation_uses_correct_pool_dimension_and_roundoff_audit(monkeypatch, encoder):
    rt = runtime()
    spec = p.ENCODERS[encoder]
    seeds, seen = [], []
    pl = ModuleType("lightning")
    pl.seed_everything = lambda seed, **kw: seeds.append(seed)
    monkeypatch.setitem(sys.modules, "lightning", pl)
    data = ModuleType("stable_cp.data")
    data.create_transforms = lambda *a, **kw: ("augmented", "clean")
    indices = list(range(1000))

    def loaders(args, config, train_tf, test_tf, cache_dir, **kw):
        assert kw["indices"] is indices and test_tf == "clean"
        seen.append(train_tf)
        return "test", train_tf, indices

    data.create_eval_loaders = loaders
    monkeypatch.setitem(sys.modules, "stable_cp.data", data)
    evaluation = ModuleType("stable_cp.evaluation.zero_shot_eval")
    def extract(model, loader, device, *, pool_strategy):
        assert model.device == device == "cuda:0"
        assert pool_strategy == spec["pool"]
        n = 148 if loader == "test" else 1000
        return np.ones((n, spec["embed_dim"])), np.arange(n) % 30

    def knn(x, y, tx, ty, *, k):
        assert len(x) == 1000 and len(tx) == 148 and k == 20
        return dict(knn_f1=1.0000001192092896, knn_acc=1.)

    def lp(x, y, tx, ty, **kw):
        assert kw == dict(device="cuda:0", lr=1e-3, min_epochs=150, min_steps=10000, batch_size=512)
        return dict(linear_pytorch_f1=.9, linear_pytorch_acc=.95)

    evaluation.extract_features, evaluation.knn_evaluate = extract, knn
    evaluation.linear_probe_pytorch_evaluate = lp
    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", evaluation)
    model = SimpleNamespace(device="cpu")
    model.to = lambda device: setattr(model, "device", device)
    scores, audit = rt.evaluate(model, "cuda:0", {}, rt.evaluation_args(encoder, "aid", 43, "/cache", 0), indices)
    assert scores["knn_f1"] == 1 and audit["raw_scores"]["knn_f1"] > 1
    assert audit["tolerance"] == 1e-6
    assert seeds == [43, 43] and seen == ["augmented", "clean"]


@pytest.mark.parametrize("profile,name,valid", [
    ("a100", "NVIDIA A100-SXM4-40GB", True),
    ("a100", "NVIDIA A100-SXM4-80GB", True),
    ("v100", "Tesla V100-SXM2-32GB", True),
    ("a100", "Tesla V100-SXM2-32GB", False),
    ("v100", "NVIDIA A100-SXM4-40GB", False),
    ("a100", "NVIDIA A100 MIG 1g.5gb", False),
])
def test_gpu_profile_without_80gb_restriction(monkeypatch, profile, name, valid):
    rt = runtime()
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 get_device_name=lambda i: name)
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True))
    torch.set_float32_matmul_precision = lambda value: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    datasets = ModuleType("stable_datasets")
    datasets.images = SimpleNamespace(**{n: object() for n in (
        "MedMNIST", "AID", "RESISC45", "StanfordDogs", "JenaFlowers30", "Flavia", "IP102")})
    monkeypatch.setitem(sys.modules, "stable_datasets", datasets)
    if valid:
        assert rt.check_environment(profile) == name
        assert not torch.backends.cuda.matmul.allow_tf32
    else:
        with pytest.raises(RuntimeError):
            rt.check_environment(profile)


def test_training_args_and_serial_seeds(monkeypatch, tmp_path):
    assert (p.ROOT / "eval/heldout_extensions/run.py").is_file(), "Extension training runner is missing"
    run = importlib.import_module("eval.heldout_extensions.run")
    task = dict(task_id=0, preparation_id=0, encoder="SigLIP", dataset="aid", method="SimCLR",
                **p.ENCODERS["SigLIP"], recipe=original.cp_recipe("SimCLR", 1000), seeds=[42, 43, 44], gpu="v100")
    args = run.training_args(task, 42, tmp_path, 0, tmp_path)
    assert args.pool_strategy == "map" and args.normalization_mode == "pretrained"
    assert args.skip_baseline and not args.resume and not args.random_init
    assert not args.pre_cp_sft and not args.post_cp_sft
    calls = []
    monkeypatch.setattr(p, "freeze_predictions", lambda *a: None)
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    run.run_task(tmp_path / "manifest", {}, task, cache_dir=tmp_path, num_workers=0)
    assert len(calls) == 3
    for (command, kwargs), seed in zip(calls, task["seeds"]):
        assert command[:5] == [sys.executable, "-u", "-m", "eval.heldout_extensions", "fit"]
        assert command[command.index("--seed")+1] == str(seed)
        assert kwargs["check"] is True
