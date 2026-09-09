"""Exercise the SFT boundary with real torch parameters and a tiny trainer."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn


@pytest.fixture
def sft(monkeypatch):
    state = {"managers": 0, "trainers": [], "heads": [], "trainable": [], "instances": []}

    class Module(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            forward = kwargs.pop("forward")
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.forward = types.MethodType(forward, self)
            self.log = lambda *a, **k: None
            state["heads"].append(self.classifier.weight.detach().clone())

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.ckpt_path = "auto-hpc"
            state["trainers"].append(kwargs)
            state["instances"].append(self)

        def save_checkpoint(self, filepath, **kwargs):
            Path(filepath).write_bytes(b"unexpected FT weights")

        def fit(self, module, datamodule=None, **kwargs):
            state["trainable"].append({n: p.requires_grad for n, p in module.named_parameters()})
            optimizer = torch.optim.AdamW(
                [p for p in module.parameters() if p.requires_grad], lr=1e-3
            )
            module.train()
            batch = {"image": torch.ones(4, 4), "label": torch.tensor([0, 1, 0, 1])}
            loss = module(batch, "fit")["loss"]
            loss.backward()
            state["grads"] = {n: p.grad is not None for n, p in module.named_parameters()}
            optimizer.step()

    class Manager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["managers"] += 1

        def __call__(self):
            self.kwargs["trainer"].fit(self.kwargs["module"], self.kwargs["data"])

    lightning = types.ModuleType("lightning")
    lightning.Trainer = Trainer
    lightning.seed_everything = lambda seed, **kwargs: torch.manual_seed(seed)
    environments = types.ModuleType("lightning.pytorch.plugins.environments")

    class SLURMEnvironment:
        def __init__(self, auto_requeue=True):
            self.auto_requeue = auto_requeue

        @staticmethod
        def detect():
            return "SLURM_NTASKS" in os.environ

    environments.SLURMEnvironment = SLURMEnvironment
    library = types.ModuleType("stable_pretraining")
    library.Module = Module
    library.Manager = Manager
    package = types.ModuleType("_sft_test_package")
    package.__path__ = []
    evaluation = types.ModuleType("_sft_test_package.zero_shot_eval")
    evaluation.finetune_evaluate = lambda *a, **k: {
        "finetune_acc": 0.5, "finetune_f1": 0.5, "finetune_auroc": 0.5
    }
    for name, value in {"lightning": lightning, "stable_pretraining": library,
                        "lightning.pytorch.plugins.environments": environments,
                        "_sft_test_package": package,
                        "_sft_test_package.zero_shot_eval": evaluation}.items():
        monkeypatch.setitem(sys.modules, name, value)
    path = Path(__file__).resolve().parents[1] / "stable_cp/evaluation/sft_eval.py"
    spec = importlib.util.spec_from_file_location("_sft_test_package.sft_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, state


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
        self.norm = nn.LayerNorm(4)

    def forward_features(self, images):
        return self.norm(self.blocks(self.patch_embed(images)))


def evaluate(module, backbone, **kwargs):
    return module.sft_evaluate(backbone, object(), [], torch.device("cpu"),
                               num_classes=2, embed_dim=4, n_samples=32,
                               verbose=False, **kwargs)


def test_partial_cp_mask_is_removed_before_optimizer_creation(sft):
    module, state = sft
    backbone = Backbone().requires_grad_(False)
    backbone.blocks[-1].requires_grad_(True)
    evaluate(module, backbone)
    assert all(state["trainable"][0].values()), state["trainable"]
    assert all(state["grads"].values()), state["grads"]


def test_full_ft_does_not_mutate_original_weights_or_mask(sft):
    module, _ = sft
    backbone = Backbone().requires_grad_(False).eval()
    weights = {n: p.detach().clone() for n, p in backbone.named_parameters()}
    evaluate(module, backbone)
    assert not backbone.training
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(torch.equal(weights[n], p) for n, p in backbone.named_parameters())


def test_no_checkpoint_callback_or_manager_is_enabled(sft, tmp_path):
    module, state = sft
    existing = tmp_path / "old.ckpt"
    existing.write_bytes(b"do not alter")
    evaluate(module, Backbone(), ckpt_path=str(existing))
    assert state["trainers"][0].get("enable_checkpointing") is False
    assert state["managers"] == 0
    assert existing.read_bytes() == b"do not alter"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.ckpt"]


def test_classifier_is_seeded_before_construction(sft):
    module, state = sft
    backbone = Backbone()
    evaluate(module, backbone, seed=43)
    torch.rand(131)
    evaluate(module, backbone, seed=43)
    evaluate(module, backbone, seed=44)
    assert torch.equal(state["heads"][0], state["heads"][1])
    assert not torch.equal(state["heads"][0], state["heads"][2])


def test_results_identify_corrected_full_ft_protocol(sft):
    module, _ = sft
    results = evaluate(module, Backbone())
    assert results.get("sft_protocol") == "full_ft_v1"
    assert results["sft_trainable_params"] == results["sft_total_params"] > 0


def test_slurm_never_auto_resumes_or_saves_an_hpc_checkpoint(sft, monkeypatch, tmp_path):
    module, state = sft
    monkeypatch.setenv("SLURM_NTASKS", "1")
    evaluate(module, Backbone())
    assert state["trainers"][0]["plugins"][0].auto_requeue is False
    trainer = state["instances"][0]
    assert trainer.ckpt_path is None
    with pytest.raises(RuntimeError, match="checkpoint"):
        trainer.save_checkpoint(tmp_path / "hpc_ckpt_1.ckpt")
    assert not list(tmp_path.iterdir())


class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, tokens):
        return self.proj(tokens.mean(dim=1))


class MapBackbone(Backbone):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*(nn.Linear(4, 4) for _ in range(12)))
        self.attn_pool = AttentionPool()
        self.fc_norm = nn.LayerNorm(4)

    def forward_features(self, images):
        return super().forward_features(images).unsqueeze(1)


def test_siglip_map_ft_unfreezes_and_backpropagates_through_pool(sft):
    module, state = sft
    backbone = MapBackbone().requires_grad_(False)
    backbone.blocks[-2:].requires_grad_(True)
    results = evaluate(module, backbone, pool_strategy="map")
    assert all(state["trainable"][0].values())
    assert all(state["grads"].values())
    assert state["grads"]["backbone.attn_pool.proj.weight"]
    assert state["grads"]["backbone.patch_embed.weight"]
    assert results["sft_total_params"] == results["sft_trainable_params"]


@pytest.mark.parametrize("depth", [2, 4, 6, -1])
def test_cp_callback_mainrule_mask_on_map_backbone(monkeypatch, depth):
    lightning = types.ModuleType("lightning")
    lightning.Callback = object
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    path = Path(__file__).resolve().parents[1] / "stable_cp/callbacks/common_callback.py"
    spec = importlib.util.spec_from_file_location("_cp_mask_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backbone = MapBackbone()
    wrapper = types.SimpleNamespace(backbone=backbone)
    callback = module.FreezeBackboneCallback(freeze_epochs=15, num_trained_blocks=depth)
    callback.on_train_start(types.SimpleNamespace(current_epoch=0), wrapper)
    callback.on_train_epoch_start(types.SimpleNamespace(current_epoch=14), wrapper)
    assert all(not p.requires_grad for p in backbone.parameters())
    callback.on_train_epoch_start(types.SimpleNamespace(current_epoch=15), wrapper)
    for name, param in backbone.named_parameters():
        expected = depth == -1 or (name.startswith("blocks.") and int(name.split(".")[1]) >= 12 - depth)
        assert param.requires_grad == expected, name
