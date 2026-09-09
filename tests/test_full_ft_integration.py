"""Optional real Lightning/stable-pretraining CPU smoke; no production data."""
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import tempfile

import pytest
import torch
from torch import nn

pl = pytest.importorskip("lightning")
spt = pytest.importorskip("stable_pretraining")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))

    def forward_features(self, image):
        return self.blocks(self.patch_embed(image))


@pytest.mark.parametrize("slurm", [False, True])
def test_real_full_ft_updates_originally_frozen_layers_without_any_saves(tmp_path, monkeypatch, slurm):
    from stable_cp.evaluation import sft_eval

    monkeypatch.chdir(tmp_path)
    if slurm:
        for key, value in dict(SLURM_NTASKS="1", SLURM_NTASKS_PER_NODE="1", SLURM_JOB_NAME="ft-smoke",
                               SLURM_JOB_ID="12345", SLURM_PROCID="0", SLURM_NODEID="0",
                               SLURM_LOCALID="0", SLURM_NODELIST="localhost").items():
            monkeypatch.setenv(key, value)
    else:
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
    sentinel = tmp_path / "hpc_ckpt_1.ckpt"
    sentinel.write_bytes(b"must not load or overwrite")
    samples = [{"image": torch.randn(4), "label": i % 2} for i in range(64)]
    loader = torch.utils.data.DataLoader(samples, batch_size=32)
    data = spt.data.DataModule(train=loader, val=loader)
    backbone = TinyBackbone().requires_grad_(False)
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    captured = []
    setup = sft_eval._setup_sft_module

    def capture(*args, **kwargs):
        result = setup(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(sft_eval, "_setup_sft_module", capture)
    monkeypatch.setattr(sft_eval, "SFT_EPOCHS", 2)
    monkeypatch.setattr(sft_eval, "SFT_WARMUP_EPOCHS", 0)
    metrics = sft_eval.sft_evaluate(backbone, data, loader, torch.device("cpu"),
                                    num_classes=2, embed_dim=4, n_samples=64, verbose=False)
    assert metrics["sft_total_params"] == metrics["sft_trainable_params"]
    for name, value in captured[0].backbone.state_dict().items():
        assert not torch.equal(before[name], value), name
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(torch.equal(before[name], value) for name, value in backbone.state_dict().items())
    assert sentinel.read_bytes() == b"must not load or overwrite"
    assert list(tmp_path.rglob("*.ckpt")) == [sentinel]
    assert not list(tmp_path.rglob("*.pt"))


def test_real_cp_manager_saves_canonical_path_and_resumes_full_state(tmp_path, monkeypatch):
    from stable_cp.evaluation.sft_eval import _setup_sft_module
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    from lightning.pytorch.loggers import CSVLogger

    monkeypatch.chdir(tmp_path)
    trainer_type = pl.Trainer

    class CPUTrainer(trainer_type):
        def __init__(self, **kwargs):
            kwargs.update(accelerator="cpu", devices=1, precision="32-true")
            super().__init__(**kwargs)

    # Load only the production orchestration function; cluster image datasets
    # are not a dependency of this synthetic training test.
    source = Path(__file__).resolve().parents[1] / "continued_pretraining.py"
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_training")
    namespace = dict(Path=Path, inspect=inspect, tempfile=tempfile, spt=spt,
                     pl=SimpleNamespace(Trainer=CPUTrainer), ModelCheckpoint=ModelCheckpoint,
                     SLURMEnvironment=SLURMEnvironment, LearningRateMonitor=LearningRateMonitor,
                     FreezeBackboneCallback=lambda **kw: pl.Callback(),
                     create_cp_evaluation_callbacks=lambda *a, **kw: [])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(num_trained_blocks=2, n_samples=64, knn_k=20, cp_method="diet",
                           epochs=1, seed=42, resume=True)
    rows = [{"image": torch.randn(4), "label": i % 2} for i in range(64)]
    loader = torch.utils.data.DataLoader(rows, batch_size=32)
    data = spt.data.DataModule(train=loader, val=loader)
    checkpoint = tmp_path / "canonical" / "cp.ckpt"
    original_cache = spt.get_config().cache_dir
    for epochs in (1, 2):
        args.epochs = epochs
        model = _setup_sft_module(TinyBackbone(), 4, {
            "optimizer": {"type": "AdamW", "lr": 1e-4},
            "scheduler": {"type": "CosineAnnealingLR", "T_max": 4},
            "interval": "step",
        }, 2)
        namespace["run_training"](model, data, args, {"num_classes": 2}, 4, 0,
                                  CSVLogger(str(tmp_path / "logs")), str(checkpoint))
        assert checkpoint.is_file()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["epoch"] == epochs
        assert state["global_step"] == 2 * epochs
        assert state["optimizer_states"]
        assert spt.get_config().cache_dir == original_cache
    assert list((tmp_path / "canonical").glob("*.ckpt")) == [checkpoint]
