import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FORWARDS = (
    "stable_cp.methods.diet.diet_forward",
    "stable_cp.methods.lejepa.lejepa_forward",
    "stable_cp.methods.simclr.simclr_cp_forward",
)


@pytest.fixture
def isolated_queues(monkeypatch):
    from stable_pretraining.callbacks.queue import OnlineQueue

    monkeypatch.setattr(OnlineQueue, "_shared_queues", {})
    monkeypatch.setattr(OnlineQueue, "_queue_info", {})
    return OnlineQueue


@pytest.fixture
def isolated_spt_config(monkeypatch):
    import stable_pretraining as spt

    monkeypatch.setattr(spt.get_config(), "cache_dir", None)


def test_method_setup_modules_have_no_separate_training_entrypoints():
    for path in (ROOT / "stable_cp/methods").glob("*/*_cp.py"):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
        ), path
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "continued_pretraining"
            for node in ast.walk(tree)
        ), path


def test_only_cp_methods_and_frozen_evaluation_callbacks_remain():
    assert not list((ROOT / "stable_cp/methods/tent").glob("*.py"))
    assert not (ROOT / "stable_cp/callbacks/lejepa_metrics.py").exists()
    source = (ROOT / "stable_cp/callbacks/continued_pretraining_metrics.py").read_text()
    assert "RankMe" not in source


@pytest.mark.parametrize("module_name", FORWARDS)
def test_feature_pooling_preserves_cls_patch_mean_and_siglip_map(module_name):
    pool = importlib.import_module(module_name)._extract_embedding
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    backbone = SimpleNamespace(
        attn_pool=lambda values: values.mean(dim=1),
        fc_norm=lambda values: values + 2,
    )
    torch.testing.assert_close(pool(tokens, "cls"), tokens[:, 0])
    torch.testing.assert_close(pool(tokens, "mean"), tokens[:, 1:].mean(dim=1))
    torch.testing.assert_close(pool(tokens, "map", backbone), tokens.mean(dim=1) + 2)
    torch.testing.assert_close(pool(tokens[:, 0], "cls"), tokens[:, 0])


@pytest.mark.parametrize("wrapped", [False, True])
def test_freezing_retains_only_requested_blocks_and_mae_masking(wrapped):
    from stable_cp.callbacks.common_callback import FreezeBackboneCallback

    backbone = nn.Module()
    backbone.stem = nn.Linear(3, 3)
    backbone.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
    backbone.norm = nn.BatchNorm1d(3)
    if wrapped:
        encoder = nn.Module()
        encoder.vit = backbone
        encoder.masking = nn.Identity()
    else:
        encoder = backbone
    module = SimpleNamespace(backbone=encoder)
    trainer = SimpleNamespace(current_epoch=0)
    callback = FreezeBackboneCallback(freeze_epochs=2, num_trained_blocks=2)
    callback.on_train_start(trainer, module)
    assert not any(param.requires_grad for param in encoder.parameters())
    assert not backbone.training
    assert not backbone.norm.training
    if wrapped:
        assert encoder.training
        assert encoder.masking.training

    trainer.current_epoch = 2
    callback.on_train_epoch_start(trainer, module)
    assert all(param.requires_grad for block in backbone.blocks[2:] for param in block.parameters())
    assert not any(
        param.requires_grad for block in backbone.blocks[:2] for param in block.parameters()
    )
    assert not any(param.requires_grad for param in backbone.stem.parameters())


@pytest.mark.parametrize("blocks,expected", [(0, False), (-1, True)])
def test_freezing_supports_head_only_and_full_cp(blocks, expected):
    from stable_cp.callbacks.common_callback import FreezeBackboneCallback

    module = SimpleNamespace(backbone=nn.Linear(3, 3))
    trainer = SimpleNamespace(current_epoch=0)
    callback = FreezeBackboneCallback(freeze_epochs=1, num_trained_blocks=blocks)
    callback.on_train_start(trainer, module)
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, module)
    assert all(param.requires_grad == expected for param in module.backbone.parameters())


def test_mae_pooling_excludes_all_prefix_tokens():
    from stable_cp.methods.mae.mae_cp_forward import _extract_embedding

    tokens = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    torch.testing.assert_close(_extract_embedding(tokens, 2, "mean"), tokens[:, 2:].mean(dim=1))
    torch.testing.assert_close(_extract_embedding(tokens, 2, "cls"), tokens[:, 0])


@pytest.mark.parametrize("method", ["diet", "lejepa", "simclr", "mae"])
def test_training_losses_are_unscaled_for_spt_gradient_accumulation(method):
    module_name = {
        "diet": "stable_cp.methods.diet.diet_forward",
        "lejepa": "stable_cp.methods.lejepa.lejepa_forward",
        "simclr": "stable_cp.methods.simclr.simclr_cp_forward",
        "mae": "stable_cp.methods.mae.mae_cp_forward",
    }[method]
    forward_name = {"simclr": "simclr_cp_forward", "mae": "mae_forward"}.get(
        method, f"{method}_forward"
    )
    forward = getattr(importlib.import_module(module_name), forward_name)
    images = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
    batch = {"image": images, "label": torch.tensor([0, 1]), "sample_idx": torch.tensor([0, 1])}
    module = SimpleNamespace(
        training=True,
        backbone=SimpleNamespace(forward_features=lambda values: values),
        pool_strategy="mean",
        projector=nn.Identity(),
        log=lambda *args, **kwargs: None,
    )
    embedding = images[:, 1:].mean(dim=1)
    if method == "diet":
        module.diet_head = nn.Linear(3, 2, bias=False)
        module.diet_loss = nn.CrossEntropyLoss(label_smoothing=0.3)
        expected = module.diet_loss(
            module.diet_head(nn.functional.normalize(embedding, dim=1)), batch["sample_idx"]
        )
    elif method == "lejepa":
        module.sigreg_loss = lambda values: values.square().mean()
        batch = [batch, dict(batch, image=images + 0.1)]
        projections = torch.stack([embedding, (images + 0.1)[:, 1:].mean(dim=1)])
        invariance = (projections - projections.mean(dim=0)).square().mean()
        expected = 0.02 * projections.square().mean() + 0.98 * invariance
    elif method == "simclr":
        from stable_pretraining.losses import NTXEntLoss

        module.simclr_loss = NTXEntLoss(temperature=0.5)
        batch = [batch, dict(batch, image=images + 0.1)]
        expected = module.simclr_loss(embedding, (images + 0.1)[:, 1:].mean(dim=1))
    else:

        class Encoder(nn.Module):
            num_prefix_tokens = 1

            def forward(self, values):
                return SimpleNamespace(encoded=values, mask=torch.ones(2, 3), ids_keep=None)

        module.backbone = Encoder()
        module.decoder = lambda values, mask, **kwargs: values
        module.loss_fn = lambda predictions, values, mask: predictions.square().mean()
        expected = images[:, 1:].square().mean()

    output = forward(module, batch, "fit")
    assert torch.isfinite(output["loss"])
    torch.testing.assert_close(output["loss"], expected)


def test_sigreg_supports_views_and_backpropagation():
    from stable_cp.methods.lejepa.lejepa_losses import EppsPulley, SlicingUnivariateTest

    values = torch.randn(2, 8, 3, requires_grad=True)
    loss = SlicingUnivariateTest(EppsPulley(), num_slices=8)(values)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(values.grad).all()


def test_online_knn_keeps_all_classes_when_the_bank_contains_only_one_class():
    from stable_cp.callbacks.continued_pretraining_metrics import create_cp_knn_probe

    callback = create_cp_knn_probe(num_classes=4, embedding_dim=2, queue_length=8, k=2)
    predictions = callback._compute_knn_predictions(
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        cached_features=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        cached_labels=torch.zeros(3, dtype=torch.long),
        current_targets=torch.tensor([0, 1]),
    )
    assert callback.num_classes == 4
    assert predictions.shape == (2, 4)
    assert torch.isfinite(predictions).all()
    assert (predictions[:, 0] > 0).all()
    assert torch.count_nonzero(predictions[:, 1:]) == 0


@pytest.mark.parametrize("accumulation", [2, 4])
def test_manager_accumulation_matches_full_batch_gradients_and_updates(
    accumulation, tmp_path, isolated_spt_config
):
    import lightning as pl
    import stable_pretraining as spt
    from torch.utils.data import DataLoader

    class CaptureGradients(pl.Callback):
        def __init__(self):
            self.gradients = []

        def on_before_optimizer_step(self, trainer, module, optimizer):
            self.gradients.append(
                torch.cat([param.grad.flatten().clone() for param in module.parameters()])
            )

    def regression_forward(self, batch, stage):
        predictions = self.backbone(batch["image"])
        return {"loss": nn.functional.mse_loss(predictions, batch["target"])}

    generator = torch.Generator().manual_seed(19)
    samples = [
        {
            "image": torch.randn(4, generator=generator),
            "target": torch.randn(2, generator=generator),
        }
        for _ in range(16)
    ]

    def fit(batch_size, frequency):
        torch.manual_seed(42)
        module = spt.Module(
            backbone=nn.Linear(4, 2),
            forward=regression_forward,
            optim={
                "optimizer": {"type": "SGD", "lr": 0.1, "momentum": 0.9},
                "scheduler": {"type": "ConstantLR", "factor": 1.0},
            },
        )
        initial = torch.cat([param.detach().flatten().clone() for param in module.parameters()])
        recorder = CaptureGradients()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            accumulate_grad_batches=frequency,
            callbacks=[recorder],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path / str(frequency),
        )
        data = spt.data.DataModule(train=DataLoader(samples, batch_size=batch_size, shuffle=False))
        spt.Manager(trainer=trainer, module=module, data=data, seed=42)()
        final = torch.cat([param.detach().flatten().clone() for param in module.parameters()])
        assert len(recorder.gradients) == 2
        assert not torch.equal(final, initial)
        return recorder.gradients, final

    full_gradients, full_parameters = fit(batch_size=8, frequency=1)
    accumulated_gradients, accumulated_parameters = fit(
        batch_size=8 // accumulation, frequency=accumulation
    )
    for full, accumulated in zip(full_gradients, accumulated_gradients):
        torch.testing.assert_close(accumulated, full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(accumulated_parameters, full_parameters, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["diet", "lejepa", "simclr", "mae"])
def test_cp_methods_fit_on_cpu_with_real_training_dependencies(
    method, tmp_path, isolated_queues, isolated_spt_config
):
    import lightning as pl
    import stable_pretraining as spt
    from timm.models.vision_transformer import VisionTransformer
    from torch.utils.data import DataLoader

    from stable_cp.callbacks import FreezeBackboneCallback, create_cp_evaluation_callbacks
    from stable_cp.methods.diet.diet_cp import setup_diet
    from stable_cp.methods.lejepa.lejepa_cp import setup_lejepa
    from stable_cp.methods.lejepa.lejepa_losses import EppsPulley, SlicingUnivariateTest
    from stable_cp.methods.mae.mae_cp import setup_mae
    from stable_cp.methods.simclr.simclr_cp import setup_simclr

    class CountOptimizerSteps(pl.Callback):
        def __init__(self):
            self.steps = {}

        def on_before_optimizer_step(self, trainer, module, optimizer):
            self.steps[optimizer] = self.steps.get(optimizer, 0) + 1

    def make_trainer(module, max_epochs):
        step_counter = CountOptimizerSteps()
        callbacks = [
            step_counter,
            FreezeBackboneCallback(freeze_epochs=1, num_trained_blocks=2),
            *create_cp_evaluation_callbacks(module, 2, 32, knn_queue_length=8, knn_k=2),
        ]
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            accumulate_grad_batches=2,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path,
        )
        return trainer, step_counter

    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        backbone_config = dict(
            img_size=16, patch_size=8, embed_dim=32, depth=4, num_heads=4, num_classes=0
        )
        backbone = VisionTransformer(**backbone_config)
        original_stem = backbone.patch_embed.proj.weight.detach().clone()
        original_last = backbone.blocks[-1].attn.qkv.weight.detach().clone()
        optim = {"optimizer": {"type": "AdamW", "lr": 1e-3}, "frequency": 2}
        setups = {
            "diet": setup_diet,
            "lejepa": setup_lejepa,
            "simclr": setup_simclr,
            "mae": setup_mae,
        }
        options = {
            "diet": {"num_samples": 8, "mixup_alpha": 0, "cutmix_alpha": 0},
            "lejepa": {
                "sigreg_loss": SlicingUnivariateTest(EppsPulley(), 8),
                "hidden_dim": 16,
                "proj_dim": 8,
                "lamb": 0.02,
            },
            "simclr": {"hidden_dim": 16, "proj_dim": 8},
            "mae": {"decoder_dim": 16, "decoder_depth": 1},
        }
        module = setups[method](backbone, 32, optim, pool_strategy="cls", **options[method])
        samples = [
            {"image": torch.randn(3, 16, 16), "label": index % 2, "sample_idx": index}
            for index in range(8)
        ]
        train_samples = samples
        if method in {"lejepa", "simclr"}:
            train_samples = [
                {"view_0": row, "view_1": dict(row, image=row["image"] + 0.1)} for row in samples
            ]
        trainer, step_counter = make_trainer(module, max_epochs=2)
        data = spt.data.DataModule(
            train=DataLoader(train_samples, batch_size=2),
            val=DataLoader(samples, batch_size=2),
        )
        spt.Manager(trainer=trainer, module=module, data=data, seed=42)()
        torch.testing.assert_close(backbone.patch_embed.proj.weight, original_stem)
        assert not torch.equal(backbone.blocks[-1].attn.qkv.weight, original_last)
        assert all(torch.isfinite(param).all() for param in module.parameters())
        assert trainer.current_epoch == 2
        assert len(trainer.optimizers) == 2
        assert step_counter.steps[trainer.optimizers[0]] == 4
        assert step_counter.steps[trainer.optimizers[1]] == 8
        assert any("cp_knn_probe" in name for name in trainer.callback_metrics)
        assert any("cp_linear_probe" in name for name in trainer.callback_metrics)
        checkpoint_path = tmp_path / "cp.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prefix = "backbone.vit." if method == "mae" else "backbone."
        restored = VisionTransformer(**backbone_config)
        restored.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith(prefix)
            }
        )
        backbone.eval()
        restored.eval()
        with torch.no_grad():
            images = torch.stack([row["image"] for row in samples])
            torch.testing.assert_close(
                restored.forward_features(images),
                backbone.forward_features(images),
                rtol=0,
                atol=0,
            )

        class CheckRestoredCheckpoint(pl.Callback):
            def on_train_start(self, trainer, module):
                torch.testing.assert_close(
                    module.state_dict(), checkpoint["state_dict"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    [optimizer.state_dict()["state"] for optimizer in trainer.optimizers],
                    [optimizer["state"] for optimizer in checkpoint["optimizer_states"]],
                    rtol=0,
                    atol=0,
                )

        isolated_queues._shared_queues.clear()
        isolated_queues._queue_info.clear()
        resumed_backbone = VisionTransformer(**backbone_config)
        resumed = setups[method](
            resumed_backbone, 32, optim, pool_strategy="cls", **options[method]
        )
        resumed_trainer, resumed_counter = make_trainer(resumed, max_epochs=3)
        resumed_trainer.callbacks.append(CheckRestoredCheckpoint())
        spt.Manager(
            trainer=resumed_trainer,
            module=resumed,
            data=data,
            seed=42,
            ckpt_path=str(checkpoint_path),
            weights_only=False,
        )()
        assert resumed_trainer.current_epoch == 3
        assert resumed_trainer.global_step > checkpoint["global_step"]
        assert resumed_counter.steps[resumed_trainer.optimizers[0]] == 2
        assert resumed_counter.steps[resumed_trainer.optimizers[1]] == 4
        torch.testing.assert_close(resumed_backbone.patch_embed.proj.weight, original_stem)
        assert not torch.equal(
            resumed_backbone.blocks[-1].attn.qkv.weight, backbone.blocks[-1].attn.qkv.weight
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_run_training_saves_and_resumes_through_real_manager(
    tmp_path, monkeypatch, isolated_queues, isolated_spt_config
):
    import lightning as pl
    import stable_pretraining as spt
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    import continued_pretraining as cp
    from stable_cp.methods.diet.diet_cp import setup_diet

    class TinyBackbone(nn.Linear):
        def forward_features(self, images):
            return self(images)

    class CPUTrainer(pl.Trainer):
        def __init__(self, **kwargs):
            kwargs.update(
                accelerator="cpu",
                devices=1,
                precision="32-true",
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            super().__init__(**kwargs)

    monkeypatch.setattr(cp.pl, "Trainer", CPUTrainer)
    samples = [
        {"image": torch.randn(4), "label": index % 2, "sample_idx": index} for index in range(8)
    ]
    loader = DataLoader(samples, batch_size=2)
    data = spt.data.DataModule(train=loader, val=loader)
    args = SimpleNamespace(
        epochs=2,
        seed=42,
        n_samples=8,
        knn_k=2,
        accumulate_grad_batches=2,
        num_trained_blocks=-1,
        resume=False,
    )
    optim = {"optimizer": {"type": "AdamW", "lr": 1e-3}, "frequency": 2}
    checkpoint_path = tmp_path / "cp.ckpt"
    module = setup_diet(TinyBackbone(4, 4), 4, optim, num_samples=8, mixup_alpha=0, cutmix_alpha=0)
    cp.run_training(
        module,
        data,
        args,
        {"num_classes": 2},
        4,
        1,
        CSVLogger(tmp_path, name="first"),
        str(checkpoint_path),
    )
    first = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert first["epoch"] == 2
    torch.testing.assert_close(first["state_dict"]["backbone.weight"], module.backbone.weight)

    args.epochs = 3
    args.resume = True
    # A resumed CLI job starts in a new process, without previous online queues.
    isolated_queues._shared_queues.clear()
    isolated_queues._queue_info.clear()
    resumed = setup_diet(TinyBackbone(4, 4), 4, optim, num_samples=8, mixup_alpha=0, cutmix_alpha=0)
    cp.run_training(
        resumed,
        data,
        args,
        {"num_classes": 2},
        4,
        1,
        CSVLogger(tmp_path, name="resumed"),
        str(checkpoint_path),
    )
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert final["epoch"] == 3
    assert final["global_step"] > first["global_step"]
    assert not torch.equal(
        final["state_dict"]["backbone.weight"], first["state_dict"]["backbone.weight"]
    )
