"""Full supervised fine-tuning on a copy of the pre- or post-CP backbone."""

import copy
import tempfile
import warnings

import lightning as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
import torch
import torch.nn as nn

import stable_pretraining as spt

from .zero_shot_eval import finetune_evaluate

# Fixed full-FT protocol.
SFT_EPOCHS = 150
SFT_LR = 1e-4
SFT_BATCH_SIZE = 32
SFT_WEIGHT_DECAY = 0.05
SFT_WARMUP_EPOCHS = 0.1 * SFT_EPOCHS
SFT_LABEL_SMOOTHING = 0.0
SFT_PROTOCOL = "full_ft_v1"


class _NoCheckpointTrainer(pl.Trainer):
    """Also block signal/plugin-triggered saves, not just ModelCheckpoint."""

    def save_checkpoint(self, *args, **kwargs):
        raise RuntimeError("FT checkpoint writes are disabled by the evaluation protocol")


def _extract_embedding(backbone_output, pool_strategy="cls", backbone=None):
    """Read CLS, patch-mean, or SigLIP attention-pooled features."""
    if backbone_output.ndim == 3:
        if pool_strategy == "map":  # SigLIP's native attention-pool readout
            return backbone.fc_norm(backbone.attn_pool(backbone_output))
        if pool_strategy == "mean":
            return backbone_output[:, 1:, :].mean(dim=1)
        return backbone_output[:, 0, :]  # CLS token
    return backbone_output


def _sft_forward(self, batch, stage):
    """Compute supervised logits and loss for ``spt.Module``."""
    out = {}
    pool_strategy = getattr(self, "pool_strategy", "cls")
    prefix = getattr(self, "metric_prefix", "sft")
    features = self.backbone.forward_features(batch["image"])
    out["embedding"] = _extract_embedding(features, pool_strategy, backbone=self.backbone)
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        out["label"] = batch["label"]
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])
        self.log(
            f"{stage}/{prefix}_loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        preds = out["logits"].detach().argmax(dim=-1)
        acc = (preds == batch["label"]).float().mean()
        self.log(
            f"{stage}/{prefix}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    return out


def _setup_sft_module(
    backbone,
    embed_dim,
    optim_config,
    num_classes,
    label_smoothing=SFT_LABEL_SMOOTHING,
    pool_strategy="cls",
    metric_prefix="sft",
):
    """Create an ``spt.Module`` configured for supervised fine-tuning."""
    classifier = nn.Linear(embed_dim, num_classes)
    return spt.Module(
        backbone=backbone,
        classifier=classifier,
        supervised_loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        pool_strategy=pool_strategy,
        metric_prefix=metric_prefix,
        forward=_sft_forward,
        optim=optim_config,
    )


def sft_evaluate(
    backbone: nn.Module,
    sft_data,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    embed_dim: int,
    n_samples: int,
    *,
    pool_strategy: str = "cls",
    seed: int = 42,
    ckpt_path: str = None,
    logger=None,
    prefix: str = "sft",
    verbose: bool = True,
) -> dict:
    """Fine-tune a copy without modifying the original model or writing FT weights.

    Training uses the fixed SFT_* configuration and a single-view, batch-32 loader.
    """
    if verbose:
        print("=" * 50)
        print(
            f"SFT Evaluation [{prefix}]: {num_classes} classes | "
            f"{SFT_EPOCHS} ep | lr={SFT_LR} | bs={SFT_BATCH_SIZE}"
        )
        print("=" * 50)

    if ckpt_path is not None:
        warnings.warn(
            "FT checkpointing is disabled; ckpt_path is ignored.", UserWarning, stacklevel=2
        )
    pl.seed_everything(seed, workers=True)
    # CP's requires_grad mask survives deepcopy. Reset it before optimizers exist.
    backbone_copy = copy.deepcopy(backbone).requires_grad_(True)

    steps_per_epoch = max(n_samples // SFT_BATCH_SIZE, 1)
    total_steps = SFT_EPOCHS * steps_per_epoch
    warmup_steps = SFT_WARMUP_EPOCHS * steps_per_epoch
    optim_config = {
        "optimizer": {
            "type": "AdamW",
            "lr": SFT_LR,
            "weight_decay": SFT_WEIGHT_DECAY,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": warmup_steps,
            "max_steps": total_steps,
            "eta_min": 0.0,
        },
        "interval": "step",
    }

    module = _setup_sft_module(
        backbone_copy,
        embed_dim,
        optim_config,
        num_classes,
        label_smoothing=SFT_LABEL_SMOOTHING,
        pool_strategy=pool_strategy,
        metric_prefix=prefix,
    )
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if trainable_params != total_params or total_params == 0:
        raise RuntimeError("Full FT requires every backbone and head parameter to be trainable")
    device = torch.device(device)

    # A private empty root prevents Lightning from discovering unrelated HPC
    # checkpoints. ckpt_path=None alone does not disable that SLURM behavior.
    with tempfile.TemporaryDirectory(prefix="full-ft-") as trainer_root:
        trainer = _NoCheckpointTrainer(
            max_epochs=SFT_EPOCHS,
            max_steps=total_steps,
            num_sanity_val_steps=0,
            precision="16-mixed" if device.type == "cuda" else "32-true",
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=[device.index or 0] if device.type == "cuda" else 1,
            logger=False,
            default_root_dir=trainer_root,
            enable_checkpointing=False,
            plugins=[SLURMEnvironment(auto_requeue=False)] if SLURMEnvironment.detect() else None,
        )
        trainer.ckpt_path = None
        # Manager can install checkpoint callbacks even with ckpt_path=None.
        trainer.fit(module, datamodule=sft_data)

    if verbose:
        print(f"SFT [{prefix}]: evaluating on test set...")
    raw = finetune_evaluate(
        backbone_copy,
        module.classifier,
        test_loader,
        device,
        pool_strategy=pool_strategy,
        verbose=verbose,
    )

    results = {
        f"{prefix}_acc": raw["finetune_acc"],
        f"{prefix}_f1": raw["finetune_f1"],
        f"{prefix}_auroc": raw.get("finetune_auroc", 0.0),
        "sft_protocol": SFT_PROTOCOL,
        "sft_trainable_params": trainable_params,
        "sft_total_params": total_params,
    }

    if verbose:
        print(
            f"SFT [{prefix}] Results: acc={results[f'{prefix}_acc']:.4f}  "
            f"f1={results[f'{prefix}_f1']:.4f}"
        )

    return results
