"""SFT (Supervised Fine-Tuning) evaluation for continued pretraining.

SFT is an evaluation protocol, not a pretraining method.  It trains a
backbone + classifier end-to-end on the target dataset and reports
classification metrics (accuracy, F1, AUROC).

Usage from continued_pretraining.py:
    from stable_cp.evaluation.sft_eval import sft_evaluate
    results = sft_evaluate(backbone, sft_data, test_loader, device, ...)
"""
import copy

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor

import stable_pretraining as spt

from .zero_shot_eval import finetune_evaluate


# ---------------------------------------------------------------------------
# Fixed SFT evaluation hyper-parameters (not user-configurable)
# ---------------------------------------------------------------------------
SFT_EPOCHS = 5
SFT_LR = 1e-4
SFT_BATCH_SIZE = 32
SFT_WEIGHT_DECAY = 0.05
SFT_WARMUP_EPOCHS = 0.1 * SFT_EPOCHS
SFT_LABEL_SMOOTHING = 0.0


# ---------------------------------------------------------------------------
# Forward & Module helpers (moved from stable_cp/methods/supervised/)
# ---------------------------------------------------------------------------

def _extract_embedding(backbone_output, pool_strategy="cls"):
    """Extract embedding from backbone output (handles ViT and CNN).

    Args:
        backbone_output: Raw output from backbone.forward_features().
            - 3-D [B, T, D] for ViT token sequences
            - 2-D [B, D]    for CNNs / already-pooled features
        pool_strategy: 'cls' (default) uses the CLS token; 'mean' averages
            over patch tokens (excluding CLS).
    """
    if backbone_output.ndim == 3:
        if pool_strategy == "mean":
            return backbone_output[:, 1:, :].mean(dim=1)
        return backbone_output[:, 0, :]  # CLS token
    return backbone_output


def _sft_forward(self, batch, stage):
    """Forward function compatible with ``spt.Module``.

    Required module attributes set via ``spt.Module(**kwargs)``:
        backbone, classifier, supervised_loss, pool_strategy, metric_prefix

    Returns a dict with ``embedding``, ``logits``, and (if labels present)
    ``label`` and ``loss``.
    """
    out = {}
    pool_strategy = getattr(self, "pool_strategy", "cls")
    prefix = getattr(self, "metric_prefix", "sft")
    features = self.backbone.forward_features(batch["image"])
    out["embedding"] = _extract_embedding(features, pool_strategy)
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


def _setup_sft_module(backbone, embed_dim, optim_config, num_classes,
                      label_smoothing=SFT_LABEL_SMOOTHING, pool_strategy="cls",
                      metric_prefix="sft"):
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


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

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
    """Run SFT evaluation: fine-tune a *copy* of the backbone, then evaluate.

    The backbone is deep-copied so the caller's original model is never
    modified.  This is critical for the pre-CP phase where the original
    backbone must remain intact for subsequent continued pretraining.

    All training hyper-parameters (epochs, lr, weight_decay, etc.) are
    **fixed** module-level constants -- see ``SFT_*`` at the top of this file.

    Args:
        backbone:       Backbone model to evaluate (will be deep-copied).
        sft_data:       ``spt.data.DataModule`` with augmented training data
                        (single-view, ``n_views=1``).
        test_loader:    DataLoader for the test split (val transform).
        device:         Target device (e.g. ``torch.device("cuda")``).
        num_classes:    Number of target classes.
        embed_dim:      Embedding dimension of the backbone.
        n_samples:      Number of training samples (used for LR schedule).
        pool_strategy:  ``'cls'`` or ``'mean'`` for ViT embedding extraction.
        seed:           Random seed.
        ckpt_path:      Optional checkpoint path for spt.Manager
                        (enables resume on crash).
        logger:         Optional ``WandbLogger`` for metric logging.
        prefix:         Metric prefix for wandb logging. Use ``'pre_sft'`` for
                        pre-CP and ``'post_sft'`` for post-CP to produce
                        distinct charts (e.g. ``fit/pre_sft_loss``).
        verbose:        Print progress to stdout.

    Returns:
        dict with keys ``{prefix}_acc``, ``{prefix}_f1``, ``{prefix}_auroc``.
    """
    if verbose:
        print("=" * 50)
        print(f"SFT Evaluation [{prefix}]: {num_classes} classes | "
              f"{SFT_EPOCHS} ep | lr={SFT_LR} | bs={SFT_BATCH_SIZE}")
        print("=" * 50)

    # Deep-copy backbone so the original is never modified
    backbone_copy = copy.deepcopy(backbone)

    # ---- optimiser / scheduler config (fixed) ----
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

    # ---- build spt.Module ----
    module = _setup_sft_module(
        backbone_copy,
        embed_dim,
        optim_config,
        num_classes,
        label_smoothing=SFT_LABEL_SMOOTHING,
        pool_strategy=pool_strategy,
        metric_prefix=prefix,
    )

    # ---- train (minimal callbacks – no KNN/LP probes) ----
    callbacks = [LearningRateMonitor(logging_interval="step")]
    trainer = pl.Trainer(
        max_epochs=SFT_EPOCHS,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision="16-mixed",
        logger=logger,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=sft_data,
        ckpt_path=ckpt_path,
        seed=seed,
    )()

    # ---- final evaluation on test set ----
    if verbose:
        print(f"SFT [{prefix}] → evaluating on test set …")
    raw = finetune_evaluate(
        backbone_copy,
        module.classifier,
        test_loader,
        device,
        pool_strategy=pool_strategy,
        verbose=verbose,
    )

    # Rename keys: finetune_* → {prefix}_*
    results = {
        f"{prefix}_acc": raw["finetune_acc"],
        f"{prefix}_f1": raw["finetune_f1"],
        f"{prefix}_auroc": raw.get("finetune_auroc", 0.0),
    }

    if verbose:
        print(f"SFT [{prefix}] Results: acc={results[f'{prefix}_acc']:.4f}  "
              f"f1={results[f'{prefix}_f1']:.4f}")

    return results
