from typing import Dict, Any


def create_optim_config(
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> Dict[str, Any]:
    """Create optimizer and scheduler configuration.

    Args:
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        steps_per_epoch: Number of optimization steps per epoch

    Returns:
        Configuration dictionary for stable-pretraining
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    return {
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": warmup_steps,
            "max_steps": total_steps,
            "eta_min": 0.0,
        },
        "interval": "step",
    }
