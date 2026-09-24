"""Versioned frozen-encoder linear-probe configuration without ML imports."""

import math

LP_PROTOCOL = "frozen_online_lp_v1"
LP_DIRECTORY = "lp_online_v1"


def lp_config(epochs=150, batch_size=512, lr=1e-3, forward_batch_size=None):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("LP epochs must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("LP batch size must be a positive integer")
    if forward_batch_size is not None and (
        type(forward_batch_size) is not int or forward_batch_size < 1
    ):
        raise ValueError("LP forward batch size must be a positive integer or None")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("LP learning rate must be finite and positive")
    return {
        "protocol": LP_PROTOCOL,
        "epochs": epochs,
        "batch_size": batch_size,
        "forward_batch_size": forward_batch_size,
        "lr": float(lr),
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "feature_normalization": "l2",
        "head": "linear",
        "loss": "cross_entropy",
        "train_transform": "random_resized_crop_horizontal_flip_v1",
        "encoder_mode": "eval_frozen",
        "feature_cache": False,
    }
