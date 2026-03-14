#!/usr/bin/env python
"""MAE Continued Pretraining.

Uses the full MAE pipeline from stable-pretraining: image-level patch masking
via MaskedEncoder, reconstruction via MAEDecoder, and MSE loss on masked patches.
The pretrained backbone is wrapped inside MaskedEncoder so that masking happens
*before* encoding — matching the original MAE paper.
"""
import stable_pretraining as spt
from stable_pretraining.backbone import MaskedEncoder, PatchMasking
from stable_pretraining.backbone.vit import MAEDecoder
from stable_pretraining.utils import MAELoss

from .mae_cp_forward import mae_forward


def setup_mae(backbone, embed_dim, optim_config, **kwargs):
    mask_ratio = kwargs.get("mask_ratio", 0.75)
    decoder_dim = kwargs.get("decoder_dim", 512)
    decoder_depth = kwargs.get("decoder_depth", 4)
    pool_strategy = kwargs.get("pool_strategy", "mean")

    masking = PatchMasking(mask_ratio=mask_ratio)
    encoder = MaskedEncoder(backbone, masking=masking)

    patch_size = encoder.patch_size_h
    num_patches = encoder.default_grid_h * encoder.default_grid_w
    in_chans = encoder.patch_embed.proj.in_channels
    patch_dim = patch_size * patch_size * in_chans

    decoder = MAEDecoder(
        embed_dim=embed_dim,
        decoder_embed_dim=decoder_dim,
        output_dim=patch_dim,
        num_patches=num_patches,
        depth=decoder_depth,
    )

    loss_fn = MAELoss(
        patch_size=patch_size,
        loss_type="mse",
        mask_only=True,
        patch_normalize=True,
    )

    return spt.Module(
        backbone=encoder,
        decoder=decoder,
        loss_fn=loss_fn,
        pool_strategy=pool_strategy,
        forward=mae_forward,
        optim=optim_config,
    )
