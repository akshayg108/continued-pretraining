"""Patch masking that preserves the supported timm ViTs' native token forward."""

import torch
from torch import nn
from timm.layers import apply_keep_indices_nlc
from stable_pretraining.backbone import MaskedEncoder
from stable_pretraining.backbone.vit import MaskedEncoderOutput


class NativeMaskedEncoder(MaskedEncoder):
    """Keep native positions and pre-normalization for ViT and DINOv3 EVA."""

    def __init__(self, backbone, masking):
        super().__init__(backbone, masking=masking)
        patch_drop = getattr(backbone, "patch_drop", None)
        if patch_drop is not None and not isinstance(patch_drop, nn.Identity):
            raise ValueError("MAE masking requires native patch dropout to be disabled")
        if getattr(backbone, "rope_mixed", False):
            raise ValueError("MAE masking does not support depth-dependent mixed RoPE")

    def forward(self, images):
        batch_size = images.shape[0]
        grid_h, grid_w = self._get_grid_size(images)
        num_patches = grid_h * grid_w
        positioned = self.vit._pos_embed(self.patch_embed(images))
        tokens, rope = positioned if isinstance(positioned, tuple) else (positioned, None)

        if self.training and self.masking is not None:
            prefix = tokens[:, : self.num_prefix_tokens]
            masked = self.masking(tokens[:, self.num_prefix_tokens :], grid_h, grid_w)
            tokens = torch.cat((prefix, masked.visible), dim=1)
            mask, ids_keep = masked.mask, masked.ids_keep
            if rope is not None:
                # Keep each visible patch's original position, never its packed index.
                rope = apply_keep_indices_nlc(tokens, rope, ids_keep).unsqueeze(1)
        else:
            mask = torch.zeros(batch_size, num_patches, device=images.device)
            ids_keep = torch.arange(num_patches, device=images.device).expand(batch_size, -1)

        tokens = self.vit.norm_pre(tokens)
        for block in self.vit.blocks:
            tokens = block(tokens, rope=rope) if rope is not None else block(tokens)
        return MaskedEncoderOutput(
            encoded=self.vit.norm(tokens),
            mask=mask,
            ids_keep=ids_keep,
            grid_size=(grid_h, grid_w),
        )
