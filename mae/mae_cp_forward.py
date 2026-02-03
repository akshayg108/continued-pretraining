# MAE CP forward - reconstruct pixels from pretrained backbone latents
import torch
import stable_pretraining as spt

def patchify(imgs, patch_size):
    """Convert images to patches.

    Args:
        imgs: (B, 3, H, W)
        patch_size: int, size of each patch

    Returns:
        patches: (B, N, patch_size^2 * 3)
    """
    B, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0, f"Image size {H} must be divisible by patch_size {patch_size}"
    
    h = w = H // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(B, h * w, patch_size ** 2 * C)
    return x

def mae_cp_forward(self, batch, stage):
    """MAE Continued Pretraining forward pass.
    
    Pipeline:
    1. Encoder outputs tokens (remove CLS token)
    2. Randomly mask a portion of patches
    3. Decoder reconstructs masked patches from visible patches
    4. Compute reconstruction loss only on masked positions
    """
    # 1. Encoder forward (remove CLS token)
    backbone_out = self.backbone(batch["image"])
    tokens = (backbone_out.last_hidden_state if hasattr(backbone_out, "last_hidden_state") else backbone_out)[:, 1:]
    B, T, D = tokens.shape
    
    # 2. Random masking (MAE convention: 1=masked, 0=visible)
    num_mask = int(T * self.mask_ratio)
    num_keep = T - num_mask
    
    noise = torch.rand(B, T, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    # Create binary mask: 1 = masked, 0 = visible
    mask = torch.zeros(B, T, device=tokens.device)
    mask.scatter_(1, ids_shuffle[:, num_keep:], 1)
    
    # 3. Decoder forward
    # Pass full sequence; decoder will automatically extract visible tokens
    # This is simpler than manual extraction and verified to produce identical results
    pred = self.decoder(tokens, mask, output_masked_only=False)  # [B, T, output_dim]
    
    # 4. Store embedding for evaluation callbacks (KNN, linear probe)
    batch["embedding"] = tokens.mean(dim=1)
    
    # 5. Compute reconstruction loss on masked patches
    if self.training:
        # Patchify: convert image to patches
        target = patchify(batch["image"], self.patch_size)  # [B, T, patch_size^2 * 3]
        
        # Loss: compute MSE only on mask=1 positions
        batch["loss"] = spt.losses.mae(
            target=target,
            pred=pred,
            mask=mask,
            norm_pix_loss=False
        )
        
        self.log(f"{stage}/loss", batch["loss"], on_step=True, on_epoch=True, sync_dist=True)
    
    return batch


# def patchify(imgs, patch_size):
#     """Convert images to patches.
    
#     Args:
#         imgs: [B, 3, H, W]
    
#     Returns:
#         patches: [B, N, patch_size^2 * 3]
#     """
#     B, C, H, W = imgs.shape
#     h, w = H // patch_size, W // patch_size
#     x = imgs.reshape(B, C, h, patch_size, w, patch_size)
#     return x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size ** 2 * C)

# def mae_cp_forward(self, batch, stage):
#     # Mask backbone latents, reconstruct pixels via MAEDecoder
#     backbone_out = self.backbone(batch["image"])
#     tokens = (backbone_out.last_hidden_state if hasattr(backbone_out, "last_hidden_state") else backbone_out)[:, 1:]
#     B, T, _ = tokens.shape
#     # Mask: 1=masked, 0=visible (MAEDecoder convention)
#     num_mask = int(T * self.mask_ratio)
#     noise = torch.rand(B, T, device=tokens.device)
#     ids_shuffle = torch.argsort(noise, dim=1)
#     mask = torch.zeros(B, T, device=tokens.device)
#     mask.scatter_(1, ids_shuffle[:, :num_mask], 1)
#     # Decode masked positions
#     pred = self.decoder(self.projector(tokens), mask, output_masked_only=True)
#     batch["embedding"] = tokens.mean(dim=1)
#     if self.training:
#         target = patchify(batch["image"], self.patch_size)
#         # Extract targets for masked positions (same order as decoder output)
#         masked_idx = torch.argsort((~mask.bool()).int(), dim=1, stable=True)[:, :num_mask]
#         masked_target = torch.gather(target, 1, masked_idx.unsqueeze(-1).expand(-1, -1, target.shape[-1]))
#         batch["loss"] = F.mse_loss(pred, masked_target)
#         self.log(f"{stage}/loss", batch["loss"], on_step=True, on_epoch=True, sync_dist=True)
#     return batch