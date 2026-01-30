# MAE CP forward - reconstruct pixels from pretrained backbone latents
import torch
import torch.nn.functional as F


def patchify(imgs, patch_size):
    # [B, 3, H, W] -> [B, N, patch_size^2 * 3]
    B, C, H, W = imgs.shape
    h, w = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size ** 2 * C)


def mae_cp_forward(self, batch, stage):
    # Mask backbone latents, reconstruct pixels via MAEDecoder
    backbone_out = self.backbone(batch["image"])
    tokens = (backbone_out.last_hidden_state if hasattr(backbone_out, "last_hidden_state") else backbone_out)[:, 1:]
    B, T, _ = tokens.shape
    # Mask: 1=masked, 0=visible (MAEDecoder convention)
    num_mask = int(T * self.mask_ratio)
    noise = torch.rand(B, T, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(B, T, device=tokens.device)
    mask.scatter_(1, ids_shuffle[:, :num_mask], 1)
    # Decode masked positions
    pred = self.decoder(self.projector(tokens), mask, output_masked_only=True)
    batch["embedding"] = tokens.mean(dim=1)
    if self.training:
        target = patchify(batch["image"], self.patch_size)
        # Extract targets for masked positions (same order as decoder output)
        masked_idx = torch.argsort((~mask.bool()).int(), dim=1, stable=True)[:, :num_mask]
        masked_target = torch.gather(target, 1, masked_idx.unsqueeze(-1).expand(-1, -1, target.shape[-1]))
        batch["loss"] = F.mse_loss(pred, masked_target)
        self.log(f"{stage}/loss", batch["loss"], on_step=True, on_epoch=True, sync_dist=True)
    return batch
