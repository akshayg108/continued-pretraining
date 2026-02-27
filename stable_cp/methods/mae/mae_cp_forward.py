import torch


def _extract_embedding(encoded_tokens, num_prefix_tokens, pool_strategy="cls"):
    """Extract a single embedding vector from encoder output tokens.

    Args:
        encoded_tokens: [B, num_prefix + N_tokens, D] from MaskedEncoder
        num_prefix_tokens: number of prefix tokens (CLS + registers)
        pool_strategy: 'cls' for CLS token, 'mean' for mean of patch tokens
    """
    if pool_strategy == "mean":
        return encoded_tokens[:, num_prefix_tokens:, :].mean(dim=1)
    return encoded_tokens[:, 0, :]


def mae_cp_forward(self, batch, stage):
    """MAE Continued Pretraining forward pass.

    Training:
        1. MaskedEncoder masks patches at image level, encodes only visible patches
        2. MAEDecoder reconstructs all patches from visible patch embeddings
        3. MAELoss computes MSE on masked patches against original pixel values

    Eval:
        MaskedEncoder runs without masking (full image), producing standard
        embeddings for KNN / linear-probe evaluation callbacks.
    """
    out = {}
    images = batch["image"]
    pool_strategy = getattr(self, "pool_strategy", "mean")

    enc_out = self.backbone(images)

    out["embedding"] = _extract_embedding(
        enc_out.encoded, self.backbone.num_prefix_tokens, pool_strategy
    )

    if "label" in batch:
        out["label"] = batch["label"]

    if self.training:
        encoded_patches = enc_out.encoded[:, self.backbone.num_prefix_tokens :]
        predictions = self.decoder(
            encoded_patches,
            enc_out.mask,
            ids_keep=enc_out.ids_keep,
            output_masked_only=False,
        )
        out["loss"] = self.loss_fn(
            predictions, images.to(predictions.dtype), enc_out.mask
        )
        self.log(
            f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True
        )

    return out
