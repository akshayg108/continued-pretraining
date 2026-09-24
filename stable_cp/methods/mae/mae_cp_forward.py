def _extract_embedding(encoded_tokens, num_prefix_tokens, pool_strategy="cls", backbone=None):
    """Use native MAP, CLS, or visible-patch mean for online probes."""
    if pool_strategy == "map":
        if backbone is None or getattr(backbone, "attn_pool", None) is None:
            raise ValueError("MAP pooling requires the backbone's native attention pool")
        return backbone.fc_norm(backbone.attn_pool(encoded_tokens))
    if pool_strategy == "mean":
        return encoded_tokens[:, num_prefix_tokens:, :].mean(dim=1)
    return encoded_tokens[:, 0, :]


def mae_forward(self, batch, stage):
    """Reconstruct masked patches during training; expose full-image features in eval."""
    out = {}
    images = batch["image"]
    pool_strategy = getattr(self, "pool_strategy", "mean")

    enc_out = self.backbone(images)

    out["embedding"] = _extract_embedding(
        enc_out.encoded,
        self.backbone.num_prefix_tokens,
        pool_strategy,
        backbone=self.backbone.vit,
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
        out["loss"] = self.loss_fn(predictions, images.to(predictions.dtype), enc_out.mask)
        self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)

    return out
