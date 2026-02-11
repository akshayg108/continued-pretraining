import torch


def _extract_embedding(backbone_output, pool_strategy="cls"):
    """Extract embedding from backbone output (handles ViT and CNN)."""
    if backbone_output.ndim == 3:
        # Token sequence from TIMM ViT [B, T, D]
        if pool_strategy == "mean":
            return backbone_output[:, 1:, :].mean(dim=1)  # Mean over patch tokens
        return backbone_output[:, 0, :]  # CLS token (default)
    # Already 2D (ResNet or pooled)
    return backbone_output


def sft_forward(self, batch, stage):
    """Forward function for supervised fine-tuning.

    Uses actual labels for training (CrossEntropyLoss).
    Supports TIMM ViT (CLS token extraction) and CNN backbones.

    Required module attributes:
        - backbone: Feature extraction network
        - classifier: Classification head (nn.Linear)
        - supervised_loss: Loss function (nn.CrossEntropyLoss)

    Args:
        batch: dict with 'image' [N, C, H, W] and 'label' [N]
        stage: 'fit', 'validate', or 'test'

    Returns:
        dict with 'embedding', 'logits', and 'loss' (if labels provided)
    """
    out = {}
    pool_strategy = getattr(self, "pool_strategy", "cls")
    out["embedding"] = _extract_embedding(self.backbone(batch["image"]), pool_strategy)
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        out["label"] = batch["label"]
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])
        self.log(
            f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True
        )

    return out
