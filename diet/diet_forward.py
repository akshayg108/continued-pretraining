# DIET forward - instance discrimination with optional MixUp/CutMix
import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def _extract_embedding(backbone_output, pool_strategy="cls"):
    # Extract embedding: CLS token for DINOv2/MAE, mean pooling for I-JEPA
    if hasattr(backbone_output, "last_hidden_state"):
        tokens = backbone_output.last_hidden_state
    elif backbone_output.ndim == 3:
        tokens = backbone_output
    else:
        return backbone_output  # Already 2D (ResNet)

    if pool_strategy == "mean":
        return tokens[:, 1:, :].mean(dim=1)  # Mean over patch tokens (exclude CLS)
    return tokens[:, 0, :]  # CLS token (default)


def diet_forward(self, batch, stage):
    out = {}
    images, sample_idx = batch["image"], batch["sample_idx"]

    # MixUp/CutMix (lazy init with probability and switch_prob control)
    mixup_prob = getattr(self, "mixup_cutmix_prob", 0.8)
    if self.training and (getattr(self, "mixup_alpha", 0) > 0 or getattr(self, "cutmix_alpha", 0) > 0):
        if torch.rand(1).item() < mixup_prob:
            if not hasattr(self, "_mixup_cutmix"):
                t = []
                if self.mixup_alpha > 0:
                    t.append(v2.MixUp(alpha=self.mixup_alpha, num_classes=self.num_samples))
                if self.cutmix_alpha > 0:
                    t.append(v2.CutMix(alpha=self.cutmix_alpha, num_classes=self.num_samples))
                if len(t) > 1:
                    switch_prob = getattr(self, "mixup_cutmix_switch_prob", 0.5)
                    self._mixup_cutmix = v2.RandomChoice(t, p=[1 - switch_prob, switch_prob])
                else:
                    self._mixup_cutmix = t[0]
            images, sample_idx = self._mixup_cutmix(images, sample_idx)

    # Extract embedding (pool_strategy: "cls" for DINOv2/MAE, "mean" for I-JEPA)
    pool_strategy = getattr(self, "pool_strategy", "cls")
    embedding = _extract_embedding(self.backbone(images), pool_strategy)
    out["embedding"] = embedding

    if "label" in batch:
        out["label"] = batch["label"]

    if self.training:
        logits = self.diet_head(F.normalize(embedding, p=2, dim=1))
        # Handle both hard labels [B] and soft labels [B, C] from MixUp/CutMix
        if sample_idx.ndim == 2:
            out["loss"] = F.cross_entropy(logits, sample_idx)  # soft targets
        else:
            out["loss"] = self.diet_loss(logits, sample_idx)  # hard targets with label_smoothing
        self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)

    return out
