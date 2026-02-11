import torch
import torch.nn as nn


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.

    Adapted from tent/tent.py
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def tent_cp_forward(self, batch, stage):
    """TENT continued pretraining forward pass.

    1. Forward through backbone to get embeddings
    2. Pass through classifier to get logits
    3. Compute entropy loss for test-time adaptation
    4. Save embedding for online linear probe

    Adapted from tent/tent.py forward_and_adapt logic.
    """
    images = batch["image"]

    # 1. Forward through backbone to get token sequence
    tokens = self.backbone.forward_features(images)

    # 2. Use CLS token for classification
    cls_token = tokens[:, 0]  # [B, D]
    batch["embedding"] = cls_token  # For online linear probe

    # 3. Get logits through classifier head
    logits = self.classifier(cls_token)  # [B, num_classes]

    # 4. Compute entropy loss (adapted from tent/tent.py)
    entropy = softmax_entropy(logits)  # [B]
    loss = entropy.mean(0)

    batch["loss"] = loss
    return batch
