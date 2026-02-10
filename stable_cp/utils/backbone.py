import torch
import torch.nn as nn
from stable_pretraining.backbone.utils import from_timm
from typing import Tuple


# Known backbone embedding dimensions
BACKBONE_DIMS = {
    # DINOv2 models (TIMM)
    "vit_small_patch14_dinov2.lvd142m": 384,
    "vit_base_patch14_dinov2.lvd142m": 768,
    "vit_large_patch14_dinov2.lvd142m": 1024,
    "vit_giant_patch14_dinov2.lvd142m": 1536,
    # DINOv3 models (TIMM)
    "vit_small_patch16_dinov3.lvd1689m": 384,
    "vit_base_patch16_dinov3.lvd1689m": 768,
    "vit_large_patch16_dinov3.lvd1689m": 1024,
    "vit_large_patch16_dinov3.sat493m": 1024,
    "vit_huge_plus_patch16_dinov3.lvd1689m": 1536,
    "vit_7b_patch16_dinov3.lvd1689m": 4096,
    # Base ViT models (TIMM)
    "vit_base_patch16_224": 768,
    "vit_large_patch16_224": 1024,
    "vit_huge_patch14_224": 1280,
    # MAE models (TIMM)
    "vit_base_patch16_224.mae": 768,
    "vit_large_patch16_224.mae": 1024,
    "vit_huge_patch14_224.mae": 1280,
}


def load_backbone(
    backbone_name: str, pretrained: bool = True
) -> Tuple[nn.Module, torch.device]:
    """Load a backbone network from TIMM.

    Args:
        backbone_name: Name of the TIMM model
        pretrained: Whether to load pretrained weights

    Returns:
        Tuple of (backbone, device)
    """
    backbone = from_timm(backbone_name, pretrained=pretrained)

    # Ensure all parameters are trainable
    for p in backbone.parameters():
        p.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone, device


def get_backbone_dim(backbone_name: str) -> int:
    """Get the embedding dimension for a backbone.

    Args:
        backbone_name: Name of the backbone model

    Returns:
        Embedding dimension

    Raises:
        ValueError: If backbone is unknown (falls back to 384)
    """
    if backbone_name in BACKBONE_DIMS:
        return BACKBONE_DIMS[backbone_name]
    else:
        print(f"Warning: Unknown backbone {backbone_name}, assuming embed_dim=384")
        return 384


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
