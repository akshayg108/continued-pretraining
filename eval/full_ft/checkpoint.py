"""Strict extraction of a timm backbone from trusted local CP checkpoints."""
from collections.abc import Mapping
from pathlib import Path


def discard_native_head(backbone):
    """FT reads forward_features plus a new classifier, not timm's native head.

    In particular, MAE CP's MaskedEncoder already removes CLIP's old projection.
    Keep attention pooling and normalization: these ARE used by SigLIP FT.
    """
    from torch import nn

    if not hasattr(backbone, "head"):
        raise ValueError("Expected one of the supported timm ViT backbones")
    backbone.head = nn.Identity()


def load_backbone_state(backbone, path):
    """Load all backbone tensors, never silently fill gaps with random weights.

    CP training used Lightning checkpoints containing optimizer state and Python
    metadata. These are trusted project artifacts, not arbitrary uploaded files.
    Prefix inference handles MaskedEncoder's ``backbone.vit`` wrapper.
    """
    import torch

    path = Path(path)
    if any(part in {"sft_pre", "sft_post"} for part in path.parts):
        raise ValueError(f"Expected a CP checkpoint, not an FT checkpoint: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Missing state_dict in {path}")
    if any(k.startswith(("classifier.", "module.classifier.")) for k in state):
        raise ValueError(f"Supervised classifier found in CP input: {path}")
    target = backbone.state_dict()
    if not target:
        raise ValueError("Cannot load a backbone with no state")
    # A single anchor generates candidates; every target key must then match.
    anchor = next(iter(target))
    prefixes = {k[:-len(anchor)] for k in state if isinstance(k, str) and k.endswith(anchor)}
    complete = [p for p in prefixes if all(p + k in state for k in target)]
    if len(complete) != 1:
        counts = {p: sum(p + k in state for k in target) for p in sorted(prefixes)}
        raise ValueError(f"Expected one complete backbone, got {len(complete)}; "
                         f"matched tensors {counts} / {len(target)} in {path}")
    prefix = complete[0]
    if any(token in prefix.lower().split(".") for token in ("teacher", "ema")):
        raise ValueError(f"Refusing a teacher/EMA backbone: {prefix}")
    selected = {}
    for key, expected in target.items():
        value = state[prefix + key]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
            raise ValueError(f"Backbone tensor shape mismatch: {prefix}{key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite backbone tensor: {prefix}{key}")
        selected[key] = value
    backbone.load_state_dict(selected, strict=True)
    return {"prefix": prefix, "n_tensors": len(selected)}
