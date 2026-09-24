"""Default feature readouts used by the supported pretrained ViTs."""


def default_pool_strategy(backbone_name):
    if "siglip" in backbone_name:
        return "map"
    if backbone_name.endswith(".mae"):
        return "mean"
    return "cls"


def feature_readout(backbone_name, pool_strategy):
    """Identify readouts whose normalization order affects stored features."""
    if backbone_name.endswith(".mae"):
        if pool_strategy == "mean":
            return "mae_patch_mean_pretrained_ln_v1"
        if pool_strategy == "cls":
            return "mae_cls_pretrained_ln_v1"
    return pool_strategy


def is_mae_backbone(backbone):
    return getattr(backbone, "pretrained_cfg", {}).get("tag") == "mae"


def forward_embedding(backbone, images, pool_strategy="cls"):
    """Read global features without changing the backbone's token-level forward."""
    if is_mae_backbone(backbone) and pool_strategy == "mean":
        # timm intermediates exclude prefix tokens and precede the final LayerNorm.
        patches = backbone.forward_intermediates(
            images, indices=1, norm=False, output_fmt="NLC", intermediates_only=True
        )[0]
        return backbone.norm(patches.mean(dim=1))

    tokens = backbone.forward_features(images)
    if tokens.ndim != 3:
        return tokens
    if pool_strategy == "map":
        return backbone.fc_norm(backbone.attn_pool(tokens))
    if pool_strategy == "mean":
        return tokens[:, 1:, :].mean(dim=1)
    return tokens[:, 0, :]
