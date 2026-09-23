"""Default feature readouts used by the supported pretrained ViTs."""


def default_pool_strategy(backbone_name):
    if "siglip" in backbone_name:
        return "map"
    if backbone_name.endswith(".mae"):
        return "mean"
    return "cls"
