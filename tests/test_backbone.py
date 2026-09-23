import pytest

from stable_cp.utils import backbone


@pytest.mark.parametrize(
    "name,expected",
    [
        ("vit_base_patch16_dinov3.lvd1689m", "cls"),
        ("vit_large_patch16_dinov3.lvd1689m", "cls"),
        ("vit_base_patch16_clip_224.openai", "cls"),
        ("vit_base_patch16_224.mae", "mean"),
        ("vit_base_patch16_siglip_224.v2_webli", "map"),
    ],
)
def test_pretrained_encoder_pooling(name, expected):
    assert backbone.default_pool_strategy(name) == expected
