"""Gradient equivalence for masked-encoder activation recomputation."""

import copy
import unittest
from unittest.mock import patch

import torch
from timm.models.vision_transformer import VisionTransformer

from stable_cp.methods.mae.masked_encoder import NativeMaskedEncoder


class MaskedCheckpointingTests(unittest.TestCase):
    def test_checkpointed_forward_and_gradients_match(self):
        torch.manual_seed(42)
        vit = VisionTransformer(img_size=32, patch_size=16, embed_dim=32,
                                depth=2, num_heads=4, num_classes=0)
        original = NativeMaskedEncoder(vit, masking=None).train()
        recomputed = copy.deepcopy(original)
        recomputed.vit.set_grad_checkpointing(True)
        images = torch.randn(2, 3, 32, 32)
        expected = original(images).encoded
        with patch("stable_cp.methods.mae.masked_encoder.checkpoint",
                   wraps=torch.utils.checkpoint.checkpoint, create=True) as checkpoint:
            actual = recomputed(images).encoded
            self.assertEqual(checkpoint.call_count, 2)
            actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for left, right in zip(original.parameters(), recomputed.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad)


if __name__ == "__main__":
    unittest.main()
