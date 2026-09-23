import lightning as pl
import torch


class FreezeBackboneCallback(pl.Callback):
    """Freeze the backbone during warmup, then train the requested final blocks."""

    def __init__(
        self,
        freeze_epochs: int = 0,
        num_trained_blocks: int = -1,
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.num_trained_blocks = num_trained_blocks
        self._backbone_frozen = False

    def on_train_start(self, trainer, pl_module):
        if self.freeze_epochs > 0:
            self._freeze_backbone(pl_module)
            self._backbone_frozen = True
            print(f"FreezeBackboneCallback: Backbone frozen for first {self.freeze_epochs} epochs")

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if self._backbone_frozen and current_epoch >= self.freeze_epochs:
            self._apply_selective_unfreezing(pl_module)
            self._backbone_frozen = False

            if self.num_trained_blocks == -1:
                print(f"Epoch {current_epoch}: All backbone parameters unfrozen")
            elif self.num_trained_blocks == 0:
                print(f"Epoch {current_epoch}: Backbone remains frozen (head-only)")
            else:
                print(f"Epoch {current_epoch}: Training last {self.num_trained_blocks} blocks")

    def _freeze_backbone(self, pl_module):
        if not hasattr(pl_module, "backbone"):
            print("Warning: Module has no 'backbone' attribute, skipping freeze")
            return

        pl_module.backbone.eval()
        for param in pl_module.backbone.parameters():
            param.requires_grad = False

        for module in pl_module.backbone.modules():
            if isinstance(
                module,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
            ):
                module.eval()

        # MAE masking remains active during warmup while the wrapped ViT stays frozen.
        if hasattr(pl_module.backbone, "masking") and pl_module.backbone.masking is not None:
            pl_module.backbone.training = True
            pl_module.backbone.masking.training = True

    def _apply_selective_unfreezing(self, pl_module):
        if not hasattr(pl_module, "backbone"):
            return

        if self.num_trained_blocks == 0:
            return

        if self.num_trained_blocks == -1:
            pl_module.backbone.train()
            for param in pl_module.backbone.parameters():
                param.requires_grad = True
            return

        layers = self._find_transformer_layers(pl_module.backbone)

        if layers is not None:
            total_blocks = len(layers)
            blocks_to_train = min(self.num_trained_blocks, total_blocks)
            start_idx = total_blocks - blocks_to_train

            pl_module.backbone.train()
            for i in range(start_idx, total_blocks):
                for param in layers[i].parameters():
                    param.requires_grad = True

            print(f"Selectively training blocks {start_idx} to {total_blocks - 1}")
        else:
            print("Warning: Could not find transformer layers, unfreezing all parameters")
            pl_module.backbone.train()
            for param in pl_module.backbone.parameters():
                param.requires_grad = True

    def _find_transformer_layers(self, backbone):
        if hasattr(backbone, "blocks"):
            return backbone.blocks

        if hasattr(backbone, "model") and hasattr(backbone.model, "blocks"):
            return backbone.model.blocks

        if hasattr(backbone, "vit") and hasattr(backbone.vit, "blocks"):
            return backbone.vit.blocks

        if hasattr(backbone, "layer4"):
            layers = []
            for i in range(1, 5):
                layer = getattr(backbone, f"layer{i}", None)
                if layer is not None:
                    layers.append(layer)
            if layers:
                return layers

        return None
