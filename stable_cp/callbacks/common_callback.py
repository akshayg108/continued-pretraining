# Common callbacks
import lightning as pl
import torch


class FreezeBackboneCallback(pl.Callback):
    # Backbone freezing for continued pretraining
    def __init__(
        self,
        freeze_epochs: int = 0,
        num_trained_blocks: int = -1,
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.num_trained_blocks = num_trained_blocks
        self._backbone_frozen = False
        self._initial_freeze_applied = False

    def on_train_start(self, trainer, pl_module):
        # Apply initial freeze if freeze_epochs > 0
        if self.freeze_epochs > 0:
            self._freeze_backbone(pl_module)
            self._backbone_frozen = True
            self._initial_freeze_applied = True
            print(f"FreezeBackboneCallback: Backbone frozen for first {self.freeze_epochs} epochs")
        elif self.num_trained_blocks != -1:
            # If freeze_epochs=0 but num_trained_blocks is set, apply selective training immediately
            self._apply_selective_unfreezing(pl_module)
            if self.num_trained_blocks == 0:
                print(f"FreezeBackboneCallback: Backbone frozen (head-only training)")
            else:
                print(f"FreezeBackboneCallback: Training last {self.num_trained_blocks} blocks")

    def on_train_epoch_start(self, trainer, pl_module):
        # Handle freezing/unfreezing at epoch boundaries
        current_epoch = trainer.current_epoch

        # Check if we should transition from frozen to selective training
        if self._backbone_frozen and current_epoch >= self.freeze_epochs:
            self._apply_selective_unfreezing(pl_module)
            self._backbone_frozen = False

            if self.num_trained_blocks == -1:
                print(f"Epoch {current_epoch}: Backbone unfrozen (full fine-tuning)")
            elif self.num_trained_blocks == 0:
                print(f"Epoch {current_epoch}: Backbone remains frozen (head-only)")
            else:
                print(f"Epoch {current_epoch}: Training last {self.num_trained_blocks} blocks")

    def _freeze_backbone(self, pl_module):
        # Freeze all backbone parameters
        if not hasattr(pl_module, "backbone"):
            print("Warning: Module has no 'backbone' attribute, skipping freeze")
            return

        for param in pl_module.backbone.parameters():
            param.requires_grad = False

        # Ensure BatchNorm layers stay in eval mode
        for module in pl_module.backbone.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()

    def _apply_selective_unfreezing(self, pl_module):
        # Apply selective parameter training based on num_trained_blocks setting
        if not hasattr(pl_module, "backbone"):
            return

        if self.num_trained_blocks == 0:
            # Keep all frozen (or freeze if not already frozen)
            if not self._backbone_frozen:
                self._freeze_backbone(pl_module)
            return

        if self.num_trained_blocks == -1:
            # Train all blocks
            pl_module.backbone.train()
            for param in pl_module.backbone.parameters():
                param.requires_grad = True
            return

        # Selective training: train only last N blocks
        # Step 1: Freeze all parameters first
        for param in pl_module.backbone.parameters():
            param.requires_grad = False
        
        # Step 2: Find transformer blocks and unfreeze last N blocks
        layers = self._find_transformer_layers(pl_module.backbone)

        if layers is not None:
            total_blocks = len(layers)
            blocks_to_train = min(self.num_trained_blocks, total_blocks)
            start_idx = total_blocks - blocks_to_train

            # Unfreeze specific layers
            for i in range(start_idx, total_blocks):
                for param in layers[i].parameters():
                    param.requires_grad = True

            # Set backbone to train mode (for dropout, etc.)
            pl_module.backbone.train()
            print(f"Selectively training blocks {start_idx} to {total_blocks - 1} (out of {total_blocks} total)")
        else:
            # Fallback: unfreeze everything with warning
            print("Warning: Could not find transformer layers, unfreezing all parameters")
            pl_module.backbone.train()
            for param in pl_module.backbone.parameters():
                param.requires_grad = True

    def _find_transformer_layers(self, backbone):
        # Find transformer layers in various model architectures
        
        # MaskedEncoder (stable_pretraining wrapper)
        if hasattr(backbone, "vit") and hasattr(backbone.vit, "blocks"):
            return backbone.vit.blocks
        
        # HuggingFace ViT wrapped in a container
        if hasattr(backbone, "model"):
            inner = backbone.model
            if hasattr(inner, "encoder") and hasattr(inner.encoder, "layer"):
                return inner.encoder.layer
            if hasattr(inner, "layer"):
                return inner.layer
            if hasattr(inner, "vit") and hasattr(inner.vit, "encoder"):
                return inner.vit.encoder.layer

        # Direct HuggingFace ViT
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            return backbone.encoder.layer

        # timm ViT (direct, not wrapped)
        if hasattr(backbone, "blocks"):
            return backbone.blocks

        # ResNet layers
        if hasattr(backbone, "layer4"):
            # Return list of ResNet layers
            layers = []
            for i in range(1, 5):
                layer = getattr(backbone, f"layer{i}", None)
                if layer is not None:
                    layers.append(layer)
            if layers:
                return layers

        return None


class GradientClipCallback(pl.Callback):
    # Gradient clipping during training
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )
