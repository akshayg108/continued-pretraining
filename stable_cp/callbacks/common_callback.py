# Common callbacks
import lightning as pl
import torch
import torch.nn as nn


class FreezeBackboneCallback(pl.Callback):
    # Backbone freezing for continued pretraining
    def __init__(
        self,
        freeze_epochs: int = 0,
        num_trained_blocks: int = -1,
        all_norm: bool = False,
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.num_trained_blocks = num_trained_blocks
        self.all_norm = all_norm
        self._backbone_frozen = False
        self._initial_freeze_applied = False

    def on_train_start(self, trainer, pl_module):
        # Apply initial freeze if freeze_epochs > 0
        if self.freeze_epochs > 0:
            self._freeze_backbone(pl_module)
            self._backbone_frozen = True
            self._initial_freeze_applied = True
            norm_info = " (norm layers will be enabled after freeze)" if self.all_norm else ""
            print(f"FreezeBackboneCallback: Backbone frozen for first {self.freeze_epochs} epochs{norm_info}")
        elif self.num_trained_blocks != -1 or self.all_norm:
            # If freeze_epochs=0 but num_trained_blocks or all_norm is set, apply selective training immediately
            self._apply_selective_unfreezing(pl_module)
            self._print_training_mode()

    def on_train_epoch_start(self, trainer, pl_module):
        # Handle freezing/unfreezing at epoch boundaries
        current_epoch = trainer.current_epoch

        # Check if we should transition from frozen to selective training
        if self._backbone_frozen and current_epoch >= self.freeze_epochs:
            self._apply_selective_unfreezing(pl_module)
            self._backbone_frozen = False
            print(f"Epoch {current_epoch}: ", end="")
            self._print_training_mode()

    def _print_training_mode(self):
        """Print current training mode based on configuration"""
        if self.all_norm and self.num_trained_blocks == 0:
            print("Training norm layers only (TENT norm_only mode)")
        elif self.all_norm and self.num_trained_blocks > 0:
            print(f"Training norm layers + last {self.num_trained_blocks} blocks (TENT combined mode)")
        elif self.num_trained_blocks == -1:
            print("Backbone unfrozen (full fine-tuning)")
        elif self.num_trained_blocks == 0:
            print("Backbone frozen (head-only training)")
        else:
            print(f"Training last {self.num_trained_blocks} blocks")

    def _freeze_backbone(self, pl_module):
        # Freeze all backbone parameters
        if not hasattr(pl_module, "backbone"):
            print("Warning: Module has no 'backbone' attribute, skipping freeze")
            return

        pl_module.backbone.eval()  # Set entire backbone to eval mode (for Dropout, etc.)
        for param in pl_module.backbone.parameters():
            param.requires_grad = False

        # Ensure BatchNorm layers stay in eval mode
        for module in pl_module.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def _apply_selective_unfreezing(self, pl_module):
        """Apply selective parameter training based on num_trained_blocks and all_norm settings"""
        if not hasattr(pl_module, "backbone"):
            return

        # Step 1: Freeze all parameters first
        for param in pl_module.backbone.parameters():
            param.requires_grad = False

        # Step 2: Handle normalization layers if all_norm=True
        if self.all_norm:
            pl_module.backbone.train()
            self._enable_norm_layers(pl_module.backbone)

        # Step 3: Handle block-wise unfreezing based on num_trained_blocks
        if self.num_trained_blocks == 0:
            # Only norm layers (if all_norm=True) or completely frozen
            if not self.all_norm:
                # Standard head-only training: ensure backbone is in eval mode
                pl_module.backbone.eval()
                # Ensure BatchNorm layers stay in eval mode
                for module in pl_module.backbone.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.eval()
            # else: TENT mode - backbone in train mode, norm layers enabled
            return

        if self.num_trained_blocks == -1:
            # Train all blocks (in addition to norm layers if all_norm=True)
            pl_module.backbone.train()
            for param in pl_module.backbone.parameters():
                param.requires_grad = True
            return

        # Step 4: Selective training - train last N blocks (+ norm layers if all_norm=True)
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

    def _enable_norm_layers(self, backbone):
        """Enable all normalization layers for training (TENT-style adaptation)"""
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.train()  # Set to train mode (override previous .eval() call)
                # Force use of batch statistics (TENT requirement)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm always uses batch statistics
                m.requires_grad_(True)
                m.train()  # Set to train mode

    def _find_transformer_layers(self, backbone):
        """Find transformer/CNN blocks for TIMM models.
        
        This function supports:
        - TIMM ViT models (direct): backbone.blocks
        - Wrapped ViT (e.g., MaskedEncoder): backbone.vit.blocks
        - TIMM ResNet models: backbone.layer1-4
        """
        # MaskedEncoder (stable_pretraining wrapper for ViT)
        if hasattr(backbone, "vit") and hasattr(backbone.vit, "blocks"):
            return backbone.vit.blocks
        
        # Direct TIMM ViT (e.g., from_timm("vit_base_patch16_224"))
        if hasattr(backbone, "blocks"):
            return backbone.blocks

        # TIMM ResNet models (e.g., from_timm("resnet50"))
        if hasattr(backbone, "layer4"):
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
