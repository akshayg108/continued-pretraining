"""
TENT Continued Pretraining - Forward pass and loss computation.

Core functions adapted from tent/tent.py for test-time entropy minimization.
"""
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
    
    # 1. Forward through backbone
    if hasattr(self.backbone, 'forward_features'):
        # timm models: use forward_features to get token sequence
        tokens = self.backbone.forward_features(images)
    else:
        # HuggingFace models: regular forward
        backbone_out = self.backbone(images)
        tokens = backbone_out.last_hidden_state if hasattr(backbone_out, "last_hidden_state") else backbone_out
    
    # 2. Use CLS token for classification (same as MAE)
    cls_token = tokens[:, 0]  # [B, D]
    batch["embedding"] = cls_token  # For online linear probe
    
    # 3. Get logits through classifier head
    logits = self.classifier(cls_token)  # [B, num_classes]
    
    # 4. Compute entropy loss (adapted from tent/tent.py)
    entropy = softmax_entropy(logits)  # [B]
    loss = entropy.mean(0)
    
    batch["loss"] = loss
    return batch


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    
    Adapted from tent/tent.py - Extended to support LayerNorm for ViT.
    
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # Support both BatchNorm2d (CNNs) and LayerNorm (ViT)
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        elif isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # normalized_shape parameters
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model_for_tent(backbone, classifier, tent_mode='norm_only', num_trained_blocks=-1):
    """Configure model for TENT adaptation based on mode.
    
    tent_mode:
    - 'norm_only': Only update normalization layers + classifier (from tent/tent.py)
    - 'combined': Update last N blocks + all norm layers + classifier
    
    Args:
        backbone: The backbone model
        classifier: The classifier head
        tent_mode: Update mode
        num_trained_blocks: Number of last blocks to train (for combined mode)
    
    Note: Classifier is always updated to avoid random logits affecting entropy loss.
    """
    if tent_mode == 'norm_only':
        # Adapted from tent/tent.py configure_model
        # Freeze backbone parameters
        backbone.requires_grad_(False)
        
        # Enable classifier 
        classifier.requires_grad_(True)
        
        # Enable only normalization layers in backbone
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # Force use of batch statistics
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm always uses batch statistics
                m.requires_grad_(True)
        
        print(f"TENT mode: norm_only - Updating classifier + normalization layers")
    
    elif tent_mode == 'combined':
        # Update last N blocks + all norm layers + classifier
        # 1. Freeze backbone
        backbone.requires_grad_(False)
        
        # 2. Enable classifier
        classifier.requires_grad_(True)
        
        # 3. Enable all normalization layers
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        
        # 4. Enable last N transformer blocks
        if num_trained_blocks > 0:
            layers = _find_transformer_layers(backbone)
            if layers is not None:
                total_blocks = len(layers)
                blocks_to_train = min(num_trained_blocks, total_blocks)
                start_idx = total_blocks - blocks_to_train
                
                for i in range(start_idx, total_blocks):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                
                print(f"TENT mode: combined - Training blocks {start_idx}-{total_blocks-1} + all norm layers + classifier")
            else:
                print(f"TENT mode: combined - Could not find transformer layers, only norm layers + classifier enabled")
        else:
            print(f"TENT mode: combined - Only norm layers + classifier enabled (num_trained_blocks={num_trained_blocks})")
    
    return backbone, classifier


def _find_transformer_layers(backbone):
    """Find transformer layers in the backbone.
    
    Helper function for combined mode.
    """
    # MaskedEncoder (stable_pretraining wrapper)
    if hasattr(backbone, "vit") and hasattr(backbone.vit, "blocks"):
        return backbone.vit.blocks
    
    # Direct timm ViT
    if hasattr(backbone, "blocks"):
        return backbone.blocks
    
    # HuggingFace ViT
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    
    return None

