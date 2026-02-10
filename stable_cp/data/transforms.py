"""Transform utilities for data augmentation."""
import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.transforms import MultiViewTransform


def create_transforms(ds_cfg, n_views=1, strong_aug=False):
    """Create training and validation transforms.
    
    Args:
        ds_cfg: Dataset configuration dictionary
        n_views: Number of augmented views to generate
        strong_aug: Whether to use strong augmentation (for contrastive methods)
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    if strong_aug:
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                (ds_cfg["input_size"], ds_cfg["input_size"]), 
                scale=(0.2, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    else:
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((ds_cfg["input_size"], ds_cfg["input_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0), p=0.2),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
        
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((ds_cfg["input_size"], ds_cfg["input_size"])),
        transforms.ToImage(**ds_cfg["normalization"]),
    )
    
    if n_views > 1:
        train_transform = MultiViewTransform(
            {f"view_{i}": base_aug for i in range(n_views)}
        )
    else:
        train_transform = base_aug
        
    return train_transform, val_transform
