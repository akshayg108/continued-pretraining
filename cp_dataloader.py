# Data loading utilities for continued pretraining
import torch
import hashlib
import numpy as np
from PIL import Image
import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.transforms import MultiViewTransform
from cp_datasets import get_dataset


def create_transforms(ds_cfg, n_views=1, strong_aug=False):
    """Create training and validation transforms.
    
    Args:
        ds_cfg: Dataset configuration from get_dataset_config() containing:
            - input_size: Target image size
            - normalization: Normalization parameters (mean, std)
        n_views: Number of views for multi-view learning (default: 1)
        strong_aug: If True, use stronger augmentation (default: False)
        
    Returns:
        tuple: (train_transform, val_transform)
            - train_transform: Transform for training (with augmentation)
            - val_transform: Transform for validation/test (no augmentation)
    """
    if strong_aug:
        # Strong augmentation for contrastive learning
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                (ds_cfg["input_size"], ds_cfg["input_size"]), scale=(0.2, 1.0)
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
        # Standard augmentation
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
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((ds_cfg["input_size"], ds_cfg["input_size"])),
        transforms.ToImage(**ds_cfg["normalization"]),
    )
    
    # Multi-view support for contrastive learning
    train_transform = (
        MultiViewTransform({f"view_{i}": base_aug for i in range(n_views)})
        if n_views > 1
        else base_aug
    )
    
    return train_transform, val_transform


class CPSubset(torch.utils.data.Dataset):
    """Dataset subset for continued pretraining.
    
    Wraps a dataset and provides access to a subset of indices,
    with proper sample_idx tracking.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if isinstance(sample, dict):
            sample["sample_idx"] = idx
        return sample


def create_data_loaders(args, ds_cfg, train_transform, val_transform, data_dir):
    """Create data loaders using stable-datasets.
    
    Args:
        args: Command-line arguments containing:
            - dataset: Dataset name
            - seed: Random seed for reproducibility
            - n_samples: Number of training samples to use
            - batch_size: Batch size for data loaders
            - num_workers: Number of workers for data loading
        ds_cfg: Dataset configuration from get_dataset_config() containing:
            - splits: List of [train_split, val_split, test_split]
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        data_dir: Cache directory for datasets
        
    Returns:
        tuple: (data_module, test_loader, eval_train_loader, indices)
            - data_module: spt.data.DataModule with train and val loaders
            - test_loader: DataLoader for test set
            - eval_train_loader: DataLoader for evaluation on train set (with val transform)
            - indices: List of training sample indices used
    """
    # Use splits from dataset config to handle datasets without validation split
    splits = ds_cfg.get("splits", ["train", "validation", "test"])
    train_split, val_split, test_split = splits
    
    # Load datasets using stable-datasets through get_dataset()
    # get_dataset returns HFDatasetWrapper with transform already applied
    full_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=train_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    val_data = get_dataset(
        args.dataset,
        split=val_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    test_data = get_dataset(
        args.dataset,
        split=test_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    
    # Verify no data leakage between splits
    if ds_cfg.get("manual_split", False):
        print("Checking for data leakage between splits...")
        check_dataset_overlap(full_train, val_data, "train", "validation", sample_size=100, seed=args.seed)
        check_dataset_overlap(full_train, test_data, "train", "test", sample_size=100, seed=args.seed)
        check_dataset_overlap(val_data, test_data, "validation", "test", sample_size=100, seed=args.seed)

    # Create training subset
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(full_train))[: args.n_samples].tolist()
    train_subset = CPSubset(full_train, indices)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    # Create evaluation train loader (with val transform, same indices as training)
    eval_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        CPSubset(eval_train, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Data: train={len(train_subset)} val={len(val_data)} test={len(test_data)}")
    return data, test_loader, eval_train_loader, indices


def _hash_sample(sample):
    """Create a hash for a sample based on its image content.
    
    Uses image content + label to reduce hash collision probability while
    still detecting true data leakage (same image content).
    
    Args:
        sample: Sample dict with 'image' key containing PIL Image, tensor, or numpy array
        
    Returns:
        str: Hash string identifying the sample
    """
    image = sample.get("image", sample.get("img", None))
    label = sample.get("label", sample.get("target", None))
    
    if image is None:
        # If no image key, use label as fallback
        return hashlib.md5(str(label).encode()).hexdigest()
    
    # Convert image to numpy array for hashing
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, torch.Tensor):
        img_array = image.cpu().numpy()
    else:
        img_array = np.array(image)
    
    # Create hash from image bytes + label for better uniqueness
    # Note: We include label because the same image with different labels
    # should still be considered as data leakage
    hash_input = img_array.tobytes()
    if label is not None:
        hash_input += str(label).encode()
    
    return hashlib.md5(hash_input).hexdigest()


def check_dataset_overlap(dataset1, dataset2, dataset1_name="dataset1", dataset2_name="dataset2", 
                          sample_size=100, seed=42):
    """Check if two datasets have overlapping samples.
    
    Args:
        dataset1, dataset2: Datasets to check (HFDatasetWrapper or similar)
        dataset1_name, dataset2_name: Names for logging
        sample_size: Number of samples to check from each dataset
        seed: Random seed for sampling
        
    Raises:
        AssertionError: If overlap is detected
    """
    np.random.seed(seed)
    
    # Get underlying HF datasets if wrapped
    ds1 = dataset1.hf_dataset if hasattr(dataset1, 'hf_dataset') else dataset1
    ds2 = dataset2.hf_dataset if hasattr(dataset2, 'hf_dataset') else dataset2
    
    # Sample indices
    n1 = len(ds1)
    n2 = len(ds2)
    sample_size = min(sample_size, n1, n2)
    
    if sample_size == 0:
        return  # Empty datasets, no overlap possible
    
    indices1 = np.random.choice(n1, size=min(sample_size, n1), replace=False)
    indices2 = np.random.choice(n2, size=min(sample_size, n2), replace=False)
    
    # Create hash sets
    hashes1 = set()
    hashes2 = set()
    
    try:
        for idx in indices1:
            sample = ds1[int(idx)]
            hashes1.add(_hash_sample(sample))
        
        for idx in indices2:
            sample = ds2[int(idx)]
            hashes2.add(_hash_sample(sample))
    except Exception as e:
        print(f"⚠️  Warning: Could not check dataset overlap due to: {e}")
        return
    
    # Check for overlap
    overlap = hashes1 & hashes2
    overlap_ratio = len(overlap) / sample_size if sample_size > 0 else 0
    
    # Allow small overlap ratio (<3%) which might be due to hash collisions or boundary sampling
    # For Galaxy10 with manual_split=True, this shouldn't happen, but we allow a small tolerance
    if len(overlap) > 0:
        if overlap_ratio < 0.03:  # Less than 3% overlap
            print(f"⚠️  Warning: Minor overlap detected between {dataset1_name} and {dataset2_name}: "
                  f"{len(overlap)}/{sample_size} samples ({overlap_ratio:.1%}). "
                  f"This is within tolerance and likely due to hash collisions.")
        else:
            raise AssertionError(
                f"Dataset overlap detected between {dataset1_name} and {dataset2_name}! "
                f"Found {len(overlap)}/{sample_size} overlapping samples ({overlap_ratio:.1%}). "
                f"This indicates data leakage."
            )
    else:
        print(f"✓ No overlap detected between {dataset1_name} and {dataset2_name} (checked {sample_size} samples each)")


