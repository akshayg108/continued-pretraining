# Dataset registry for continued pretraining (using stable-datasets)
from pathlib import Path

import stable_pretraining as spt
from stable_datasets import images as stable_ds

# Dataset configuration registry
DATASETS = {
    # CIFAR datasets
    "cifar10": {
        "dataset_class": stable_ds.CIFAR10,
        "config_name": None,
        "num_classes": 10,
        "input_size": 224,
        "normalization": "cifar10",
        "splits": ["train", "test", "test"],
    },
    "cifar100": {
        "dataset_class": stable_ds.CIFAR100,
        "config_name": None,
        "num_classes": 100,
        "input_size": 224,
        "normalization": "cifar100",
        "splits": ["train", "test", "test"],
    },
    # Food and Objects
    "food101": {
        "dataset_class": stable_ds.Food101,
        "config_name": None,
        "num_classes": 101,
        "input_size": 224,
        "normalization": "imagenet",
        "splits": ["train", "test", "test"],
    },
    # Fine-Grained Classification
    "fgvc_aircraft": {
        "dataset_class": stable_ds.FGVCAircraft,
        "config_name": "variant",  # Options: "variant" (100), "family" (70), "manufacturer" (30)
        "num_classes": 100,
        "input_size": 224,
        "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
    # Galaxy dataset
    "galaxy10": {
        "dataset_class": stable_ds.Galaxy10Decal,
        "config_name": None,
        "num_classes": 10,
        "input_size": 224,
        "normalization": "galaxy10",
        "splits": ["train", "validation", "test"],
        "manual_split": True,  # Force manual splitting from full dataset (avoid data leakage)
    },
    # MedMNIST datasets
    "bloodmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "bloodmnist",
        "num_classes": 8,
        "input_size": 224,
        "normalization": "bloodmnist",
        "splits": ["train", "validation", "test"],
    },
    "tissuemnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "tissuemnist",
        "num_classes": 8,
        "input_size": 224,
        "normalization": "tissuemnist",
        "splits": ["train", "validation", "test"],
    },
    "pathmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "pathmnist",
        "num_classes": 9,
        "input_size": 224,
        "normalization": "pathmnist",
        "splits": ["train", "validation", "test"],
    },
    "dermamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "dermamnist",
        "num_classes": 7,
        "input_size": 224,
        "normalization": "dermamnist",
        "splits": ["train", "validation", "test"],
    },
    "octmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "octmnist",
        "num_classes": 4,
        "input_size": 224,
        "normalization": "octmnist",
        "splits": ["train", "validation", "test"],
    },
    "pneumoniamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "pneumoniamnist",
        "num_classes": 2,
        "input_size": 224,
        "normalization": "pneumoniamnist",
        "splits": ["train", "validation", "test"],
    },
    "retinamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "retinamnist",
        "num_classes": 5,
        "input_size": 224,
        "normalization": "retinamnist",
        "splits": ["train", "validation", "test"],
    },
    "breastmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "breastmnist",
        "num_classes": 2,
        "input_size": 224,
        "normalization": "breastmnist",
        "splits": ["train", "validation", "test"],
    },
    "organamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organamnist",
        "num_classes": 11,
        "input_size": 224,
        "normalization": "organamnist",
        "splits": ["train", "validation", "test"],
    },
    "organcmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organcmnist",
        "num_classes": 11,
        "input_size": 224,
        "normalization": "organcmnist",
        "splits": ["train", "validation", "test"],
    },
    "organsmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organsmnist",
        "num_classes": 11,
        "input_size": 224,
        "normalization": "organsmnist",
        "splits": ["train", "validation", "test"],
    },
}

# Normalization presets
# Using custom statistics from temp.py for better performance on each dataset
NORMALIZATIONS = {
    # Standard presets from stable-pretraining
    "imagenet": spt.data.static.ImageNet,  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    "cifar10": spt.data.static.CIFAR10,  # mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    "cifar100": spt.data.static.CIFAR100,  # mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    # Datasets using ImageNet statistics (for reference, not yet in stable-datasets)
    "plantnet300k": spt.data.static.ImageNet,
    "crop14_balance": spt.data.static.ImageNet,
    "fgvc_aircraft": spt.data.static.ImageNet,
    # Custom statistics computed for specific datasets
    "galaxy10": {"mean": [0.097, 0.097, 0.097], "std": [0.174, 0.164, 0.156]},
    # MedMNIST custom statistics (RGB datasets)
    "bloodmnist": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "pathmnist": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "dermamnist": {"mean": [0.7634, 0.5423, 0.5698], "std": [0.0841, 0.1246, 0.1043]},
    "retinamnist": {"mean": [0.1706, 0.1706, 0.1706], "std": [0.1946, 0.1946, 0.1946]},
    "organamnist": {"mean": [0.4996, 0.4996, 0.4996], "std": [0.1731, 0.1731, 0.1731]},
    "organcmnist": {"mean": [0.4996, 0.4996, 0.4996], "std": [0.1731, 0.1731, 0.1731]},
    "organsmnist": {"mean": [0.4996, 0.4996, 0.4996], "std": [0.1731, 0.1731, 0.1731]},
    # MedMNIST custom statistics (Grayscale datasets - single channel)
    "tissuemnist": {"mean": [0.5], "std": [0.5]},
    "chestmnist": {"mean": [0.4984], "std": [0.2483]},
    "octmnist": {"mean": [0.1778], "std": [0.1316]},
    "pneumoniamnist": {"mean": [0.5060], "std": [0.2537]},
    "breastmnist": {"mean": [0.4846], "std": [0.2522]},
}


class HFDatasetWrapper(spt.data.Dataset):
    """Wrapper for HuggingFace datasets with transform support.

    This wrapper enables stable-datasets (which return HF Dataset objects) to be
    used with stable-pretraining's transform pipeline.
    """

    def __init__(self, hf_dataset, transform=None):
        super().__init__(transform)
        self.hf_dataset = hf_dataset
        # Add sample_idx if not present (required by some stable-pretraining callbacks)
        if "sample_idx" not in hf_dataset.column_names:
            self.hf_dataset = hf_dataset.add_column(
                "sample_idx", list(range(hf_dataset.num_rows))
            )

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        return self.process_sample(sample)

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def column_names(self):
        return self.hf_dataset.column_names


def get_dataset_config(name):
    """Get dataset configuration.

    Args:
        name: Dataset name from DATASETS registry

    Returns:
        dict: Dataset configuration with normalization preset resolved
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    cfg = DATASETS[name].copy()
    cfg["normalization"] = NORMALIZATIONS[cfg["normalization"]]
    return cfg


def get_dataset(name, split, transform, cache_dir="/.cache", seed=42):
    """Load dataset by name using stable-datasets.

    Args:
        name: Dataset name from DATASETS registry
        split: Split name ("train", "validation", or "test")
        transform: Transform to apply to samples
        cache_dir: Base cache directory for downloads and processed data
        seed: Random seed for splitting datasets that only have a single split

    Returns:
        HFDatasetWrapper: Wrapped dataset compatible with stable-pretraining
    """
    cfg = DATASETS[name]
    cache_dir = Path(cache_dir)

    # Create cache subdirectories
    download_dir = cache_dir / "stable_datasets" / "downloads"
    processed_cache_dir = cache_dir / "stable_datasets" / "processed"
    download_dir.mkdir(parents=True, exist_ok=True)
    processed_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_class = cfg["dataset_class"]
    config_name = cfg["config_name"]

    # For datasets that need manual splitting (e.g., Galaxy10 with only train split),
    # force loading full dataset and split manually to avoid data leakage
    if cfg.get("manual_split", False):
        if config_name is not None:
            hf_ds = dataset_class(
                split=None,
                config_name=config_name,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
        else:
            hf_ds = dataset_class(
                split=None,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
        hf_ds = _handle_split_from_dict(hf_ds, split, seed)
        return HFDatasetWrapper(hf_ds, transform=transform)

    # Load the raw HF dataset from stable-datasets
    if config_name is not None:
        # Dataset with config (e.g., MedMNIST variants)
        try:
            hf_ds = dataset_class(
                split=split,
                config_name=config_name,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
        except (ValueError, KeyError):
            # If requested split doesn't exist, load all and split manually
            hf_ds = dataset_class(
                split=None,
                config_name=config_name,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
            hf_ds = _handle_split_from_dict(hf_ds, split, seed)
    else:
        # Dataset without config
        try:
            hf_ds = dataset_class(
                split=split,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
        except (ValueError, KeyError):
            # If requested split doesn't exist, load all and split manually
            hf_ds = dataset_class(
                split=None,
                download_dir=str(download_dir),
                processed_cache_dir=str(processed_cache_dir),
            )
            hf_ds = _handle_split_from_dict(hf_ds, split, seed)

    # Wrap in HFDatasetWrapper for stable-pretraining compatibility
    return HFDatasetWrapper(hf_ds, transform=transform)


def _handle_split_from_dict(dataset_dict, split, seed=42):
    """Handle DatasetDict returned when split=None.

    Args:
        dataset_dict: HuggingFace DatasetDict
        split: Requested split name
        seed: Random seed for splitting

    Returns:
        Dataset: The requested split
    """
    # Map common split name variations
    split_map = {
        "validation": ["validation", "val", "valid"],
        "val": ["validation", "val", "valid"],
        "test": ["test"],
        "train": ["train"],
    }

    # Try to find the split in the DatasetDict
    possible_names = split_map.get(split, [split])
    for name in possible_names:
        if name in dataset_dict:
            return dataset_dict[name]

    # If split not found, create it from train split
    if "train" in dataset_dict:
        return _split_single_dataset(dataset_dict["train"], split, seed)

    raise ValueError(f"Split '{split}' not found in dataset and cannot be created")


def _split_single_dataset(hf_dataset, split, seed=42, val_ratio=0.1, test_ratio=0.1):
    """Split a single dataset into train/val/test.

    Args:
        hf_dataset: HuggingFace Dataset (single split)
        split: Requested split ("train", "validation", or "test")
        seed: Random seed for splitting
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Dataset: The requested split
    """
    # For Galaxy10 and similar datasets with only one split
    if split == "train":
        # Keep (1 - val_ratio - test_ratio) for train
        temp_split = hf_dataset.train_test_split(
            test_size=val_ratio + test_ratio, seed=seed
        )
        return temp_split["train"]
    elif split in ["validation", "val"]:
        # Take val_ratio
        temp_split = hf_dataset.train_test_split(
            test_size=val_ratio + test_ratio, seed=seed
        )
        val_test = temp_split["test"]
        val_test_split = val_test.train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=seed
        )
        return val_test_split["train"]
    elif split == "test":
        # Take test_ratio
        temp_split = hf_dataset.train_test_split(
            test_size=val_ratio + test_ratio, seed=seed
        )
        val_test = temp_split["test"]
        val_test_split = val_test.train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=seed
        )
        return val_test_split["test"]
    else:
        raise ValueError(f"Unknown split: {split}")
