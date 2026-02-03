# Dataset registry for continued pretraining (HuggingFace + custom loaders)
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import stable_pretraining as spt


class Galaxy10Dataset(Dataset):
    # Galaxy10 DECaLS dataset from HDF5
    def __init__(self, h5_path, indices=None):
        import h5py
        from PIL import Image
        self._Image = Image
        with h5py.File(h5_path, 'r') as f:
            self.images, self.labels = f['images'][:], f['ans'][:]
        if indices is not None:
            self.images, self.labels = self.images[indices], self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._Image.fromarray(self.images[idx], 'RGB'), int(self.labels[idx])


def load_galaxy10(split, transform, cache_dir, val_split=0.1, seed=42):
    # Load Galaxy10 dataset with train/val/test splits
    import h5py
    h5_path = Path(cache_dir) / "galaxy10" / "Galaxy10_DECals.h5"

    with h5py.File(h5_path, 'r') as f:
        n_samples = len(f['ans'])

    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_size = int(n_samples * val_split)

    splits = {
        "train": indices[val_size:],
        "validation": indices[:val_size],
        "test": indices[:val_size],  # Use val as test for galaxy10
    }

    base_dataset = Galaxy10Dataset(h5_path, indices=splits[split])
    return spt.data.FromTorchDataset(base_dataset, names=["image", "label"], transform=transform)


def load_hf_dataset(hf_name, split, transform, cache_dir, hf_config=None, **kwargs):
    # Load HuggingFace dataset
    return spt.data.HFDataset(
        hf_name, name=hf_config, split=split, transform=transform,
        trust_remote_code=True, cache_dir=str(cache_dir), **kwargs
    )


DATASETS = {
    # HuggingFace datasets
    "fgvc-aircraft": {
        "loader": "hf",
        "hf_name": "randall-lab/fgvc-aircraft",
        "num_classes": 100,
        "input_size": 224,
        "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
    "cifar10": {
        "loader": "hf",
        "hf_name": "cifar10",
        "num_classes": 10,
        "input_size": 224,  # Resize to 224 for ViT/MAE models (original is 32×32)
        "normalization": "cifar",
        "splits": ["train", "test", "test"],  # no val split, use test
        "rename_columns": {"img": "image"},  # CIFAR uses 'img' instead of 'image'
    },
    "cifar100": {
        "loader": "hf",
        "hf_name": "cifar100",
        "num_classes": 100,
        "input_size": 224,  # Resize to 224 for ViT/MAE models (original is 32×32)
        "normalization": "cifar",
        "splits": ["train", "test", "test"],
        "rename_columns": {"img": "image"},  # CIFAR uses 'img' instead of 'image'
    },
    "breastmnist": {
        "loader": "hf",
        "hf_name": "randall-lab/medmnist",
        "hf_config": "breastmnist",
        "num_classes": 2,
        "input_size": 224,  # Resize to 224 for ViT backbones
        "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
    # Custom datasets
    "galaxy10": {
        "loader": "custom",
        "loader_fn": load_galaxy10,
        "num_classes": 10,
        "input_size": 224,
        "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
}

# Normalization presets
NORMALIZATIONS = {
    "imagenet": spt.data.static.ImageNet,
    "cifar": spt.data.static.CIFAR10,
}


def get_dataset_config(name):
    # Get dataset configuration
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    cfg = DATASETS[name].copy()
    cfg["normalization"] = NORMALIZATIONS[cfg["normalization"]]
    return cfg


def get_dataset(name, split, transform, cache_dir="/.cache"):
    # Load dataset by name
    cfg = DATASETS[name]
    cache_dir = Path(cache_dir)

    if cfg["loader"] == "hf":
        return load_hf_dataset(cfg["hf_name"], split, transform, cache_dir / "huggingface" / "datasets")
    elif cfg["loader"] == "custom":
        return cfg["loader_fn"](split, transform, cache_dir)
    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")
