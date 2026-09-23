# Dataset registry for continued pretraining (using stable-datasets)
from pathlib import Path

import numpy as np
import stable_pretraining as spt
from stable_datasets import images as stable_ds
from .heldout import HELDOUT_DATASETS, IndexedSplit, load_heldout_split, numeric_column

# Dataset configuration registry
DATASETS = {
    # CIFAR datasets
    "cifar10": {
        "dataset_class": stable_ds.CIFAR10,
        "config_name": None,
        "num_classes": 10,
        "input_size": 224,
        "splits": ["train", "test", "test"],
    },
    "cifar100": {
        "dataset_class": stable_ds.CIFAR100,
        "config_name": None,
        "num_classes": 100,
        "input_size": 224,
        "splits": ["train", "test", "test"],
    },
    # Food and Objects
    "food101": {
        "dataset_class": stable_ds.Food101,
        "config_name": None,
        "num_classes": 101,
        "input_size": 224,
        "splits": ["train", "test", "test"],
    },
    # Fine-Grained Classification
    "fgvc_aircraft": {
        "dataset_class": stable_ds.FGVCAircraft,
        "config_name": "variant",  # Options: "variant" (100), "family" (70), "manufacturer" (30)
        "num_classes": 100,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
    },
    # Galaxy dataset
    "galaxy10": {
        "dataset_class": stable_ds.Galaxy10Decal,
        "config_name": None,
        "num_classes": 10,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "manual_split": True,  # Force manual splitting from full dataset (avoid data leakage)
    },
    # MedMNIST datasets (size=224 for native high-resolution images)
    "bloodmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "bloodmnist",
        "num_classes": 8,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "tissuemnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "tissuemnist",
        "num_classes": 8,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "pathmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "pathmnist",
        "num_classes": 9,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "dermamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "dermamnist",
        "num_classes": 7,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "octmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "octmnist",
        "num_classes": 4,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "pneumoniamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "pneumoniamnist",
        "num_classes": 2,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "retinamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "retinamnist",
        "num_classes": 5,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "breastmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "breastmnist",
        "num_classes": 2,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "organamnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organamnist",
        "num_classes": 11,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "organcmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organcmnist",
        "num_classes": 11,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    "organsmnist": {
        "dataset_class": stable_ds.MedMNIST,
        "config_name": "organsmnist",
        "num_classes": 11,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
        "dataset_kwargs": {"size": 224},
    },
    # Fine-grained additions
    "cars196": {
        "dataset_class": stable_ds.Cars196,
        "config_name": None,
        "num_classes": 196,
        "input_size": 224,
        "splits": ["train", "test", "test"],  # no validation; reuse test
    },
    "cub200": {
        "dataset_class": stable_ds.CUB200,
        "config_name": None,
        "num_classes": 200,
        "input_size": 224,
        "splits": ["train", "test", "test"],  # no validation; reuse test
    },
    "flowers102": {
        "dataset_class": stable_ds.Flowers102,
        "config_name": None,
        "num_classes": 102,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
    },
    "oxford_pet": {
        "dataset_class": stable_ds.OxfordPet,
        "config_name": None,
        "num_classes": 37,
        "input_size": 224,
        "splits": ["train", "test", "test"],  # no validation; reuse test
    },
    # OOD additions
    "dtd": {
        "dataset_class": stable_ds.DTD,
        "config_name": None,
        "num_classes": 47,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
    },
    "eurosat": {
        "dataset_class": stable_ds.EuroSAT,
        "config_name": None,
        "num_classes": 10,
        "input_size": 224,
        "splits": ["train", "validation", "test"],
    },
    "plant_village": {
        "dataset_class": stable_ds.PlantVillage,
        "config_name": "color",  # RGB variant (default); alternatives: "grayscale", "segmented"
        "num_classes": 38,
        "input_size": 224,
        "splits": ["train", "test", "test"],  # no validation; reuse test
    },
}

# Preserve legacy entries; new readers are resolved lazily by the held-out adapter.
for _name, _config in HELDOUT_DATASETS.items():
    DATASETS.setdefault(_name, _config.copy())


class HFDatasetWrapper(spt.data.Dataset):
    """Adapt stable-datasets rows to stable-pretraining transforms."""

    def __init__(self, hf_dataset, transform=None):
        super().__init__(transform)
        self.hf_dataset = hf_dataset

    def __getitem__(self, idx):
        sample = dict(self.hf_dataset[idx])
        sample.setdefault("sample_idx", int(idx))
        return self.process_sample(sample)

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def column_names(self):
        return list(dict.fromkeys([*self.hf_dataset.features, "sample_idx"]))

    @property
    def labels(self):
        if isinstance(self.hf_dataset, IndexedSplit):
            return self.hf_dataset.labels
        return numeric_column(self.hf_dataset, "label").ravel()


def get_dataset_config(name):
    """Return a copy; the encoder supplies normalization before loading data."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name].copy()


def get_dataset(name, split, transform, cache_dir="/.cache", seed=42):
    """Load an official or reproducibly constructed stable-datasets split."""
    if name in HELDOUT_DATASETS:
        raw = load_heldout_split(name, split, str(Path(cache_dir).expanduser().resolve()))
        return HFDatasetWrapper(raw, transform=transform)
    cfg = DATASETS[name]
    cache_dir = Path(cache_dir)

    download_dir = cache_dir / "stable_datasets" / "downloads"
    processed_cache_dir = cache_dir / "stable_datasets" / "processed"
    download_dir.mkdir(parents=True, exist_ok=True)
    processed_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_class = cfg["dataset_class"]
    kwargs = dict(
        download_dir=str(download_dir),
        processed_cache_dir=str(processed_cache_dir),
        **cfg.get("dataset_kwargs", {}),
    )
    if cfg["config_name"] is not None:
        kwargs["config_name"] = cfg["config_name"]

    if cfg.get("manual_split", False):
        raw = _split_single_dataset(dataset_class(split="train", **kwargs), split, seed)
    else:
        try:
            raw = dataset_class(split=split, **kwargs)
        except (ValueError, KeyError):
            raw = _handle_split_from_dict(dataset_class(split=None, **kwargs), split, seed)
    return HFDatasetWrapper(raw, transform=transform)


def _handle_split_from_dict(dataset_dict, split, seed=42):
    """Resolve split aliases, or partition a train-only dataset."""
    split_map = {
        "validation": ["validation", "val", "valid"],
        "val": ["validation", "val", "valid"],
        "test": ["test"],
        "train": ["train"],
    }

    possible_names = split_map.get(split, [split])
    for name in possible_names:
        if name in dataset_dict:
            return dataset_dict[name]

    if "train" in dataset_dict:
        return _split_single_dataset(dataset_dict["train"], split, seed)

    raise ValueError(f"Split '{split}' not found in dataset and cannot be created")


def _split_single_dataset(hf_dataset, split, seed=42, val_ratio=0.1, test_ratio=0.1):
    """Preserve the original seeded 80/10/10 partition for train-only datasets."""
    if split not in {"train", "validation", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    # Match stable-datasets' two seeded permutations without gathering image bytes.
    indices = np.random.RandomState(seed).permutation(len(hf_dataset))
    train_size = int(len(indices) * (1 - (val_ratio + test_ratio)))
    if split == "train":
        indices = indices[:train_size]
    else:
        remaining = indices[train_size:]
        remaining = remaining[np.random.RandomState(seed).permutation(len(remaining))]
        val_size = int(len(remaining) * (1 - test_ratio / (val_ratio + test_ratio)))
        indices = remaining[val_size:] if split == "test" else remaining[:val_size]
    return IndexedSplit(
        hf_dataset, indices, partition=f"random_split_seed{seed}", source_split="train"
    )
