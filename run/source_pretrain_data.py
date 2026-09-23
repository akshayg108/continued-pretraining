"""Source datasets, LeJEPA crops, and deterministic step-based sampling."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import re
import signal
import tempfile

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from run.data_cache import _terminate, prepare_dataset, staged_dataset, staged_directory

TARGETS = ("octmnist", "pathmnist", "galaxy10")
DOMAIN_NAMES = ("imagenet", *TARGETS)
SPLIT_SEED = 42
VALIDATION_CLASSES = 1000
VALIDATION_PER_CLASS = 5
VALIDATION_CACHE = "source_imagenet_validation_5000"
IMAGENET_REPO = "ILSVRC/imagenet-1k"
IMAGENET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
IMAGENET_SPLIT_SIZES = {"train": 1_281_167, "validation": 50_000}


def download_imagenet(imagenet_dir, validation_dir, cache_dir):
    """Download authorized HF splits once and save self-contained offline datasets."""
    from datasets import ClassLabel, Features, Image, load_dataset
    from timm.data import ImageNetInfo
    import numpy as np

    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=ImageNetInfo("imagenet-1k").label_names()),
        }
    )
    for split, destination in (("train", imagenet_dir), ("validation", validation_dir)):
        destination = Path(destination).expanduser().resolve()
        if destination.is_dir() and any(destination.iterdir()):
            print(f"REUSE ImageNet {split}: {destination}", flush=True)
            continue
        if destination.is_dir():
            destination.rmdir()
        print(
            f"DOWNLOAD {IMAGENET_REPO} split={split} revision={IMAGENET_REVISION}", flush=True
        )
        source = load_dataset(
            IMAGENET_REPO,
            revision=IMAGENET_REVISION,
            data_files={split: f"data/{split}-*.parquet"},
            split=split,
            features=features,
            token=True,
            cache_dir=str(cache_dir),
            # The repo metadata lists all splits; validate this split below instead.
            verification_mode="no_checks",
        )
        if len(source) != IMAGENET_SPLIT_SIZES[split] or not np.array_equal(
            np.unique(source["label"]), np.arange(1000)
        ):
            raise ValueError(f"Incomplete ImageNet {split}: {len(source):,} rows")
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous_handler = signal.signal(signal.SIGTERM, _terminate)
        try:
            with tempfile.TemporaryDirectory(
                prefix=".imagenet-", dir=destination.parent
            ) as temporary:
                prepared = Path(temporary) / "dataset"
                source.cast_column("image", Image(decode=False)).save_to_disk(str(prepared))
                metadata = {
                    "repo_id": IMAGENET_REPO,
                    "revision": IMAGENET_REVISION,
                    "split": split,
                    "num_images": len(source),
                    "label_mapping": "official_numeric_ids_to_sorted_wnids",
                }
                (prepared / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
                prepared.rename(destination)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        print(f"DOWNLOADED ImageNet {split}: {destination}", flush=True)


def _imagenet_dataset(imagenet_dir):
    from torchvision.datasets import ImageFolder

    imagenet_dir = Path(imagenet_dir).expanduser().resolve()
    if not imagenet_dir.is_dir():
        raise FileNotFoundError(imagenet_dir)
    if (imagenet_dir / "state.json").is_file():
        from datasets import Image, load_from_disk
        from timm.data import ImageNetInfo

        dataset = load_from_disk(str(imagenet_dir)).cast_column("image", Image())
        dataset.classes = dataset.features["label"].names
        if dataset.classes != ImageNetInfo("imagenet-1k").label_names():
            raise ValueError("Prepared HF ImageNet labels must use canonical WNID order")
    else:
        dataset = ImageFolder(str(imagenet_dir))
    if len(dataset) != IMAGENET_SPLIT_SIZES["train"] or len(dataset.classes) != 1000:
        raise ValueError(
            "ImageNet must be the complete training dataset with 1,281,167 "
            f"images and 1,000 classes, not validation: {imagenet_dir} has "
            f"{len(dataset):,} images and {len(dataset.classes):,} classes"
        )
    return dataset


def prepare_sources(cache_dir: Path, imagenet_dir: Path):
    """Validate full ImageNet training data and prepare all target caches."""
    from stable_cp.data.datasets import get_dataset

    cache_dir = Path(cache_dir).expanduser().resolve()
    sizes = {"imagenet": len(_imagenet_dataset(imagenet_dir))}
    for name in TARGETS:
        prepare_dataset(cache_dir, name)
        sizes[name] = len(
            get_dataset(name, split="train", transform=None, cache_dir=cache_dir, seed=SPLIT_SEED)
        )
    return sizes


def _load_prepared_validation(path, metadata=None):
    from datasets import load_from_disk
    import numpy as np

    recorded = json.loads((path / "metadata.json").read_text())
    if metadata is not None and recorded != metadata:
        raise ValueError(f"Incompatible prepared ImageNet validation: {path}")
    source = load_from_disk(str(path / "dataset"))
    labels = np.asarray(source["label"], dtype=np.int64)
    if (
        len(source) != VALIDATION_CLASSES * VALIDATION_PER_CLASS
        or len(recorded["indices"]) != len(source)
        or source.features["label"].names != recorded["train_classes"]
        or len(recorded["train_classes"]) != VALIDATION_CLASSES
        or not np.array_equal(
            np.bincount(labels, minlength=VALIDATION_CLASSES),
            np.full(VALIDATION_CLASSES, VALIDATION_PER_CLASS),
        )
    ):
        raise ValueError(f"Invalid prepared ImageNet validation labels: {path}")
    return source, recorded


def prepare_imagenet_validation(cache_dir, imagenet_dir, validation_dir):
    """Prepare a fixed labeled validation subset for online ImageNet probes."""
    from datasets import ClassLabel, DatasetDict, Image, load_from_disk
    import numpy as np

    validation_dir = Path(validation_dir).expanduser().resolve()
    train_classes = _imagenet_dataset(imagenet_dir).classes
    if any(re.fullmatch(r"n\d{8}", name) is None for name in train_classes):
        raise ValueError("ImageNet training folders must use canonical WNID class names")
    source = load_from_disk(str(validation_dir))
    if isinstance(source, DatasetDict):
        source = source["validation"]
    if source.split is not None and str(source.split) not in {"val", "validation"}:
        raise ValueError(f"Expected ImageNet validation data, received {source.split}")
    if not {"image", "label"}.issubset(source.column_names):
        raise ValueError("ImageNet validation data must contain image and label columns")
    labels = np.asarray(source["label"])
    if (
        not np.issubdtype(labels.dtype, np.integer)
        or len(labels) < VALIDATION_CLASSES * VALIDATION_PER_CLASS
        or labels.min() < 0
        or labels.max() >= VALIDATION_CLASSES
    ):
        raise ValueError("ImageNet validation labels must be integer IDs in [0, 999]")
    names = getattr(source.features["label"], "names", None)
    if names and all(re.fullmatch(r"n\d{8}", name) is not None for name in names):
        if len(names) != VALIDATION_CLASSES or set(names) != set(train_classes):
            raise ValueError("ImageNet validation WNIDs differ from training classes")
        mapping = {name: index for index, name in enumerate(train_classes)}
        labels = np.asarray([mapping[name] for name in names])[labels]
        policy = "remap_validation_wnids_to_imagefolder_class_indices"
    else:
        if train_classes != sorted(train_classes):
            raise ValueError("Canonical numeric labels require lexicographically ordered WNIDs")
        policy = "assume_official_imagenet1k_numeric_ids_match_sorted_wnids"
    rng = np.random.RandomState(SPLIT_SEED)
    indices = []
    for label in range(VALIDATION_CLASSES):
        available = np.flatnonzero(labels == label)
        if len(available) < VALIDATION_PER_CLASS:
            raise ValueError(f"ImageNet validation class {label} has fewer than 5 samples")
        indices.extend(rng.choice(available, VALIDATION_PER_CLASS, replace=False).tolist())
    indices.sort()
    metadata = {
        "protocol": "source_online_imagenet_validation_v1",
        "source": str(validation_dir),
        "source_fingerprint": source._fingerprint,
        "n_source_images": len(source),
        "sampling_seed": SPLIT_SEED,
        "samples_per_class": VALIDATION_PER_CLASS,
        "indices": indices,
        "train_classes": train_classes,
        "label_mapping": policy,
    }
    path = Path(cache_dir).expanduser().resolve() / VALIDATION_CACHE
    if path.exists():
        _load_prepared_validation(path, metadata)
        return path
    selected = source.select(indices).select_columns(["image"])
    selected = selected.add_column("label", labels[indices].tolist())
    selected = selected.cast_column("label", ClassLabel(names=train_classes))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".source-validation-") as temporary:
        prepared = Path(temporary) / "prepared"
        # Embed selected image bytes so the node-local subset is self-contained.
        selected.cast_column("image", Image(decode=False)).save_to_disk(str(prepared / "dataset"))
        (prepared / "metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        _load_prepared_validation(prepared, metadata)
        prepared.rename(path)
    return path


def build_validation_dataset(prepared_dir, transform):
    from datasets import Image

    source, metadata = _load_prepared_validation(Path(prepared_dir))
    dataset = _DomainDataset(source.cast_column("image", Image()), 0, transform)
    dataset.metadata = metadata
    return dataset


@contextmanager
def staged_sources(cache_dir: Path, imagenet_dir: Path):
    """Stage full ImageNet training data and all targets on node-local storage."""
    with ExitStack() as stack:
        local_imagenet = stack.enter_context(staged_directory(imagenet_dir))
        target_caches = {
            name: stack.enter_context(staged_dataset(cache_dir, name)) for name in TARGETS
        }
        local_validation = stack.enter_context(staged_directory(Path(cache_dir) / VALIDATION_CACHE))
        yield local_imagenet, target_caches, local_validation


def make_transforms():
    """Return two global crops, eight local crops, and clean evaluation inputs."""
    from stable_pretraining.data import transforms

    normalization = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    def crop(size, scale):
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((size, size), scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToImage(**normalization),
        )

    train = transforms.MultiViewTransform(
        {
            **{f"global_{i}": crop(224, (0.3, 1.0)) for i in range(2)},
            **{f"local_{i}": crop(96, (0.05, 0.3)) for i in range(8)},
        }
    )
    evaluation = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((224, 224)),
        transforms.ToImage(**normalization),
    )
    return train, evaluation


class _DomainDataset(Dataset):
    """Give ImageFolder tuples and target rows the same pre-transform schema."""

    def __init__(self, dataset, domain_id, transform):
        self.dataset = dataset
        self.domain_id = domain_id
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw = self.dataset[idx]
        image, label = (raw["image"], raw["label"]) if isinstance(raw, dict) else raw
        sample = {
            "image": image,
            "label": int(torch.as_tensor(label).item()),
            "sample_idx": int(idx),
            "domain_id": self.domain_id,
        }
        return self.transform(sample) if self.transform is not None else sample


def build_train_dataset(imagenet_dir, target_cache_dirs, condition, transform):
    """Concatenate complete training splits without domain-balanced sampling."""
    if condition not in {"imagenet", "mixed"}:
        raise ValueError(f"Unknown source condition: {condition}")
    datasets = [_DomainDataset(_imagenet_dataset(imagenet_dir), 0, transform)]
    sizes = {"imagenet": len(datasets[0])}
    if condition == "mixed":
        from stable_cp.data.datasets import get_dataset

        for domain_id, name in enumerate(TARGETS, start=1):
            dataset = get_dataset(
                name,
                split="train",
                transform=None,
                cache_dir=target_cache_dirs[name],
                seed=SPLIT_SEED,
            )
            datasets.append(_DomainDataset(dataset, domain_id, transform))
            sizes[name] = len(dataset)
    return ConcatDataset(datasets), sizes


class ShuffledStepSampler(Sampler):
    """Stream full shuffled cycles, resuming at an exact optimizer-step offset."""

    def __init__(self, data_source, total_steps, batch_size, seed, start_step=0):
        self.size = len(data_source)
        if self.size <= 0:
            raise ValueError("Source training dataset must not be empty")
        if batch_size <= 0 or not 0 <= start_step <= total_steps:
            raise ValueError("Require batch_size > 0 and 0 <= start_step <= total_steps")
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step

    def __len__(self):
        return (self.total_steps - self.start_step) * self.batch_size

    def __iter__(self):
        cycle, offset = divmod(self.start_step * self.batch_size, self.size)
        remaining = len(self)
        while remaining:
            generator = torch.Generator().manual_seed(self.seed + cycle)
            permutation = torch.randperm(self.size, generator=generator)
            count = min(remaining, self.size - offset)
            yield from permutation[offset : offset + count].tolist()
            remaining -= count
            cycle += 1
            offset = 0
