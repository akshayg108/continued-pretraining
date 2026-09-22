"""Fixed, disjoint partitions for the eight new target datasets."""

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SPLIT_SEED = 42
HELDOUT_DATASETS = {
    "bloodmnist": dict(
        dataset_class="MedMNIST",
        config_name="bloodmnist",
        num_classes=8,
        dataset_kwargs={"size": 224},
        group="OOD",
        partition="official",
    ),
    "tissuemnist": dict(
        dataset_class="MedMNIST",
        config_name="tissuemnist",
        num_classes=8,
        dataset_kwargs={"size": 224},
        group="OOD",
        partition="official",
    ),
    "aid": dict(
        dataset_class="AID",
        config_name=None,
        num_classes=30,
        group="OOD",
        partition="custom_80_10_10_seed42",
    ),
    "resisc45": dict(
        dataset_class="RESISC45",
        config_name=None,
        num_classes=45,
        group="OOD",
        partition="custom_80_10_10_seed42",
    ),
    "stanford_dogs": dict(
        dataset_class="StanfordDogs",
        config_name=None,
        num_classes=120,
        group="FG",
        partition="official_test_train90_val10_seed42",
    ),
    "jena_flowers30": dict(
        dataset_class="JenaFlowers30",
        config_name="all",
        num_classes=30,
        group="FG",
        partition="custom_80_10_10_seed42",
    ),
    "flavia": dict(
        dataset_class="Flavia",
        config_name=None,
        num_classes=32,
        group="FG",
        partition="custom_80_10_10_seed42",
    ),
    "ip102": dict(
        dataset_class="IP102",
        config_name=None,
        num_classes=102,
        group="FG",
        partition="official",
    ),
}
for _cfg in HELDOUT_DATASETS.values():
    _cfg.update(
        input_size=224, normalization="imagenet", splits=["train", "validation", "test"]
    )


def split_indices(labels):
    """Fixed 80/10/10 partition, independent of the training seed."""
    labels = np.asarray(labels).ravel()
    train, holdout = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=SPLIT_SEED
    )
    val, test = train_test_split(
        holdout, test_size=0.5, stratify=labels[holdout], random_state=SPLIT_SEED
    )
    return {"train": np.sort(train), "validation": np.sort(val), "test": np.sort(test)}


def exact_train_indices(labels, n_samples, seed):
    """Preserve ordinary stratification, but never shrink a requested budget."""
    labels = np.asarray(labels).ravel()
    classes, counts = np.unique(labels, return_counts=True)
    if type(n_samples) is not int or not len(classes) <= n_samples <= len(labels):
        raise ValueError(
            "Budget must cover all classes and not exceed the training pool"
        )
    if n_samples == len(labels):
        return list(range(len(labels)))
    try:
        chosen, _ = train_test_split(
            np.arange(len(labels)),
            train_size=n_samples,
            stratify=labels,
            random_state=seed,
        )
        if len(np.unique(labels[chosen])) == len(classes):
            return chosen.tolist()
    except ValueError:
        pass
    # Reserve each class before allocating the remaining proportional quota.
    rng = np.random.RandomState(seed)
    ideal = n_samples * counts / counts.sum()
    quotas = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    tie_order = rng.permutation(len(classes))
    while quotas.sum() != n_samples:
        adding = quotas.sum() < n_samples
        eligible = quotas < counts if adding else quotas > 1
        score = ideal - quotas if adding else quotas - ideal
        candidates = tie_order[eligible[tie_order]]
        index = candidates[np.argmax(score[candidates])]
        quotas[index] += 1 if adding else -1
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == cls), size=int(quota), replace=False)
            for cls, quota in zip(classes, quotas)
        ]
    )
    rng.shuffle(selected)
    return selected.tolist()


def numeric_column(source, name):
    """Read a numeric Arrow column without decoding or concatenating images."""
    paths = getattr(source, "_shard_paths", None)
    if paths is None:
        return source.table.column(name).to_numpy(zero_copy_only=False).copy()
    import pyarrow as pa
    import pyarrow.ipc as ipc

    columns = []
    for path in paths:
        with pa.memory_map(str(path), "r") as mapped:
            column = ipc.open_file(mapped).read_all().column(name)
            columns.append(column.to_numpy(zero_copy_only=False).copy())
    return np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)


class IndexedSplit:
    """Lazy row view: keep Arrow shards intact rather than copying all images."""

    heldout = True

    def __init__(self, source, indices, partition, source_split, labels=None):
        self.source = source
        self.source_indices = np.asarray(indices, dtype=np.int64)
        self.partition = partition
        self.source_split = source_split
        self.column_names = list(dict.fromkeys([*source.features, "sample_idx"]))
        self.num_rows = len(self.source_indices)
        self.labels = (numeric_column(source, "label") if labels is None else labels)[
            self.source_indices
        ]

    def __len__(self):
        return self.num_rows

    def __getitem__(self, index):
        if isinstance(index, str):
            if index == "sample_idx":
                return np.arange(len(self))
            if index == "label":
                return self.labels
            raise KeyError(
                f"Only numeric label and sample_idx columns are exposed: {index}"
            )
        sample = dict(self.source[int(self.source_indices[index])])
        sample["sample_idx"] = int(index)
        return sample


@lru_cache(maxsize=32)
def _source(name, source_split, cache_dir):
    from stable_datasets import images

    cfg = HELDOUT_DATASETS[name]
    cls = getattr(images, cfg["dataset_class"], None)
    if cls is None:
        raise RuntimeError(
            f"stable-datasets lacks {cfg['dataset_class']}; install commit cc01e36 or newer"
        )
    kwargs = dict(cfg.get("dataset_kwargs", {}))
    if cfg["config_name"] is not None:
        kwargs["config_name"] = cfg["config_name"]
    root = Path(cache_dir).expanduser() / "stable_datasets"
    return cls(
        split=source_split,
        download_dir=str(root / "downloads"),
        processed_cache_dir=str(root / "processed"),
        **kwargs,
    )


@lru_cache(maxsize=24)
def load_heldout_split(name, split, cache_dir):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown held-out split: {split}")
    cfg = HELDOUT_DATASETS[name]
    partition = cfg["partition"]
    source_split = split
    if partition.startswith("custom_") or (name == "stanford_dogs" and split != "test"):
        source_split = "train"
    source = _source(name, source_split, str(Path(cache_dir).expanduser().resolve()))
    labels = numeric_column(source, "label").ravel()
    if not np.array_equal(np.unique(labels), np.arange(cfg["num_classes"])):
        raise ValueError(f"Missing or invalid classes in {name}/{source_split}")
    indices = np.arange(len(source))
    if partition.startswith("custom_"):
        indices = split_indices(labels)[split]
    elif name == "stanford_dogs" and split != "test":
        train, val = train_test_split(
            indices, test_size=0.1, stratify=labels, random_state=SPLIT_SEED
        )
        indices = np.sort(train if split == "train" else val)
    return IndexedSplit(source, indices, partition, source_split, labels=labels)


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def data_signature(
    train_indices, train_source_indices, train_labels, test_source_indices, test_labels
):
    return dict(
        n_train_actual=len(train_indices),
        n_test=len(test_labels),
        train_indices_sha256=array_hash(train_indices),
        train_source_indices_sha256=array_hash(train_source_indices),
        train_labels_sha256=array_hash(train_labels),
        test_source_indices_sha256=array_hash(test_source_indices),
        test_labels_sha256=array_hash(test_labels),
    )
