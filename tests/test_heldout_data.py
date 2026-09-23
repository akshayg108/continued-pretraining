"""Held-out partitions and sample identity must not depend on the CP objective."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "stable_cp/data/heldout.py"
    assert path.is_file(), "Held-out data adapter is missing"
    spec = importlib.util.spec_from_file_location("heldout_data_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_eight_datasets_and_class_counts():
    data = module()
    assert {name: cfg["num_classes"] for name, cfg in data.HELDOUT_DATASETS.items()} == {
        "bloodmnist": 8,
        "tissuemnist": 8,
        "aid": 30,
        "resisc45": 45,
        "stanford_dogs": 120,
        "jena_flowers30": 30,
        "flavia": 32,
        "ip102": 102,
    }
    assert data.HELDOUT_DATASETS["jena_flowers30"]["config_name"] == "all"


def test_custom_partitions_are_disjoint_complete_and_reproducible():
    data = module()
    labels = np.repeat(np.arange(30), 50)
    splits = data.split_indices(labels)
    assert {name: len(idx) for name, idx in splits.items()} == {
        "train": 1200,
        "validation": 150,
        "test": 150,
    }
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(1500))
    for name, indices in splits.items():
        assert len(np.unique(labels[indices])) == 30
        assert indices.tolist() == data.split_indices(labels)[name].tolist()
    assert len(data.exact_train_indices(labels[splits["train"]], 1000, 42)) == 1000


def test_exact_sampling_covers_rare_classes_without_shrinking_budget():
    data = module()
    labels = np.concatenate([np.repeat(0, 6000), np.repeat(np.arange(1, 102), 2)])
    for seed in (42, 43, 44):
        indices = data.exact_train_indices(labels, 1000, seed)
        assert len(indices) == len(set(indices)) == 1000
        assert len(np.unique(labels[indices])) == 102
        assert indices == data.exact_train_indices(labels, 1000, seed)
    assert data.exact_train_indices(labels, 1000, 42) != data.exact_train_indices(labels, 1000, 43)


def test_ordinary_sampling_preserves_legacy_stratified_selection():
    from sklearn.model_selection import train_test_split

    labels = np.repeat(np.arange(8), 200)
    expected, _ = train_test_split(
        np.arange(len(labels)), train_size=1000, stratify=labels, random_state=43
    )
    assert module().exact_train_indices(labels, 1000, 43) == expected.tolist()


@pytest.mark.parametrize("n", [0, 3, 2001])
def test_impossible_budget_is_rejected(n):
    with pytest.raises(ValueError):
        module().exact_train_indices(np.repeat(np.arange(8), 200), n, 42)


def test_index_view_keeps_source_identity_without_copying_images():
    data = module()

    class Source:
        import pyarrow as pa

        features = {"image": None, "label": None}
        table = pa.table({"label": list(range(10))})
        reads = 0

        def __len__(self):
            return 10

        def __getitem__(self, index):
            if isinstance(index, str):
                assert index == "label"
                return list(range(10))
            self.reads += 1
            return {"image": f"image-{index}", "label": index}

    source = Source()
    view = data.IndexedSplit(source, [1, 4, 6], "custom_80_10_10_seed42", "train")
    assert view["label"].tolist() == [1, 4, 6]
    assert source.reads == 0
    assert "sample_idx" in view.column_names
    assert view[1] == {"image": "image-4", "label": 4, "sample_idx": 1}
    assert view.source_indices.tolist() == [1, 4, 6]
    assert source.reads == 1


def test_real_shard_backed_dataset_supports_labels_without_decoding_images(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "stable-datasets"))
    from PIL import Image as PILImage
    from stable_datasets.arrow_dataset import StableDataset
    from stable_datasets.cache import write_sharded_arrow_cache
    from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image

    features = Features({"image": Image(), "label": ClassLabel(num_classes=2)})
    rows = ((i, {"image": PILImage.new("RGB", (4, 4)), "label": i % 2}) for i in range(10))
    meta = write_sharded_arrow_cache(
        rows, features, tmp_path / "shards", batch_size=2, shard_size_bytes=100
    )
    source = StableDataset(
        features,
        DatasetInfo(features),
        shard_dir=meta.cache_dir,
        shard_paths=[meta.cache_dir / p for p in meta.shard_filenames],
        shard_row_counts=meta.shard_row_counts,
    )
    view = module().IndexedSplit(source, [1, 4, 6], "fixed", "train")
    assert view["label"].tolist() == [1, 0, 0]
    assert source._table is None
    assert len(source._shard_lru) == 0
    assert view[1]["image"].size == (4, 4)
    assert view[1]["label"] == 0 and view[1]["sample_idx"] == 1
