"""Core data adapters, sampling, and native encoder normalization."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from sklearn.model_selection import train_test_split
from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features

ROOT = Path(__file__).resolve().parents[1]


def definitions(filename, names, **namespace):
    path = ROOT / "stable_cp/data" / filename
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def data_api():
    class TransformDataset:
        def __init__(self, transform=None):
            self.transform = transform

        def process_sample(self, sample):
            return self.transform(sample) if self.transform else sample

    heldout = definitions(
        "heldout.py",
        {"numeric_column", "exact_train_indices", "IndexedSplit"},
        np=np,
        train_test_split=train_test_split,
    )
    datasets = definitions(
        "datasets.py",
        {"HFDatasetWrapper"},
        spt=SimpleNamespace(data=SimpleNamespace(Dataset=TransformDataset)),
        numeric_column=heldout.numeric_column,
        IndexedSplit=heldout.IndexedSplit,
    )
    loaders = definitions(
        "loaders.py",
        {"CPSubset", "BalancedRepeatSampler", "_sample_shared_train_indices_by_class"},
        torch=torch,
        np=np,
        train_test_split=train_test_split,
        exact_train_indices=heldout.exact_train_indices,
    )
    return SimpleNamespace(**vars(datasets), loaders=loaders, heldout=heldout)


def arrow_dataset(labels):
    features = Features({"label": ClassLabel(num_classes=len(set(labels)))})
    return StableDataset(features, DatasetInfo(features), table=pa.table({"label": labels}))


def test_wrapper_accepts_current_arrow_dataset_and_does_not_mutate_rows(data_api):
    source = arrow_dataset([0, 1, 0, 1])
    dataset = data_api.HFDatasetWrapper(source, transform=lambda row: dict(row, transformed=True))
    assert dataset[2] == {"label": 0, "sample_idx": 2, "transformed": True}
    assert source[2] == {"label": 0}
    assert set(dataset.column_names) == {"label", "sample_idx"}
    assert dataset.labels.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("counts", [[20, 20, 20, 20], [70, 20, 10]])
def test_sampling_reads_arrow_labels_and_preserves_selection(data_api, counts):
    labels = np.repeat(np.arange(len(counts)), counts)
    source = arrow_dataset(labels.tolist())
    dataset = data_api.HFDatasetWrapper(source)
    args = SimpleNamespace(n_samples=20, seed=43)
    expected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=20,
        stratify=labels,
        random_state=43,
    )
    selected = data_api.loaders._sample_shared_train_indices_by_class(args, dataset)
    assert selected == expected.tolist()
    assert len(set(selected)) == 20


@pytest.mark.parametrize(
    "counts,budget,expected",
    [
        ([6000] + [2] * 29, 1000, [971] + [1] * 29),
        ([100, 2, 2], 10, [8, 1, 1]),
        ([100, 1, 1], 10, [8, 1, 1]),
        ([8, 1, 1], 9, [7, 1, 1]),
        ([100, 2, 2], 3, [1, 1, 1]),
        ([8, 1, 1], 10, [8, 1, 1]),
    ],
)
def test_sampling_preserves_budget_and_covers_every_class(data_api, counts, budget, expected):
    labels = np.repeat(np.arange(len(counts)), counts)
    dataset = data_api.HFDatasetWrapper(arrow_dataset(labels.tolist()))
    args = SimpleNamespace(n_samples=budget, seed=42)

    selected = data_api.loaders._sample_shared_train_indices_by_class(args, dataset)

    assert len(selected) == len(set(selected)) == budget
    assert np.bincount(labels[selected], minlength=len(counts)).tolist() == expected
    assert selected == data_api.loaders._sample_shared_train_indices_by_class(args, dataset)


def test_fixed_split_uses_exact_sampling_and_subset_local_ids(data_api):
    source = arrow_dataset([0] * 100 + [1] * 2 + [2] * 2)
    view = data_api.heldout.IndexedSplit(source, np.arange(104), "fixed", "train")
    assert view.features == source.features
    dataset = data_api.HFDatasetWrapper(view)
    args = SimpleNamespace(n_samples=10, seed=42)
    selected = data_api.loaders._sample_shared_train_indices_by_class(args, dataset)
    assert len(set(selected)) == 10
    assert set(dataset.labels[selected]) == {0, 1, 2}
    subset = data_api.loaders.CPSubset(dataset, selected)
    assert [subset[i]["sample_idx"] for i in range(len(subset))] == list(range(10))
    assert [subset[i]["label"] for i in range(len(subset))] == dataset.labels[selected].tolist()


def test_repeat_sampler_covers_all_samples_and_padding(data_api):
    sampler = data_api.loaders.BalancedRepeatSampler(
        list(range(5)),
        num_samples=12,
        generator=torch.Generator().manual_seed(42),
    )
    samples = list(sampler)
    assert len(samples) == len(sampler) == 12
    assert sorted(np.bincount(samples)) == [2, 2, 2, 3, 3]


def test_sampling_rejects_oversized_budget(data_api):
    dataset = data_api.HFDatasetWrapper(arrow_dataset([0, 1]))
    with pytest.raises(ValueError, match="dataset size"):
        data_api.loaders._sample_shared_train_indices_by_class(
            SimpleNamespace(n_samples=3, seed=42),
            dataset,
        )


def test_dataset_registry_keeps_all_datasets_without_normalization_presets():
    path = ROOT / "stable_cp/data/datasets.py"
    tree = ast.parse(path.read_text())
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DATASETS" for target in node.targets)
    )
    names = {ast.literal_eval(key) for key in registry.keys}
    assert names | {"aid", "resisc45", "stanford_dogs", "jena_flowers30", "flavia", "ip102"} == {
        "cifar10",
        "cifar100",
        "food101",
        "fgvc_aircraft",
        "galaxy10",
        "bloodmnist",
        "tissuemnist",
        "pathmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "retinamnist",
        "breastmnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
        "cars196",
        "cub200",
        "flowers102",
        "oxford_pet",
        "dtd",
        "eurosat",
        "plant_village",
        "aid",
        "resisc45",
        "stanford_dogs",
        "jena_flowers30",
        "flavia",
        "ip102",
    }
    assert all(
        "normalization" not in [ast.literal_eval(key) for key in entry.keys]
        for entry in registry.values
    )


def test_dataset_config_is_copied_without_dataset_normalization():
    registry = {"example": {"input_size": 224, "num_classes": 2}}
    api = definitions("datasets.py", {"get_dataset_config"}, DATASETS=registry)
    config = api.get_dataset_config("example")
    config["normalization"] = {"mean": [0.5] * 3, "std": [0.5] * 3}
    assert "normalization" not in registry["example"]


def test_manual_splits_are_disjoint_and_reproducible():
    api = definitions("datasets.py", {"_split_single_dataset"})
    source = arrow_dataset(list(range(100)))
    splits = {
        name: api._split_single_dataset(source, name, seed=42)
        for name in ("train", "validation", "test")
    }
    assert {name: len(split) for name, split in splits.items()} == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    labels = {
        name: {split[i]["label"] for i in range(len(split))} for name, split in splits.items()
    }
    assert labels["train"].isdisjoint(labels["validation"] | labels["test"])
    assert labels["validation"].isdisjoint(labels["test"])
    repeated = api._split_single_dataset(source, "test", seed=42)
    assert [repeated[i] for i in range(len(repeated))] == [
        splits["test"][i] for i in range(len(splits["test"]))
    ]


def test_real_transforms_use_checkpoint_normalization_for_all_views():
    # Isolate actual training-stack imports from the lightweight test doubles.
    code = """
from types import SimpleNamespace
import torch
from PIL import Image
from timm.models import get_pretrained_cfg
from continued_pretraining import configure_normalization
from stable_cp.data import create_transforms

torch.set_num_threads(2)
models = (
    'vit_base_patch16_dinov3.lvd1689m',
    'vit_large_patch16_dinov3.lvd1689m',
    'vit_base_patch16_clip_224.openai',
    'vit_base_patch16_siglip_224.v2_webli',
    'vit_base_patch16_224.mae',
)
rgb = (64, 128, 192)
for model in models:
    native = get_pretrained_cfg(model).to_dict()
    config = configure_normalization(
        {'input_size': 224}, SimpleNamespace(pretrained_cfg=native),
    )
    expected = (torch.tensor(rgb) / 255 - torch.tensor(native['mean'])) / torch.tensor(native['std'])
    for n_views in (1, 2):
        for strong_aug in (False, True):
            train, clean = create_transforms(config, n_views, strong_aug)
            image = Image.new('RGB', (8, 8), rgb)
            actual = clean({'image': image.copy()})['image']
            assert actual.shape == (3, 224, 224), model
            torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))
            branches = train.transforms.values() if n_views > 1 else (train,)
            for branch in branches:
                normalize = branch.args[-1].t.transforms[-1]
                assert tuple(normalize.mean) == tuple(native['mean']), model
                assert tuple(normalize.std) == tuple(native['std']), model
            outputs = train({'image': image.copy()})
            samples = outputs.values() if n_views > 1 else (outputs,)
            for sample in samples:
                assert sample['image'].shape == (3, 224, 224), model
                assert torch.isfinite(sample['image']).all(), model
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
