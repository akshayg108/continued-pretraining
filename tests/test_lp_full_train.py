"""Check LP and kNN data scope without importing the GPU training stack."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


class FakeDataset:
    def __init__(self, size, transform):
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"image": index, "label": index % 2}


class FakeLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.options = kwargs


class FullTrainLPTests(unittest.TestCase):
    def setUp(self):
        self.sample = Mock(side_effect=lambda args, dataset: (
            list(range(len(dataset))) if args.n_samples == len(dataset) else [4, 1]
        ))
        self.namespace = {
            "argparse": argparse,
            "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(
                Dataset=object, DataLoader=FakeLoader,
            ))),
            "get_dataset": lambda name, split, transform, **kwargs: FakeDataset(
                6 if split == "train" else 3, transform
            ),
            "_sample_shared_train_indices_by_class": self.sample,
            "create_transforms": Mock(return_value=("augmented", "clean")),
        }
        load_definitions(ROOT / "stable_cp/data/loaders.py",
                         {"CPSubset", "create_eval_loaders"}, self.namespace)
        load_definitions(ROOT / "continued_pretraining.py",
                         {"_create_shared_eval_data"}, self.namespace)

    def prepare(self, budget):
        args = SimpleNamespace(dataset="target", n_samples=budget, seed=42,
                               batch_size=256, num_workers=0,
                               eval_batch_size=32, eval_num_workers=0)
        result = self.namespace["_create_shared_eval_data"](args, {}, "/data")
        self.assertEqual(args.n_samples, budget)
        self.assertEqual(args.batch_size, 256)
        return result

    def test_small_cp_budget_uses_full_train_for_lp_and_knn(self):
        _, test, lp, knn, cp_indices = self.prepare(2)
        self.assertEqual(len(lp.dataset), 6)
        self.assertEqual([lp.dataset[i]["image"] for i in range(6)], list(range(6)))
        self.assertEqual(cp_indices, [4, 1])
        self.assertEqual(len(knn.dataset), 6)
        self.assertEqual([knn.dataset[i]["image"] for i in range(6)], list(range(6)))
        self.assertEqual(len(test.dataset), 3)
        self.assertEqual(lp.dataset.dataset.transform, "augmented")
        self.assertEqual(knn.dataset.dataset.transform, "clean")
        self.assertEqual(test.dataset.transform, "clean")
        self.assertEqual(lp.options["batch_size"], 32)
        self.assertEqual(self.sample.call_count, 1)

    def test_full_budget_preserves_existing_sample_order(self):
        _, test, lp, knn, cp_indices = self.prepare(6)
        self.assertEqual(cp_indices, list(range(6)))
        self.assertEqual(lp.dataset.indices, cp_indices)
        self.assertEqual(knn.dataset.indices, cp_indices)
        self.assertEqual(len(test.dataset), 3)
        self.assertEqual(self.sample.call_count, 1)

    def test_default_loader_still_respects_cp_budget(self):
        args = SimpleNamespace(dataset="target", n_samples=2, seed=42,
                               batch_size=32, num_workers=0)
        _, train, indices = self.namespace["create_eval_loaders"](
            args, {}, "augmented", "clean", "/data"
        )
        self.assertEqual(len(train.dataset), 2)
        self.assertEqual(indices, [4, 1])


if __name__ == "__main__":
    unittest.main()
