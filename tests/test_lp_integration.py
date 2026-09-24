"""CPU contracts for the image-based LP loaders and entry point."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

import continued_pretraining as entry
from stable_cp.data import loaders
from stable_cp.utils.lp_protocol import lp_config


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.labels = np.arange(7) % 2
        self.image = Image.fromarray(np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {"image": self.image.copy(), "label": int(self.labels[index])}
        return self.transform(sample) if self.transform else sample


class LPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "input_size": 16,
            "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
            "splits": ["train", "validation", "test"],
            "num_classes": 2,
        }
        self.args = argparse.Namespace(
            dataset="breastmnist",
            n_samples=7,
            seed=42,
            batch_size=2,
            num_workers=0,
            eval_batch_size=None,
            eval_num_workers=None,
            lp_batch_size=3,
            lp_epochs=2,
            lp_lr=0.005,
            lp_forward_batch_size=1,
            knn_k=1,
            pool_strategy="cls",
            skip_baseline=False,
            skip_final_eval=False,
        )

    def test_lp_transform_is_weak_and_resampled_but_test_is_deterministic(self):
        train, test = loaders.create_lp_transforms(self.cfg)
        self.assertEqual(
            [type(t).__name__ for t in train.args],
            ["RGB", "RandomResizedCrop", "RandomHorizontalFlip", "ToImage"],
        )
        self.assertEqual(train.args[1].scale, (0.08, 1.0))
        self.assertEqual(train.args[2].p, 0.5)
        dataset = ImageDataset(train)
        torch.manual_seed(42)
        views = [dataset[0]["image"] for _ in range(4)]
        self.assertTrue(any(not torch.equal(views[0], v) for v in views[1:]))
        clean = ImageDataset(test)
        self.assertTrue(torch.equal(clean[0]["image"], clean[0]["image"]))

    def test_lp_batch_shuffle_and_tail_do_not_change_clean_loaders(self):
        with patch.object(
            loaders, "get_dataset", side_effect=lambda *a, **kw: ImageDataset(kw["transform"])
        ):
            _, test, train, knn, indices = entry._create_shared_eval_data(
                self.args, self.cfg, "/unused"
            )
        self.assertEqual(train.batch_size, 3)
        self.assertEqual(test.batch_size, 2)
        self.assertEqual(knn.batch_size, 2)
        self.assertIsInstance(train.sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(knn.sampler, torch.utils.data.SequentialSampler)
        self.assertFalse(train.drop_last)
        self.assertEqual(sorted(train.dataset.indices), list(range(7)))
        self.assertEqual(list(train.dataset.indices), list(knn.dataset.indices))
        self.assertEqual(len(indices), 7)
        self.assertEqual(sum(len(batch["label"]) for batch in train), 7)

    def test_cli_lp_defaults_are_independent_of_cp(self):
        args = entry.create_base_parser().parse_args(["--dataset", "dtd", "--backbone", "test"])
        self.assertEqual(
            (args.lp_epochs, args.lp_batch_size, args.lp_lr, args.lp_forward_batch_size),
            (150, 512, 1e-3, None),
        )
        self.assertEqual(args.batch_size, 32)
        args = entry.create_base_parser().parse_args(
            [
                "--dataset",
                "dtd",
                "--backbone",
                "test",
                "--lp-epochs",
                "2",
                "--lp-batch-size",
                "4",
                "--lp-lr",
                "0.01",
                "--lp-forward-batch-size",
                "2",
            ]
        )
        self.assertEqual(
            (args.lp_epochs, args.lp_batch_size, args.lp_lr, args.lp_forward_batch_size),
            (2, 4, 0.01, 2),
        )

    def test_both_stages_pass_same_lp_protocol_and_class_count(self):
        expected = dict(
            lp_epochs=2, lp_lr=0.005, lp_forward_batch_size=1, lp_num_classes=2, lp_seed=42
        )
        metrics = {
            "knn_f1": 0.5,
            "knn_acc": 0.5,
            "linear_pytorch_f1": 0.5,
            "linear_pytorch_acc": 0.5,
            "lp": lp_config(2, 3, 0.005, 1),
        }
        logger = MagicMock()
        with patch.object(entry, "zero_shot_eval", return_value=metrics) as evaluate:
            pre = entry.run_baseline(None, None, None, "cpu", self.args, logger, num_classes=2)
            post = entry.run_final_eval(
                None, None, None, "cpu", self.args, logger, pre, num_classes=2
            )
        self.assertEqual(evaluate.call_count, 2)
        for call in evaluate.call_args_list:
            for key, value in expected.items():
                self.assertEqual(call.kwargs[key], value)
        self.assertEqual(pre["lp"], post["lp"])
        self.assertNotIn("baseline/lp", logger.experiment.log.call_args.args[0])

    def test_lp_protocol_validates_configuration(self):
        self.assertEqual(lp_config()["protocol"], "frozen_online_lp_v1")
        for kwargs in (
            {"epochs": 0},
            {"batch_size": -1},
            {"forward_batch_size": True},
            {"lr": float("nan")},
            {"lr": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lp_config(**kwargs)

    def test_entry_point_exports_lp_metadata_only_for_evaluated_stage(self):
        metrics = {
            "knn_f1": 0.5,
            "knn_acc": 0.5,
            "linear_pytorch_f1": 0.5,
            "linear_pytorch_acc": 0.5,
            "lp": lp_config(),
        }
        for post_only in (False, True):
            with (
                self.subTest(post_only=post_only),
                tempfile.TemporaryDirectory() as temp,
                ExitStack() as stack,
            ):
                output = Path(temp) / "result.json"
                argv = [
                    "continued_pretraining.py",
                    "--dataset",
                    "breastmnist",
                    "--backbone",
                    "test",
                    "--results-json",
                    str(output),
                    "--cache-dir",
                    temp,
                    "--n-samples",
                    "7",
                ]
                argv += ["--cp-method", "simclr", "--skip-baseline"] if post_only else ["--no-cp"]
                backbone = MagicMock(num_features=4)
                stack.enter_context(patch("sys.argv", argv))
                stack.enter_context(
                    patch.object(
                        entry, "_get_methods", return_value={"simclr": {"setup": MagicMock()}}
                    )
                )
                stack.enter_context(
                    patch.object(entry, "load_backbone", return_value=(backbone, "cpu"))
                )
                stack.enter_context(
                    patch.object(entry, "get_config", return_value=(self.cfg, 15, 15))
                )
                stack.enter_context(
                    patch.object(entry, "configure_normalization", return_value=self.cfg)
                )
                stack.enter_context(patch.object(entry, "WandbLogger"))
                stack.enter_context(
                    patch.object(
                        entry,
                        "_create_shared_eval_data",
                        return_value=(
                            None,
                            MagicMock(dataset=list(range(3))),
                            None,
                            None,
                            list(range(7)),
                        ),
                    )
                )
                stack.enter_context(patch.object(entry, "_create_cp_data", return_value=(None, 2)))
                stack.enter_context(patch.object(entry, "run_training"))
                stack.enter_context(patch.object(entry, "zero_shot_eval", return_value=metrics))
                entry.main()
                row = json.loads(output.read_text())
                self.assertEqual(row["post_lp" if post_only else "pre_lp"], lp_config())
                self.assertNotIn("pre_lp" if post_only else "post_lp", row)

    def test_real_completed_checkpoint_restores_weights_without_rewriting(self):
        args = argparse.Namespace(epochs=2, n_samples=7, batch_size=3, accumulate_grad_batches=1)
        model = torch.nn.Linear(4, 2)
        expected = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        saved = {
            "epoch": 1,
            "global_step": 6,
            "loops": {"fit_loop": {"epoch_progress": {"current": {"processed": 2}}}},
            "state_dict": dict(expected, **{"callbacks_modules.probe.weight": torch.zeros(2, 4)}),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cp.ckpt"
            torch.save(saved, path)
            original = path.read_bytes()
            self.assertTrue(entry._load_completed_checkpoint(model, path, args))
            self.assertEqual(original, path.read_bytes())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(expected[key], value))


if __name__ == "__main__":
    unittest.main()
