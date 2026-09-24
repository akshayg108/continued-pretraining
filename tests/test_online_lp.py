"""CPU-only contracts for the frozen online linear probe."""

import importlib
import multiprocessing
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, Subset

from stable_cp import evaluation
from stable_cp.utils.lp_protocol import lp_config


class RecordingDataset(Dataset):
    def __init__(self, size=5, augmented=False, as_dict=True):
        self.labels = [index % 2 for index in range(size)]
        self.augmented = augmented
        self.as_dict = as_dict
        self.reads = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = torch.tensor([float(label == 0), float(label == 1), 1.0])
        if self.augmented:
            image = image + torch.rand(3) * 0.1
        self.reads.append((index, image.clone()))
        if self.as_dict:
            return {"image": image, "label": label}
        return image, label


class RecordingEncoder(nn.Module):
    def __init__(self, fail_at=None):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.norm = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.5)
        self.fail_at = fail_at
        self.calls = []

    def forward_features(self, images):
        self.calls.append(
            {
                "batch_size": len(images),
                "grad_enabled": torch.is_grad_enabled(),
                "train_flags": [module.training for module in self.modules()],
                "requires_grad": [param.requires_grad for param in self.parameters()],
                "images": images.detach().clone(),
            }
        )
        if self.fail_at == len(self.calls):
            raise RuntimeError("synthetic encoder failure")
        return self.dropout(self.norm(self.projection(images)))


class OnlineLinearProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.train = RecordingDataset(augmented=True)
        self.test = RecordingDataset(size=4, as_dict=False)
        self.train_loader = DataLoader(self.train, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(self.test, batch_size=3, shuffle=False)
        self.model = RecordingEncoder()

    def evaluate(self, **kwargs):
        evaluate = getattr(evaluation, "linear_probe_online_evaluate", None)
        self.assertIsNotNone(evaluate, "The online LP evaluator must be exported")
        options = {
            "epochs": 2,
            "forward_batch_size": 2,
            "verbose": False,
        }
        options.update(kwargs)
        return evaluate(
            self.model,
            self.train_loader,
            self.test_loader,
            torch.device("cpu"),
            **options,
        )

    def test_each_epoch_reads_fresh_images_and_keeps_tail_batches(self):
        result = self.evaluate()
        self.assertEqual([index for index, _ in self.train.reads], list(range(5)) * 2)
        self.assertFalse(torch.equal(self.train.reads[0][1], self.train.reads[5][1]))
        self.assertEqual([index for index, _ in self.test.reads], list(range(4)))
        self.assertEqual(
            [call["batch_size"] for call in self.model.calls], [2, 2, 1, 2, 2, 1, 2, 1, 1]
        )
        self.assertEqual(result["lp"]["protocol"], "frozen_online_lp_v1")
        self.assertEqual(result["lp"]["batch_size"], 4)
        self.assertEqual(result["lp"]["epochs"], 2)
        self.assertEqual(result["lp"]["forward_batch_size"], 2)
        self.assertEqual(
            set(result),
            {
                "linear_pytorch_acc",
                "linear_pytorch_f1",
                "linear_pytorch_auroc",
                "lp",
            },
        )

    def test_default_encoder_forward_uses_the_complete_classifier_batch(self):
        self.train = RecordingDataset(size=513, augmented=True)
        self.train_loader = DataLoader(self.train, batch_size=512)
        result = evaluation.linear_probe_online_evaluate(
            self.model,
            self.train_loader,
            self.test_loader,
            torch.device("cpu"),
            epochs=1,
            verbose=False,
        )
        self.assertEqual([call["batch_size"] for call in self.model.calls], [512, 1, 3, 1])
        self.assertIsNone(result["lp"]["forward_batch_size"])
        self.assertIsNone(lp_config()["forward_batch_size"])

    def test_none_forward_size_is_equivalent_to_explicit_full_batch(self):
        full = self.evaluate(forward_batch_size=None, seed=41)
        full_views = [call["images"].clone() for call in self.model.calls]
        self.model.calls.clear()
        explicit = self.evaluate(forward_batch_size=4, seed=41)
        self.assertEqual(full["linear_pytorch_f1"], explicit["linear_pytorch_f1"])
        for left, right in zip(full_views, self.model.calls):
            self.assertTrue(torch.equal(left, right["images"]))

    def test_encoder_parameters_buffers_gradients_and_mixed_modes_are_restored(self):
        self.model.train()
        self.model.norm.eval()
        self.model.projection.bias.requires_grad_(False)
        self.model.projection.weight.grad = torch.ones_like(self.model.projection.weight)
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        masks = [param.requires_grad for param in self.model.parameters()]
        modes = [module.training for module in self.model.modules()]
        gradients = [param.grad for param in self.model.parameters()]
        self.evaluate()
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual(masks, [param.requires_grad for param in self.model.parameters()])
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        for param, old_grad in zip(self.model.parameters(), gradients):
            self.assertIs(param.grad, old_grad)
        self.assertTrue(
            torch.equal(
                self.model.projection.weight.grad, torch.ones_like(self.model.projection.weight)
            )
        )
        for call in self.model.calls:
            self.assertFalse(call["grad_enabled"])
            self.assertFalse(any(call["train_flags"]))
            self.assertFalse(any(call["requires_grad"]))

    def test_encoder_state_is_restored_when_forward_fails(self):
        self.model = RecordingEncoder(fail_at=2)
        self.model.train()
        self.model.dropout.eval()
        self.model.projection.bias.requires_grad_(False)
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        modes = [module.training for module in self.model.modules()]
        masks = [param.requires_grad for param in self.model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
            self.evaluate()
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertEqual(masks, [param.requires_grad for param in self.model.parameters()])
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(param.grad is None for param in self.model.parameters()))

    def test_nonfinite_features_fail_fast_and_restore_encoder_state(self):
        self.model.train()
        self.model.norm.eval()
        self.model.projection.bias.requires_grad_(False)
        with torch.no_grad():
            self.model.projection.weight.fill_(float("nan"))
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        modes = [module.training for module in self.model.modules()]
        masks = [param.requires_grad for param in self.model.parameters()]
        with self.assertRaisesRegex(ValueError, "Non-finite.*features"):
            self.evaluate()
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertEqual(masks, [param.requires_grad for param in self.model.parameters()])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(before[name], value, equal_nan=True, rtol=0, atol=0)
        self.assertEqual(len(self.train.reads), 4)
        self.assertEqual(self.test.reads, [])
        self.assertTrue(all(param.grad is None for param in self.model.parameters()))

    def test_nonfinite_loss_fails_before_an_optimizer_update(self):
        class NonfiniteLoss(nn.Module):
            def forward(self, logits, labels):
                return logits.sum() * float("nan")

        with (
            patch("torch.nn.CrossEntropyLoss", NonfiniteLoss),
            patch("torch.optim.Adam.step") as step,
        ):
            with self.assertRaisesRegex(ValueError, "Non-finite.*loss"):
                self.evaluate()
        step.assert_not_called()
        self.assertEqual(len(self.train.reads), 4)
        self.assertTrue(all(param.grad is None for param in self.model.parameters()))

    def test_only_a_plain_linear_head_is_optimized_with_adam(self):
        original_adam = torch.optim.Adam
        optimizer_instances = []
        updates = []

        class RecordingAdam(original_adam):
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                optimizer_instances.append(self)

            def step(self, *args, **kwargs):
                params = self.param_groups[0]["params"]
                before = [param.detach().clone() for param in params]
                result = super().step(*args, **kwargs)
                updates.append(any(not torch.equal(old, new) for old, new in zip(before, params)))
                return result

        with patch("torch.optim.Adam", RecordingAdam):
            self.evaluate(lr=0.02)
        self.assertEqual(len(optimizer_instances), 1)
        optimizer = optimizer_instances[0]
        self.assertEqual(len(updates), 4)
        self.assertTrue(all(updates))
        self.assertEqual(optimizer.defaults["lr"], 0.02)
        self.assertEqual(optimizer.defaults["weight_decay"], 0)
        self.assertEqual(
            [tuple(param.shape) for param in optimizer.param_groups[0]["params"]], [(2, 3), (2,)]
        )
        encoder_ids = {id(param) for param in self.model.parameters()}
        self.assertTrue(
            all(id(param) not in encoder_ids for param in optimizer.param_groups[0]["params"])
        )
        self.assertTrue(all(param.grad is None for param in self.model.parameters()))

    def test_clean_test_predictions_are_deterministic_and_features_normalized(self):
        online = getattr(evaluation, "linear_probe_online_evaluate", None)
        self.assertIsNotNone(online, "The online LP evaluator must be exported")
        module = importlib.import_module(online.__module__)
        original_linear = nn.Linear
        seen_features = []
        heads = []

        class RecordingLinear(original_linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                heads.append(self)

            def forward(self, features):
                seen_features.append(features.detach().clone())
                return super().forward(features)

        with patch.object(module.nn, "Linear", RecordingLinear):
            torch.manual_seed(22)
            first = self.evaluate()
        first_features = [features.clone() for features in seen_features]
        self.train.reads.clear()
        self.test.reads.clear()
        seen_features.clear()
        with patch.object(module.nn, "Linear", RecordingLinear):
            torch.manual_seed(22)
            second = self.evaluate()
        self.assertEqual(first, second)
        for left, right in zip(first_features, seen_features):
            self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.allclose(right.norm(dim=1), torch.ones(len(right))))
        self.assertFalse(heads[-1].training)

    def test_subset_and_explicit_class_count_need_no_augmented_prepass(self):
        self.train_loader = DataLoader(Subset(self.train, [0, 1, 4]), batch_size=2)
        self.evaluate(epochs=1)
        self.assertEqual([index for index, _ in self.train.reads], [0, 1, 4])
        self.train.reads.clear()
        self.evaluate(epochs=1, num_classes=2)
        self.assertEqual([index for index, _ in self.train.reads], [0, 1, 4])

    def test_streaming_metrics_match_existing_full_tensor_definitions(self):
        online = getattr(evaluation, "linear_probe_online_evaluate", None)
        self.assertIsNotNone(online, "The online LP evaluator must be exported")
        module = importlib.import_module(online.__module__)
        original_linear = nn.Linear
        test_logits = []

        class RecordingLinear(original_linear):
            def forward(self, features):
                logits = super().forward(features)
                if not self.training:
                    test_logits.append(logits.detach().cpu())
                return logits

        self.test = RecordingDataset(size=5)
        self.test_loader = DataLoader(self.test, batch_size=3)
        with patch.object(module.nn, "Linear", RecordingLinear):
            result = self.evaluate()
        logits = torch.cat(test_logits)
        labels = torch.tensor(self.test.labels)
        expected = {
            "linear_pytorch_acc": module.MulticlassAccuracy(num_classes=2)(
                logits.argmax(dim=1), labels
            ).item(),
            "linear_pytorch_f1": module.MulticlassF1Score(num_classes=2, average="macro")(
                logits.argmax(dim=1), labels
            ).item(),
            "linear_pytorch_auroc": module.MulticlassAUROC(num_classes=2, average="macro")(
                logits.softmax(dim=1), labels
            ).item(),
        }
        for name, value in expected.items():
            self.assertEqual(result[name], value)

    def test_rejects_invalid_protocol_before_reading_images(self):
        for kwargs in ({"epochs": 0}, {"lr": 0}, {"forward_batch_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.evaluate(**kwargs)
        self.assertEqual(self.train.reads, [])
        self.train_loader = DataLoader(self.train, batch_size=4, drop_last=True)
        with self.assertRaisesRegex(ValueError, "drop_last"):
            self.evaluate()
        self.assertEqual(self.train.reads, [])

    def test_seed_is_reproducible_and_all_cpu_rng_states_are_restored(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        first = self.evaluate(seed=11)
        first_views = [image.clone() for _, image in self.train.reads]
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        random.random()
        np.random.rand()
        torch.rand(2)
        self.train.reads.clear()
        second = self.evaluate(seed=11)
        self.assertEqual(first, second)
        for before, (_, after) in zip(first_views, self.train.reads):
            self.assertTrue(torch.equal(before, after))

    def test_rng_states_are_restored_after_failure(self):
        self.model.fail_at = 2
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
            self.evaluate(seed=123)
        self.assertEqual(before_python, random.getstate())
        np.testing.assert_array_equal(before_numpy[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_repeated_seeded_calls_preserve_caller_generator_and_match_shuffled_views(self):
        generator = torch.Generator().manual_seed(17)
        self.train_loader = DataLoader(self.train, batch_size=4, shuffle=True, generator=generator)
        before = generator.get_state().clone()
        first = self.evaluate(seed=29)
        first_reads = [(index, image.clone()) for index, image in self.train.reads]
        self.train.reads.clear()
        torch.rand(9)
        second = self.evaluate(seed=29)
        self.assertEqual(
            [index for index, _ in first_reads], [index for index, _ in self.train.reads]
        )
        for (_, left), (_, right) in zip(first_reads, self.train.reads):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, generator.get_state()))

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "Requires fork workers")
    def test_repeated_seeded_calls_restart_persistent_worker_augmentation(self):
        worker_pids_before = {process.pid for process in multiprocessing.active_children()}
        generator = torch.Generator().manual_seed(17)
        self.train_loader = DataLoader(
            self.train,
            batch_size=4,
            shuffle=True,
            generator=generator,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=3,
            multiprocessing_context="fork",
        )
        before = generator.get_state().clone()
        first = self.evaluate(seed=29)
        first_images = [call["images"].clone() for call in self.model.calls]
        self.model.calls.clear()
        second = self.evaluate(seed=29)
        self.assertEqual(len(first_images), len(self.model.calls))
        for left, right in zip(first_images, self.model.calls):
            self.assertTrue(torch.equal(left, right["images"]))
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, generator.get_state()))
        self.assertEqual(
            worker_pids_before, {process.pid for process in multiprocessing.active_children()}
        )

    def test_reproducible_loader_preserves_public_settings_without_mutating_source(self):
        module = importlib.import_module("stable_cp.evaluation.linear_probe")
        fresh_loader = getattr(module, "_fresh_loader", None)
        self.assertIsNotNone(fresh_loader, "An evaluator-local loader must be constructed")
        generator = torch.Generator().manual_seed(17)
        source = DataLoader(
            self.train,
            batch_size=4,
            shuffle=True,
            generator=generator,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=3,
            timeout=7,
            multiprocessing_context="spawn",
            pin_memory=True,
        )
        before = generator.get_state().clone()
        fresh = fresh_loader(source, seed=29)
        self.assertIsNot(fresh, source)
        self.assertIs(fresh.dataset, source.dataset)
        self.assertIs(fresh.collate_fn, source.collate_fn)
        self.assertIs(fresh.worker_init_fn, source.worker_init_fn)
        for name in (
            "batch_size",
            "drop_last",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "timeout",
            "pin_memory",
            "pin_memory_device",
            "in_order",
        ):
            self.assertEqual(getattr(fresh, name, None), getattr(source, name, None), name)
        self.assertEqual(fresh.multiprocessing_context.get_start_method(), "spawn")
        self.assertIsInstance(fresh.sampler, RandomSampler)
        self.assertIsNot(fresh.sampler, source.sampler)
        self.assertIsNot(fresh.generator, generator)
        self.assertEqual(fresh.generator.initial_seed(), 29)
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_standard_settings_on_a_loader_subclass_are_supported(self):
        class ImageLoader(DataLoader):
            pass

        self.train_loader = ImageLoader(self.train, batch_size=4, shuffle=True)
        self.evaluate(epochs=1, seed=29)
        self.assertEqual(sorted(index for index, _ in self.train.reads), list(range(5)))

    def test_rejects_unsupported_custom_sampler_without_reading_images(self):
        class ReversedSampler(Sampler):
            def __iter__(self):
                return iter(range(4, -1, -1))

            def __len__(self):
                return 5

        self.train_loader = DataLoader(self.train, batch_size=4, sampler=ReversedSampler())
        with self.assertRaisesRegex(ValueError, "sampler"):
            self.evaluate(seed=29)
        self.assertEqual(self.train.reads, [])

    def test_zero_shot_uses_clean_bank_and_never_caches_augmented_train_features(self):
        module = importlib.import_module("stable_cp.evaluation.zero_shot_eval")
        clean = DataLoader(RecordingDataset(), batch_size=4)
        extracted_loaders = []

        def extract(model, loader, *args, **kwargs):
            extracted_loaders.append(loader)
            return np.eye(2, dtype=np.float32), np.array([0, 1])

        with (
            patch.object(module, "extract_features", side_effect=extract),
            patch.object(module, "knn_evaluate", return_value={"knn_f1": 0.5}),
            patch.object(module, "linear_probe_pytorch_evaluate") as cached_lp,
        ):
            result = module.zero_shot_eval(
                self.model,
                self.train_loader,
                self.test_loader,
                torch.device("cpu"),
                knn_train_loader=clean,
                verbose=False,
                lp_epochs=2,
                lp_forward_batch_size=2,
                lp_num_classes=2,
            )
        self.assertEqual(len(extracted_loaders), 2)
        self.assertIn(clean, extracted_loaders)
        self.assertIn(self.test_loader, extracted_loaders)
        self.assertNotIn(self.train_loader, extracted_loaders)
        cached_lp.assert_not_called()
        self.assertEqual(len(self.train.reads), 10)
        self.assertEqual(result["lp"]["epochs"], 2)


if __name__ == "__main__":
    unittest.main()
