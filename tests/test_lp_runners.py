"""Versioned LP runners must never reuse legacy evaluation artifacts."""

import argparse
import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(REPO))

import cp_full
import precp
from stable_cp.utils.lp_protocol import LP_DIRECTORY, LP_PROTOCOL, lp_config


class LPRunnerTests(unittest.TestCase):
    task = ("DINOv3-B", "breastmnist", "LeJEPA-CP")
    seed = 42

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
        return path

    def legacy_config(self, digest="old-baseline"):
        encoder, dataset, method = self.task
        return dict(
            protocol=cp_full.PROTOCOL,
            encoder=encoder,
            dataset=dataset,
            method=method,
            seed=self.seed,
            full_train=True,
            sft=False,
            gpu=cp_full.gpu_for(self.task),
            backbone=precp.ENCODERS[cp_full.ENCODERS[encoder]],
            baseline_sha256=digest,
            n_train=cp_full.DATASETS[dataset],
            recipe=cp_full.recipe(self.task),
        )

    def pre_row(self):
        return dict(
            dataset="breastmnist",
            backbone=precp.ENCODERS["DINOv3"],
            seed=self.seed,
            full_train=True,
            no_cp=True,
            normalization_mode="pretrained",
            normalization={"mean": [0.5] * 3, "std": [0.5] * 3},
            n_samples=546,
            n_train_actual=546,
            n_test=78,
            num_classes=2,
            feature_readout="cls",
            cp_config={"knn_k": 20, "pool_strategy": "cls"},
            pre_lp=lp_config(),
            geometry=dict(
                protocol="precp_geometry_5000_v1",
                n_geometry=546,
                n_reference=5000,
                **{key: 0.1 for key in precp.GEOMETRY_METRICS},
            ),
            **{key: 0.6 for key in precp.METRICS},
        )

    def post_row(self, attached=True):
        pre = self.pre_row()
        row = dict(
            dataset=pre["dataset"],
            backbone=pre["backbone"],
            method="lejepa",
            seed=self.seed,
            full_train=True,
            no_cp=False,
            epochs=150,
            n_samples=546,
            n_train_actual=546,
            n_test=pre["n_test"],
            num_classes=pre["num_classes"],
            normalization_mode="pretrained",
            normalization=pre["normalization"],
            feature_readout="cls",
            cp_config=cp_full.recipe(self.task),
            post_lp=lp_config(),
            post_geometry=dict(pre["geometry"], phase="post", reference_encoder="post_cp"),
            **{key: 0.7 for key in cp_full.POST_METRICS},
        )
        if attached:
            row.update(pre_lp=lp_config(), baseline_sha256="new-baseline")
        return row

    def evaluation_dir(self):
        return cp_full.seed_dir(self.root, self.task, self.seed) / LP_DIRECTORY

    def write_checkpoint(self, config=None):
        directory = cp_full.seed_dir(self.root, self.task, self.seed)
        self.save(directory / "config.json", config or self.legacy_config())
        (directory / "cp.ckpt").write_bytes(b"existing trained checkpoint")
        return directory

    def write_completed(self):
        self.write_checkpoint()
        directory = self.evaluation_dir()
        self.save(
            directory / "config.json", cp_full.run_config(self.task, self.seed, "new-baseline")
        )
        self.save(directory / "result.json", self.post_row())
        for name in ("post_reference.npz", "post_features.npz"):
            (directory / name).write_bytes(b"geometry fixture")
        return directory

    def completed(self, **kwargs):
        return cp_full.completed_result(
            self.root, self.task, self.seed, self.pre_row(), "new-baseline", **kwargs
        )

    def test_pre_result_path_is_versioned(self):
        expected = (
            self.root
            / "outputs/precp_full"
            / LP_DIRECTORY
            / "results/DINOv3/breastmnist/seed42.json"
        )
        self.assertEqual(precp.result_path(self.root, "DINOv3", "breastmnist", self.seed), expected)

    def test_legacy_pre_results_do_not_skip_new_evaluation(self):
        legacy = self.root / "outputs/precp_full/results/DINOv3/breastmnist/seed42.json"
        self.save(legacy, self.pre_row())
        self.assertIsNone(
            precp.completed_result(
                precp.result_path(self.root, "DINOv3", "breastmnist", self.seed),
                "DINOv3",
                "breastmnist",
                self.seed,
            )
        )

    def test_pre_result_requires_exact_online_lp_metadata(self):
        path = precp.result_path(self.root, "DINOv3", "breastmnist", self.seed)
        for metadata in (
            None,
            {**lp_config(), "epochs": 149},
            {**lp_config(), "feature_cache": True},
        ):
            with self.subTest(metadata=metadata):
                self.save(path, {**self.pre_row(), "pre_lp": metadata})
                with self.assertRaises(ValueError):
                    precp.completed_result(path, "DINOv3", "breastmnist", self.seed)

    def test_cp_command_keeps_checkpoint_but_versions_evaluation_outputs(self):
        command = cp_full.command_for(
            self.root, self.task, self.seed, self.root / "data", self.root / "reference", 2
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(Path(value("--checkpoint-path")), self.evaluation_dir().parent / "cp.ckpt")
        self.assertEqual(Path(value("--results-json")), self.evaluation_dir() / "result.json")
        self.assertEqual(Path(value("--post-geometry-dir")), self.evaluation_dir())
        self.assertIn("--resume", command)

    def test_runner_commands_explicitly_fix_lp_hyperparameters(self):
        cp_command = cp_full.command_for(
            self.root, self.task, self.seed, self.root / "data", self.root / "reference", 2
        )
        args = argparse.Namespace(root=self.root, dry_run=False, num_workers=2)
        with (
            patch.object(
                precp.subprocess, "run", return_value=SimpleNamespace(returncode=1)
            ) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            precp.run_dataset(args, "breastmnist", [("DINOv3", self.seed)], self.root / "data")
        for command in (cp_command, run.call_args.args[0]):
            for flag, value in (
                ("--lp-epochs", "150"),
                ("--lp-batch-size", "512"),
                ("--lp-lr", "0.001"),
                ("--lp-forward-batch-size", "32"),
            ):
                with self.subTest(flag=flag, command=command):
                    self.assertIn(flag, command)
                    self.assertEqual(command[command.index(flag) + 1], value)
        self.assertTrue(
            (
                self.root
                / "outputs/precp_full"
                / LP_DIRECTORY
                / "logs/DINOv3/breastmnist/seed42.log"
            ).is_file()
        )

    def test_cp_run_config_contains_both_lp_metadata_and_current_digest(self):
        config = cp_full.run_config(self.task, self.seed, "new-baseline")
        self.assertEqual(config.get("lp_protocol"), LP_PROTOCOL)
        self.assertEqual(config.get("pre_lp"), lp_config())
        self.assertEqual(config.get("post_lp"), lp_config())
        self.assertEqual(config["baseline_sha256"], "new-baseline")
        self.assertEqual(config["recipe"], self.legacy_config()["recipe"])

    def test_matching_checkpoint_reused_without_overwriting_legacy_artifacts(self):
        directory = self.write_checkpoint()
        self.save(directory / "result.json", {"legacy": True})
        (directory / "run.log").write_text("legacy log")
        originals = {path.name: path.read_bytes() for path in directory.iterdir()}
        cp_full.prepare_run(self.root, self.task, self.seed, "new-baseline")
        self.assertEqual(
            json.loads((self.evaluation_dir() / "config.json").read_text()),
            cp_full.run_config(self.task, self.seed, "new-baseline"),
        )
        for name, content in originals.items():
            self.assertEqual((directory / name).read_bytes(), content)

    def test_checkpoint_recipe_and_identity_mismatches_are_rejected(self):
        for key, value in (
            ("seed", 43),
            ("backbone", "different"),
            ("dataset", "dtd"),
            ("gpu", "v100"),
            ("recipe", {**cp_full.recipe(self.task), "lr": 9e-4}),
        ):
            with self.subTest(key=key):
                self.write_checkpoint({**self.legacy_config(), key: value})
                with self.assertRaises(ValueError):
                    cp_full.prepare_run(self.root, self.task, self.seed, "new-baseline")

    def test_checkpoint_without_provenance_is_rejected(self):
        directory = cp_full.seed_dir(self.root, self.task, self.seed)
        directory.mkdir(parents=True)
        (directory / "cp.ckpt").write_bytes(b"unattributed")
        with self.assertRaises(ValueError):
            cp_full.prepare_run(self.root, self.task, self.seed, "new-baseline")

    def test_new_evaluation_config_cannot_reuse_another_baseline(self):
        self.write_checkpoint()
        self.save(
            self.evaluation_dir() / "config.json",
            cp_full.run_config(self.task, self.seed, "other-baseline"),
        )
        with self.assertRaises(ValueError):
            cp_full.prepare_run(self.root, self.task, self.seed, "new-baseline")

    def test_new_evaluation_config_requires_exact_lp_metadata(self):
        self.write_checkpoint()
        config = cp_full.run_config(self.task, self.seed, "new-baseline")
        config["post_lp"] = {**lp_config(), "epochs": 149}
        self.save(self.evaluation_dir() / "config.json", config)
        with self.assertRaises(ValueError):
            cp_full.prepare_run(self.root, self.task, self.seed, "new-baseline")

    def test_legacy_cp_result_is_pending(self):
        directory = self.write_checkpoint(self.legacy_config("new-baseline"))
        self.save(directory / "result.json", self.post_row())
        for name in ("post_reference.npz", "post_features.npz"):
            (directory / name).write_bytes(b"legacy geometry")
        self.assertIsNone(self.completed())

    def test_new_complete_result_is_accepted_with_original_checkpoint(self):
        self.write_completed()
        self.assertEqual(self.completed()["post_lp"], lp_config())

    def test_new_result_cannot_skip_a_mismatched_checkpoint_recipe(self):
        self.write_completed()
        changed = self.legacy_config()
        changed["recipe"]["lr"] = 9e-4
        self.save(self.evaluation_dir().parent / "config.json", changed)
        with self.assertRaises(ValueError):
            self.completed()

    def test_lp_cli_fields_in_result_do_not_change_cp_recipe_identity(self):
        directory = self.write_completed()
        row = self.post_row()
        row["cp_config"].update(
            lp_epochs=150, lp_batch_size=512, lp_lr=1e-3, lp_forward_batch_size=32
        )
        self.save(directory / "result.json", row)
        self.assertIsNotNone(self.completed())
        self.assertFalse(any(key.startswith("lp_") for key in cp_full.recipe(self.task)))

    def test_new_result_with_legacy_or_changed_lp_is_rejected(self):
        directory = self.write_completed()
        for key, value in (
            ("pre_lp", None),
            ("post_lp", None),
            ("post_lp", {**lp_config(), "batch_size": 32}),
            ("baseline_sha256", "old-baseline"),
        ):
            with self.subTest(key=key, value=value):
                self.save(directory / "result.json", {**self.post_row(), key: value})
                with self.assertRaises(ValueError):
                    self.completed()

    def test_unattached_cli_result_is_pending_until_subprocess_succeeds(self):
        directory = self.write_completed()
        self.save(directory / "result.json", self.post_row(attached=False))
        self.assertIsNone(self.completed())
        self.assertIsNotNone(self.completed(allow_unattached=True))

    def test_attach_baseline_propagates_lp_metadata(self):
        row = cp_full.attach_baseline(
            self.post_row(attached=False), self.pre_row(), Path("baseline.json"), "new-baseline"
        )
        self.assertEqual(row.get("pre_lp"), lp_config())
        self.assertEqual(row["baseline_sha256"], "new-baseline")
        self.assertAlmostEqual(row["delta"]["linear_f1"], 0.1)

    def test_manual_baseline_attachment_rejects_mixed_lp_protocols(self):
        for legacy_pre in (False, True):
            with self.subTest(legacy_pre=legacy_pre):
                row, pre = self.post_row(attached=False), self.pre_row()
                if legacy_pre:
                    pre.pop("pre_lp")
                else:
                    row.pop("post_lp")
                with self.assertRaises(ValueError):
                    cp_full.attach_baseline(row, pre, Path("baseline.json"), "new-baseline")

    def test_manual_attachment_rejects_existing_different_pre_lp(self):
        row = self.post_row(attached=False)
        row["pre_lp"] = {**lp_config(), "epochs": 149}
        with self.assertRaises(ValueError):
            cp_full.attach_baseline(row, self.pre_row(), Path("baseline.json"), "new-baseline")

    def test_failed_raw_run_is_retried_before_completed_run_is_skipped(self):
        checkpoint = self.write_checkpoint()
        legacy_result = self.save(checkpoint / "result.json", {"legacy": True})
        original_config = (checkpoint / "config.json").read_bytes()
        self.save(precp.result_path(self.root, "DINOv3", "breastmnist", self.seed), self.pre_row())
        args = argparse.Namespace(root=self.root, task_id=0, dry_run=False, num_workers=2)
        modules = {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True, get_device_name=lambda index: "test GPU"
                )
            ),
            "filelock": SimpleNamespace(FileLock=lambda *args, **kwargs: contextlib.nullcontext()),
            "data_cache": SimpleNamespace(
                staged_dataset=lambda path, dataset: contextlib.nullcontext(path),
                staged_directory=contextlib.nullcontext,
            ),
        }

        returncodes = iter((1, 0))

        def evaluate(command, **kwargs):
            output = Path(command[command.index("--results-json") + 1])
            self.save(output, self.post_row(attached=False))
            for name in ("post_reference.npz", "post_features.npz"):
                (output.parent / name).write_bytes(b"new geometry")
            return SimpleNamespace(returncode=next(returncodes))

        with (
            patch.dict(sys.modules, modules),
            patch.object(cp_full, "TASKS", (self.task,)),
            patch.object(cp_full, "SEEDS", (self.seed,)),
            patch.object(cp_full, "validate_reference", return_value=self.root / "reference"),
            patch.object(cp_full.subprocess, "run", side_effect=evaluate) as run,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaises(SystemExit):
                cp_full.run_task(args)
            self.assertNotIn("SKIP", output.getvalue())
            self.assertNotIn(
                "baseline_sha256", json.loads((self.evaluation_dir() / "result.json").read_text())
            )
            cp_full.run_task(args)
            cp_full.run_task(args)
        self.assertEqual(run.call_count, 2)
        self.assertIn("DONE", output.getvalue())
        self.assertIn("SKIP", output.getvalue())
        self.assertTrue((self.evaluation_dir() / "run.log").is_file())
        self.assertEqual(json.loads(legacy_result.read_text()), {"legacy": True})
        self.assertEqual((checkpoint / "config.json").read_bytes(), original_config)
        result = json.loads((self.evaluation_dir() / "result.json").read_text())
        self.assertEqual(result["pre_lp"], lp_config())
        self.assertEqual(result["post_lp"], lp_config())

    def test_cp_report_lists_missing_new_runs_even_without_new_baselines(self):
        legacy = self.root / "outputs/results/cp_full_results.DINOv3-B.csv"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("legacy report\n")
        with (
            patch.object(cp_full, "TASKS", (self.task,)),
            patch.object(cp_full, "SEEDS", (self.seed,)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cp_full.report(self.root, (self.task[0],))
        path = self.root / "outputs/results" / LP_DIRECTORY / "cp_full_results.DINOv3-B.csv"
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(records[0]["status"], "MISSING")
        self.assertEqual(Path(records[0]["source"]), self.evaluation_dir() / "result.json")
        self.assertEqual(legacy.read_text(), "legacy report\n")

    def test_pre_report_versions_outputs_and_does_not_count_legacy_results(self):
        legacy = self.root / "outputs/precp_full/results/DINOv3/breastmnist/seed42.json"
        self.save(legacy, self.pre_row())
        with (
            patch.object(precp, "GROUPS", (("breastmnist",),)),
            patch.object(precp, "SEEDS", (self.seed,)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            precp.report(self.root, ("DINOv3",))
        path = self.root / "outputs/precp_full" / LP_DIRECTORY / "results.DINOv3.csv"
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(records[0]["status"], "MISSING")

    def test_both_runner_clis_dry_run_without_ml_imports(self):
        for runner, flags in (("precp.py", ["--encoder", "DINOv3"]), ("cp_full.py", [])):
            with self.subTest(runner=runner):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "run" / runner),
                        "run",
                        "--task-id",
                        "0",
                        "--root",
                        str(self.root),
                        "--dry-run",
                        *flags,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(LP_DIRECTORY, result.stdout)
                self.assertNotIn("SKIP", result.stdout)
                self.assertFalse((self.root / "outputs").exists())


class CompletedCheckpointTests(unittest.TestCase):
    """Exercise the CLI restore gate without importing heavyweight ML packages."""

    @classmethod
    def setUpClass(cls):
        source = REPO / "continued_pretraining.py"
        tree = ast.parse(source.read_text())
        names = {"get_steps_per_epoch", "_load_completed_checkpoint", "run_training"}
        cls.definitions = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ],
            type_ignores=[],
        )

    def setUp(self):
        self.args = argparse.Namespace(
            epochs=150,
            n_samples=546,
            batch_size=256,
            accumulate_grad_batches=1,
            checkpoint_path="cp.ckpt",
            resume=True,
        )
        self.saved = {
            "epoch": 149,
            "global_step": 450,
            "loops": {
                "fit_loop": {"epoch_progress": {"current": {"processed": 150, "completed": 149}}}
            },
            "state_dict": {"backbone.weight": "encoder weights"},
        }
        self.torch = SimpleNamespace(load=Mock(return_value=self.saved))
        self.trainer = Mock(
            side_effect=AssertionError("Completed checkpoints must not create a Trainer")
        )
        self.namespace = {
            "torch": self.torch,
            "Path": Path,
            "pl": SimpleNamespace(Trainer=self.trainer),
            "print": Mock(),
        }
        exec(compile(self.definitions, "continued_pretraining.py", "exec"), self.namespace)
        self.module = Mock()

    def restore(self):
        return self.namespace["_load_completed_checkpoint"](self.module, "cp.ckpt", self.args)

    def test_epoch_end_checkpoint_before_completed_counter_is_accepted(self):
        self.assertTrue(self.restore())
        self.module.load_state_dict.assert_called_once_with(self.saved["state_dict"], strict=True)

    def test_post_fit_checkpoint_is_accepted(self):
        self.saved["epoch"] = 150
        self.saved["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = 150
        self.assertTrue(self.restore())

    def test_partial_progress_does_not_fast_restore(self):
        for epoch, processed, updates in ((148, 149, 447), (149, 149, 450)):
            with self.subTest(epoch=epoch, processed=processed, updates=updates):
                self.saved.update(epoch=epoch, global_step=updates)
                self.saved["loops"]["fit_loop"]["epoch_progress"]["current"][
                    "processed"
                ] = processed
                self.assertFalse(self.restore())
        self.module.load_state_dict.assert_not_called()

    def test_completed_epochs_with_inconsistent_main_updates_are_rejected(self):
        for updates in (449, 900):
            with self.subTest(updates=updates):
                self.saved["global_step"] = updates
                with self.assertRaises(ValueError):
                    self.restore()
        self.module.load_state_dict.assert_not_called()

    def test_completed_accumulation_counts_main_optimizer_updates(self):
        self.args.batch_size = 128
        self.args.accumulate_grad_batches = 2
        self.assertTrue(self.restore())

    def test_callback_filter_preserves_all_method_state_with_strict_loading(self):
        method_keys = {
            "lejepa": ("backbone.weight", "projector.0.weight", "sigreg_loss.univariate_test.t"),
            "simclr": ("backbone.weight", "projector.0.weight"),
            "diet": ("backbone.weight", "diet_head.weight"),
            "mae": ("backbone.vit.weight", "backbone.patch_embed.weight", "decoder.weight"),
        }
        for method, keys in method_keys.items():
            with self.subTest(method=method):
                state = dict.fromkeys(keys, "trained weight")
                self.saved["state_dict"] = dict(
                    state,
                    **{
                        "callbacks_modules.cp_linear_probe.weight": "probe",
                        "callbacks_metrics.cp_knn_probe.state": "metric",
                    },
                )
                self.module.reset_mock()
                self.assertTrue(self.restore())
                self.module.load_state_dict.assert_called_once_with(state, strict=True)

    def test_incompatible_state_is_not_silently_treated_as_untrained(self):
        self.module.load_state_dict.side_effect = RuntimeError("Missing CP decoder key")
        with self.assertRaises(RuntimeError):
            self.restore()

    def test_completed_restore_never_constructs_trainer_or_rewrites_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "cp.ckpt"
            checkpoint.write_bytes(b"immutable completed checkpoint")
            self.namespace["run_training"](
                self.module, None, self.args, {}, 768, 15, None, checkpoint, num_trained_blocks=2
            )
            self.assertEqual(checkpoint.read_bytes(), b"immutable completed checkpoint")
        self.trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
