"""Check CP cohort selection and commands without training or cluster access."""

from contextlib import nullcontext, redirect_stdout
import csv
import io
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from filelock import FileLock, Timeout

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "run"))
import cp_full as cp


SMALL = (
    "breastmnist", "dermamnist", "dtd", "fgvc_aircraft", "cars196", "cub200",
    "flowers102", "oxford_pet", "aid", "jena_flowers30", "flavia",
)
FOUR_BLOCK = {
    "bloodmnist": 11959, "galaxy10": 14188, "eurosat": 16200, "stanford_dogs": 10800,
}
SIX_BLOCK = {
    "resisc45": 25200, "organamnist": 34561, "plant_village": 43596, "ip102": 45095,
}
ALL_BLOCK = {
    "food101": 75750, "pathmnist": 89996, "octmnist": 97477, "tissuemnist": 165466,
}
ENCODERS = ("DINOv3-B", "CLIP", "SigLiP-2", "DINOv3-L", "MAE")
METHODS = ("LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP")


def cli(*args):
    output = io.StringIO()
    with patch.object(sys, "argv", ["cp_full.py", *args]), redirect_stdout(output):
        cp.main()
    return output.getvalue()


class FullGridTests(unittest.TestCase):
    def test_original_ids_and_default_selection_are_unchanged(self):
        self.assertEqual(cp.TASKS[:220], tuple(product(ENCODERS, SMALL, METHODS)))
        self.assertIn("220 jobs; 660 CP runs; Full; no FT.", cli("list"))
        self.assertEqual(len(cli("array", "--gpu", "v100").strip().split(",")), 132)
        self.assertEqual(len(cli("array", "--gpu", "a100").strip().split(",")), 88)
        for task in cp.TASKS[:220]:
            self.assertEqual(cp.recipe(task)["num_trained_blocks"], 2)
            self.assertEqual(cp.run_config(task, 42, "sha")["protocol"], "cp_full_small_v1")

    def test_four_block_tasks_are_appended_not_interleaved(self):
        self.assertEqual(cp.TASKS[220:300], tuple(product(ENCODERS, FOUR_BLOCK, METHODS)))
        self.assertIn("80 jobs; 240 CP runs; Full; no FT.", cli("list", "--group", "four-block"))
        a100 = cli("array", "--group", "four-block", "--gpu", "a100").strip().split(",")
        large = cli("array", "--group", "four-block", "--gpu", "a100-80gb").strip().split(",")
        self.assertEqual(len(a100), 64)
        self.assertEqual(len(large), 16)
        self.assertEqual({int(i) for i in a100 + large}, set(range(220, 300)))
        self.assertTrue(all(cp.TASKS[int(i)][0] == "DINOv3-L" for i in large))
        self.assertEqual(cli("array", "--group", "four-block", "--gpu", "v100"), "\n")

    def test_only_unfreezing_changes_in_training_recipe(self):
        for encoder, dataset, method in product(ENCODERS, FOUR_BLOCK, METHODS):
            task = encoder, dataset, method
            with self.subTest(task=task):
                expected = cp.recipe((encoder, "breastmnist", method))
                expected["num_trained_blocks"] = 4
                self.assertEqual(cp.recipe(task), expected)
                config = cp.run_config(task, 42, "sha")
                self.assertEqual(config["n_train"], FOUR_BLOCK[dataset])
                self.assertEqual(config["protocol"], "cp_full_four_block_v1")
                self.assertEqual(config["gpu"], "a100-80gb" if encoder == "DINOv3-L" else "a100")
                args = cp.command_for(Path("/root"), task, 42, "/local/data", "/local/reference", 8)
                self.assertEqual(args[args.index("--num-trained-blocks") + 1], "4")
                self.assertEqual(args[args.index("--cache-dir") + 1], "/local/data")
                self.assertEqual(args[args.index("--post-geometry-reference-data") + 1], "/local/reference")
                self.assertIn("--full-train", args)
                self.assertIn("--skip-baseline", args)
                self.assertNotIn("--post-cp-sft", args)
                self.assertEqual(cp.seed_dir(Path("/root"), task, 42),
                                 Path(f"/root/outputs/results/{dataset}/{encoder}/{method}/Full/42"))

    def test_check_uses_only_selected_baselines(self):
        with patch.object(cp, "baseline") as baseline, patch.object(cp, "validate_reference"):
            self.assertIn("READY: 60", cli("check", "--group", "four-block", "--root", "/tmp/cp"))
            self.assertEqual(baseline.call_count, 60)
            self.assertEqual({call.args[1][1] for call in baseline.call_args_list}, set(FOUR_BLOCK))
        with patch.object(cp, "baseline") as baseline, patch.object(cp, "validate_reference"):
            self.assertIn("READY: 12", cli("check", "--group", "four-block", "--encoder", "MAE"))
            self.assertEqual(baseline.call_count, 12)
            self.assertTrue(all(call.args[1][0] == "MAE" for call in baseline.call_args_list))

    def test_four_block_report_does_not_replace_small_grid_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "outputs/results"
            destination.mkdir(parents=True)
            original = destination / "cp_full_results.csv"
            original.write_text("original\n")
            with patch.object(cp, "baseline", return_value=({}, "source", "sha")), \
                    patch.object(cp, "completed_result", return_value=None):
                output = cli("report", "--group", "four-block", "--root", str(root))
            self.assertIn("Completed: 0/240", output)
            self.assertEqual(original.read_text(), "original\n")
            with (destination / "cp_full_results.four-block.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 240)
            self.assertEqual({row["dataset"] for row in rows}, set(FOUR_BLOCK))

    def test_four_block_mae_result_accepts_matched_mean_readout(self):
        task = "MAE", "bloodmnist", "LeJEPA-CP"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = cp.seed_dir(root, task, 42)
            path.mkdir(parents=True)
            pre = {"n_test": 3421, "num_classes": 8, "normalization": {"mean": [0.485, 0.456, 0.406]}}
            geometry = dict(protocol="precp_geometry_5000_v1", n_geometry=5000, n_reference=5000,
                            phase="post", reference_encoder="post_cp", feature_readout=cp.encoder_readout("MAE"))
            geometry.update({key: 0.5 for key in cp.GEOMETRY_METRICS})
            row = dict(dataset="bloodmnist", backbone=cp.PRE_ENCODERS["MAE-Mean"], method="lejepa",
                       seed=42, full_train=True, no_cp=False, epochs=150, n_samples=11959,
                       n_train_actual=11959, normalization_mode="pretrained", **pre,
                       feature_readout=cp.encoder_readout("MAE"), cp_config=cp.recipe(task), post_geometry=geometry)
            row.update({key: 0.5 for key in cp.POST_METRICS})
            (path / "result.json").write_text(json.dumps(row))
            (path / "config.json").write_text(json.dumps(cp.run_config(task, 42, "sha")))
            for name in ("cp.ckpt", "post_reference.npz", "post_features.npz"):
                (path / name).touch()
            self.assertEqual(cp.completed_result(root, task, 42, pre, "sha"), row)
            pre.update({key: 0.4 for key in cp.PRE_METRICS}, geometry=geometry)
            merged = cp.attach_baseline(row, pre, "source", "sha")
            self.assertEqual(merged["protocol"], "cp_full_four_block_v1")
            self.assertEqual(merged["baseline_sha256"], "sha")

    def test_six_block_ids_cover_each_combination_and_seed_once(self):
        self.assertEqual(len(cp.TASKS), 732)
        self.assertEqual(len(cp.TASK_SEEDS), len(cp.TASKS))
        self.assertTrue(all(seeds == cp.SEEDS for seeds in cp.TASK_SEEDS[:300]))
        runs = []
        for task, seeds in zip(cp.TASKS[300:540], cp.TASK_SEEDS[300:540]):
            self.assertEqual(len(seeds), 1)
            runs.append((*task, seeds[0]))
        self.assertEqual(runs, list(product(ENCODERS, SIX_BLOCK, METHODS, (42, 43, 44))))
        output = cli("list", "--group", "six-block")
        self.assertIn("240 jobs; 240 CP runs; Full; no FT.", output)
        self.assertNotIn("seeds=42,43,44", output)
        self.assertEqual(
            cli("array", "--group", "six-block", "--gpu", "a100-80gb").strip(),
            ",".join(map(str, range(300, 540))),
        )
        for gpu in ("v100", "a100"):
            self.assertEqual(cli("array", "--group", "six-block", "--gpu", gpu), "\n")
        self.assertIn("48 jobs; 48 CP runs; Full; no FT.",
                      cli("list", "--group", "six-block", "--encoder", "MAE"))

    def test_all_block_cohort_has_four_encoders_and_one_seed_per_job(self):
        encoders = tuple(e for e in ENCODERS if e != "DINOv3-L")
        runs = [(*task, seeds[0]) for task, seeds in zip(cp.TASKS[540:], cp.TASK_SEEDS[540:])]
        self.assertEqual(runs, list(product(encoders, ALL_BLOCK, METHODS, (42, 43, 44))))
        self.assertTrue(all(len(s) == 1 for s in cp.TASK_SEEDS[540:]))
        self.assertIn("192 jobs; 192 CP runs; Full; no FT.", cli("list", "--group", "all-block"))
        for task in cp.TASKS[540:]:
            expected = cp.recipe((task[0], "breastmnist", task[2]))
            expected["num_trained_blocks"] = -1
            self.assertEqual(cp.recipe(task), expected)
            self.assertEqual(cp.gpu_for(task), "a100-80gb")
            self.assertEqual(cp.protocol_for(task[1]), "cp_full_all_block_v1")
        with patch.object(cp, "baseline") as baseline, patch.object(cp, "validate_reference"):
            self.assertIn("READY: 48", cli("check", "--group", "all-block"))
            self.assertEqual(baseline.call_count, 48)

    def test_large_memory_workloads_enable_activation_checkpointing(self):
        for task in [("DINOv3-L", "eurosat", "LeJEPA-CP"),
                     ("CLIP", "food101", "SimCLR-CP"),
                     ("MAE", "pathmnist", "MAE-CP")]:
            args = cp.command_for(Path("/root"), task, 42, "/local/data", "/local/ref", 16)
            self.assertIn("--activation-checkpointing", args)

    def test_six_block_recipe_changes_only_depth(self):
        for encoder, dataset, method in product(ENCODERS, SIX_BLOCK, METHODS):
            task = encoder, dataset, method
            with self.subTest(task=task):
                expected = cp.recipe((encoder, "breastmnist", method))
                expected["num_trained_blocks"] = 6
                self.assertEqual(cp.recipe(task), expected)
                config = cp.run_config(task, 44, "sha")
                self.assertEqual(config["protocol"], "cp_full_six_block_v1")
                self.assertEqual(config["n_train"], SIX_BLOCK[dataset])
                self.assertEqual(config["gpu"], "a100-80gb")
                args = cp.command_for(Path("/root"), task, 44, "/local/data", "/local/ref", 16)
                self.assertEqual(args[args.index("--num-trained-blocks") + 1], "6")
                self.assertEqual(args[args.index("--num-workers") + 1], "16")
                self.assertEqual(args[args.index("--cache-dir") + 1], "/local/data")
                self.assertEqual(args[args.index("--post-geometry-reference-data") + 1], "/local/ref")
                self.assertNotIn("--post-cp-sft", args)
                self.assertEqual(cp.seed_dir(Path("/root"), task, 44),
                                 Path(f"/root/outputs/results/{dataset}/{encoder}/{method}/Full/44"))

    def test_six_block_dry_run_launches_only_assigned_seed(self):
        for task_id, seed in ((300, 42), (301, 43), (302, 44), (539, 44)):
            output = cli("run", "--task-id", str(task_id), "--num-workers", "16", "--dry-run")
            commands = [line for line in output.splitlines() if "continued_pretraining.py" in line]
            self.assertEqual(len(commands), 1)
            self.assertIn(f"--seed {seed} ", commands[0])
            self.assertIn("--num-trained-blocks 6", commands[0])
        self.assertEqual(cli("run", "--task-id", "299", "--dry-run").count("--seed "), 3)

    def test_six_block_check_and_report_do_not_duplicate_seeds(self):
        with patch.object(cp, "baseline") as baseline, patch.object(cp, "validate_reference"):
            self.assertIn("READY: 60", cli("check", "--group", "six-block"))
            self.assertEqual(baseline.call_count, 60)
            self.assertEqual({call.args[1][1] for call in baseline.call_args_list}, set(SIX_BLOCK))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(cp, "baseline", return_value=({}, "source", "sha")), \
                    patch.object(cp, "completed_result", return_value=None):
                output = cli("report", "--group", "six-block", "--root", str(root))
            self.assertIn("Completed: 0/240", output)
            destination = root / "outputs/results"
            with (destination / "cp_full_results.six-block.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 240)
            self.assertEqual(len({(r["encoder"], r["dataset"], r["method"], r["seed"]) for r in rows}), 240)
            with (destination / "cp_full_summary.six-block.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 80)
            self.assertFalse((destination / "cp_full_results.csv").exists())

    def test_single_seed_execution_uses_independent_locks_and_local_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            task = ("DINOv3-B", "resisc45", "LeJEPA-CP")
            directory = cp.seed_dir(root, task, 42).parent
            directory.mkdir(parents=True)
            data_cache = Mock()
            data_cache.staged_dataset.return_value = nullcontext(Path("/local/data"))
            data_cache.staged_directory.return_value = nullcontext(Path("/local/reference"))
            with FileLock(str(directory / ".seed42.run.lock"), timeout=0), \
                    patch.dict(sys.modules, {"torch": Mock(), "data_cache": data_cache}), \
                    patch.object(cp, "baseline", return_value=({}, "source", "sha")) as baseline, \
                    patch.object(cp, "validate_reference", return_value=root / "data/imagenet_reference_5000"), \
                    patch.object(cp, "completed_result", side_effect=[None, {"complete": True}]), \
                    patch.object(cp, "attach_baseline", return_value={"complete": True}), \
                    patch.object(cp.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                output = cli("run", "--task-id", "301", "--root", str(root), "--num-workers", "16")
                self.assertIn("DONE DINOv3-B/resisc45/LeJEPA-CP/43", output)
                baseline.assert_called_once_with(root, task, 43)
                self.assertEqual(run.call_count, 1)
                args = run.call_args.args[0]
                self.assertEqual(args[args.index("--seed") + 1], "43")
                self.assertEqual(args[args.index("--cache-dir") + 1], "/local/data")
                self.assertEqual(args[args.index("--post-geometry-reference-data") + 1], "/local/reference")
                data_cache.staged_dataset.assert_called_once_with(root / "data", "resisc45")
                self.assertFalse(cp.seed_dir(root, task, 42).exists())
                self.assertFalse(cp.seed_dir(root, task, 44).exists())
                with self.assertRaises(Timeout):
                    cli("run", "--task-id", "300", "--root", str(root))

    def test_completed_single_seed_skips_staging_and_training(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {"torch": Mock(), "data_cache": Mock()}) as modules, \
                patch.object(cp, "baseline", return_value=({}, "source", "sha")), \
                patch.object(cp, "completed_result", return_value={"complete": True}), \
                patch.object(cp, "attach_baseline", return_value={"complete": True}), \
                patch.object(cp.subprocess, "run") as run:
            output = cli("run", "--task-id", "302", "--root", folder)
            self.assertIn("SKIP DINOv3-B/resisc45/LeJEPA-CP/44", output)
            run.assert_not_called()
            modules["data_cache"].staged_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
