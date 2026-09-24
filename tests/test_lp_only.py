"""LP reruns must neither train CP nor evaluate unrelated metrics."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(REPO))

import cp_full
import lp_only


class LPOnlyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = cp_full.TASKS[1]

    def args(self, phase="post", task_id=1, **kwargs):
        return argparse.Namespace(
            phase=phase,
            task_id=task_id,
            root=self.root,
            epochs=150,
            batch_size=512,
            lr=0.001,
            forward_batch_size=None,
            num_workers=2,
            dry_run=False,
            **kwargs,
        )

    def checkpoint(self, complete=True, task=None, seed=42):
        import torch

        task = task or self.task
        recipe = cp_full.recipe(task)
        saved = {
            "epoch": recipe["epochs"] - 1 if complete else 12,
            "global_step": lp_only.expected_steps(task) if complete else 39,
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "current": {
                            "processed": recipe["epochs"] if complete else 12,
                        }
                    }
                }
            },
            "state_dict": {"backbone.weight": torch.ones(2, 2)},
        }
        path = cp_full.seed_dir(self.root, task, seed) / "cp.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, path)
        return path, saved

    def test_pre_has_twenty_three_individual_datasets(self):
        self.assertEqual(len(lp_only.PRE_TASKS), 23)
        self.assertEqual(len(set(lp_only.PRE_TASKS)), 23)
        self.assertEqual(lp_only.PRE_TASKS[:7], lp_only.precp.GROUPS[0])
        runs = lp_only.evaluations("pre", 0)
        self.assertEqual(len(runs), 15)
        self.assertEqual({run[0] for run in runs}, set(lp_only.PRE_ENCODERS))
        self.assertNotIn("MAE-CLS", lp_only.PRE_ENCODERS)

    def test_post_uses_original_task_ids_and_three_seeds(self):
        self.assertEqual(len(cp_full.TASKS), 220)
        self.assertEqual(
            lp_only.evaluations("post", 69),
            [("CLIP", "flowers102", "SimCLR-CP", seed) for seed in (42, 43, 44)],
        )

    def test_checkpoint_status_requires_completed_epochs_and_steps(self):
        import torch

        path, saved = self.checkpoint()
        self.assertEqual(lp_only.checkpoint_status(path, self.task), "complete")
        saved["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"] = 149
        torch.save(saved, path)
        self.assertEqual(lp_only.checkpoint_status(path, self.task), "incomplete")
        path.unlink()
        self.assertEqual(lp_only.checkpoint_status(path, self.task), "missing")

    def test_mismatched_completed_step_count_fails(self):
        import torch

        path, saved = self.checkpoint()
        saved["global_step"] -= 1
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, "updates"):
            lp_only.checkpoint_status(path, self.task)

    def test_only_available_completed_post_tasks_enter_array(self):
        self.checkpoint()
        self.checkpoint(complete=False, task=cp_full.TASKS[3])
        self.assertEqual(lp_only.available_tasks(self.root, "post"), [1])
        self.assertEqual(lp_only.available_tasks(self.root, "post", "a100"), [])
        self.assertEqual(lp_only.available_tasks(self.root, "pre"), list(range(23)))

    def test_restore_regular_encoder_ignores_projector(self):
        import torch

        model = torch.nn.Linear(2, 2)
        saved = {
            "state_dict": {
                "backbone.weight": torch.full((2, 2), 3.0),
                "backbone.bias": torch.ones(2),
                "projector.weight": torch.zeros(8, 2),
            }
        }
        lp_only.restore_encoder(model, saved, "SimCLR-CP")
        self.assertTrue(torch.equal(model.weight, torch.full((2, 2), 3.0)))

    def test_restore_mae_cp_uses_vit_prefix_and_identity_head(self):
        import torch

        model = torch.nn.Module()
        model.linear = torch.nn.Linear(2, 2)
        model.head = torch.nn.Linear(2, 8)
        saved = {
            "state_dict": {
                "backbone.vit.linear.weight": torch.ones(2, 2),
                "backbone.vit.linear.bias": torch.zeros(2),
                "backbone.patch_embed.unused": torch.ones(1),
                "decoder.unused": torch.ones(1),
            }
        }
        lp_only.restore_encoder(model, saved, "MAE-CP")
        self.assertIsInstance(model.head, torch.nn.Identity)
        self.assertTrue(torch.equal(model.linear.weight, torch.ones(2, 2)))

    def test_timm_mae_patch_mean_matches_checkpoint_encoder(self):
        import torch
        from timm.models.vision_transformer import VisionTransformer
        from stable_cp.utils.backbone import forward_embedding

        def backbone():
            model = VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=16,
                depth=1,
                num_heads=2,
                num_classes=8,
            )
            model.pretrained_cfg = {"tag": "mae"}
            return model

        original = backbone().eval()
        original.head = torch.nn.Identity()
        saved = {
            "state_dict": {
                "backbone.vit." + key: value for key, value in original.state_dict().items()
            }
        }
        restored = backbone().eval()
        lp_only.restore_encoder(restored, saved, "MAE-CP")
        images = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = forward_embedding(original, images, "mean")
            actual = forward_embedding(restored, images, "mean")
        torch.testing.assert_close(actual, expected)

    def test_restore_rejects_partial_encoder_weights(self):
        import torch

        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            lp_only.restore_encoder(
                torch.nn.Linear(2, 2),
                {"state_dict": {"backbone.weight": torch.ones(2, 2)}},
                "SimCLR-CP",
            )

    def test_missing_checkpoint_does_not_stage_or_start_process(self):
        with (
            patch.object(lp_only, "stage_dataset") as stage,
            patch.object(lp_only.subprocess, "run") as run,
        ):
            lp_only.run_task(self.args())
        stage.assert_not_called()
        run.assert_not_called()

    def test_only_completed_seed_is_launched_and_staging_happens_once(self):
        self.checkpoint()
        self.checkpoint(complete=False, seed=43)
        args = self.args()

        @contextmanager
        def staged(*_):
            yield self.root / "local"

        with (
            patch.object(lp_only, "stage_dataset", side_effect=staged) as stage,
            patch.object(lp_only.subprocess, "run", return_value=Mock(returncode=0)) as run,
            patch.object(lp_only, "completed_result", side_effect=[False, True]),
        ):
            lp_only.run_task(args)
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("evaluate", command)
        self.assertNotIn("continued_pretraining.py", " ".join(command))
        self.assertNotIn("--resume", command)
        self.assertNotIn("--forward-batch-size", command)

    def test_outputs_are_lp_only_sidecars(self):
        pre = lp_only.output_path(self.root, "pre", "MAE-Mean", "dtd", None, 42)
        post = lp_only.output_path(self.root, "post", *self.task, 42)
        self.assertEqual(
            pre.relative_to(self.root).as_posix(),
            "outputs/precp_full/lp_online_v1/lp_results/MAE-Mean/dtd/seed42.json",
        )
        self.assertEqual(post.name, "lp.json")
        self.assertEqual(post.parent.name, "lp_online_v1")

    def test_settings_change_does_not_reuse_previous_lp(self):
        args = self.args("pre", 0)
        run = lp_only.evaluations("pre", 0)[0]
        path = lp_only.output_path(self.root, "pre", *run)
        path.parent.mkdir(parents=True)
        content = lp_only.identity(args, *run)
        content.update(pre_linear_f1=0.5, pre_linear_acc=0.5, pre_linear_auroc=0.5)
        path.write_text(json.dumps(content))
        self.assertTrue(lp_only.completed_result(path, content))
        args.epochs = 2
        self.assertFalse(lp_only.completed_result(path, lp_only.identity(args, *run)))

    def test_evaluate_writes_only_lp_metrics_without_touching_previous_results(self):
        import torch
        import timm

        datasets = ModuleType("stable_cp.data.datasets")
        loaders = ModuleType("stable_cp.data.loaders")
        evaluation = ModuleType("stable_cp.evaluation.linear_probe")
        config = dict(input_size=32, num_classes=2, splits=("train", "validation", "test"))
        datasets.get_dataset_config = Mock(return_value=config)
        samples = torch.utils.data.TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 0]))
        datasets.get_dataset = Mock(return_value=samples)
        loaders.create_lp_transforms = Mock(return_value=("random_train", "fixed_test"))
        evaluation.linear_probe_online_evaluate = Mock(
            return_value={
                "linear_pytorch_acc": 0.8,
                "linear_pytorch_f1": 0.7,
                "linear_pytorch_auroc": 0.9,
            }
        )
        args = self.args("pre", 0, encoder="DINOv3", seed=42, cache_dir=self.root / "data")
        args.num_workers = 0
        model = Mock(pretrained_cfg={"mean": [0.5] * 3, "std": [0.5] * 3})
        original = self.root / "outputs/precp_full/results/DINOv3/breastmnist/seed42.json"
        original.parent.mkdir(parents=True)
        original.write_text('{"pre_knn_f1": 0.6, "geometry": {}}')
        before = original.read_bytes()
        with (
            patch.dict(
                sys.modules,
                {
                    "stable_cp.data.datasets": datasets,
                    "stable_cp.data.loaders": loaders,
                    "stable_cp.evaluation.linear_probe": evaluation,
                },
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("timm.create_model", return_value=model) as create,
        ):
            lp_only.evaluate(args)
        create.assert_called_once_with(
            lp_only.precp.ENCODERS["DINOv3"], pretrained=True, img_size=32
        )
        evaluation.linear_probe_online_evaluate.assert_called_once()
        call = evaluation.linear_probe_online_evaluate.call_args
        self.assertEqual(call.kwargs["epochs"], 150)
        self.assertIsNone(call.kwargs["forward_batch_size"])
        self.assertEqual(call.args[1].batch_size, 512)
        self.assertFalse(call.args[1].drop_last)
        output = lp_only.output_path(self.root, "pre", "DINOv3", "breastmnist", None, 42)
        row = json.loads(output.read_text())
        self.assertEqual(row["pre_linear_f1"], 0.7)
        self.assertNotIn("geometry", row)
        self.assertNotIn("pre_knn_f1", row)
        self.assertEqual(original.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
