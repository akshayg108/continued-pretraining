"""Check evaluation-only control flow without importing the training stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock


class EvaluationOnlyTests(unittest.TestCase):
    def test_missing_or_incomplete_checkpoint_cannot_start_training(self):
        source = Path(__file__).resolve().parents[1] / "continued_pretraining.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_training")
        namespace = {"Path": Path, "_load_completed_checkpoint": Mock(return_value=False)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "cp.ckpt"
            args = SimpleNamespace(num_trained_blocks=2, resume=True,
                                   checkpoint_path=str(checkpoint), require_completed_checkpoint=True)
            for exists in (False, True):
                if exists:
                    checkpoint.touch()
                with self.assertRaisesRegex(ValueError, "completed CP checkpoint"):
                    namespace["run_training"](None, None, args, {}, 32, 15, None, checkpoint)
            namespace["_load_completed_checkpoint"].return_value = True
            namespace["run_training"](None, None, args, {}, 32, 15, None, checkpoint)


if __name__ == "__main__":
    unittest.main()
