"""Test CP save/resume separation without importing the optional cluster stack."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest


def runner():
    source = Path(__file__).resolve().parents[1] / "continued_pretraining.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
    calls = []

    class ModelCheckpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Trainer:
        def __init__(self, **kwargs):
            calls.append(("trainer", kwargs))

        def save_checkpoint(self, path):
            calls.append(("save", str(path)))
            Path(path).write_bytes(b"completed CP weights")

    class Manager:
        def __init__(self, *, trainer, module, data, ckpt_path, seed, weights_only=True):
            if ckpt_path is not None and not Path(ckpt_path).is_file():
                raise FileNotFoundError(ckpt_path)
            calls.append(("manager", dict(path=ckpt_path, weights_only=weights_only)))

        def __call__(self):
            calls.append(("fit", None))

    namespace = dict(Path=Path, inspect=inspect, tempfile=tempfile, ModelCheckpoint=ModelCheckpoint,
                     SLURMEnvironment=SimpleNamespace(detect=lambda: False),
                     pl=SimpleNamespace(Trainer=Trainer), spt=SimpleNamespace(Manager=Manager),
                     FreezeBackboneCallback=lambda **kw: object(),
                     create_cp_evaluation_callbacks=lambda *a, **kw: [],
                     LearningRateMonitor=lambda **kw: object())
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(num_trained_blocks=2, n_samples=500, knn_k=20, cp_method="diet",
                           epochs=150, seed=42, resume=True)
    return namespace["run_training"], calls, args, ModelCheckpoint


def test_fresh_cp_has_a_save_destination_but_no_restore_path(tmp_path):
    run, calls, args, saver_type = runner()
    path = tmp_path / "cp" / "fresh.ckpt"
    run(object(), object(), args, {"num_classes": 2}, 768, 15, None, str(path))
    assert next(value for kind, value in calls if kind == "manager")["path"] is None
    callbacks = next(value for kind, value in calls if kind == "trainer")["callbacks"]
    saver = next(cb for cb in callbacks if isinstance(cb, saver_type))
    assert saver.kwargs["filename"] == "fresh"
    assert saver.kwargs["save_last"] is False
    assert saver.kwargs["enable_version_counter"] is False
    assert path.read_bytes() == b"completed CP weights"


def test_existing_cp_is_resumed_with_optimizer_state(tmp_path):
    run, calls, args, _ = runner()
    path = tmp_path / "existing.ckpt"
    path.write_bytes(b"old progress")
    run(object(), object(), args, {"num_classes": 2}, 768, 15, None, str(path))
    manager = next(value for kind, value in calls if kind == "manager")
    assert manager["path"] == str(path)
    assert manager["weights_only"] is False


def test_existing_cp_is_never_deleted_for_a_non_resume_launch(tmp_path):
    run, calls, args, _ = runner()
    args.resume = False
    path = tmp_path / "existing.ckpt"
    path.write_bytes(b"keep CP weights")
    with pytest.raises(FileExistsError):
        run(object(), object(), args, {"num_classes": 2}, 768, 15, None, str(path))
    assert path.read_bytes() == b"keep CP weights"
    assert not any(kind == "fit" for kind, value in calls)


def test_resume_rejects_a_directory_before_training(tmp_path):
    run, calls, args, _ = runner()
    path = tmp_path / "directory.ckpt"
    path.mkdir()
    with pytest.raises(ValueError, match="file"):
        run(object(), object(), args, {"num_classes": 2}, 768, 15, None, str(path))
    assert not any(kind == "fit" for kind, value in calls)
