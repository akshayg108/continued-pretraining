"""Check opt-in native normalization without the optional cluster stack."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "continued_pretraining.py"


def function(name):
    tree = ast.parse(SOURCE.read_text())
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert node is not None, f"Missing {name}"
    namespace = dict(argparse=argparse, DATASETS={"food101": {}}, json=__import__("json"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_dataset_default_is_unchanged_and_pretrained_mode_is_explicit():
    parser = function("create_base_parser")()
    args = parser.parse_args(["--dataset", "food101", "--backbone", "siglip"])
    assert getattr(args, "normalization_mode", None) == "dataset"
    args = parser.parse_args(["--dataset", "food101", "--backbone", "siglip",
                              "--normalization-mode", "pretrained"])
    assert args.normalization_mode == "pretrained"


def test_native_override_uses_model_configuration_without_mutating_dataset():
    apply = function("configure_normalization")
    cfg = dict(input_size=224, normalization={"mean": [.1] * 3, "std": [.2] * 3})
    model = SimpleNamespace(pretrained_cfg={"mean": (.5,) * 3, "std": (.5,) * 3})
    assert apply(cfg, model, "dataset") is cfg
    result = apply(cfg, model, "pretrained")
    assert result == dict(input_size=224, normalization={"mean": [.5] * 3, "std": [.5] * 3})
    assert cfg["normalization"]["mean"] == [.1] * 3


@pytest.mark.parametrize("native", [{}, {"mean": [.5] * 3, "std": [0.] * 3},
    {"mean": [float("nan")] * 3, "std": [.5] * 3},
    {"mean": [.5], "std": [.5] * 3}, {"mean": [.5] * 3, "std": [-.5] * 3}])
def test_bad_checkpoint_normalization_fails_before_data_creation(native):
    apply = function("configure_normalization")
    with pytest.raises(ValueError, match="normalization"):
        apply({"normalization": {}}, SimpleNamespace(pretrained_cfg=native), "pretrained")


def test_override_precedes_all_loaders_and_is_recorded_in_json():
    text = SOURCE.read_text()
    main = text[text.index("def main("):]
    assert "ds_cfg = configure_normalization(" in main
    assert main.index("ds_cfg = configure_normalization(") < main.index("_create_shared_eval_data(")
    assert main.index("ds_cfg = configure_normalization(") < main.index("_create_cp_data(")
    tree = ast.parse(text)
    keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict)
            for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert {"normalization", "normalization_mode", "cp_config"} <= keys
