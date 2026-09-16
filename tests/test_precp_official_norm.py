"""CPU-only protocol and launcher checks for the four-encoder baseline audit."""

import importlib
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "run/slurm/pre-cp-official"


def audit():
    assert (ROOT / "eval/precp_official_norm.py").is_file(), "Audit runner is missing"
    return importlib.import_module("eval.precp_official_norm")


def test_plan_has_exactly_four_encoder_jobs_and_max_seeds():
    module = audit()
    assert module.ENCODER_ORDER == ("SigLIP", "CLIP", "DINOv3", "MAE")
    plans = {encoder: module.build_plan(encoder) for encoder in module.ENCODER_ORDER}
    assert [len(p) for p in plans.values()] == [43, 45, 45, 45]
    for encoder, plan in plans.items():
        assert len({(t["dataset"], t["seed"]) for t in plan}) == len(plan)
        assert len({t["dataset"] for t in plan}) == 15
        for task in plan:
            assert task["budget"] == "MAX" and task["encoder"] == encoder
            assert task["n_samples"] == module.DATASET_META[task["dataset"]][2]
            assert task["seed"] in (42, 43, 44)
    all_tasks = [task for plan in plans.values() for task in plan]
    geometry = [t for t in all_tasks if t["geometry_only"]]
    assert len(geometry) == 1
    assert (geometry[0]["encoder"], geometry[0]["dataset"], geometry[0]["seed"]) == (
        "SigLIP", "food101", 42)
    assert sum(not t["geometry_only"] for t in all_tasks) == 177
    with pytest.raises(ValueError):
        module.build_plan("DINOv3L")


@pytest.mark.parametrize("encoder,mean,std", [
    ("SigLIP", [.5] * 3, [.5] * 3),
    ("CLIP", [.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]),
    ("DINOv3", [.485, .456, .406], [.229, .224, .225]),
    ("MAE", [.485, .456, .406], [.229, .224, .225]),
])
def test_normalization_must_match_checkpoint_configuration(encoder, mean, std):
    module = audit()
    assert module.official_normalization(encoder, {"mean": tuple(mean), "std": tuple(std)}) == {
        "mean": mean, "std": std}
    for bad in ({}, {"mean": mean, "std": [1., 1., 1.]},
                {"mean": [float("nan")] * 3, "std": std}):
        with pytest.raises(ValueError, match="normalization"):
            module.official_normalization(encoder, bad)


@pytest.mark.parametrize("block_size", [1, 2, 7, 128])
def test_full_uniformity_matches_dense_distinct_pairs(block_size):
    module = audit()
    pytest.importorskip("torch")
    features = np.random.RandomState(12).normal(size=(19, 9)).astype(np.float32)
    normalized = features.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    distances = ((normalized[:, None] - normalized[None, :]) ** 2).sum(axis=-1)
    expected = math.log(np.exp(-2 * distances[np.triu_indices(19, 1)]).mean())
    actual = module.full_uniformity(features, device="cpu", block_size=block_size)
    assert actual == pytest.approx(expected, abs=2e-6)


def test_uniformity_rejects_invalid_features_and_excludes_self_pairs():
    module = audit()
    pytest.importorskip("torch")
    assert module.full_uniformity(np.array([[1., 0.], [-1., 0.]]), device="cpu") == pytest.approx(-8)
    assert module.full_uniformity(np.ones((3, 5)), device="cpu") == pytest.approx(0, abs=1e-6)
    for features in (np.ones((1, 3)), np.zeros((3, 3)),
                     np.array([[1., 2.], [float("nan"), 0.]])):
        with pytest.raises(ValueError):
            module.full_uniformity(features, device="cpu")
    with pytest.raises(ValueError):
        module.full_uniformity(np.ones((3, 5)), device="cpu", block_size=0)


def test_subset_selection_is_deterministic_and_matches_historical_estimator():
    module = audit()
    from sklearn.model_selection import train_test_split
    labels = np.repeat(np.arange(10), 601)
    selected, _ = train_test_split(np.arange(len(labels)), train_size=5000,
                                   stratify=labels, random_state=42)
    expected = np.sort(selected)[np.random.RandomState(42).choice(5000, 3000, replace=False)]
    assert np.array_equal(module.subset_indices(labels), expected)
    assert np.array_equal(module.subset_indices(np.arange(8)), np.arange(8))


def test_json_publication_is_no_clobber_and_rejects_nan(tmp_path):
    module = audit()
    path = tmp_path / "nested/seed42.json"
    module.write_new_json(path, {"status": "complete", "value": 1.})
    with pytest.raises(FileExistsError):
        module.write_new_json(path, {"value": 2.})
    assert json.loads(path.read_text()) == {"status": "complete", "value": 1.}
    bad = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        module.write_new_json(bad, {"value": float("nan")})
    assert not bad.exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("task_id,encoder", enumerate(("SigLIP", "CLIP", "DINOv3", "MAE")))
def test_array_dry_run_is_pure_and_does_not_load_cluster_environment(tmp_path, task_id, encoder):
    script = DIRECTORY / "array.sh"
    assert script.is_file(), "Array launcher is missing"
    out = tmp_path / "outputs"
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task_id), PRECP_NORM_REPO_ROOT=str(ROOT),
               PRECP_NORM_OUTPUT_BASE=str(out), PYTHON=sys.executable)
    result = subprocess.run(["bash", str(script), "--dry-run"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"encoder={encoder}" in result.stdout
    assert "normalization=official" in result.stdout and "budget=MAX" in result.stdout
    assert "seed=42" in result.stdout and "seed=43" in result.stdout and "seed=44" in result.stdout
    assert not out.exists()


def test_submit_dry_run_requests_four_v100_jobs_without_submitting(tmp_path):
    script = DIRECTORY / "submit.sh"
    assert script.is_file(), "Submission launcher is missing"
    out = tmp_path / "outputs"
    result = subprocess.run(["bash", str(script), "--dry-run"],
                            env=dict(os.environ, PRECP_NORM_OUTPUT_BASE=str(out), PYTHON=sys.executable),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--array=0-3%4" in result.stdout
    assert "--gres=gpu:v100:1" in result.stdout
    assert "--time=96:00:00" in result.stdout
    assert result.stdout.count("JOB encoder=") == 4
    assert not out.exists()


@pytest.mark.parametrize("name", ["submit.sh", "array.sh"])
def test_shell_syntax_and_unknown_options(name):
    script = DIRECTORY / name
    assert script.is_file()
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    result = subprocess.run(["bash", str(script), "--invalid", "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 2


@pytest.mark.parametrize("encoder,geometry_only", [
    ("SigLIP", False), ("SigLIP", True), ("CLIP", False),
    ("DINOv3", False), ("MAE", False),
])
def test_runner_preserves_readout_and_evaluators_and_writes_provenance(
        tmp_path, monkeypatch, encoder, geometry_only):
    module = audit()
    torch = pytest.importorskip("torch")
    task = dict(module.build_plan(encoder)[0], n_samples=8, geometry_only=geometry_only)
    events = []
    native = module.EXPECTED_NORMALIZATIONS[encoder]
    cfg = dict(input_size=224, num_classes=2, splits=["train", "validation", "test"],
               normalization={"mean": [.1] * 3, "std": [.2] * 3})
    model = SimpleNamespace(pretrained_cfg=native, state_dict=lambda: {"weight": torch.ones(2)},
                            requires_grad_=lambda value: events.append(("freeze", value)),
                            eval=lambda: events.append("eval"), to=lambda device: None,
                            modules=lambda: [])
    labels = np.arange(8) % 2
    features = np.random.RandomState(10).normal(size=(8, 768)).astype(np.float32)
    loaders = {
        "lp": SimpleNamespace(dataset=SimpleNamespace(
            dataset=SimpleNamespace(transform="augmented", hf_dataset=SimpleNamespace(_fingerprint="train"),
                                    __len__=lambda: 8))),
        "knn": SimpleNamespace(), "test": SimpleNamespace(dataset=range(4)),
    }
    class BaseDataset:
        transform = "augmented"
        hf_dataset = SimpleNamespace(_fingerprint="train")

        def __len__(self):
            return 8

    loaders["lp"].dataset.dataset = BaseDataset()

    def load(args, *, img_size, pretrained):
        assert args.backbone == task["model_id"] and args.pool_strategy == task["pool"]
        assert args.n_samples == 8 and args.seed == 42 and img_size == 224 and pretrained
        events.append("public_model")
        return model, SimpleNamespace(type="cuda")

    def create(args, config, cache):
        assert config["normalization"] == native
        assert config["splits"] == cfg["splits"]
        return "eval_tf", loaders["test"], loaders["lp"], loaders["knn"], list(range(8))

    def extract(backbone, loader, device, *, pool_strategy, verbose):
        assert backbone is model and pool_strategy == task["pool"]
        key = next(k for k, value in loaders.items() if loader is value)
        events.append("extract_" + key)
        return (features[:4], labels[:4]) if key == "test" else (features, labels)

    def knn(train, train_y, test, test_y, *, k):
        assert k == 20 and train is features
        events.append("knn")
        return dict(knn_f1=.6, knn_acc=.7)

    def lp(train, train_y, test, test_y, **kwargs):
        assert kwargs["min_epochs"] == 150 and kwargs["min_steps"] == 10000
        assert kwargs["batch_size"] == 512 and kwargs["lr"] == 1e-3
        events.append("lp")
        return dict(linear_pytorch_f1=.8, linear_pytorch_acc=.9)

    monkeypatch.setitem(sys.modules, "lightning", SimpleNamespace(
        seed_everything=lambda seed, workers: events.append(("seed", seed))))
    monkeypatch.setitem(sys.modules, "continued_pretraining", SimpleNamespace(
        _create_shared_eval_data=create, get_dataset_config=lambda name: cfg, load_backbone=load))
    monkeypatch.setitem(sys.modules, "stable_cp.evaluation.zero_shot_eval", SimpleNamespace(
        extract_features=extract, knn_evaluate=knn, linear_probe_pytorch_evaluate=lp))
    monkeypatch.setattr(module, "check_gpu", lambda: "Tesla V100")
    monkeypatch.setattr(module, "full_uniformity", lambda *args, **kwargs: -.75)
    path = module.run_task(task, cache_dir=tmp_path / "cache", outdir=tmp_path / "output", num_workers=0)
    record = json.loads(path.read_text())
    module.validate_result(record, task)
    assert record["no_cp"] and record["no_ft"]
    assert record["normalization"] == native
    assert record["n_samples"] == 8 and record["seed"] == 42
    assert record["code_sha256"] and record["weights_sha256"] and record["train_indices_sha256"]
    assert record["uniformity_t2"] == -.75 and record["uniformity_t2_subset"] == -.75
    assert cfg["normalization"] != native
    if geometry_only:
        assert "pre_knn_f1" not in record and "pre_linear_f1" not in record
        assert "knn" not in events and "lp" not in events
        assert "extract_test" not in events and "extract_lp" not in events
    else:
        assert record["pre_knn_f1"] == .6 and record["pre_linear_f1"] == .8
        assert events.index("extract_lp") < events.index("extract_test") < events.index("extract_knn")
        assert events.count("knn") == events.count("lp") == 1
    with pytest.raises(FileExistsError):
        module.run_task(task, cache_dir=tmp_path / "cache", outdir=tmp_path / "output", num_workers=0)


def test_result_validation_cannot_accept_a_wrong_model_or_missing_metric():
    module = audit()
    task = module.build_plan("DINOv3")[0]
    record = dict(protocol=module.PROTOCOL, status="complete", encoder=task["encoder"],
                  dataset=task["dataset"], seed=task["seed"], n_samples=task["n_samples"],
                  backbone=task["model_id"], pool_strategy=task["pool"], budget="MAX",
                  geometry_only=False, no_cp=True, no_ft=True, initialization="public_pretrained",
                  normalization=module.EXPECTED_NORMALIZATIONS["DINOv3"],
                  uniformity_t2=-1., uniformity_t2_subset=-1.,
                  pre_knn_f1=.5, pre_knn_acc=.5, pre_linear_f1=.5, pre_linear_acc=.5)
    module.validate_result(record, task)
    for key, value in (("backbone", "wrong"), ("pre_knn_f1", float("nan")),
                       ("pre_linear_f1", None), ("uniformity_t2", 3.)):
        with pytest.raises(ValueError):
            module.validate_result(dict(record, **{key: value}), task)


def test_planning_does_not_import_training_dependencies():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; from eval.precp_official_norm import build_plan; "
                             "assert len(build_plan('SigLIP')) == 43; "
                             "assert not {'torch', 'timm', 'lightning', 'stable_pretraining'} & sys.modules.keys()"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("task_id", ["-1", "4", "00", "not-an-index"])
def test_array_rejects_invalid_identity(task_id):
    result = subprocess.run(["bash", str(DIRECTORY / "array.sh"), "--dry-run"],
                            env=dict(os.environ, SLURM_ARRAY_TASK_ID=task_id),
                            capture_output=True, text=True)
    assert result.returncode == 2


@pytest.mark.parametrize("fail_first", [False, True])
def test_staging_is_private_cleans_up_and_continues_after_a_failed_seed(tmp_path, fail_first):
    for name in ("one", "two"):
        source = tmp_path / "shared/stable_datasets/processed" / name
        source.mkdir(parents=True)
        (source / "data.bin").write_bytes(name.encode())
    stage = tmp_path / "stage"
    stage.mkdir()
    marker = tmp_path / "calls.jsonl"
    fake = tmp_path / "fake-python"
    fake.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
from pathlib import Path
import sys

if sys.argv[1] == "-c":
    raise SystemExit(0)
if sys.argv[3] == "plan":
    print("one\tone\t42\t8\tFalse")
    print("one\tone\t43\t8\tFalse")
    print("two\ttwo\t42\t8\tFalse")
    raise SystemExit(0)
args = dict(zip(sys.argv[4::2], sys.argv[5::2]))
cache = Path(args["--cache-dir"])
dataset = args["--dataset"]
assert (cache / "stable_datasets/processed" / dataset / "data.bin").read_bytes() == dataset.encode()
with Path(os.environ["CALL_MARKER"]).open("a") as handle:
    handle.write(json.dumps(dict(args, cache=str(cache))) + "\n")
if os.environ["FAIL_FIRST"] == "1" and dataset == "one" and args["--seed"] == "42":
    raise SystemExit(9)
''')
    fake.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    df = bin_dir / "df"
    df.write_text("#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\\nmock 2000000000 1 1000000000 1%% /tmp\\n'\n")
    df.chmod(0o755)
    env = dict(os.environ, PRECP_NORM_REPO_ROOT=str(ROOT), PRECP_NORM_SKIP_ENV_SETUP="1",
               PRECP_NORM_CACHE_DIR=str(tmp_path / "shared"),
               PRECP_NORM_OUTPUT_BASE=str(tmp_path / "outputs"), SLURM_JOB_ID="123",
               SLURM_ARRAY_JOB_ID="122", SLURM_ARRAY_TASK_ID="0", PYTHON=str(fake),
               TMPDIR=str(stage), CALL_MARKER=str(marker), FAIL_FIRST="1" if fail_first else "0",
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(DIRECTORY / "array.sh")], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == (1 if fail_first else 0), result.stdout + result.stderr
    calls = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(calls) == 3
    assert calls[0]["cache"] == calls[1]["cache"] != calls[2]["cache"]
    assert all(Path(call["cache"]).parent == stage and not Path(call["cache"]).exists() for call in calls)
    assert not list(stage.iterdir())
    for name in ("one", "two"):
        assert (tmp_path / "shared/stable_datasets/processed" / name / "data.bin").read_bytes() == name.encode()
    if fail_first:
        assert "FAIL encoder=SigLIP dataset=one seed=42 exit=9" in result.stderr
        assert "SUCCESS encoder=SigLIP dataset=two seed=42" in result.stdout


def test_summary_keeps_missing_and_malformed_records_visible(tmp_path, capsys):
    module = audit()
    path = module.result_path(tmp_path, module.build_plan("SigLIP")[0])
    path.parent.mkdir(parents=True)
    path.write_text("invalid json")
    assert module.summarize(tmp_path) == 1
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert len(rows) == 178 and all(row["status"] == "CHECK" for row in rows)
    assert all(row["pre_knn_f1"] == row["uniformity_t2"] == "" for row in rows)
    assert "Validated records: 0/178" in captured.err
