"""Frozen configuration, artifacts, and summaries for held-out target CP."""

from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import statistics

from eval.full_ft.run import (
    atomic_json,
    digest_json,
    file_sha256,
    seed_lock as seed_lock,
)
from eval.full_ft.manifest import ENCODERS, SEEDS
from eval.precp_official_norm import EXPECTED_NORMALIZATIONS
from eval.siglip_mainrule.protocol import cp_recipe


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "heldout_cp_1000_v1"
DATASETS = (
    "bloodmnist",
    "tissuemnist",
    "aid",
    "resisc45",
    "stanford_dogs",
    "jena_flowers30",
    "flavia",
    "ip102",
)
ENCODER_ORDER = ("DINOv3", "CLIP")
METHODS = ("LeJEPA", "DIET", "SimCLR")
METRICS = ("knn_f1", "linear_f1", "knn_acc", "linear_acc")
DEFAULT_OUTPUT_BASE = Path("/scratch/gs4133/zhd/CP/outputs")


def implementation_sha256():
    files = [
        ROOT / "continued_pretraining.py",
        ROOT / "eval/precp_official_norm.py",
        ROOT / "eval/siglip_mainrule/protocol.py",
        ROOT / "eval/full_ft/run.py",
        ROOT / "eval/full_ft/manifest.py",
        *sorted((ROOT / "stable_cp").rglob("*.py")),
        *sorted(Path(__file__).parent.glob("*.py")),
    ]
    return digest_json({str(p.relative_to(ROOT)): file_sha256(p) for p in files})


def build_manifest(output_base=DEFAULT_OUTPUT_BASE):
    tasks = []
    for encoder in ENCODER_ORDER:
        model, pool = ENCODERS[encoder]
        for dataset in DATASETS:
            for method in METHODS:
                tasks.append(
                    dict(
                        task_id=len(tasks),
                        encoder=encoder,
                        dataset=dataset,
                        method=method,
                        model_id=model,
                        pool=pool,
                        n_samples=1000,
                        seeds=list(SEEDS),
                        gpu="v100",
                        recipe=cp_recipe(method, 1000),
                        normalization=EXPECTED_NORMALIZATIONS[encoder],
                    )
                )
    return dict(
        protocol=PROTOCOL,
        schema_version=2,
        datasets=list(DATASETS),
        tasks=tasks,
        output_root=str(Path(output_base).expanduser().resolve() / PROTOCOL),
        implementation_sha256=implementation_sha256(),
        no_ft=True,
        split_seed=42,
        geometry_max_samples=3000,
        evaluation_batch_size=64,
        geometry_sampling="5000_stratified_then_3000_uniform_seed42",
        transform_contract="official_mean_std_with_existing_224_transforms",
        preparation_dependency="same_dataset_only",
        prediction_contract="fixed_rule_and_dataset_local_pre_cp_geometry",
        hypothesis="Higher initial uniformity ranks higher CP macro-F1 gains within each encoder.",
    )


def load_manifest(path):
    doc = json.loads(Path(path).read_text())
    root = Path(doc["output_root"])
    if (
        not root.is_absolute()
        or root.name != PROTOCOL
        or doc != build_manifest(root.parent)
    ):
        raise ValueError(
            "Held-out manifest differs from the fixed protocol or current code"
        )
    return doc


def root_path(doc):
    return Path(doc["output_root"])


def pre_path(doc, encoder, dataset, seed):
    return root_path(doc) / "pre" / encoder / dataset / f"seed{seed}.json"


def geometry_path(doc, encoder, dataset):
    return root_path(doc) / "geometry" / encoder / f"{dataset}.json"


def predictions_path(doc, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown held-out dataset: {dataset}")
    return root_path(doc) / "predictions" / f"{dataset}.json"


def result_path(doc, task, seed):
    return (
        root_path(doc)
        / "cp_results"
        / task["encoder"]
        / task["method"]
        / task["dataset"]
        / f"seed{seed}.json"
    )


def identity(doc, encoder, dataset, seed=None):
    return dict(
        protocol=PROTOCOL,
        encoder=encoder,
        dataset=dataset,
        seed=seed,
        model_id=ENCODERS[encoder][0],
        pool="cls",
        normalization=EXPECTED_NORMALIZATIONS[encoder],
        implementation_sha256=doc["implementation_sha256"],
    )


def check_identity(row, expected):
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Held-out artifact identity/configuration mismatch")


def check_metrics(row, prefix):
    for metric in METRICS:
        key = f"{prefix}_{metric}"
        value = row.get(key)
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid or missing metric: {key}")


def validate_pre(doc, encoder, dataset, seed):
    path = pre_path(doc, encoder, dataset, seed)
    row = json.loads(path.read_text())
    check_identity(row, identity(doc, encoder, dataset, seed))
    check_metrics(row, "pre")
    indices = row.get("train_indices", [])
    if (
        row.get("status") != "complete"
        or len(indices) != 1000
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != 1000
        or row.get("data", {}).get("n_train_actual") != 1000
    ):
        raise ValueError(f"Incomplete 1000-image baseline: {path}")
    return row


@contextmanager
def shared_lock(path):
    """Serialize shared publication; concurrent array tasks wait rather than fail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def freeze_predictions(doc, dataset):
    """Freeze this target's initial scores before its CP, without a global barrier."""
    target = predictions_path(doc, dataset)
    inputs, initial_uniformity = {}, {}
    for encoder in ENCODER_ORDER:
        path = geometry_path(doc, encoder, dataset)
        row = json.loads(path.read_text())
        check_identity(row, identity(doc, encoder, dataset))
        value = row.get("uniformity_t2")
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not -8.000001 <= value <= 0.000001
            or not 2 <= row.get("n_geometry", 0) <= 3000
            or row.get("status") != "complete"
        ):
            raise ValueError(f"Invalid initial geometry: {path}")
        inputs[str(path.relative_to(root_path(doc)))] = file_sha256(path)
        for seed in SEEDS:
            validate_pre(doc, encoder, dataset, seed)
            path = pre_path(doc, encoder, dataset, seed)
            inputs[str(path.relative_to(root_path(doc)))] = file_sha256(path)
        initial_uniformity[encoder] = value
    expected = dict(
        protocol=PROTOCOL,
        dataset=dataset,
        implementation_sha256=doc["implementation_sha256"],
        hypothesis=doc["hypothesis"],
        inputs=inputs,
        initial_uniformity=initial_uniformity,
    )
    with shared_lock(target):
        if target.exists():
            record = json.loads(target.read_text())
            check_identity(record, expected)
        else:
            if any((root_path(doc) / "attempts").glob(f"*/*/{dataset}/seed*/*")):
                raise ValueError(
                    f"Cannot create prospective predictions after CP attempts exist for {dataset}"
                )
            atomic_json(
                target,
                dict(expected, frozen_at_utc=datetime.now(timezone.utc).isoformat()),
            )
    return target


def validate_result(doc, task, seed, row):
    check_identity(
        row,
        dict(
            identity(doc, task["encoder"], task["dataset"], seed),
            status="complete",
            method=task["method"],
            recipe=task["recipe"],
            no_ft=True,
            initialization="public_pretrained",
        ),
    )
    baseline = validate_pre(doc, task["encoder"], task["dataset"], seed)
    if row.get("data") != baseline["data"] or row.get("pre_sha256") != file_sha256(
        pre_path(doc, task["encoder"], task["dataset"], seed)
    ):
        raise ValueError("CP and baseline use different data or baseline records")
    for metric in METRICS:
        if row.get(f"pre_{metric}") != baseline[f"pre_{metric}"]:
            raise ValueError("Baseline score changed in CP result")
    check_metrics(row, "pre")
    check_metrics(row, "post")
    prediction = predictions_path(doc, task["dataset"])
    if not prediction.is_file() or row.get("predictions_sha256") != file_sha256(
        prediction
    ):
        raise ValueError("Missing or changed pre-training predictions")
    return row


def summarize(rows):
    summary = dict(n_seeds=len(rows))
    for metric in ("knn_f1", "linear_f1"):
        for phase in ("pre", "post", "delta"):
            values = [
                (r[f"post_{metric}"] - r[f"pre_{metric}"])
                if phase == "delta"
                else r[f"{phase}_{metric}"]
                for r in rows
            ]
            summary[f"{phase}_{metric}_mean"] = (
                statistics.mean(values) if values else None
            )
            summary[f"{phase}_{metric}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
    return summary
