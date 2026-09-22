"""Pin extension recipes and reuse the original held-out preparation samples."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from eval.heldout_cp import protocol as base

ROOT = base.ROOT
PROTOCOL = "heldout_extensions_1000_v1"
DATASETS, SEEDS, METHODS, METRICS = base.DATASETS, base.SEEDS, base.METHODS, base.METRICS
DEFAULT_OUTPUT_BASE = base.DEFAULT_OUTPUT_BASE
ENCODER_ORDER = ("SigLIP", "DINOv3L")
ENCODERS = {
    "SigLIP": dict(model_id=base.ENCODERS["SigLIP"][0], pool="map", embed_dim=768,
                   normalization=base.EXPECTED_NORMALIZATIONS["SigLIP"]),
    "DINOv3L": dict(model_id="vit_large_patch16_dinov3.lvd1689m", pool="cls", embed_dim=1024,
                    normalization=base.EXPECTED_NORMALIZATIONS["DINOv3"]),
}
atomic_json, digest_json, file_sha256 = base.atomic_json, base.digest_json, base.file_sha256
root_path, pre_path, geometry_path, result_path = (
    base.root_path, base.pre_path, base.geometry_path, base.result_path,
)
check_identity, check_metrics = base.check_identity, base.check_metrics
seed_lock, shared_lock, summarize = base.seed_lock, base.shared_lock, base.summarize


def implementation_sha256():
    files = [*sorted(Path(__file__).parent.glob("*.py")),
             ROOT / "eval/heldout_metric_roundoff.py"]
    return digest_json(dict(base=base.implementation_sha256(), files={
        str(path.relative_to(ROOT)): file_sha256(path) for path in files
    }))


def source_snapshot(manifest):
    path = Path(manifest).expanduser().resolve()
    doc = base.load_manifest(path)
    inputs = {}
    for dataset in DATASETS:
        prediction = base.predictions_path(doc, dataset)
        if not prediction.is_file():
            raise ValueError(f"Original preparation is not complete: {dataset}")
        base.freeze_predictions(doc, dataset)
        paths = [prediction]
        for seed in SEEDS:
            rows = [base.validate_pre(doc, e, dataset, seed) for e in base.ENCODER_ORDER]
            if any(rows[0][key] != rows[1][key] for key in ("train_indices", "data")):
                raise ValueError(f"Original encoders use different data: {dataset} seed={seed}")
            paths.extend(base.pre_path(doc, e, dataset, seed) for e in base.ENCODER_ORDER)
        geometries = [json.loads(base.geometry_path(doc, e, dataset).read_text())
                      for e in base.ENCODER_ORDER]
        if geometries[0].get("geometry_indices") != geometries[1].get("geometry_indices"):
            raise ValueError(f"Original encoders use different geometry samples: {dataset}")
        indices = geometries[0].get("geometry_indices", [])
        if (not 2 <= len(indices) <= 3000 or len(set(indices)) != len(indices)
                or any(type(i) is not int or i < 0 for i in indices)):
            raise ValueError(f"Missing or invalid original geometry indices: {dataset}")
        paths.extend(base.geometry_path(doc, e, dataset) for e in base.ENCODER_ORDER)
        inputs.update({str(p.relative_to(base.root_path(doc))): file_sha256(p) for p in paths})
    return dict(manifest=str(path), manifest_sha256=file_sha256(path),
                output_root=doc["output_root"], inputs=inputs)


def build_manifest(output_base, source_manifest):
    preparations, tasks = [], []
    for encoder in ENCODER_ORDER:
        for dataset in DATASETS:
            prep_id = len(preparations)
            preparations.append(dict(preparation_id=prep_id, encoder=encoder, dataset=dataset,
                                     gpu="a100" if encoder == "DINOv3L" else "v100"))
            for method in METHODS:
                tasks.append(dict(task_id=len(tasks), preparation_id=prep_id,
                                  encoder=encoder, dataset=dataset, method=method,
                                  **ENCODERS[encoder], seeds=list(SEEDS), n_samples=1000,
                                  gpu="a100" if encoder == "DINOv3L" or method == "LeJEPA" else "v100",
                                  recipe=base.cp_recipe(method, 1000)))
    return dict(protocol=PROTOCOL, schema_version=1, datasets=list(DATASETS),
                preparations=preparations, tasks=tasks,
                output_root=str(Path(output_base).expanduser().resolve() / PROTOCOL),
                source=source_snapshot(source_manifest), implementation_sha256=implementation_sha256(),
                no_ft=True, split_seed=42, geometry_max_samples=3000, evaluation_batch_size=64,
                geometry_sampling="reuse_original_5000_stratified_then_3000_uniform_seed42",
                transform_contract="official_mean_std_with_existing_224_transforms",
                preparation_dependency="same_encoder_and_dataset_only",
                hypothesis="Higher initial uniformity ranks higher CP macro-F1 gains within each encoder.")


def load_manifest(path):
    doc = json.loads(Path(path).read_text())
    root = Path(doc["output_root"])
    if (not root.is_absolute() or root.name != PROTOCOL
            or doc != build_manifest(root.parent, doc["source"]["manifest"])):
        raise ValueError("Extension manifest differs from the fixed protocol, source preparations, or code")
    return doc


def source_artifact(doc, relative):
    source = doc["source"]
    if file_sha256(source["manifest"]) != source["manifest_sha256"]:
        raise ValueError("Original manifest changed")
    path = Path(source["output_root"]) / relative
    if file_sha256(path) != source["inputs"][str(relative)]:
        raise ValueError(f"Original preparation changed: {path}")
    return json.loads(path.read_text())


def source_baseline(doc, dataset, seed):
    rows = [source_artifact(doc, Path("pre") / e / dataset / f"seed{seed}.json")
            for e in base.ENCODER_ORDER]
    if any(rows[0][key] != rows[1][key] for key in ("train_indices", "data")):
        raise ValueError("Original baseline subsets differ across encoders")
    return rows[0]


def source_geometry(doc, dataset):
    return source_artifact(doc, Path("geometry/DINOv3") / f"{dataset}.json")


def identity(doc, encoder, dataset, seed=None):
    return dict(protocol=PROTOCOL, encoder=encoder, dataset=dataset, seed=seed,
                **ENCODERS[encoder], implementation_sha256=doc["implementation_sha256"],
                source_preparation_sha256=digest_json(doc["source"]))


def predictions_path(doc, encoder, dataset):
    if encoder not in ENCODERS or dataset not in DATASETS:
        raise ValueError("Unknown encoder/target for frozen predictions")
    return root_path(doc) / "predictions" / encoder / f"{dataset}.json"


def validate_pre(doc, encoder, dataset, seed):
    row = json.loads(pre_path(doc, encoder, dataset, seed).read_text())
    source = source_baseline(doc, dataset, seed)
    check_identity(row, dict(identity(doc, encoder, dataset, seed), status="complete",
                             train_indices=source["train_indices"], data=source["data"]))
    check_metrics(row, "pre")
    return row


def freeze_predictions(doc, encoder, dataset):
    target = predictions_path(doc, encoder, dataset)
    path = geometry_path(doc, encoder, dataset)
    geometry = json.loads(path.read_text())
    check_identity(geometry, dict(identity(doc, encoder, dataset), status="complete"))
    value = geometry.get("uniformity_t2")
    indices = source_geometry(doc, dataset)["geometry_indices"]
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not -8.000001 <= value <= .000001
            or geometry.get("geometry_indices") != indices
            or geometry.get("n_geometry") != len(indices)):
        raise ValueError("Invalid or mismatched initial geometry")
    inputs = {str(path.relative_to(root_path(doc))): file_sha256(path)}
    for seed in SEEDS:
        validate_pre(doc, encoder, dataset, seed)
        path = pre_path(doc, encoder, dataset, seed)
        inputs[str(path.relative_to(root_path(doc)))] = file_sha256(path)
    expected = dict(identity(doc, encoder, dataset), inputs=inputs,
                    hypothesis=doc["hypothesis"], initial_uniformity=value)
    with shared_lock(target):
        if target.exists():
            check_identity(json.loads(target.read_text()), expected)
        else:
            if any((root_path(doc) / "attempts" / encoder).glob(f"*/{dataset}/seed*/*")):
                raise ValueError("Cannot freeze initial predictions after CP attempts exist")
            atomic_json(target, dict(expected, frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    return target


def validate_result(doc, task, seed, row):
    baseline = validate_pre(doc, task["encoder"], task["dataset"], seed)
    check_identity(row, dict(identity(doc, task["encoder"], task["dataset"], seed),
                             status="complete", method=task["method"], recipe=task["recipe"],
                             no_ft=True, initialization="public_pretrained", data=baseline["data"],
                             software=baseline["software"],
                             pretrained_weights_sha256=baseline["pretrained_weights_sha256"],
                             pre_sha256=file_sha256(pre_path(doc, task["encoder"], task["dataset"], seed))))
    for metric in METRICS:
        if row.get(f"pre_{metric}") != baseline[f"pre_{metric}"]:
            raise ValueError("Baseline scores changed in CP result")
    check_metrics(row, "pre")
    check_metrics(row, "post")
    prediction = predictions_path(doc, task["encoder"], task["dataset"])
    if not prediction.is_file() or row.get("predictions_sha256") != file_sha256(prediction):
        raise ValueError("Missing or changed pre-training predictions")
    if task["gpu"].upper() not in row.get("gpu", "").upper():
        raise ValueError("Result used the wrong GPU profile")
    return row
