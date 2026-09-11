"""Build a dataset-selectable ViT-L MAX manifest without importing torch."""
import argparse
import json
from pathlib import Path

from eval.full_ft.manifest import DATASET_META, SEEDS
from eval.full_ft.run import atomic_json, digest_json

PROTOCOL = "vitl_completion_v1"
DEFAULT_OUTPUT_BASE = Path("/scratch/gs4133/zhd/CP/outputs")
MODEL = "vit_large_patch16_dinov3.lvd1689m"
DATASETS = ("breastmnist", "octmnist", "organamnist", "pathmnist",
            "plant_village", "food101", "flowers102", "oxford_pet")
METHODS = ("DIET", "LeJEPA", "SimCLR")


def trained_blocks(n):
    if type(n) is not int or n <= 0:
        raise ValueError("Sample count must be a positive integer")
    if n < 10000:
        return 2
    if n <= 25000:
        return 4
    if n <= 50000:
        return 6
    return -1


def cp_recipe(method, n):
    recipe = dict(epochs=150, freeze_epochs=15, warmup_epochs=15,
                  num_trained_blocks=trained_blocks(n), lr=1e-4,
                  weight_decay=.05, knn_k=20)
    if method in {"LeJEPA", "SimCLR"}:
        # Preserve the existing ViT-L microbatch, including SimCLR's negative pool.
        recipe.update(batch_size=128, accumulate_grad_batches=2,
                      proj_dim=128, hidden_dim=2048)
        if method == "LeJEPA":
            recipe.update(n_views=8, lamb=.02)
        else:
            recipe.update(temperature=.5)
    elif method == "DIET":
        recipe.update(batch_size=32, accumulate_grad_batches=1, label_smoothing=.3,
                      mixup_alpha=1.0, cutmix_alpha=1.0, mixup_cutmix_prob=0.0,
                      mixup_cutmix_switch_prob=.5)
    else:
        raise ValueError(f"Unsupported objective: {method}")
    return recipe


def selected_datasets(datasets):
    if datasets is None:
        datasets = DATASETS
    if (not isinstance(datasets, (list, tuple)) or not datasets
            or any(not isinstance(d, str) for d in datasets)):
        raise ValueError("Select a nonempty list of dataset keys")
    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate dataset keys are not allowed")
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Not in the eight-dataset completion panel: {', '.join(sorted(unknown))}. "
                         f"Choose from: {', '.join(DATASETS)}")
    return sorted(datasets)


def build_manifest(output_base=DEFAULT_OUTPUT_BASE, *, datasets=None):
    selection = selected_datasets(datasets)
    root = Path(output_base).expanduser().resolve() / PROTOCOL
    configs = [(method, dataset) for dataset in selection for method in METHODS]
    configs.sort(key=lambda c: (-DATASET_META[c[1]][2], c[1], c[0]))
    tasks = []
    for task_id, (method, dataset) in enumerate(configs):
        _display, subpath, n = DATASET_META[dataset]
        checkpoints = {str(seed): str(root / "checkpoints" / method / dataset / "cp" /
                                     f"{dataset}_{MODEL}_n{n}_s{seed}.ckpt") for seed in SEEDS}
        tasks.append(dict(task_id=task_id, encoder="DINOv3L", method=method,
                          dataset=dataset, size="MAX", n_samples=n, model_id=MODEL,
                          pool="cls", processed_subpath=subpath, checkpoints=checkpoints,
                          cp_recipe=cp_recipe(method, n)))
    return dict(schema_version=1, protocol=PROTOCOL, output_root=str(root),
                datasets=selection, tasks=tasks)


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    try:
        if not isinstance(document, dict) or not isinstance(document["output_root"], str):
            raise ValueError("Malformed ViT-L completion manifest")
        root = Path(document["output_root"])
        if not root.is_absolute() or root.name != PROTOCOL:
            raise ValueError("Output root must use the isolated completion namespace")
        if not isinstance(document["datasets"], list):
            raise ValueError("Manifest must contain its explicit dataset selection")
        expected = build_manifest(root.parent, datasets=document["datasets"])
        if digest_json(document) != digest_json(expected):
            raise ValueError("Manifest does not match the selected ViT-L grid, recipes, or paths")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed ViT-L completion manifest") from exc
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="Dataset keys (default: all eight missing datasets)")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite a manifest; choose a new manifest filename")
    try:
        document = build_manifest(args.output_base, datasets=args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    atomic_json(args.output, document)
    count = len(document["tasks"])
    print(f"MANIFEST {args.output}: {count} tasks, {count * len(SEEDS)} CP fits, 0 FT fits")
    for task in document["tasks"]:
        print(f"TASK {task['task_id']:02d} {task['method']} {task['dataset']} "
              f"n={task['n_samples']} blocks={task['cp_recipe']['num_trained_blocks']} "
              "seeds=42,43,44")


if __name__ == "__main__":
    main()
