"""Build the isolated, fixed 29-task SigLIP MAX rerun manifest (no torch)."""
import argparse
import json
from pathlib import Path

from eval.full_ft.manifest import DATASET_META, ENCODERS, SEEDS
from eval.full_ft.run import atomic_json, digest_json

PROTOCOL = "siglip_mainrule_v1"
DEFAULT_OUTPUT_BASE = Path("/scratch/gs4133/zhd/CP/outputs")
AFFECTED = frozenset({"galaxy10", "eurosat", "organamnist", "plant_village",
                      "food101", "pathmnist", "octmnist"})
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
    depth = trained_blocks(n)
    recipe = dict(epochs=150, freeze_epochs=15, warmup_epochs=15,
                  num_trained_blocks=depth, lr=1e-4, weight_decay=.05, knn_k=20)
    if method == "LeJEPA":
        accumulation = {2: 1, 4: 2, 6: 2, -1: 4}[depth]
        recipe.update(batch_size=256 // accumulation, accumulate_grad_batches=accumulation,
                      n_views=8, proj_dim=128, hidden_dim=2048, lamb=.02)
    elif method == "SimCLR":
        recipe.update(batch_size=256, accumulate_grad_batches=1,
                      proj_dim=128, hidden_dim=2048, temperature=.5)
    elif method == "DIET":
        recipe.update(batch_size=32, accumulate_grad_batches=1, label_smoothing=.3,
                      mixup_alpha=1.0, cutmix_alpha=1.0, mixup_cutmix_prob=0.0,
                      mixup_cutmix_switch_prob=.5)
    else:
        raise ValueError(f"Unsupported objective: {method}")
    return recipe


def build_manifest(output_base=DEFAULT_OUTPUT_BASE):
    root = Path(output_base).expanduser().resolve() / PROTOCOL
    model, pool = ENCODERS["SigLIP"]
    configs = [(method, dataset) for method in METHODS for dataset in DATASET_META
               if method == "DIET" or dataset in AFFECTED]
    configs.sort(key=lambda c: (-DATASET_META[c[1]][2], c[1], c[0]))
    tasks = []
    for task_id, (method, dataset) in enumerate(configs):
        _display, subpath, n = DATASET_META[dataset]
        checkpoints = {str(seed): [str(root / "checkpoints" / method / dataset / "cp" /
                                      f"{dataset}_{model}_n{n}_s{seed}.ckpt")]
                       for seed in SEEDS}
        tasks.append(dict(task_id=task_id, phase="post", scope="siglip", encoder="SigLIP",
                          method=method, dataset=dataset, budget="MAX", n_samples=n,
                          model_id=model, pool=pool, processed_subpath=subpath,
                          checkpoints=checkpoints, cp_recipe=cp_recipe(method, n)))
    return dict(schema_version=1, protocol=PROTOCOL, output_root=str(root), tasks=tasks)


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    try:
        root = Path(document["output_root"])
        if not root.is_absolute() or root.name != PROTOCOL:
            raise ValueError("Output root must use the isolated protocol namespace")
        expected = build_manifest(root.parent)
        if digest_json(document) != digest_json(expected):
            raise ValueError("Manifest does not match the frozen 29-task protocol/recipes/paths")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed SigLIP main-rule manifest") from exc
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite a manifest; choose a new manifest filename")
    atomic_json(args.output, build_manifest(args.output_base))
    print(f"MANIFEST {args.output}: 29 tasks, 87 CP fits, 87 full post-CP FT fits")


if __name__ == "__main__":
    main()
