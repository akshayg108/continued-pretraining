#!/usr/bin/env python3
"""Build grouped, torch-free manifests for full-parameter evaluation."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_CHECKPOINT_ROOT = Path("/scratch/gs4133/zhd/CP/outputs/ckpts")
SEEDS = (42, 43, 44)
ENCODERS = {
    "DINOv3": ("vit_base_patch16_dinov3.lvd1689m", "cls"),
    "CLIP": ("vit_base_patch16_clip_224.openai", "cls"),
    "MAE": ("vit_base_patch16_224.mae", "mean"),
    "SigLIP": ("vit_base_patch16_siglip_224.v2_webli", "map"),
}
DATASETS = [
    ("breastmnist", "BreastMNIST", "med_mnist/breastmnist-size=224", 546),
    ("dermamnist", "DermaMNIST", "med_mnist/dermamnist-size=224", 7007),
    ("octmnist", "OctMNIST", "med_mnist/octmnist-size=224", 97477),
    ("organamnist", "OrganAMNIST", "med_mnist/organamnist-size=224", 34561),
    ("pathmnist", "PathMNIST", "med_mnist/pathmnist-size=224", 89996),
    ("galaxy10", "Galaxy10", "galaxy10", 14188),
    ("eurosat", "EuroSAT", "eurosat", 16200),
    ("plant_village", "PlantVillage", "plant_village", 43596),
    ("dtd", "DTD", "dtd", 1880),
    ("food101", "Food101", "food101", 75750),
    ("fgvc_aircraft", "FGVC_Aircraft", "fgvc_aircraft", 3334),
    ("cars196", "Cars196", "cars196", 8144),
    ("cub200", "CUB200", "cub200", 5994),
    ("flowers102", "Flowers102", "flowers102", 1020),
    ("oxford_pet", "OxfordPet", "oxford_pet", 3680),
]
DATASET_META = {key: (display, subpath, maximum)
                for key, display, subpath, maximum in DATASETS}
SOURCE_FILES = (
    "eval/outputs/cp_long_refreshed.csv",
    "eval/outputs/postcp_sweep_fixed.csv",
    "eval/outputs/nd12_operator.csv",
)


def _budget(dataset, size, is_max=False):
    if is_max or size == DATASET_META[dataset][2] or (dataset == "fgvc_aircraft" and size == 3400):
        return "MAX"
    if size in {100, 101, 102, 196, 200}:
        return "100"
    return str(size)


def _relocate(path, checkpoint_root):
    value = str(path)
    old = str(DEFAULT_CHECKPOINT_ROOT)
    if value.startswith(old + "/"):
        return str(checkpoint_root) + value[len(old):]
    return value


def _inventory(repo_root, checkpoint_root):
    found = {}
    for rel in SOURCE_FILES[1:]:
        path = repo_root / rel
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                ckpt = row.get("ckpt", "")
                if not ckpt or "sft" in ckpt.lower() or "teacher" in ckpt.lower():
                    continue
                try:
                    method = row["method"].replace("-CP", "")
                    encoder = row["encoder"]
                    dataset = row["dataset"]
                    size = int(float(row["size"]))
                    seed = int(row["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if dataset == "fgvc_aircraft" and size == 3400:
                    size = 3334
                key = (method, encoder, dataset, size, seed)
                candidate = _relocate(ckpt, checkpoint_root)
                expected = f"_n{size}_s{seed}.ckpt"
                if candidate.endswith(expected):
                    found.setdefault(key, []).append(candidate)
    return found


MULTI_BUDGET_DIR_DATASETS = {
    "eurosat", "food101", "galaxy10", "octmnist", "organamnist",
    "pathmnist", "plant_village",
}


def _fallback(checkpoint_root, scope, method, display, encoder, dataset, model_id,
              n_samples, seed, budget):
    if scope == "siglip":
        parent = checkpoint_root / "cp-siglip" / "cp" / method / display / encoder
    else:
        parent = checkpoint_root / "cp" / method / "pretrained" / display / encoder
    basename = f"{dataset}_{model_id}_n{n_samples}_s{seed}.ckpt"
    candidates = []
    if scope == "main" and dataset in MULTI_BUDGET_DIR_DATASETS:
        budget_dir = {"100": "small", "500": "small", "1000": "small",
                      "10000": "10k", "25000": "25k", "MAX": "all"}.get(budget)
        if budget_dir:
            candidates.append(str(parent / budget_dir / "cp" / basename))
    candidates.append(str(parent / "cp" / basename))
    return candidates


def _checkpoint_map(inventory, checkpoint_root, scope, method, display, encoder,
                    dataset, model_id, n_samples, budget):
    result = {}
    for seed in SEEDS:
        known = inventory.get((method, encoder, dataset, n_samples, seed), [])
        fallback = _fallback(checkpoint_root, scope, method, display, encoder,
                             dataset, model_id, n_samples, seed, budget)
        result[str(seed)] = list(dict.fromkeys(known + fallback))
    return result


def _main_configs(repo_root):
    path = repo_root / SOURCE_FILES[0]
    configs = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            encoder = row["Backbone"]
            if encoder not in {"DINOv3", "CLIP", "MAE"}:
                continue
            dataset = row["dataset_key"]
            size = int(float(row["size"]))
            is_max = row.get("is_max", "").lower() == "true"
            if dataset == "fgvc_aircraft" and is_max:
                size = 3334
            configs.add((row["Method"].removesuffix("-CP"), encoder, dataset,
                         _budget(dataset, size, is_max), size))
    return sorted(configs, key=lambda x: (x[1], x[0], x[2], x[4]))


def build_tasks(repo_root, *, methods=None, budgets=None, encoders=None,
                phases=("post",), checkpoint_root=None):
    repo_root = Path(repo_root)
    checkpoint_root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
    methods = set(methods) if methods is not None else None
    budgets = {str(value).upper() for value in budgets} if budgets is not None else None
    encoders = set(encoders) if encoders is not None else None
    phases = tuple(phases)
    invalid = set(phases) - {"pre", "post"}
    if invalid:
        raise ValueError(f"unsupported phases: {sorted(invalid)}")
    inventory = _inventory(repo_root, checkpoint_root)

    post_configs = list(_main_configs(repo_root))
    for dataset, _display, _subpath, maximum in DATASETS:
        for method in ("DIET", "LeJEPA", "SimCLR"):
            post_configs.append((method, "SigLIP", dataset, "MAX", maximum))
    post_configs = [c for c in post_configs
                    if (methods is None or c[0] in methods)
                    and (encoders is None or c[1] in encoders)
                    and (budgets is None or c[3].upper() in budgets)]

    tasks = []
    if "pre" in phases:
        pre_configs = sorted({(encoder, dataset, budget, size)
                              for _method, encoder, dataset, budget, size in post_configs})
        for encoder, dataset, budget, size in pre_configs:
            display, subpath, _maximum = DATASET_META[dataset]
            model_id, pool = ENCODERS[encoder]
            tasks.append(dict(phase="pre", scope="siglip" if encoder == "SigLIP" else "main",
                              encoder=encoder, method="PRE", dataset=dataset, budget=budget,
                              n_samples=size, model_id=model_id, pool=pool,
                              processed_subpath=subpath,
                              checkpoints={str(seed): [] for seed in SEEDS}))
    if "post" in phases:
        for method, encoder, dataset, budget, size in post_configs:
            display, subpath, _maximum = DATASET_META[dataset]
            model_id, pool = ENCODERS[encoder]
            scope = "siglip" if encoder == "SigLIP" else "main"
            tasks.append(dict(phase="post", scope=scope, encoder=encoder, method=method,
                              dataset=dataset, budget=budget, n_samples=size,
                              model_id=model_id, pool=pool, processed_subpath=subpath,
                              checkpoints=_checkpoint_map(
                                  inventory, checkpoint_root, scope, method, display,
                                  encoder, dataset, model_id, size, budget)))
    for task_id, task in enumerate(tasks):
        task["task_id"] = task_id
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--budgets", nargs="+")
    parser.add_argument("--encoders", nargs="+")
    parser.add_argument("--phases", nargs="+", default=["post"])
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks = build_tasks(args.repo_root, methods=args.methods, budgets=args.budgets,
                        encoders=args.encoders, phases=args.phases,
                        checkpoint_root=args.checkpoint_root)
    document = {
        "schema_version": 1,
        "source": {"repo_root": str(args.repo_root.resolve()), "files": list(SOURCE_FILES)},
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
