#!/usr/bin/env python3
"""Extract fixed ImageNet reference features from a local validation cache."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from precp import ENCODERS, REPO

PROTOCOL = "precp_geometry_5000_v1"
N_REFERENCE = 5000
SAMPLING_SEED = 42


class ReferenceImages:
    """Apply the clean evaluation transform without requiring source labels."""

    def __init__(self, source, transform):
        self.source = source
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.transform({"image": self.source[index]["image"], "label": 0})


def existing_reference(path, metadata, indices, feature_dim):
    import numpy as np

    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as saved:
        recorded = json.loads(saved["metadata"].item())
        features = saved["features"]
        valid = all(recorded.get(key) == value for key, value in metadata.items())
        valid = valid and np.array_equal(saved["indices"], indices)
        valid = valid and features.shape == (N_REFERENCE, feature_dim)
        valid = valid and features.dtype == np.float32 and np.isfinite(features).all()
        valid = valid and np.all(np.linalg.norm(features, axis=1) > 0)
    if not valid:
        raise ValueError(f"Incompatible reference cache; move it aside before regenerating: {path}")
    return True


def save_reference(path, features, indices, metadata):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                features=np.asarray(features, dtype=np.float32),
                indices=np.asarray(indices, dtype=np.int64),
                metadata=json.dumps(metadata, sort_keys=True, allow_nan=False),
            )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_references(args, encoders):
    import numpy as np
    import torch
    from datasets import DatasetDict, load_from_disk
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(REPO))
    from continued_pretraining import configure_normalization, load_backbone
    from stable_cp.data import create_transforms
    from stable_cp.evaluation.zero_shot_eval import extract_features
    from stable_cp.utils.backbone import default_pool_strategy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit reference extraction through Slurm.")
    if not args.imagenet_dir.is_dir():
        raise FileNotFoundError(f"Missing local ImageNet validation cache: {args.imagenet_dir}")
    source = load_from_disk(str(args.imagenet_dir))
    if isinstance(source, DatasetDict):
        if "validation" not in source:
            raise ValueError("ImageNet DatasetDict must contain the validation split")
        source = source["validation"]
    split = getattr(source, "split", None)
    if split is not None and str(split) not in {"validation", "val"}:
        raise ValueError(f"Expected ImageNet validation data, received split={split}")
    if len(source) < N_REFERENCE or "image" not in source.column_names:
        raise ValueError("ImageNet validation cache must contain at least 5000 images")
    indices = np.sort(
        np.random.RandomState(SAMPLING_SEED).choice(len(source), N_REFERENCE, replace=False)
    ).astype(np.int64)
    selected = source.select(indices.tolist())
    source_metadata = {
        "source": str(args.imagenet_dir),
        "n_source_images": len(source),
    }
    fingerprint = getattr(source, "_fingerprint", None)
    if fingerprint is not None:
        source_metadata["source_fingerprint"] = fingerprint

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for encoder in encoders:
        backbone_name = ENCODERS[encoder]
        pool = default_pool_strategy(backbone_name)
        backbone, device = load_backbone(SimpleNamespace(backbone=backbone_name), img_size=224)
        config = configure_normalization({"input_size": 224}, backbone)
        metadata = {
            "protocol": PROTOCOL,
            "encoder": encoder,
            "backbone": backbone_name,
            "pool_strategy": pool,
            "normalization": config["normalization"],
            "n_reference": N_REFERENCE,
            "sampling_seed": SAMPLING_SEED,
            "input_size": 224,
            **source_metadata,
        }
        path = args.root / "outputs/precp_full/reference" / f"{encoder}.npz"
        if existing_reference(path, metadata, indices, backbone.num_features):
            print(f"SKIP {encoder}: {path}", flush=True)
            del backbone
            continue
        _, transform = create_transforms(config)
        loader = DataLoader(
            ReferenceImages(selected, transform),
            batch_size=32,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        backbone.requires_grad_(False)
        backbone.eval().to(device)
        features, _ = extract_features(backbone, loader, device, pool_strategy=pool)
        if (
            features.shape != (N_REFERENCE, backbone.num_features)
            or not np.isfinite(features).all()
            or np.any(np.linalg.norm(features, axis=1) <= 0)
        ):
            raise ValueError(f"Invalid extracted reference features for {encoder}")
        save_reference(path, features, indices, metadata)
        print(f"DONE {encoder}: {path}", flush=True)
        del backbone, loader, features
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("CP_ROOT", REPO.parent)))
    parser.add_argument("--imagenet-dir", type=Path)
    parser.add_argument("--encoder", choices=tuple(ENCODERS))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    args.root = args.root.expanduser().resolve()
    args.imagenet_dir = (
        (args.imagenet_dir or args.root / "data/imagenet_val").expanduser().resolve()
    )
    encoders = [args.encoder] if args.encoder else list(ENCODERS)
    if args.dry_run:
        print(f"ImageNet validation cache: {args.imagenet_dir}")
        for encoder in encoders:
            path = args.root / "outputs/precp_full/reference" / f"{encoder}.npz"
            print(f"{encoder}: {ENCODERS[encoder]} -> {path}")
        return
    extract_references(args, encoders)


if __name__ == "__main__":
    main()
