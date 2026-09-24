"""Post-CP geometry with the fixed ImageNet images re-encoded by the CP model."""

import json
from pathlib import Path
import tempfile

import numpy as np

from .geometry import GEOMETRY_PROTOCOL


class ReferenceImages:
    """Read prepared reference images with the same clean transform as the target."""

    def __init__(self, source, transform):
        self.source = source
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.transform({"image": self.source[index]["image"], "label": 0})


def create_post_geometry(backbone, device, args, ds_cfg):
    """Extract a fresh CP reference bank and return arguments for evaluate_geometry."""
    from datasets import Image, load_from_disk
    from torch.utils.data import DataLoader

    from stable_cp.data import create_transforms
    from stable_cp.utils.backbone import feature_readout
    from .zero_shot_eval import extract_features

    prepared = Path(args.post_geometry_reference_data).expanduser().resolve()
    manifest = json.loads((prepared / "metadata.json").read_text())
    expected = dict(protocol=GEOMETRY_PROTOCOL, n_reference=5000, sampling_seed=42)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible prepared ImageNet reference metadata: {prepared}")
    source_metadata = manifest.get("source_metadata", {})
    n_source = source_metadata.get("n_source_images")
    if type(n_source) is not int or n_source < 5000:
        raise ValueError("Prepared reference must record at least 5000 source images")
    indices = np.asarray(manifest.get("indices", []))
    expected_indices = np.sort(np.random.RandomState(42).choice(n_source, 5000, replace=False))
    if (
        indices.shape != (5000,)
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.array_equal(indices, expected_indices)
    ):
        raise ValueError(
            "Prepared reference source indices do not match the fixed 5000-image sample"
        )
    source = load_from_disk(str(prepared / "dataset"))
    if len(source) != 5000 or source.column_names != ["image"]:
        raise ValueError("Prepared ImageNet reference must contain exactly 5000 image rows")
    source = source.cast_column("image", Image())
    _, transform = create_transforms(ds_cfg, n_views=1, strong_aug=False)
    loader = DataLoader(
        ReferenceImages(source, transform),
        batch_size=32,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
    )
    backbone.eval().to(device)
    features, _ = extract_features(backbone, loader, device, pool_strategy=args.pool_strategy)
    if (
        features.shape != (5000, backbone.num_features)
        or not np.isfinite(features).all()
        or np.any(np.linalg.norm(features, axis=1) <= 0)
    ):
        raise ValueError("Invalid post-CP ImageNet reference features")
    metadata = dict(
        **source_metadata,
        **expected,
        backbone=args.backbone,
        pool_strategy=args.pool_strategy,
        normalization=ds_cfg["normalization"],
        input_size=ds_cfg["input_size"],
        dataset=args.dataset,
        seed=args.seed,
        phase="post",
        reference_encoder="post_cp",
    )
    if args.backbone.endswith(".mae"):
        metadata["feature_readout"] = feature_readout(args.backbone, args.pool_strategy)
    directory = Path(args.post_geometry_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference_path = directory / "post_reference.npz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                features=np.asarray(features, dtype=np.float32),
                indices=indices.astype(np.int64),
                metadata=json.dumps(metadata, sort_keys=True, allow_nan=False),
            )
        temporary.replace(reference_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(
        reference_path=str(reference_path),
        output_path=str(directory / "post_features.npz"),
        metadata=metadata,
    )
