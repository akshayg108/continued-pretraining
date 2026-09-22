"""GPU primitives shared by held-out baseline and CP jobs."""

import json
from pathlib import Path
from types import SimpleNamespace

from eval.full_ft.run import digest_json, file_sha256, software_versions
from eval.heldout_cp import protocol as p


def check_environment():
    import torch
    from stable_datasets import images

    for name in (
        "MedMNIST",
        "AID",
        "RESISC45",
        "StanfordDogs",
        "JenaFlowers30",
        "Flavia",
        "IP102",
    ):
        if not hasattr(images, name):
            raise RuntimeError(
                f"Missing stable-datasets reader {name}; install cc01e36 or newer"
            )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This experiment requires exactly one allocated CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if "V100" not in gpu.upper():
        raise RuntimeError(f"Expected a V100 allocation, received {gpu}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return gpu


def software():
    import stable_datasets

    root = Path(stable_datasets.__file__).parent
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*.py"))
    }
    return dict(versions=software_versions(), stable_datasets_sha256=digest_json(files))


def evaluation_args(encoder, dataset, seed, cache_dir, num_workers):
    return SimpleNamespace(
        dataset=dataset,
        seed=seed,
        n_samples=1000,
        batch_size=64,
        num_workers=num_workers,
        cache_dir=str(cache_dir),
        backbone=p.ENCODERS[encoder][0],
        pool_strategy="cls",
    )


def load_model(encoder, args):
    from continued_pretraining import (
        load_backbone,
        get_dataset_config,
        configure_normalization,
    )
    from eval.precp_official_norm import official_normalization, _weights_hash

    model, device = load_backbone(args, img_size=224, pretrained=True)
    expected = official_normalization(encoder, model.pretrained_cfg)
    config = configure_normalization(
        get_dataset_config(args.dataset), model, "pretrained"
    )
    if config["normalization"] != expected:
        raise ValueError("Native normalization was not applied")
    if getattr(model, "num_features", None) != 768:
        raise ValueError("Expected a 768-dimensional ViT-B encoder")
    return model.to(device), device, config, _weights_hash(model)


def source_fingerprint(source, cache_dir):
    """Hash Arrow contents once; reuse hashes only while file identities match."""
    shard_dir = getattr(source, "_shard_dir", None)
    if shard_dir is None:
        raise RuntimeError(
            "Expected the current shard-backed stable-datasets implementation"
        )
    root = Path(shard_dir).resolve()
    paths = [
        root / "_metadata.json",
        *[Path(path).resolve() for path in source._shard_paths],
    ]

    def identities():
        result = {}
        for path in paths:
            stat = path.stat()
            result[str(path)] = [
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ]
        return result

    cached_path = (
        Path(cache_dir) / "heldout_cp_fingerprints" / f"{digest_json(str(root))}.json"
    )
    with p.shared_lock(cached_path):
        before = identities()
        if cached_path.is_file():
            cached = json.loads(cached_path.read_text())
            if cached.get("file_identities") == before:
                return digest_json(cached["content_sha256"])
        print(f"Fingerprinting dataset shards: {root}", flush=True)
        hashes = {path.name: file_sha256(path) for path in paths}
        if before != identities():
            raise RuntimeError(
                "Dataset files changed while computing their fingerprints"
            )
        p.atomic_json(cached_path, dict(file_identities=before, content_sha256=hashes))
        return digest_json(hashes)


def dataset_record(dataset, cache_dir, indices):
    import numpy as np
    from stable_cp.data.heldout import (
        HELDOUT_DATASETS,
        load_heldout_split,
        data_signature,
        array_hash,
    )

    cfg = HELDOUT_DATASETS[dataset]
    train, val, test = [
        load_heldout_split(dataset, split, str(Path(cache_dir).expanduser().resolve()))
        for split in ("train", "validation", "test")
    ]
    idx = np.asarray(indices)
    if (
        idx.shape != (1000,)
        or not np.issubdtype(idx.dtype, np.integer)
        or len(np.unique(idx)) != 1000
        or (idx < 0).any()
        or (idx >= len(train)).any()
    ):
        raise ValueError("Training requires exactly 1000 distinct valid indices")
    labels = train["label"][idx]
    if not np.array_equal(np.unique(labels), np.arange(cfg["num_classes"])):
        raise ValueError("The 1000-image training subset is missing classes")
    for left, right in ((train, val), (train, test), (val, test)):
        if (
            left.source is right.source
            and np.intersect1d(left.source_indices, right.source_indices).size
        ):
            raise ValueError("Held-out partition leakage detected")
    result = data_signature(
        idx, train.source_indices[idx], labels, test.source_indices, test["label"]
    )
    fingerprints = {
        name: source_fingerprint(split.source, cache_dir)
        for name, split in (("train", train), ("validation", val), ("test", test))
    }
    return dict(
        result,
        partition=cfg["partition"],
        split_seed=42,
        n_train_pool=len(train),
        n_validation=len(val),
        num_classes=cfg["num_classes"],
        train_class_counts=np.bincount(
            labels.astype(int), minlength=cfg["num_classes"]
        ).tolist(),
        train_pool_indices_sha256=array_hash(train.source_indices),
        source_content_sha256=fingerprints,
    )


def evaluate(model, device, config, args, indices):
    """The existing clean kNN and augmented PyTorch-LP evaluation, and nothing else."""
    import lightning as pl
    import numpy as np
    from stable_cp.data import create_transforms, create_eval_loaders
    from stable_cp.evaluation.zero_shot_eval import (
        extract_features,
        knn_evaluate,
        linear_probe_pytorch_evaluate,
    )

    model.to(device)
    pl.seed_everything(args.seed, workers=True)
    augmented, clean = create_transforms(config, n_views=1, strong_aug=False)
    test, lp_train, _ = create_eval_loaders(
        args, config, augmented, clean, args.cache_dir, indices=indices
    )
    _, knn_train, _ = create_eval_loaders(
        args, config, clean, clean, args.cache_dir, indices=indices
    )
    lp_x, lp_y = extract_features(model, lp_train, device, pool_strategy="cls")
    test_x, test_y = extract_features(model, test, device, pool_strategy="cls")
    knn_x, knn_y = extract_features(model, knn_train, device, pool_strategy="cls")
    for name, features, labels in (
        ("LP", lp_x, lp_y),
        ("test", test_x, test_y),
        ("kNN", knn_x, knn_y),
    ):
        if features.shape != (len(labels), 768) or not np.isfinite(features).all():
            raise ValueError(f"Invalid or non-finite {name} features")
    if len(lp_x) != 1000 or len(knn_x) != 1000 or not np.array_equal(lp_y, knn_y):
        raise ValueError("CP, kNN and LP must share the same 1000 training images")
    knn = knn_evaluate(knn_x, knn_y, test_x, test_y, k=20)
    # Reinitialize the linear head identically before and after CP.
    pl.seed_everything(args.seed, workers=True)
    lp = linear_probe_pytorch_evaluate(
        lp_x,
        lp_y,
        test_x,
        test_y,
        device=device,
        lr=1e-3,
        min_epochs=150,
        min_steps=10000,
        batch_size=512,
    )
    return dict(
        knn_f1=knn["knn_f1"],
        knn_acc=knn["knn_acc"],
        linear_f1=lp["linear_pytorch_f1"],
        linear_acc=lp["linear_pytorch_acc"],
    )


def initial_geometry(model, device, config, args):
    import numpy as np
    import torch
    from stable_cp.data import create_transforms, get_dataset, CPSubset
    from stable_cp.evaluation.zero_shot_eval import extract_features
    from eval.precp_official_norm import subset_indices, full_uniformity
    from stable_cp.data.heldout import array_hash

    _, clean = create_transforms(config, n_views=1, strong_aug=False)
    train = get_dataset(args.dataset, "train", clean, args.cache_dir, seed=42)
    indices = subset_indices(train.hf_dataset["label"])
    loader = torch.utils.data.DataLoader(
        CPSubset(train, indices.tolist()),
        batch_size=64,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    features, labels = extract_features(model, loader, device, pool_strategy="cls")
    if features.shape != (len(indices), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid initial geometry features")
    return dict(
        n_geometry=len(indices),
        geometry_indices=indices.tolist(),
        geometry_indices_sha256=array_hash(indices),
        geometry_source_indices_sha256=array_hash(
            train.hf_dataset.source_indices[indices]
        ),
        geometry_labels_sha256=array_hash(labels),
        uniformity_t2=full_uniformity(features, device=device),
        sampling="5000_stratified_then_3000_uniform_seed42",
    )
