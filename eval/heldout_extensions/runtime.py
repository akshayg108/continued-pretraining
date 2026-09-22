"""Encoder-aware execution with the original held-out evaluation protocol."""

from types import SimpleNamespace

from eval.heldout_cp.runtime import dataset_record, software
from eval.heldout_extensions import protocol as p
from eval.heldout_metric_roundoff import canonical_scores, POLICY, TOLERANCE


def check_environment(profile):
    import torch
    from stable_datasets import images

    if profile not in ("a100", "v100"):
        raise ValueError(f"Unknown GPU profile: {profile}")
    for name in ("MedMNIST", "AID", "RESISC45", "StanfordDogs", "JenaFlowers30", "Flavia", "IP102"):
        if not hasattr(images, name):
            raise RuntimeError(f"Missing stable-datasets reader {name}; install cc01e36 or newer")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if profile.upper() not in gpu.upper() or "MIG" in gpu.upper():
        raise RuntimeError(f"Expected one {profile.upper()} GPU, received {gpu}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return gpu


def evaluation_args(encoder, dataset, seed, cache_dir, num_workers):
    spec = p.ENCODERS[encoder]
    return SimpleNamespace(dataset=dataset, seed=seed, n_samples=1000, batch_size=64,
                           num_workers=num_workers, cache_dir=str(cache_dir),
                           backbone=spec["model_id"], pool_strategy=spec["pool"],
                           embed_dim=spec["embed_dim"])


def load_model(encoder, args):
    from continued_pretraining import load_backbone, get_dataset_config, configure_normalization
    from eval.precp_official_norm import official_normalization, _weights_hash

    model, device = load_backbone(args, img_size=224, pretrained=True)
    native = official_normalization("DINOv3" if encoder == "DINOv3L" else "SigLIP", model.pretrained_cfg)
    config = configure_normalization(get_dataset_config(args.dataset), model, "pretrained")
    if config["normalization"] != native:
        raise ValueError("Checkpoint-native normalization was not applied")
    if getattr(model, "num_features", None) != p.ENCODERS[encoder]["embed_dim"]:
        raise ValueError(f"Unexpected {encoder} feature dimension")
    return model.to(device), device, config, _weights_hash(model)


def evaluate(model, device, config, args, indices):
    import lightning as pl
    import numpy as np
    from stable_cp.data import create_transforms, create_eval_loaders
    from stable_cp.evaluation.zero_shot_eval import (
        extract_features, knn_evaluate, linear_probe_pytorch_evaluate,
    )

    model.to(device)
    pl.seed_everything(args.seed, workers=True)
    augmented, clean = create_transforms(config, n_views=1, strong_aug=False)
    test, lp_train, _ = create_eval_loaders(args, config, augmented, clean, args.cache_dir, indices=indices)
    _, knn_train, _ = create_eval_loaders(args, config, clean, clean, args.cache_dir, indices=indices)
    arrays = [extract_features(model, loader, device, pool_strategy=args.pool_strategy)
              for loader in (lp_train, test, knn_train)]
    for features, labels in arrays:
        if features.shape != (len(labels), args.embed_dim) or not np.isfinite(features).all():
            raise ValueError("Invalid evaluation feature shape or non-finite values")
    (lp_x, lp_y), (test_x, test_y), (knn_x, knn_y) = arrays
    if len(lp_x) != 1000 or len(knn_x) != 1000 or not np.array_equal(lp_y, knn_y):
        raise ValueError("CP, kNN and LP must use the same 1000-image subset")
    knn = knn_evaluate(knn_x, knn_y, test_x, test_y, k=20)
    pl.seed_everything(args.seed, workers=True)
    lp = linear_probe_pytorch_evaluate(lp_x, lp_y, test_x, test_y, device=device,
                                       lr=1e-3, min_epochs=150, min_steps=10000, batch_size=512)
    raw = dict(knn_f1=knn["knn_f1"], knn_acc=knn["knn_acc"],
               linear_f1=lp["linear_pytorch_f1"], linear_acc=lp["linear_pytorch_acc"])
    return canonical_scores(raw), dict(policy=POLICY, tolerance=TOLERANCE, raw_scores=raw)


def initial_geometry(model, device, config, args, indices):
    import numpy as np
    import torch
    from stable_cp.data import create_transforms, get_dataset, CPSubset
    from stable_cp.data.heldout import array_hash
    from stable_cp.evaluation.zero_shot_eval import extract_features
    from eval.precp_official_norm import full_uniformity

    _, clean = create_transforms(config, n_views=1, strong_aug=False)
    train = get_dataset(args.dataset, "train", clean, args.cache_dir, seed=42)
    idx = np.asarray(indices)
    if (not 2 <= len(idx) <= 3000 or not np.issubdtype(idx.dtype, np.integer)
            or len(np.unique(idx)) != len(idx) or (idx < 0).any() or (idx >= len(train)).any()):
        raise ValueError("Invalid source geometry indices")
    loader = torch.utils.data.DataLoader(CPSubset(train, indices), batch_size=64,
                                         num_workers=args.num_workers, shuffle=False, pin_memory=True)
    model.to(device)
    features, labels = extract_features(model, loader, device, pool_strategy=args.pool_strategy)
    if features.shape != (len(idx), args.embed_dim) or not np.isfinite(features).all():
        raise ValueError("Invalid geometry feature shape or non-finite values")
    return dict(n_geometry=len(idx), geometry_indices=indices,
                geometry_indices_sha256=array_hash(idx),
                geometry_source_indices_sha256=array_hash(train.hf_dataset.source_indices[idx]),
                geometry_labels_sha256=array_hash(labels),
                uniformity_t2=full_uniformity(features, device=device),
                sampling="reuse_original_5000_stratified_then_3000_uniform_seed42")
