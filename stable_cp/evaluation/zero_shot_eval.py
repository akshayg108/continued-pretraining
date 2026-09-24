"""Frozen-feature kNN and fresh-view linear-probe evaluation."""

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)
from tqdm import tqdm

from stable_cp.utils.backbone import forward_embedding
from .linear_probe import _frozen_encoder, _preserve_rng, linear_probe_online_evaluate


def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pool_strategy: str = "cls",
    verbose: bool = True,
) -> tuple:
    """Read features without gradients or changes to the trainable-parameter mask."""
    features, labels = [], []

    iterator = tqdm(loader, desc="Extracting features") if verbose else loader

    with _frozen_encoder(model), torch.no_grad():
        for batch in iterator:
            if isinstance(batch, dict):
                x = batch["image"]
                y = batch["label"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1]
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            x = x.to(device)
            feat = forward_embedding(model, x, pool_strategy)

            features.append(feat.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

    features = np.vstack(features)
    labels = np.concatenate(labels, axis=0).ravel()

    return features, labels


def knn_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 20,
) -> dict:
    """Evaluate normalized features with inverse-cosine-distance weighted kNN."""
    k = min(k, len(train_labels))

    train_features = normalize(train_features)
    test_features = normalize(test_features)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(train_features, train_labels)

    pred = knn.predict(test_features)
    proba = knn.predict_proba(test_features)

    num_classes = len(np.unique(train_labels))
    pred_t = torch.from_numpy(pred)
    target_t = torch.from_numpy(test_labels)
    proba_t = torch.from_numpy(proba)

    results = {
        "knn_acc": MulticlassAccuracy(num_classes=num_classes)(pred_t, target_t).item(),
        "knn_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
            pred_t, target_t
        ).item(),
    }

    try:
        results["knn_auroc"] = MulticlassAUROC(num_classes=num_classes, average="macro")(
            proba_t, target_t
        ).item()
    except ValueError:
        results["knn_auroc"] = 0.0

    return results


def linear_probe_pytorch_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device = "cuda",
    lr: float = 1e-3,
    min_epochs: int = 150,
    min_steps: int = 10000,
    batch_size: int = 512,
    verbose: bool = True,
) -> dict:
    """Train an Adam linear classifier on normalized, precomputed features."""
    train_features = normalize(train_features)
    test_features = normalize(test_features)

    train_features_t = torch.from_numpy(train_features).float().to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    test_features_t = torch.from_numpy(test_features).float().to(device)
    test_labels_t = torch.from_numpy(test_labels).long()

    num_classes = len(np.unique(train_labels))
    in_dim = train_features.shape[1]

    clf = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    clf.train()
    n_samples = len(train_features_t)
    n_batches = (n_samples + batch_size - 1) // batch_size
    num_steps = max(min_steps, min_epochs * n_batches)
    indices = torch.randperm(n_samples, device=device)

    if verbose:
        effective_epochs = num_steps / n_batches
        print(
            f"    LP training: {num_steps} steps ({effective_epochs:.0f} epochs, {n_samples} samples)"
        )

    log_interval = max(num_steps // 5, 1)
    for step in range(num_steps):
        if step % n_batches == 0:
            indices = torch.randperm(n_samples, device=device)

        batch_idx = step % n_batches
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_features = train_features_t[batch_indices]
        batch_labels = train_labels_t[batch_indices]

        optimizer.zero_grad()
        logits = clf(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % log_interval == 0:
            print(f"    Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    clf.eval()
    with torch.no_grad():
        logits = clf(test_features_t)
        proba = torch.softmax(logits, dim=1).cpu()
        pred = logits.argmax(dim=1).cpu()

    results = {
        "linear_pytorch_acc": MulticlassAccuracy(num_classes=num_classes)(
            pred, test_labels_t
        ).item(),
        "linear_pytorch_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
            pred, test_labels_t
        ).item(),
    }

    try:
        results["linear_pytorch_auroc"] = MulticlassAUROC(num_classes=num_classes, average="macro")(
            proba, test_labels_t
        ).item()
    except ValueError:
        results["linear_pytorch_auroc"] = 0.0

    return results


def zero_shot_eval(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k_neighbors: int = 20,
    linear_pytorch_min_steps: int = 10000,
    linear_pytorch_lr: float = 1e-3,
    pool_strategy: str = "cls",
    knn_train_loader: torch.utils.data.DataLoader = None,
    verbose: bool = True,
    geometry: dict = None,
    *,
    lp_epochs: int = 150,
    lp_lr: float = None,
    lp_forward_batch_size: int = 32,
    lp_num_classes: int = None,
    lp_seed: int = None,
) -> dict:
    """Evaluate clean-view kNN and frozen online LP with fresh training views.

    train_loader is the LP loader; knn_train_loader must provide clean views.
    linear_pytorch_min_steps is retained for call compatibility but is unused:
    LP always makes exactly lp_epochs complete passes through train_loader.
    """
    if knn_train_loader is None:
        raise ValueError("A clean knn_train_loader is required separately from the LP train_loader")
    if linear_pytorch_min_steps != 10000:
        warnings.warn(
            "linear_pytorch_min_steps is unused by online LP; configure lp_epochs instead",
            FutureWarning,
            stacklevel=2,
        )
    with _preserve_rng(lp_seed):
        model = model.to(device)
        test_features, test_labels = extract_features(
            model, test_loader, device, pool_strategy=pool_strategy, verbose=verbose
        )
        knn_features, knn_labels = extract_features(
            model, knn_train_loader, device, pool_strategy=pool_strategy, verbose=verbose
        )

        results = knn_evaluate(knn_features, knn_labels, test_features, test_labels, k=k_neighbors)
        if geometry is not None:
            from .geometry import evaluate_geometry

            results["geometry"] = evaluate_geometry(knn_features, **geometry)
            if verbose:
                print(f"  Geometry: {results['geometry']}")
        del knn_features, knn_labels, test_features, test_labels
        results.update(
            linear_probe_online_evaluate(
                model,
                train_loader,
                test_loader,
                device=device,
                pool_strategy=pool_strategy,
                epochs=lp_epochs,
                lr=linear_pytorch_lr if lp_lr is None else lp_lr,
                forward_batch_size=lp_forward_batch_size,
                num_classes=lp_num_classes,
                seed=lp_seed,
                verbose=verbose,
            )
        )
        if verbose:
            print(f"  kNN F1: {results['knn_f1']:.4f}, LP F1: {results['linear_pytorch_f1']:.4f}")
        return results


def finetune_evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pool_strategy: str = "cls",
    verbose: bool = True,
) -> dict:
    """Evaluate a fine-tuned backbone and its supervised classifier."""
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_proba = []
    all_labels = []

    iterator = tqdm(test_loader, desc="Finetune evaluation") if verbose else test_loader

    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1]
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            features = forward_embedding(backbone, images, pool_strategy)

            logits = classifier(features)
            proba = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_proba.append(proba.cpu())
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    pred_t = torch.cat(all_preds)
    proba_t = torch.cat(all_proba)
    target_t = torch.cat(all_labels)

    num_classes = proba_t.shape[1]

    results = {
        "finetune_acc": MulticlassAccuracy(num_classes=num_classes)(pred_t, target_t).item(),
        "finetune_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
            pred_t, target_t
        ).item(),
    }

    try:
        results["finetune_auroc"] = MulticlassAUROC(num_classes=num_classes, average="macro")(
            proba_t, target_t
        ).item()
    except ValueError:
        results["finetune_auroc"] = 0.0

    if verbose:
        print(
            f"  Finetune Accuracy: {results['finetune_acc']:.4f}, F1: {results['finetune_f1']:.4f}"
        )

    return results
