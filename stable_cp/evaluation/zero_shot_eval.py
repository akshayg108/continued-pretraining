# Zero-shot evaluation utilities for continued pretraining
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# sklearn for evaluation (standard practice for post-training eval)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# torchmetrics for classification metrics (spt convention)
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
)

# stable-pretraining imports
import stable_pretraining as spt


def load_backbone_from_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> dict:
    # Load backbone weights from a stable-pretraining checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        # Lightning format: state_dict has "backbone.layer.weight" keys
        full_state_dict = checkpoint["state_dict"]
        # Extract only backbone keys and remove prefix
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in full_state_dict.items()
            if k.startswith("backbone.")
        }
        if not state_dict:
            # Maybe no prefix, try loading full state_dict
            state_dict = full_state_dict
    elif "model_state_dict" in checkpoint:
        # DIET reference format
        state_dict = checkpoint["model_state_dict"]
    elif "backbone_state_dict" in checkpoint:
        # Our custom format (if used)
        state_dict = checkpoint["backbone_state_dict"]
    else:
        # Assume it's a raw state dict
        state_dict = checkpoint

    backbone.load_state_dict(state_dict, strict=strict)
    print(f"Loaded backbone from {checkpoint_path}")

    return checkpoint


def load_backbone(
    model_name: str,
    source: str = "timm",
    checkpoint: str = None,
    low_resolution: bool = False,
    **kwargs,
) -> nn.Module:
    # Load a backbone model using stable-pretraining conventions
    if source == "torchvision":
        model = spt.backbone.from_torchvision(
            model_name, low_resolution=low_resolution, **kwargs
        )
    elif source == "timm":
        model = spt.backbone.from_timm(
            model_name, low_resolution=low_resolution, **kwargs
        )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'torchvision' or 'timm'")

    # Load checkpoint if provided
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        # Handle common checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        # Remove "backbone." prefix if present (common in spt checkpoints)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    return model


def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pool_strategy: str = "cls",
    verbose: bool = True,
) -> tuple:
    # Extract features from a data loader using forward_features
    features, labels = [], []
    model.eval()

    iterator = tqdm(loader, desc="Extracting features") if verbose else loader

    with torch.no_grad():
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
            feat = model.forward_features(x)

            if feat.dim() == 3:
                if pool_strategy == "mean":
                    feat = feat[:, 1:, :].mean(dim=1)
                else:
                    feat = feat[:, 0, :]  # CLS token

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
    # Evaluate using k-NN classifier
    # Adjust k if training set is small
    k = min(k, len(train_labels))

    # L2 normalize (standard practice)
    train_features = normalize(train_features)
    test_features = normalize(test_features)

    # Fit k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(train_features, train_labels)

    # Predictions
    pred = knn.predict(test_features)
    proba = knn.predict_proba(test_features)

    # Compute metrics using torchmetrics
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

    # AUROC (may fail if not all classes present in test set)
    try:
        results["knn_auroc"] = MulticlassAUROC(
            num_classes=num_classes, average="macro"
        )(proba_t, target_t).item()
    except ValueError:
        results["knn_auroc"] = 0.0

    return results


def linear_probe_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict:
    # Evaluate using linear probe classifier (sklearn LogisticRegression)
    # L2 normalize
    train_features = normalize(train_features)
    test_features = normalize(test_features)

    num_classes = len(np.unique(train_labels))

    # sklearn LogisticRegression (standard for post-training eval)
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(train_features, train_labels)

    # Predictions
    pred = clf.predict(test_features)
    proba = clf.predict_proba(test_features)

    # Compute metrics using torchmetrics
    pred_t = torch.from_numpy(pred)
    target_t = torch.from_numpy(test_labels)
    proba_t = torch.from_numpy(proba).float()

    results = {
        "linear_acc": MulticlassAccuracy(num_classes=num_classes)(
            pred_t, target_t
        ).item(),
        "linear_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
            pred_t, target_t
        ).item(),
    }

    try:
        results["linear_auroc"] = MulticlassAUROC(
            num_classes=num_classes, average="macro"
        )(proba_t, target_t).item()
    except ValueError:
        results["linear_auroc"] = 0.0

    return results


def linear_probe_pytorch_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device = "cuda",
    lr: float = 1e-3,
    num_steps: int = 20000,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict:
    # Evaluate using PyTorch linear probe (DIET-CP reference protocol)
    # L2 normalize
    train_features = normalize(train_features)
    test_features = normalize(test_features)

    # Convert to tensors
    train_features_t = torch.from_numpy(train_features).float().to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    test_features_t = torch.from_numpy(test_features).float().to(device)
    test_labels_t = torch.from_numpy(test_labels).long()

    num_classes = len(np.unique(train_labels))
    in_dim = train_features.shape[1]

    # Create linear classifier
    clf = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    clf.train()
    n_samples = len(train_features_t)

    for step in range(num_steps):
        # Random batch
        idx = torch.randint(0, n_samples, (min(batch_size, n_samples),))
        batch_features = train_features_t[idx]
        batch_labels = train_labels_t[idx]

        optimizer.zero_grad()
        logits = clf(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % 5000 == 0:
            print(f"    Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    # Evaluation
    clf.eval()
    with torch.no_grad():
        logits = clf(test_features_t)
        proba = torch.softmax(logits, dim=1).cpu()
        pred = logits.argmax(dim=1).cpu()

    # Compute metrics
    results = {
        "linear_pytorch_acc": MulticlassAccuracy(num_classes=num_classes)(
            pred, test_labels_t
        ).item(),
        "linear_pytorch_f1": MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )(pred, test_labels_t).item(),
    }

    try:
        results["linear_pytorch_auroc"] = MulticlassAUROC(
            num_classes=num_classes, average="macro"
        )(proba, test_labels_t).item()
    except ValueError:
        results["linear_pytorch_auroc"] = 0.0

    return results


def kmeans_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = None,
) -> dict:
    # Evaluate using K-Means clustering
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    # L2 normalize
    features = normalize(features)

    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(features)

    return {
        "kmeans_ari": adjusted_rand_score(labels, pred),
        "kmeans_nmi": normalized_mutual_info_score(labels, pred),
    }


def zero_shot_eval(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k_neighbors: int = 20,
    linear_max_iter: int = 1000,
    linear_probe_method: str = "both",
    linear_pytorch_steps: int = 20000,
    linear_pytorch_lr: float = 1e-3,
    pool_strategy: str = "cls",
    verbose: bool = True,
) -> dict:
    # Full zero-shot evaluation pipeline
    model = model.to(device)

    # Extract features
    if verbose:
        print("Extracting features...")
    train_features, train_labels = extract_features(
        model, train_loader, device, pool_strategy=pool_strategy, verbose=verbose
    )
    test_features, test_labels = extract_features(
        model, test_loader, device, pool_strategy=pool_strategy, verbose=verbose
    )

    if verbose:
        print(f"Train: {train_features.shape}, Test: {test_features.shape}")

    results = {}

    # k-NN evaluation
    if verbose:
        print("Running k-NN evaluation...")
    results.update(
        knn_evaluate(
            train_features, train_labels, test_features, test_labels, k=k_neighbors
        )
    )
    if verbose:
        print(f"  Accuracy: {results['knn_acc']:.4f}, F1: {results['knn_f1']:.4f}")

    # Linear probe evaluation - sklearn (CLIP/DINOv2 standard)
    if linear_probe_method in ["sklearn", "both"]:
        if verbose:
            print("Running linear probe evaluation (sklearn LogisticRegression)...")
        results.update(
            linear_probe_evaluate(
                train_features,
                train_labels,
                test_features,
                test_labels,
                max_iter=linear_max_iter,
            )
        )
        if verbose:
            print(
                f"  Accuracy: {results['linear_acc']:.4f}, F1: {results['linear_f1']:.4f}"
            )

    # Linear probe evaluation - PyTorch (DIET-CP reference protocol)
    if linear_probe_method in ["pytorch", "both"]:
        if verbose:
            print(
                f"Running linear probe evaluation (PyTorch, {linear_pytorch_steps} steps)..."
            )
        results.update(
            linear_probe_pytorch_evaluate(
                train_features,
                train_labels,
                test_features,
                test_labels,
                device=device,
                lr=linear_pytorch_lr,
                num_steps=linear_pytorch_steps,
                verbose=verbose,
            )
        )
        if verbose:
            print(
                f"  Accuracy: {results['linear_pytorch_acc']:.4f}, F1: {results['linear_pytorch_f1']:.4f}"
            )

    # K-means evaluation
    if verbose:
        print("Running k-means evaluation...")
    results.update(kmeans_evaluate(test_features, test_labels))
    if verbose:
        print(f"  ARI: {results['kmeans_ari']:.4f}, NMI: {results['kmeans_nmi']:.4f}")

    return results


def finetune_evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pool_strategy: str = "cls",
    verbose: bool = True,
) -> dict:
    """Evaluate a fine-tuned model (backbone + classifier) on test data.

    Unlike knn/linear probe which only use frozen backbone features,
    this evaluates the actual trained classification head.

    Args:
        backbone: Trained backbone network
        classifier: Trained classification head (nn.Linear)
        test_loader: Test data loader
        device: Device to use
        pool_strategy: 'cls' or 'mean' for ViT embedding extraction
        verbose: Whether to print progress

    Returns:
        dict: Results with finetune_acc, finetune_f1, finetune_auroc
    """
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

            features = backbone.forward_features(images)
            if features.dim() == 3:
                if pool_strategy == "mean":
                    features = features[:, 1:, :].mean(dim=1)
                else:
                    features = features[:, 0, :]  # CLS token

            logits = classifier(features)
            proba = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_proba.append(proba.cpu())
            all_labels.append(
                labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            )

    pred_t = torch.cat(all_preds)
    proba_t = torch.cat(all_proba)
    target_t = torch.cat(all_labels)

    num_classes = proba_t.shape[1]

    results = {
        "finetune_acc": MulticlassAccuracy(num_classes=num_classes)(
            pred_t, target_t
        ).item(),
        "finetune_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
            pred_t, target_t
        ).item(),
    }

    try:
        results["finetune_auroc"] = MulticlassAUROC(
            num_classes=num_classes, average="macro"
        )(proba_t, target_t).item()
    except ValueError:
        results["finetune_auroc"] = 0.0

    if verbose:
        print(
            f"  Finetune Accuracy: {results['finetune_acc']:.4f}, F1: {results['finetune_f1']:.4f}"
        )

    return results


def evaluate_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    **kwargs,
) -> dict:
    return zero_shot_eval(model, train_loader, test_loader, device, **kwargs)
