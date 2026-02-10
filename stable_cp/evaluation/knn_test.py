import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score
import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.backbone.utils import from_timm
from pathlib import Path
from tqdm import tqdm

CACHE_DIR = Path(os.path.expanduser("~/.cache"))
DATA_DIR = CACHE_DIR / "stable_datasets"

INPUT_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard transforms
transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

print("Loading DINOv2-small baseline (TIMM)...")
model = from_timm("vit_small_patch14_dinov2.lvd142m", pretrained=True).to(device)
model.eval()

print("Loading FGVC-Aircraft datasets...")
full_train = spt.data.HFDataset(
    "randall-lab/fgvc-aircraft",
    split="train",
    transform=transform,
    trust_remote_code=True,
    cache_dir=str(DATA_DIR),
)
test_data = spt.data.HFDataset(
    "randall-lab/fgvc-aircraft",
    split="test",
    transform=transform,
    trust_remote_code=True,
    cache_dir=str(DATA_DIR),
)

print(f"Full train: {len(full_train)}, Test: {len(test_data)}")

# Create N=1000 subset
torch.manual_seed(42)
indices = torch.randperm(len(full_train))[:1000].tolist()
subset_train = torch.utils.data.Subset(full_train, indices)


def extract_features(model, dataset, device):
    """Extract features from dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
    features, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            x = batch["image"].to(device)
            y = batch["label"]
            feat = model(x)
            # Handle sequence outputs (take CLS token)
            if feat.dim() == 3:
                feat = feat[:, 0]
            features.append(feat.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(features), np.concatenate(labels)


print("\n" + "=" * 60)
print("Extracting features...")
print("=" * 60)

test_features, test_labels = extract_features(model, test_data, device)
full_train_features, full_train_labels = extract_features(model, full_train, device)

# Extract N=1000 subset features
subset_features = full_train_features[indices]
subset_labels = full_train_labels[indices]

print(f"\nFull train features: {full_train_features.shape}")
print(f"Subset (N=1000) features: {subset_features.shape}")
print(f"Test features: {test_features.shape}")

# Normalize
full_train_norm = normalize(full_train_features)
subset_norm = normalize(subset_features)
test_norm = normalize(test_features)

print("\n" + "=" * 60)
print("k-NN Evaluation Comparison")
print("=" * 60)

for k in [1, 5, 20]:
    print(f"\n--- k={k} ---")

    # Full training set as neighbors
    knn_full = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn_full.fit(full_train_norm, full_train_labels)
    pred_full = knn_full.predict(test_norm)
    acc_full = accuracy_score(test_labels, pred_full)
    f1_full = f1_score(test_labels, pred_full, average="macro")
    print(
        f"Full train ({len(full_train)} samples): Acc={acc_full:.4f}, F1={f1_full:.4f}"
    )

    # N=1000 subset as neighbors
    knn_subset = KNeighborsClassifier(
        n_neighbors=k, metric="cosine", weights="distance"
    )
    knn_subset.fit(subset_norm, subset_labels)
    pred_subset = knn_subset.predict(test_norm)
    acc_subset = accuracy_score(test_labels, pred_subset)
    f1_subset = f1_score(test_labels, pred_subset, average="macro")
    print(f"Subset (N=1000 samples): Acc={acc_subset:.4f}, F1={f1_subset:.4f}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
