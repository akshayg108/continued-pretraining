#!/usr/bin/env python
"""
geometry_metrics.py — Recompute pre-CP representation geometry for 15 datasets x 3 encoders.

PORTED from the original hypothesis/ colab geometry analysis (since removed in the repo
cleanup; this file supersedes it). The loaders + extract_features + metric defs are FAITHFUL
to that version so it runs in the same environment. Changes vs the original are
marked `# FIX` / `# NEW` and are limited to:

  FIX-1  Drop the hard-coded CP_OUTCOMES + correlation half. Geometry is per
         (encoder, dataset); the Δ outcomes live in results.xlsx and depend on
         the encoder AND method AND size. Mixing a single encoder-averaged Δ into
         this file was a methodological error. This script now ONLY emits the
         geometry CSV; correlate.py joins it to results.xlsx per-encoder.
  NEW-1  mmd_rbf_components(): return MMD plus its three energy terms
         (M_PP, M_QQ, M_PQ) and gamma, so the claimed identity
         MMD^2 = M_PP + exp(L_uniform(Q)) - 2 M_PQ can be EMPIRICALLY tested
         (it only holds if uniformity-t == MMD-gamma AND the diagonal handling
         matches; hypothesis_1.md asserts it as exact — it is not).
  NEW-2  Also emit uniformity at t=gamma_mmd (per dataset) alongside t=2.0, for
         the identity check above.
  NEW-3  neighbor_overlap at k=20 (matches kNN eval) AND k=50 (matches the doc).
  FIX-2  Local-friendly default paths (./eval/data/...), and a clear import
         guard for stable_datasets.

Geometry metrics (uniformity, l2_norm_cv) need only the target datasets + the
encoders. MMD / overlap / centroid / OTCE additionally need ImageNet-val
(gated: ILSVRC/imagenet-1k, HF login + license). Use --skip-imagenet for a
fast, ImageNet-free pass that still covers Hypothesis 1's headline predictor.

Usage (run in the env where stable_datasets imports):
  cd /Users/zhanghaodong/Desktop/CP
  # full (needs ImageNet-val pre-saved by hypothesis/download_imagenet_val.py):
  python eval/geometry_metrics.py --imagenet-dir <dir> \
      --download-dir <dir> --processed-dir <dir> --output eval/outputs/geometry_15.csv
  # ImageNet-free (uniformity + norm CV only, all 15 x 3):
  python eval/geometry_metrics.py --skip-imagenet --output eval/outputs/geometry_15_noimnet.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.preprocessing import normalize

import timm

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Constants (verbatim from colab variant) ──────────────────
ENCODERS = {
    "DINOv3": {"timm_id": "vit_base_patch16_dinov3.lvd1689m", "pool": "cls"},
    "MAE":    {"timm_id": "vit_base_patch16_224.mae",         "pool": "mean"},
    "CLIP":   {"timm_id": "vit_base_patch16_clip_224.openai", "pool": "cls"},
    "SigLIP": {"timm_id": "vit_base_patch16_siglip_224.v2_webli", "pool": "map"},  # SigLIP-2; native MAP attn-pool head (no cls token; sphere-native pooled embedding)
}

TARGET_DATASETS = [
    "breastmnist", "dermamnist", "octmnist", "organamnist", "pathmnist", "galaxy10",
    "eurosat", "plant_village", "dtd",
    "food101", "fgvc_aircraft",
    "cars196", "cub200", "flowers102", "oxford_pet",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
MAX_TARGET_SAMPLES = 5000

# FIX-2: clear import guard.
try:
    from stable_datasets import images as stable_ds
except Exception as e:  # pragma: no cover
    raise ImportError(
        "stable_datasets is required for dataset loading. Install the vendored "
        "package: `pip install -e continued-pretraining/stable-datasets` (or the "
        "standalone codebase/stable-datasets). Original error: %r" % (e,)
    )

DS_REGISTRY = {
    "breastmnist":   (stable_ds.MedMNIST, "breastmnist",   ["train", "validation", "test"], {"size": 224}),
    "dermamnist":    (stable_ds.MedMNIST, "dermamnist",     ["train", "validation", "test"], {"size": 224}),
    "octmnist":      (stable_ds.MedMNIST, "octmnist",       ["train", "validation", "test"], {"size": 224}),
    "organamnist":   (stable_ds.MedMNIST, "organamnist",    ["train", "validation", "test"], {"size": 224}),
    "pathmnist":     (stable_ds.MedMNIST, "pathmnist",      ["train", "validation", "test"], {"size": 224}),
    "galaxy10":      (stable_ds.Galaxy10Decal, None,        ["train", "validation", "test"], {}),
    "food101":       (stable_ds.Food101, None,              ["train", "test", "test"],       {}),
    "fgvc_aircraft": (stable_ds.FGVCAircraft, "variant",    ["train", "validation", "test"], {}),
    "eurosat":       (stable_ds.EuroSAT, None,              ["train", "validation", "test"], {}),
    "plant_village": (stable_ds.PlantVillage, "color",      ["train", "test", "test"],       {}),
    "dtd":           (stable_ds.DTD, None,                  ["train", "validation", "test"], {}),
    "cars196":       (stable_ds.Cars196, None,              ["train", "test", "test"],       {}),
    "cub200":        (stable_ds.CUB200, None,               ["train", "test", "test"],       {}),
    "flowers102":    (stable_ds.Flowers102, None,           ["train", "validation", "test"], {}),
    "oxford_pet":    (stable_ds.OxfordPet, None,            ["train", "test", "test"],       {}),
}


class StableDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset, self.transform = hf_dataset, transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img, label = sample["image"], sample["label"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label) if not isinstance(label, int) else label


def eval_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def load_target_dataset(name, download_dir, processed_dir):
    ds_class, config_name, splits, extra_kwargs = DS_REGISTRY[name]
    kwargs = {}
    if config_name is not None:
        kwargs["config_name"] = config_name
    kwargs.update(extra_kwargs)
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    hf_ds = ds_class(split=splits[0], download_dir=str(download_dir),
                     processed_cache_dir=str(processed_dir), **kwargs)
    if len(hf_ds) > MAX_TARGET_SAMPLES:
        from sklearn.model_selection import train_test_split
        labels = np.array(hf_ds["label"]).ravel()
        selected, _ = train_test_split(np.arange(len(hf_ds)), train_size=MAX_TARGET_SAMPLES,
                                       stratify=labels, random_state=42)
        hf_ds = hf_ds.select(sorted(selected.tolist()))
    return DataLoader(StableDatasetWrapper(hf_ds, eval_transform()),
                      batch_size=256, num_workers=4, shuffle=False)


class ImageNetValDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.hf_ds, self.transform = hf_ds, transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        img = sample["image"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["label"]


def load_imagenet_val(imagenet_dir, n_samples=5000):
    from datasets import load_from_disk
    ds = load_from_disk(str(imagenet_dir))
    rng = np.random.RandomState(42)
    idx = sorted(rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False))
    ds = ds.select(idx)
    return DataLoader(ImageNetValDataset(ds, eval_transform()),
                      batch_size=256, num_workers=4, shuffle=False)


def extract_features(model, loader, device, pool_strategy="cls"):
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
            x, y = batch[0], batch[1]
            feat = model.forward_features(x.to(device))
            if pool_strategy == "map":
                feat = model.forward_head(feat)  # SigLIP MAP head: attn_pool + fc_norm (native pooled embedding); needs num_classes=0 so head=Identity
            elif feat.dim() == 3:
                feat = feat[:, 1:, :].mean(dim=1) if pool_strategy == "mean" else feat[:, 0, :]
            feats.append(feat.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.vstack(feats), np.concatenate(labels).ravel()


# ── Metrics ──────────────────────────────────────────────────
def l2_norm_stats(features):
    norms = np.linalg.norm(features, axis=1)
    return float(norms.mean()), float(norms.std()), float(norms.std() / norms.mean())


def cosine_distance_centroids(feat_a, feat_b):
    ca = normalize(feat_a.mean(axis=0, keepdims=True))
    cb = normalize(feat_b.mean(axis=0, keepdims=True))
    return float(1.0 - (ca * cb).sum())


def wang_isola_uniformity(features, t=2.0, l2_normalize=True):
    """L_uniform = log E[exp(-t ||x-y||^2)] over upper-triangle pairs. More negative = more uniform."""
    f = normalize(features) if l2_normalize else features.astype(np.float64)
    if len(f) > 3000:
        f = f[np.random.RandomState(42).choice(len(f), 3000, replace=False)]
    sq = np.sum(f**2, 1, keepdims=True) + np.sum(f**2, 1) - 2 * f @ f.T
    np.fill_diagonal(sq, np.inf)
    triu = sq[np.triu_indices(len(f), k=1)]
    return float(np.log(np.exp(-t * triu).mean()))


def mmd_rbf_components(feat_a, feat_b, gamma=None):
    """NEW-1: RBF-MMD with its energy terms exposed.

    Returns dict with mmd, m_pp (target self), m_qq (imagenet self), m_pq (cross),
    and gamma. NOTE: rbf(a,a).mean() INCLUDES the diagonal (self-pairs, value 1),
    i.e. this is the biased V-statistic — same as the original code.
    """
    a, b = normalize(feat_a), normalize(feat_b)  # a = target, b = imagenet
    if gamma is None:
        from scipy.spatial.distance import pdist
        combined = np.vstack([a[:500], b[:500]])
        dists = pdist(combined, metric="sqeuclidean")
        gamma = 1.0 / max(np.median(dists), 1e-8)

    def rbf(x, y):
        sq = np.sum(x**2, 1, keepdims=True) + np.sum(y**2, 1) - 2 * x @ y.T
        return np.exp(-gamma * sq)

    m_pp = float(rbf(a, a).mean())   # target self-similarity (== exp(uniformity) iff t==gamma & no diag)
    m_qq = float(rbf(b, b).mean())   # imagenet self-similarity
    m_pq = float(rbf(a, b).mean())   # cross
    mmd = m_pp + m_qq - 2 * m_pq
    return {"mmd_rbf": float(mmd), "mmd_m_pp_target": m_pp, "mmd_m_qq_imagenet": m_qq,
            "mmd_m_pq_cross": m_pq, "mmd_gamma": float(gamma)}


def neighbor_overlap(feat_target, feat_imagenet, k=50):
    ft, fi = normalize(feat_target), normalize(feat_imagenet)
    rng = np.random.RandomState(42)
    if len(ft) > 2000:
        ft = ft[rng.choice(len(ft), 2000, replace=False)]
    if len(fi) > 5000:
        fi = fi[rng.choice(len(fi), 5000, replace=False)]
    combined = np.vstack([ft, fi])
    is_imagenet = np.array([False] * len(ft) + [True] * len(fi))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(combined)
    _, indices = nn.kneighbors(ft)
    return float(is_imagenet[indices[:, 1:k + 1]].mean())


# ── Main ─────────────────────────────────────────────────────
FIELDNAMES = [
    "encoder", "dataset", "n_samples", "embed_dim",
    "l2_norm_mean", "l2_norm_std", "l2_norm_cv",
    "uniformity_t2", "uniformity_t2_raw",          # Wang-Isola canonical t=2 (+ raw ablation)
    "uniformity_at_gamma",                          # NEW-2: t = MMD gamma, for identity check
    "cosine_dist_centroid", "mmd_rbf",
    "mmd_m_pp_target", "mmd_m_qq_imagenet", "mmd_m_pq_cross", "mmd_gamma",  # NEW-1 components
    "neighbor_overlap_k20", "neighbor_overlap_k50",                         # NEW-3 both k
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-dir", type=str, default=str(ROOT / "eval/data/imagenet_val"))
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/geometry_15.csv"))
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--skip-imagenet", action="store_true",
                    help="uniformity + norm-CV only (no MMD/overlap/centroid); avoids gated ImageNet")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  | skip_imagenet={args.skip_imagenet}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    imagenet_loader = None if args.skip_imagenet else load_imagenet_val(
        args.imagenet_dir, n_samples=args.imagenet_samples)

    rows = []
    for enc_name in args.encoders:
        cfg = ENCODERS[enc_name]
        print(f"\n{'='*60}\nEncoder: {enc_name} ({cfg['timm_id']}, pool={cfg['pool']})\n{'='*60}")
        model = timm.create_model(cfg["timm_id"], pretrained=True, num_classes=0).eval().to(device)  # num_classes=0 -> head=Identity so forward_head returns the pooled embedding (MAP for SigLIP)

        feat_imagenet = None
        if imagenet_loader is not None:
            feat_imagenet, _ = extract_features(model, imagenet_loader, device, cfg["pool"])
            print(f"  ImageNet features: {feat_imagenet.shape}")
            nm, ns, nc = l2_norm_stats(feat_imagenet)
            rows.append({"encoder": enc_name, "dataset": "imagenet",
                         "n_samples": len(feat_imagenet), "embed_dim": feat_imagenet.shape[1],
                         "l2_norm_mean": nm, "l2_norm_std": ns, "l2_norm_cv": nc,
                         "uniformity_t2": wang_isola_uniformity(feat_imagenet)})

        for ds_name in args.datasets:
            print(f"--- {ds_name} ---")
            try:
                loader = load_target_dataset(ds_name, args.download_dir, args.processed_dir)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            feat, _ = extract_features(model, loader, device, cfg["pool"])
            nm, ns, nc = l2_norm_stats(feat)
            row = {"encoder": enc_name, "dataset": ds_name,
                   "n_samples": len(feat), "embed_dim": feat.shape[1],
                   "l2_norm_mean": nm, "l2_norm_std": ns, "l2_norm_cv": nc,
                   "uniformity_t2": wang_isola_uniformity(feat, t=2.0, l2_normalize=True),
                   "uniformity_t2_raw": wang_isola_uniformity(feat, t=2.0, l2_normalize=False)}
            if feat_imagenet is not None:
                comp = mmd_rbf_components(feat, feat_imagenet)
                row.update(comp)
                # NEW-2: uniformity evaluated at the same gamma MMD used.
                row["uniformity_at_gamma"] = wang_isola_uniformity(feat, t=comp["mmd_gamma"])
                row["cosine_dist_centroid"] = cosine_distance_centroids(feat, feat_imagenet)
                row["neighbor_overlap_k20"] = neighbor_overlap(feat, feat_imagenet, k=20)
                row["neighbor_overlap_k50"] = neighbor_overlap(feat, feat_imagenet, k=50)
                print(f"  CV={nc:.3f}  unif_t2={row['uniformity_t2']:.3f}  "
                      f"MMD={comp['mmd_rbf']:.4f}  overlap50={row['neighbor_overlap_k50']:.4f}")
            else:
                print(f"  CV={nc:.3f}  unif_t2={row['uniformity_t2']:.3f}  (ImageNet skipped)")
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"\nSaved {len(rows)} rows -> {args.output}")
    print("Next: python eval/correlate.py --geometry", args.output)


if __name__ == "__main__":
    main()
