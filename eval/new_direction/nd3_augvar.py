#!/usr/bin/env python
"""
nd3_augvar.py — ND3 (GPU pass): per-direction augmentation-to-data variance ratios.

Jing et al. 2022 (papers/new_direction/Jing2022_DimensionalCollapse.pdf), Theorem 1 /
Corollary 1: along any feature direction where the variance CAUSED BY AUGMENTATION exceeds
the variance of the data distribution, the weight component collapses — i.e. which
directions the CP suppression force kills is set by the per-direction augmentation-to-data
variance ratio. The DTD hypothesis (papers/new_direction/NEW_DIRECTION.md §H-4, A3):
texture-discriminative directions are augmentation-dominated, so DTD is suppressed exactly
where it needs signal — the pre-registered test is whether DTD is an OUTLIER among the 15
datasets in the aug/data ratio along class-discriminative directions.

Per (encoder, dataset), on PRE-CP features:
  Sigma_data : covariance of clean features (standard <=5000 eval sample)
  Sigma_aug  : E_img[ Cov_views(features of K strong-aug views of the image) ]
  ratio along a direction w:  (w' Sigma_aug w) / (w' Sigma_data w)
  discriminative directions:  top eigvecs of the between-class scatter Sigma_B
Emits per (encoder, dataset): mean/max ratio on discriminative directions, median ratio on
the top-100 data directions, and the data-variance-weighted fraction of directions with
ratio > 1.

The augmentation stack mirrors stable_cp/data/transforms.py strong_aug branch in
torchvision (RandomResizedCrop scale=(0.2,1.0), HFlip p=.5, ColorJitter(.8,.8,.8,.2) p=.8,
Grayscale p=.2, GaussianBlur k=3 sigma=(0.1,2.0) p=.5 — the contrastive/invariance stack
that drives the suppression force), with PER-DATASET training normalization
(stable_cp/data/datasets.py NORMALIZATIONS: ImageNet for 9/15 targets; custom stats for
galaxy10 + 5 MedMNIST — TRAIN_NORM below). The CLEAN pass uses the same per-dataset
normalization so both Sigma_data and Sigma_aug are the training-stack Jing quantities
(NOTE: clean features here therefore differ from geometry_15.csv, which is
ImageNet-normalized everywhere — intentional and disclosed).

Cluster (one array task per dataset):
  python eval/new_direction/nd3_augvar.py --datasets <ds> \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/nd3_augvar_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                    # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))   # eval/utils

from geometry_metrics import (ENCODERS, TARGET_DATASETS, DS_REGISTRY,
                              StableDatasetWrapper, extract_features,
                              IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
                              MAX_TARGET_SAMPLES)
from spectral_metrics import class_scatter

ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = ["encoder", "dataset", "n_imgs", "k_views", "n_classes",
          "ratio_disc_mean", "ratio_disc_max", "ratio_top100_median",
          "frac_datavar_augdom"]

# Training-time normalization per dataset, copied from stable_cp/data/datasets.py
# NORMALIZATIONS (verified 2026-07-10). 1-channel MedMNIST stats tiled to 3 channels
# (training applies RGB() before ToImage, and Normalize broadcasts identically).
# Datasets not listed use ImageNet stats, matching the training registry.
TRAIN_NORM = {
    "galaxy10":    ([0.097, 0.097, 0.097], [0.174, 0.164, 0.156]),
    "pathmnist":   ([0.5, 0.5, 0.5],       [0.5, 0.5, 0.5]),
    "dermamnist":  ([0.7634, 0.5423, 0.5698], [0.0841, 0.1246, 0.1043]),
    "octmnist":    ([0.1778, 0.1778, 0.1778], [0.1316, 0.1316, 0.1316]),
    "organamnist": ([0.4996, 0.4996, 0.4996], [0.1731, 0.1731, 0.1731]),
    "breastmnist": ([0.4846, 0.4846, 0.4846], [0.2522, 0.2522, 0.2522]),
}


def train_norm(dataset):
    return TRAIN_NORM.get(dataset, (IMAGENET_MEAN, IMAGENET_STD))


def strong_aug_transform(dataset):
    """torchvision mirror of stable_cp/data/transforms.py strong_aug=True."""
    mean, std = train_norm(dataset)
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8,
                                     saturation=0.8, hue=0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def clean_transform(dataset):
    """Eval-style transform under the same per-dataset training normalization."""
    mean, std = train_norm(dataset)
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                      T.Normalize(mean=mean, std=std)])


class MultiViewWrapper(Dataset):
    """K independently augmented views per image, stacked to (K, C, H, W)."""

    def __init__(self, hf_dataset, k_views, dataset):
        self.hf_dataset, self.k = hf_dataset, k_views
        self.aug = strong_aug_transform(dataset)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]["image"]
        return torch.stack([self.aug(img) for _ in range(self.k)]), 0


def load_raw_subset(name, download_dir, processed_dir, n_imgs):
    """The same underlying (<=MAX_TARGET_SAMPLES, stratified, seed-42) hf dataset that
    load_target_dataset wraps, subsampled to n_imgs (seed 42). Mirrors
    geometry_metrics.load_target_dataset — kept in sync with its loading conventions."""
    ds_class, config_name, splits, extra_kwargs = DS_REGISTRY[name]
    kwargs = {}
    if config_name is not None:
        kwargs["config_name"] = config_name
    kwargs.update(extra_kwargs)
    hf_ds = ds_class(split=splits[0], download_dir=str(download_dir),
                     processed_cache_dir=str(processed_dir), **kwargs)
    if len(hf_ds) > MAX_TARGET_SAMPLES:
        from sklearn.model_selection import train_test_split
        labels = np.array(hf_ds["label"]).ravel()
        selected, _ = train_test_split(np.arange(len(hf_ds)), train_size=MAX_TARGET_SAMPLES,
                                       stratify=labels, random_state=42)
        hf_ds = hf_ds.select(sorted(selected.tolist()))
    if len(hf_ds) > n_imgs:
        idx = np.random.RandomState(42).choice(len(hf_ds), n_imgs, replace=False)
        hf_ds = hf_ds.select(sorted(idx.tolist()))
    return hf_ds


@torch.no_grad()
def aug_covariance(model, hf_ds, k_views, pool, device, dataset, batch_imgs=16):
    """Sigma_aug = E_img[ Cov over K views ], accumulated in float64 on CPU."""
    loader = DataLoader(MultiViewWrapper(hf_ds, k_views, dataset), batch_size=batch_imgs,
                        num_workers=4, shuffle=False)
    d, sigma, n_img = None, None, 0
    for views, _ in tqdm(loader, desc="aug views", leave=False):
        b, k = views.shape[:2]
        flat = views.reshape(b * k, *views.shape[2:]).to(device)
        feat = model.forward_features(flat)
        if pool == "map":
            feat = model.forward_head(feat)
        elif feat.dim() == 3:
            feat = feat[:, 1:, :].mean(dim=1) if pool == "mean" else feat[:, 0, :]
        feat = feat.reshape(b, k, -1).double()
        if sigma is None:
            d = feat.shape[-1]
            sigma = torch.zeros(d, d, dtype=torch.float64, device=device)
        centered = feat - feat.mean(dim=1, keepdim=True)
        sigma += torch.einsum("bkd,bke->de", centered, centered) / (k - 1)
        n_img += b
    return (sigma / n_img).cpu().numpy(), n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", type=str, default=str(ROOT / "eval/outputs/nd3_augvar.csv"))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--n-imgs", type=int, default=512)
    ap.add_argument("--k-views", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["encoder"], r["dataset"]) for r in csv.DictReader(f)}

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for enc in args.encoders:
            todo = [d for d in args.datasets if (enc, d) not in done]
            if not todo:
                continue
            cfg = ENCODERS[enc]
            print(f"\n===== {enc} ({cfg['timm_id']}) — {len(todo)} datasets")
            model = timm.create_model(cfg["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            for ds in todo:
                # clean pass under the SAME per-dataset training normalization as the
                # aug pass (see TRAIN_NORM note in the module docstring)
                hf_full = load_raw_subset(ds, args.download_dir, args.processed_dir,
                                          MAX_TARGET_SAMPLES)
                clean_loader = DataLoader(
                    StableDatasetWrapper(hf_full, clean_transform(ds)),
                    batch_size=256, num_workers=4, shuffle=False)
                feat, labels = extract_features(model, clean_loader, device, cfg["pool"])
                sigma_b, sigma_data, *_ = class_scatter(feat, labels)

                hf_sub = load_raw_subset(ds, args.download_dir, args.processed_dir,
                                         args.n_imgs)
                sigma_aug, n_img = aug_covariance(model, hf_sub, args.k_views,
                                                  cfg["pool"], device, ds)

                lam, V = np.linalg.eigh(sigma_data)          # ascending
                lam, V = lam[::-1], V[:, ::-1]
                aug_on_data = np.einsum("dj,de,ej->j", V, sigma_aug, V)
                valid = lam > lam[0] * 1e-8
                ratio = np.where(valid, aug_on_data / np.maximum(lam, 1e-30), np.nan)
                top100 = ratio[:min(100, valid.sum())]
                frac_augdom = float(lam[valid][ratio[valid] > 1].sum() / lam[valid].sum())

                lam_b, Vb = np.linalg.eigh(sigma_b)
                lam_b, Vb = lam_b[::-1], Vb[:, ::-1]
                m = int(min(max((lam_b > lam_b[0] * 1e-8).sum(), 1), 10))
                Wd = Vb[:, :m]
                r_disc = (np.einsum("dj,de,ej->j", Wd, sigma_aug, Wd)
                          / np.einsum("dj,de,ej->j", Wd, sigma_data, Wd))

                row = dict(encoder=enc, dataset=ds, n_imgs=n_img, k_views=args.k_views,
                           n_classes=len(np.unique(labels)),
                           ratio_disc_mean=float(np.mean(r_disc)),
                           ratio_disc_max=float(np.max(r_disc)),
                           ratio_top100_median=float(np.nanmedian(top100)),
                           frac_datavar_augdom=frac_augdom)
                w.writerow(row)
                f.flush()
                print(f"  {ds:>14}: disc_mean={row['ratio_disc_mean']:.3f} "
                      f"disc_max={row['ratio_disc_max']:.3f} "
                      f"top100_med={row['ratio_top100_median']:.3f} "
                      f"frac_augdom={row['frac_datavar_augdom']:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nNext (local): python eval/new_direction/nd3_verdict.py")


if __name__ == "__main__":
    main()
