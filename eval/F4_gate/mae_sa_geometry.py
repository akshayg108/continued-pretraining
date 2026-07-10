#!/usr/bin/env python
"""
mae_sa_geometry.py — P-C2 (defensive, HIGH priority): is the MAE-encoder F1 inversion a
property of MAE's embedding geometry, or an artifact of the mean-pool READOUT?

Beyond-[cls] (Przewiezlikowski 2024) + our Exp B show MAE's fixed readout under-aggregates
patch information. A reviewer can therefore argue the whole "off-sphere encoder inverts the
law" leg reduces to "wrong pooling". This script settles it: recompute the MAE encoder's
PRE-CP geometry with a learned Selective-Aggregation readout (the Exp-B recipe: depth-1
ABMILP + BatchNorm1d(affine=False) + Adam) instead of mean-pool, then re-run the F1
correlations against the SAME behavioral Δ.

Outcomes (both publishable):
  - If uniformity/overlap → ΔkNN stays inverted/n.s. on SA features  → the inversion is a
    genuine geometry property; the double-exception attribution survives its strongest
    confound (report as an ablation).
  - If the sphere law REAPPEARS on SA features → the "off-sphere encoder" leg collapses into
    "readout failure"; the gate must be restated as objective-gating + readout, and C1's
    encoder story softens.

Per dataset: train the SA head on the dataset's <=5000 labeled train subset (labels used for
the probe only, matching Exp B), then pool BOTH the target subset and ImageNet-val through the
SAME trained head, and compute l2_norm_cv / uniformity_t2 / mmd_rbf / neighbor_overlap_k50 /
cosine_dist_centroid on the SA embeddings.

GPU. Run (cluster):
  python eval/adjudicate/mae_sa_geometry.py --imagenet-dir <dir> --download-dir <raw> \
      --processed-dir <arrow> --out eval/outputs/geometry_mae_sa.csv
Then (CPU):
  python eval/f1_position/correlate.py --geometry eval/outputs/geometry_mae_sa.csv
"""
import argparse
import csv
import sys
from pathlib import Path as _P

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))
from geometry_metrics import (TARGET_DATASETS, load_target_dataset, load_imagenet_val,  # noqa: E402
                              l2_norm_stats, wang_isola_uniformity, mmd_rbf_components,
                              neighbor_overlap, cosine_distance_centroids)

from stable_cp.evaluation.abmilp import ABMILPHead  # noqa: E402

ROOT = _P(__file__).resolve().parent.parent.parent
TIMM_ID, POOL = "vit_base_patch16_224.mae", "mean"

FIELDNAMES = ["encoder", "dataset", "n_samples", "embed_dim",
              "l2_norm_mean", "l2_norm_std", "l2_norm_cv",
              "uniformity_t2", "uniformity_t2_raw", "uniformity_at_gamma",
              "cosine_dist_centroid", "mmd_rbf",
              "mmd_m_pp_target", "mmd_m_qq_imagenet", "mmd_m_pq_cross", "mmd_gamma",
              "neighbor_overlap_k20", "neighbor_overlap_k50"]


@torch.no_grad()
def extract_tokens(model, loader, device):
    toks, labs = [], []
    model.eval()
    for batch in loader:
        x, y = batch[0], batch[1]
        feat = model.forward_features(x.to(device))
        toks.append(feat.cpu())
        labs.append(y if isinstance(y, torch.Tensor) else torch.as_tensor(np.array(y)))
    return torch.cat(toks), torch.cat(labs).ravel().numpy()


def train_sa_head(tokens, labels, device, lr=1e-3, batch_size=512,
                  min_epochs=150, min_steps=10000):
    """Exp-B recipe (run_exp_b.sa_lp_bn), returning the trained SA module."""
    num_patches = tokens.shape[1] - 1
    dim = tokens.shape[-1]
    num_classes = int(len(np.unique(labels)))
    sa = ABMILPHead(dim=dim, self_attention_apply_to="none", depth=1, cond="none",
                    content="patch", num_patches=num_patches).to(device)
    bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6).to(device)
    clf = nn.Linear(dim, num_classes).to(device)
    trl = torch.from_numpy(labels).long()
    opt = torch.optim.Adam(list(sa.parameters()) + list(clf.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(tokens)
    spe = max(n // batch_size, 1)
    epochs = max(min_epochs, (min_steps + spe - 1) // spe)
    sa.train(); bn.train(); clf.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(spe):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            if len(idx) < 2:
                continue
            x = tokens[idx].to(device).float()
            y = trl[idx].to(device)
            loss = crit(clf(bn(sa(x))), y)
            opt.zero_grad(); loss.backward(); opt.step()
    sa.eval()
    return sa


@torch.no_grad()
def sa_pool(sa, tokens, device, batch_size=512):
    out = []
    for i in range(0, len(tokens), batch_size):
        out.append(sa(tokens[i:i + batch_size].to(device).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-dir", type=str, default=str(ROOT / "eval/data/imagenet_val"))
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", type=str, default=str(ROOT / "eval/outputs/geometry_mae_sa.csv"))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _P(args.out).parent.mkdir(parents=True, exist_ok=True)

    import timm
    model = timm.create_model(TIMM_ID, pretrained=True, num_classes=0).eval().to(device)
    in_tokens, _ = extract_tokens(model, load_imagenet_val(
        args.imagenet_dir, n_samples=args.imagenet_samples), device)

    rows = []
    for ds in args.datasets:
        print(f"--- {ds} ---")
        try:
            loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        tokens, labels = extract_tokens(model, loader, device)
        sa = train_sa_head(tokens, labels, device)
        feat = sa_pool(sa, tokens, device)
        feat_in = sa_pool(sa, in_tokens, device)   # ImageNet through the SAME head
        nm, ns, nc = l2_norm_stats(feat)
        row = {"encoder": "MAE_SA", "dataset": ds, "n_samples": len(feat),
               "embed_dim": feat.shape[1], "l2_norm_mean": nm, "l2_norm_std": ns,
               "l2_norm_cv": nc,
               "uniformity_t2": wang_isola_uniformity(feat),
               "uniformity_t2_raw": wang_isola_uniformity(feat, l2_normalize=False)}
        comp = mmd_rbf_components(feat, feat_in)
        row.update(comp)
        row["uniformity_at_gamma"] = wang_isola_uniformity(feat, t=comp["mmd_gamma"])
        row["cosine_dist_centroid"] = cosine_distance_centroids(feat, feat_in)
        row["neighbor_overlap_k20"] = neighbor_overlap(feat, feat_in, k=20)
        row["neighbor_overlap_k50"] = neighbor_overlap(feat, feat_in, k=50)
        print(f"  CV={nc:.3f} unif={row['uniformity_t2']:.3f} "
              f"ov50={row['neighbor_overlap_k50']:.4f}")
        rows.append(row)
        del sa
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"Saved {len(rows)} rows -> {args.out}")
    print("Next (CPU): join to MAE behavioral Δ and compare F1 correlations mean-pool vs SA "
          "(eval/adjudicate/correlate_second_axis.py handles it when the CSV exists; "
          "rename encoder MAE_SA -> MAE for correlate.py compatibility if run standalone).")


if __name__ == "__main__":
    main()
