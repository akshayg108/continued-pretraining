#!/usr/bin/env python
"""
layerwise_geometry.py — Exp I (pre-CP side): every block of every encoder is a "virtual
encoder". 4 encoders x 12 blocks x 15 datasets -> per-layer geometry + per-layer internal kNN.

Purpose (eval/DESIGN_spectrum_transport.md, Design 2): the encoder axis of the gate has only
~2 effective points (sphere trio vs MAE). Layers multiply it to ~48 geometric states, enough
to test whether the position law's strength varies CONTINUOUSLY with layer geometry (L2) and
whether any MAE middle layer locally recovers the law (L3).

Per (encoder, dataset, layer) on the SAME <=5000 stratified subset as geometry_metrics.py:
  uniformity_t2, l2_norm_cv, rankme (normalized feats), numerical_rank, cdnv, center_margin,
  knn_internal.

Layer readout protocol (FIXED; disclosed in the paper):
  - block outputs (pre final-norm), pooled on the fly by forward hooks;
  - DINOv3/CLIP: cls token (token 0); MAE: mean over tokens[num_prefix_tokens:];
    SigLIP: mean over ALL tokens (no cls; the MAP head is final-layer-only).
Internal kNN protocol (NOT the production kNN; consistent across layers/pre/post so the
per-layer delta is internally valid): stratified 80/20 split (RandomState(0)), cosine k=20,
macro-F1.

No ImageNet needed. GPU. Cluster:
  python eval/layerwise_geometry.py --datasets <ds> --download-dir <raw> \
      --processed-dir <arrow> --output eval/outputs/layerwise_pre_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                              extract_features, wang_isola_uniformity, l2_norm_stats)
from geometry_class import class_manifold_stats, rankme, numerical_rank

ROOT = Path(__file__).resolve().parent.parent

# cls = token 0; mean = tokens after the prefix (cls/registers); SigLIP has no prefix tokens,
# so "mean" is the all-token mean there.
LAYER_READOUT = {"DINOv3": "cls", "CLIP": "cls", "MAE": "mean", "SigLIP": "mean"}

FIELDS = ["encoder", "dataset", "layer", "n_samples",
          "uniformity_t2", "l2_norm_cv", "rankme", "numerical_rank",
          "cdnv", "center_margin", "knn_internal"]


class LayerTap:
    """Forward hooks on model.blocks that pool each block's token output on the fly."""

    def __init__(self, model, readout):
        self.model, self.readout = model, readout
        self.n_prefix = getattr(model, "num_prefix_tokens", 1)
        self.buf = None
        self.handles = [blk.register_forward_hook(self._make_hook(i))
                        for i, blk in enumerate(model.blocks)]

    def _make_hook(self, i):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            pooled = t[:, 0] if self.readout == "cls" else t[:, self.n_prefix:].mean(1)
            self.buf[i].append(pooled.detach().float().cpu().numpy())
        return hook

    def collect(self, loader, device):
        """One forward pass; returns ({layer_1based: (N, d)}, labels)."""
        self.buf = {i: [] for i in range(len(self.model.blocks))}
        labels = []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                self.model.forward_features(x.to(device))
                labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        feats = {i + 1: np.vstack(chunks) for i, chunks in self.buf.items()}
        return feats, np.concatenate(labels).ravel()

    def close(self):
        for h in self.handles:
            h.remove()


def internal_knn(feats, labels, k=20, seed=0):
    """Stratified 80/20 cosine-kNN macro-F1 (the Exp-I internal protocol)."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score
    labels = np.asarray(labels).ravel()
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return np.nan
    tr, te = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels,
                              random_state=seed)
    if len(tr) < k:
        return np.nan
    f = normalize(feats)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(f[tr], labels[tr])
    return float(f1_score(labels[te], clf.predict(f[te]), average="macro"))


def layer_row(feat, labels):
    fn = normalize(feat)
    cm = class_manifold_stats(feat, labels)
    _, _, cv = l2_norm_stats(feat)
    return {"uniformity_t2": round(wang_isola_uniformity(feat, t=2.0), 5),
            "l2_norm_cv": round(cv, 5),
            "rankme": round(rankme(fn), 3),
            "numerical_rank": numerical_rank(fn),
            "cdnv": round(cm["cdnv"], 5),
            "center_margin": round(cm["center_margin"], 5),
            "knn_internal": round(internal_knn(feat, labels), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/layerwise_pre.csv"))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import timm
    rows = []
    for enc_name in args.encoders:
        cfg = ENCODERS[enc_name]
        print(f"\n{'='*60}\nEncoder: {enc_name} (layer readout: {LAYER_READOUT[enc_name]})\n{'='*60}")
        model = timm.create_model(cfg["timm_id"], pretrained=True, num_classes=0).eval().to(device)
        tap = LayerTap(model, LAYER_READOUT[enc_name])
        for ds in args.datasets:
            print(f"--- {ds} ---")
            try:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            feats, labels = tap.collect(loader, device)
            for layer in sorted(feats):
                row = {"encoder": enc_name, "dataset": ds, "layer": layer,
                       "n_samples": len(labels)}
                row.update(layer_row(feats[layer], labels))
                rows.append(row)
            last = rows[-1]
            print(f"  L12: unif={last['uniformity_t2']} rankme={last['rankme']} "
                  f"knn={last['knn_internal']}")
        tap.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.output}")
    print("Next (after post-CP shards land): python eval/adjudicate/layerwise_law.py")


if __name__ == "__main__":
    main()
