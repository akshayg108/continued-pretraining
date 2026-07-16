#!/usr/bin/env python
"""
nd9_capture.py — ND9 (GPU pass): task CAPTURE, centered KTA and the spectral
accessibility curve on PRE-CP frozen features.

Motivation (theory-unification 2026-07-15, adopted target #1): nd6's cC(rho) is a SHARE —
cumulative label power normalized by the label power the span captures at all. It answers
"how well is the captured task signal placed?" and discards "how much task signal is
captured?". Two encoders can share the whole cC curve while capturing different totals;
the SigLIP-2 alignment deficit (ND6-2, 15/15 vs CLIP on caucC_log) may therefore be a
capture deficit, a conditional-placement deficit, or both. ND9 measures the missing
factor and the combined readout-weighted quantity:

  capture_raw/cen   sum_i ||u_i^T Y||^2 / ||Y||^2 over the feature span (== in-sample LS R^2)
  kta_cen           centered kernel-target alignment (linear CKA features vs labels)
  acc_r{...}        A(kappa) = sum_i lam_i/(lam_i+kappa) p_cen_i / ||Yc||^2 at
                    kappa = r * mean(lam), r in KAPPA_RELS (r->0 recovers capture;
                    large r weights the leading modes — a ridge-readout-matched
                    accessibility). acc_auc_log = mean of A over the log-r grid.
  cC_K_check        in-pass recomputation of nd6's cC_K (ND9-0 protocol gate).

Protocol identical to nd6_alignment.py (same extraction, same uncentered-second-moment
eigenbasis, same class-centered Y convention). Adjudication: nd9_verdict.py.

Cluster (one array task per dataset):
  python eval/new_direction/nd9_capture.py --datasets <ds> \
      --download-dir <raw> --processed-dir <arrow> \
      --output eval/outputs/nd9_capture_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                    # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))   # eval/utils

from nd9_task_operator import _svd_powers, accessibility_curve, centered_kta

ROOT = Path(__file__).resolve().parent.parent.parent
KAPPA_RELS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
KAPPA_COLS = [f"acc_r{r:g}".replace("-", "m").replace(".", "p") for r in KAPPA_RELS]
FIELDS = (["encoder", "dataset", "n_samples", "embed_dim", "n_classes",
           "capture_raw", "capture_cen", "kta_cen"]
          + KAPPA_COLS + ["acc_auc_log", "cC_K_check"])


def capture_row(feat, labels):
    """All ND9 statistics for one (encoder, dataset) cell."""
    _, lam, p_raw, p_cen, y2_raw, y2_cen, n_cls = _svd_powers(feat, labels)
    row = {"n_samples": len(feat), "embed_dim": feat.shape[1], "n_classes": n_cls,
           "capture_raw": float(p_raw.sum() / y2_raw),
           "capture_cen": float(p_cen.sum() / y2_cen),
           "kta_cen": centered_kta(feat, labels)}
    kappas = np.array(KAPPA_RELS) * lam.mean()
    A = accessibility_curve(lam, p_cen, y2_cen, kappas)
    row.update({c: float(a) for c, a in zip(KAPPA_COLS, A)})
    grid = np.logspace(-6, 1, 30) * lam.mean()
    row["acc_auc_log"] = float(accessibility_curve(lam, p_cen, y2_cen, grid).mean())
    C = np.cumsum(p_cen) / max(p_cen.sum(), 1e-30)
    row["cC_K_check"] = float(C[min(n_cls, len(C)) - 1])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/nd9_capture.csv"))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    import timm
    import torch
    from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                                  extract_features)
    encoders = args.encoders or list(ENCODERS.keys())
    datasets = args.datasets or TARGET_DATASETS
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["encoder"], r["dataset"]) for r in csv.DictReader(f)}
        print(f"Resume: {len(done)} rows present")

    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        for enc in encoders:
            todo = [d for d in datasets if (enc, d) not in done]
            if not todo:
                continue
            cfg = ENCODERS[enc]
            print(f"\n===== {enc} ({cfg['timm_id']}) — {len(todo)} datasets")
            model = timm.create_model(cfg["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            for ds in todo:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
                feat, labels = extract_features(model, loader, device, cfg["pool"])
                row = {"encoder": enc, "dataset": ds}
                row.update(capture_row(feat, labels))
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                print(f"  {ds:>14}: capture={row['capture_cen']:.3f} "
                      f"kta={row['kta_cen']:.4f} acc_r1={row['acc_r1']:.4f} "
                      f"cC_K_check={row['cC_K_check']:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nNext (local): python eval/new_direction/nd9_verdict.py")


if __name__ == "__main__":
    main()
