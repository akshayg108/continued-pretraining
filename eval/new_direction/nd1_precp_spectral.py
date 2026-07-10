#!/usr/bin/env python
"""
nd1_precp_spectral.py — ND1 (GPU pass): spectral geometry of the PRE-CP public encoders.

Computes rankme / alpha-ReQ / coherence / VCI (+ uniformity_t2 as the anchor to the
existing level-channel measurement) for all 4 encoders x 15 datasets, from the same
features protocol as geometry_15.csv (timm pretrained, <=5000 stratified samples,
eval transform, per-encoder pool strategy).

Purpose (papers/new_direction/NEW_DIRECTION.md §H-3, A4): the SigLIP-2 level-channel
anomaly is rho(uniformity_t2, pre-CP kNN) = +0.171 while DINOv3/CLIP are negative
(findings/FINDINGS_step9 addendum). Tsitsulin et al. 2023 show most spectral scalars
can flip correlation sign from the metric-by-condition interaction alone, and that
coherence is the one sign-stable axis they tested. This pass provides the per-metric
level channels; eval/new_direction/nd1_verdict.py (CPU, local) joins pre-CP kNN and
adjudicates. Doubles as the pre-CP baseline for ND2's deltas.

Resumable: (encoder, dataset) rows already in --output are skipped.

Cluster:  python eval/new_direction/nd1_precp_spectral.py \
    --download-dir <raw> --processed-dir <arrow> \
    --output eval/outputs/nd1_precp_spectral.csv
"""
import argparse
import csv
from pathlib import Path

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                    # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))   # eval/utils

import numpy as np
import timm
import torch

from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                              extract_features, wang_isola_uniformity)
from spectral_metrics import spectral_row

ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = ["encoder", "dataset", "n_samples", "embed_dim",
          "rankme", "alpha", "alpha_r2", "coherence_mu", "coherence_mu99",
          "vci", "vci_rank_b", "uniformity_t2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/nd1_precp_spectral.csv"))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["encoder"], r["dataset"]) for r in csv.DictReader(f)}
        print(f"Resume: {len(done)} (encoder, dataset) rows already present")

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
            print(f"\n===== {enc} ({cfg['timm_id']}, pool={cfg['pool']}) — {len(todo)} datasets")
            model = timm.create_model(cfg["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            for ds in todo:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
                feat, labels = extract_features(model, loader, device, cfg["pool"])
                row = {"encoder": enc, "dataset": ds,
                       "n_samples": len(feat), "embed_dim": feat.shape[1],
                       "uniformity_t2": wang_isola_uniformity(feat)}
                row.update(spectral_row(feat, labels))
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                print(f"  {ds:>14}: rankme={row['rankme']:.1f} alpha={row['alpha']:.3f} "
                      f"(R2 {row['alpha_r2']:.2f}) mu={row['coherence_mu']:.1f} "
                      f"vci={row['vci']:.3f} unif={row['uniformity_t2']:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nNext (local): python eval/new_direction/nd1_verdict.py")


if __name__ == "__main__":
    main()
