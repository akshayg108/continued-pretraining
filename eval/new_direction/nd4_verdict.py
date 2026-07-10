#!/usr/bin/env python
"""
nd4_verdict.py — ND4 (CPU adjudicator): projector-as-buffer and the encoder gate.

Jing et al. 2022, Fig. 7b: with a projector, dimensional collapse is confined to the
post-projector space and the encoder-output spectrum does not collapse. Candidate gate
mechanism (papers/new_direction/NEW_DIRECTION.md map, A2): the CP head buffers spectral
reshaping away from the backbone; where buffering fails, the backbone absorbs it.

Quantities (normalized effective ranks so 768-d backbones and 128-d/N-d heads compare):
  nr_bb      = rankme_bb / min(emb_dim, n_samples)      backbone spectral occupancy, post-CP
  nr_head    = rankme_head / min(head_dim, n_samples)   head-output spectral occupancy
  buffer_gap = nr_bb - nr_head       > 0 = head space MORE collapsed than backbone (buffering)
  bb_shift   = |rankme_bb - rankme_pre| / rankme_pre    relative backbone spectral change
               (pre from nd1_precp_spectral.csv, same features protocol)

Pre-registered readouts (declared 2026-07-10, before nd4 data existed):
  ND4-1 BUFFER: buffer_gap > 0 in a clear majority of projector-method cells
        (LeJEPA/SimCLR) on every encoder — Jing's confinement, on our checkpoints.
  ND4-2 GATE LINK (exploratory, n=4 encoders): encoder ordering of median bb_shift has
        MAE HIGHEST (its backbone absorbs the reshaping the spheres' heads buffer away),
        and median buffer_gap ordering tracks the L12 gate ordering
        (DINOv3 .83 > CLIP .76 > SigLIP .55 > MAE .04). Point estimate — n=4 cannot
        support more than a directional statement.
  ND4-3 (descriptive): DIET (classifier head, directional logits) vs the two projector
        methods — is buffering projector-specific?

Inputs: eval/outputs/nd4_projector_spectra.csv (concat of shards), nd1_precp_spectral.csv.
Run (local): python eval/new_direction/nd4_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
GATE = {"DINOv3": 0.83, "CLIP": 0.76, "SigLIP": 0.55, "MAE": 0.04}
PROJ_METHODS = ["LeJEPA", "SimCLR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default=str(OUT / "nd4_projector_spectra.csv"))
    ap.add_argument("--pre", default=str(OUT / "nd1_precp_spectral.csv"))
    args = ap.parse_args()
    if not _P(args.spectra).exists():
        sys.exit(f"MISSING {args.spectra} — run the ND4 GPU pass first "
                 f"(run/slurm/new_direction/nd4_projector.sh), then concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd4_projector_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd4_projector_spectra.csv', index=False)\"")
    df = pd.read_csv(args.spectra)
    pre = pd.read_csv(args.pre)[["encoder", "dataset", "rankme"]]
    df = df.merge(pre.rename(columns={"rankme": "rankme_pre"}),
                  on=["encoder", "dataset"], how="left")
    if df.rankme_pre.isna().any():
        miss = df[df.rankme_pre.isna()][["encoder", "dataset"]].drop_duplicates()
        print(f"WARN: {len(miss)} (encoder, dataset) cells missing the nd1 pre-CP baseline "
              f"— bb_shift is NaN there:\n{miss.to_string(index=False)}")

    df["nr_bb"] = df.rankme_bb / np.minimum(df.emb_dim, df.n_samples)
    df["nr_head"] = df.rankme_head / np.minimum(df.head_dim, df.n_samples)
    df["buffer_gap"] = df.nr_bb - df.nr_head
    df["bb_shift"] = (df.rankme_bb - df.rankme_pre).abs() / df.rankme_pre
    df.to_csv(OUT / "nd4_buffer.csv", index=False)

    print("=" * 88)
    print(f"ND4 — {len(df)} MAX checkpoints "
          f"({df.encoder.nunique()} encoders x {df.method.nunique()} methods)")
    print("=" * 88)

    proj = df[df.method.isin(PROJ_METHODS)]
    print("\nND4-1 BUFFER  buffer_gap > 0 (projector methods), per encoder:")
    all_maj = True
    for e, g in proj.groupby("encoder"):
        n_pos = (g.buffer_gap > 0).sum()
        maj = n_pos > 0.5 * len(g)
        all_maj &= maj
        print(f"  {e:>7}: {n_pos}/{len(g)} cells positive, median gap "
              f"{g.buffer_gap.median():+.3f} {'OK' if maj else 'MISS'}")
    print(f"  ND4-1 verdict: {'PASS' if all_maj else 'FAIL'} "
          f"(pre-registered: majority positive on every encoder)")

    print("\nND4-2 GATE LINK (exploratory, n=4):")
    agg = (proj.groupby("encoder")[["bb_shift", "buffer_gap"]].median()
           .reindex(["DINOv3", "CLIP", "SigLIP", "MAE"]).dropna())
    agg["gate"] = [GATE[e] for e in agg.index]
    print(agg.round(3).to_string())
    if "MAE" in agg.index and len(agg) >= 2:
        mae_top = agg.bb_shift.idxmax() == "MAE"
        print(f"  MAE backbone shift highest: {'YES' if mae_top else 'NO'} "
              f"(prediction: YES — no projector at MAE PRE-training, backbone absorbs)")
    if len(agg) >= 3:
        r = spearmanr(agg.buffer_gap, agg.gate).correlation
        print(f"  rho(median buffer_gap, L12 gate) = {r:+.3f}  (n={len(agg)}, "
              f"point estimate only)")

    print("\nND4-3 method contrast (median buffer_gap per method x encoder):")
    print(df.pivot_table(index="method", columns="encoder", values="buffer_gap",
                         aggfunc="median").round(3).to_string())
    print(f"\nwrote -> {OUT / 'nd4_buffer.csv'}")


if __name__ == "__main__":
    main()
