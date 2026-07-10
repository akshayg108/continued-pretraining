#!/usr/bin/env python
"""
nd3_verdict.py — ND3 (CPU adjudicator): is DTD augmentation-dominated along its
class-discriminative directions?

Pre-registered readout (declared 2026-07-10, before nd3_augvar.csv existed):
  ND3-1 PRIMARY: rank the 15 datasets per encoder by ratio_disc_mean (aug/data variance
        ratio along top between-class directions). PASS if DTD is in the TOP-2 on at least
        2 of the 3 sphere encoders (DINOv3, CLIP, SigLIP). That is Jing Thm-1's texture
        prediction: the suppression force hits exactly the directions DTD needs.
  ND3-2 secondary (descriptive): DTD's rank on frac_datavar_augdom (overall
        augmentation-domination), to check the effect is direction-specific rather than
        a global property of DTD images.
Scope caveat (carried from the paper): Thm 1 is proven for linear networks under InfoNCE
with additive augmentation noise — for ViTs this is a qualitative criterion, and the
DTD link is OUR hypothesis, not Jing et al.'s.

Input: eval/outputs/nd3_augvar.csv (concat of shards if run as an array).
Run (local): python eval/new_direction/nd3_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import pandas as pd

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
SPHERE = ["DINOv3", "CLIP", "SigLIP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augvar", default=str(OUT / "nd3_augvar.csv"))
    args = ap.parse_args()
    if not _P(args.augvar).exists():
        sys.exit(f"MISSING {args.augvar} — run the ND3 GPU pass first "
                 f"(run/slurm/new_direction/nd3_augvar.sh), concat shards if arrayed:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd3_augvar_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd3_augvar.csv', index=False)\"")
    df = pd.read_csv(args.augvar)

    print("=" * 88)
    print("ND3 — aug/data variance ratio along discriminative directions (rank 1 = highest)")
    print("=" * 88)
    hits = 0
    for enc in df.encoder.unique():
        g = df[df.encoder == enc].copy()
        g["rank_disc"] = g.ratio_disc_mean.rank(ascending=False).astype(int)
        g["rank_global"] = g.frac_datavar_augdom.rank(ascending=False).astype(int)
        dtd = g[g.dataset == "dtd"]
        if dtd.empty:
            print(f"\n{enc}: no dtd row yet ({len(g)} datasets present)")
            continue
        r_disc = int(dtd.rank_disc.iloc[0])
        r_glob = int(dtd.rank_global.iloc[0])
        if enc in SPHERE and r_disc <= 2:
            hits += 1
        print(f"\n{enc} (n={len(g)}): DTD rank on ratio_disc_mean = {r_disc}, "
              f"on frac_datavar_augdom = {r_glob}")
        print(g.sort_values("rank_disc")[
            ["dataset", "ratio_disc_mean", "ratio_disc_max",
             "frac_datavar_augdom", "rank_disc"]].head(5).to_string(index=False))

    n_sphere = df[df.encoder.isin(SPHERE)].encoder.nunique()
    print(f"\nND3-1 PRIMARY: DTD in top-2 discriminative-direction ratio on {hits}/"
          f"{n_sphere} sphere encoders -> {'PASS' if hits >= 2 else 'FAIL'} "
          f"(pre-registered PASS >= 2/3)")
    print("ND3-2: compare rank_disc vs rank_global above — direction-specific means "
          "rank_disc high while rank_global unremarkable.")


if __name__ == "__main__":
    main()
