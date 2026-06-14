#!/usr/bin/env python
"""
coverage_audit.py — which post-CP checkpoints are present vs expected.

postcp_sweep.py only reads checkpoints that exist on disk, so MISSING ckpts are silently absent
from the sweep CSV (no crash, but invisible). This makes the gaps visible: it cross-checks the
sweep against the expected config grid from results.xlsx (every method×backbone×dataset×size that
has a CP result should have 3 seed ckpts). MAX-label drift (FGVC: ckpt n3334 vs results n3400) is
reconciled via size_canon, so it is NOT reported as missing.

  python eval/coverage_audit.py --sweep eval/outputs/postcp_sweep.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from load_results import load_long, add_size_canon

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "eval/outputs/postcp_sweep.csv"))
    ap.add_argument("--results", default=None)
    ap.add_argument("--encoders", nargs="+", default=["DINOv3", "CLIP", "MAE"])
    args = ap.parse_args()

    sw = pd.read_csv(args.sweep)
    sw = sw[(sw.variant == "pretrained") & (sw.encoder.isin(args.encoders))].copy()
    sw["method_cp"] = sw.method + "-CP"
    sw = add_size_canon(sw, "dataset", "size")

    df = load_long(args.results) if args.results else load_long()
    exp = df[df.Backbone.isin(args.encoders)].dropna(subset=["size"]).copy()
    exp = exp.rename(columns={"Method": "method_cp", "Backbone": "encoder", "dataset_key": "dataset"})
    exp = add_size_canon(exp, "dataset", "size")[
        ["method_cp", "encoder", "dataset", "size_canon"]].drop_duplicates()

    seeds = (sw.groupby(["method_cp", "encoder", "dataset", "size_canon"]).seed
               .nunique().reset_index(name="seeds"))
    m = exp.merge(seeds, on=["method_cp", "encoder", "dataset", "size_canon"], how="left")
    m["seeds"] = m.seeds.fillna(0).astype(int)

    print("=" * 64)
    print("COVERAGE vs results.xlsx  (have≥1 / expected; each config wants 3 seeds)")
    print("=" * 64)
    tab = m.assign(have=m.seeds > 0).pivot_table(
        index="method_cp", columns="encoder", values="have",
        aggfunc=lambda s: f"{int(s.sum())}/{len(s)}")
    print(tab.to_string())
    print(f"\nfully missing (0 ckpts): {(m.seeds==0).sum()}   "
          f"partial (1-2 seeds): {((m.seeds>0)&(m.seeds<3)).sum()}   "
          f"complete (3): {(m.seeds==3).sum()}")

    miss = m[m.seeds == 0].sort_values(["method_cp", "encoder", "dataset", "size_canon"])
    if len(miss):
        print("\n--- FULLY MISSING (no ckpt) ---")
        for r in miss.itertuples():
            print(f"  {r.method_cp:10s} {r.encoder:7s} {r.dataset:14s} {r.size_canon}")
    part = m[(m.seeds > 0) & (m.seeds < 3)].sort_values(["method_cp", "encoder", "dataset"])
    if len(part):
        print("\n--- PARTIAL (1-2 seeds) ---")
        for r in part.itertuples():
            print(f"  {r.method_cp:10s} {r.encoder:7s} {r.dataset:14s} {r.size_canon}  seeds={r.seeds}")

    m.to_csv(ROOT / "eval/outputs/coverage_audit.csv", index=False)
    print("\nsaved eval/outputs/coverage_audit.csv")


if __name__ == "__main__":
    main()
