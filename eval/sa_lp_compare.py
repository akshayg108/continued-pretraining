#!/usr/bin/env python
"""
sa_lp_compare.py — Exp B (F2.3): Selective-Aggregation LP vs [cls] LP recovery.

The SA-LP numbers themselves come from the CP repo's evaluation run with the `--aggregation`
flag on each post-CP checkpoint (see plan.md "Selective Aggregation LP 实验协议"): it reports
  pre_linear_f1   — healthy pre-CP [cls] linear probe
  post_linear_f1  — degraded post-CP [cls] linear probe
  post_sa_lp_f1   — post-CP Selective-Aggregation (AbMILP depth-1) linear probe
Put those into `--input` (CSV: config,pre_linear_f1,post_linear_f1,post_sa_lp_f1); this script
computes the recovery fraction and the aggregation-failure-vs-information-loss verdict.

  recovery = (post_sa_lp_f1 − post_linear_f1) / (pre_linear_f1 − post_linear_f1)
    ≈ 1  → AGGREGATION FAILURE (info is in patch tokens; [cls] aggregation broke)
    ≈ 0  → INFORMATION LOSS    (patch level also degraded)
    mid  → BOTH (recovery quantifies the split)

Run anywhere:
  python eval/sa_lp_compare.py --input eval/outputs/sa_lp_input.csv \
      --out eval/outputs/sa_lp.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def verdict(recovery, degraded):
    if not degraded:
        return "no degradation (config not a degraded case)"
    if recovery >= 0.75:
        return "AGGREGATION FAILURE"
    if recovery <= 0.25:
        return "INFORMATION LOSS"
    return "BOTH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "eval/outputs/sa_lp_input.csv"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/sa_lp.csv"))
    ap.add_argument("--degrade-thresh", type=float, default=0.05,
                    help="min (pre−post) [cls] LP drop to count as a degraded case")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = []
    for r in df.itertuples():
        drop = r.pre_linear_f1 - r.post_linear_f1
        degraded = drop >= args.degrade_thresh
        rec = (r.post_sa_lp_f1 - r.post_linear_f1) / drop if drop > 1e-9 else float("nan")
        out.append(dict(config=r.config,
                        pre_linear_f1=round(r.pre_linear_f1, 4),
                        post_linear_f1=round(r.post_linear_f1, 4),
                        post_sa_lp_f1=round(r.post_sa_lp_f1, 4),
                        cls_drop=round(drop, 4),
                        recovery=round(rec, 3) if rec == rec else "n/a",
                        verdict=verdict(rec, degraded)))
    res = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
