#!/usr/bin/env python
"""Test 3 — re-run Exp B for the cells whose LeJEPA reference was broken, then re-aggregate.

Exp B compares MAE-CP frozen-LP (the subject) against a healthy LeJEPA-CP reference (b_lej).
The MAE-CP side is fine, but some LeJEPA-CP references were the broken checkpoints
(food101 / octmnist / cars196), so their `b_lej` (and thus recovery_fraction) is wrong.

This test:
  1. finds the affected datasets = LeJEPA datasets among the 60 reran cells that Exp B uses;
  2. (--run-expb) drops the stale LeJEPA rows for those datasets from eval/outputs/exp_b/<ds>.csv
     and re-runs eval/run_exp_b.py --methods LeJEPA on them (re-loads the now-fixed checkpoints);
  3. re-aggregates eval/outputs/sa_lp_recovery.csv from all exp_b/*.csv.

Run (GPU, full):
  python eval/rest/test3_rerun_exp_b.py --run-expb \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp --cache-dir <.../data>
Run (CPU, re-aggregate only, if Exp B already re-run):
  python eval/rest/test3_rerun_exp_b.py --aggregate-only
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

EXP_B_DATASETS = {  # Exp B only evaluates these (MAX_N keys in run_exp_b.py)
    "food101", "octmnist", "plant_village", "organamnist", "galaxy10", "fgvc_aircraft",
    "cars196", "breastmnist", "cub200", "dermamnist", "dtd", "eurosat", "flowers102",
    "oxford_pet", "pathmnist",
}
RECOVERY_CAP = 1.5


def affected_datasets(csv_path):
    """LeJEPA datasets among the 60 reran cells that Exp B actually evaluates."""
    out = set()
    for r in csv.reader(open(csv_path)):
        if not r or r[0].strip() in ("", "ckpt"):
            continue
        ck = r[0].strip()
        if "/LeJEPA/" not in ck:
            continue
        ds = os.path.basename(ck).split("_vit_base")[0]
        if ds in EXP_B_DATASETS:
            out.add(ds)
    return sorted(out)


def drop_lejepa_rows(exp_b_dir, datasets):
    """Remove stale LeJEPA rows for the affected datasets so run_exp_b re-evaluates them."""
    for ds in datasets:
        f = os.path.join(exp_b_dir, f"{ds}.csv")
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        kept = [r for r in rows if r["method"] != "LeJEPA"]
        if len(kept) != len(rows):
            with open(f, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(kept)
            print(f"  dropped {len(rows)-len(kept)} stale LeJEPA rows from {f}")


def aggregate(exp_b_dir, out_csv):
    agg = defaultdict(lambda: {"MAE": {"b": [], "s": []}, "LeJEPA": {"b": []}})
    for f in glob.glob(os.path.join(exp_b_dir, "*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                b, s = float(r["baseline_lp_f1"]), float(r["sa_lp_f1"])
            except (ValueError, KeyError):
                continue
            cell = agg[(r["encoder"], r["dataset"])]
            if r["method"] == "MAE":
                cell["MAE"]["b"].append(b)
                cell["MAE"]["s"].append(s)
            elif r["method"] == "LeJEPA":
                cell["LeJEPA"]["b"].append(b)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "dataset", "b_mae", "s_mae", "b_lej", "degraded", "recovery_fraction"])
        recs = []
        for (enc, ds) in sorted(agg):
            c = agg[(enc, ds)]
            b_mae, s_mae, b_lej = mean(c["MAE"]["b"]), mean(c["MAE"]["s"]), mean(c["LeJEPA"]["b"])
            if b_mae is None or b_lej is None:
                continue
            degraded = b_mae < b_lej
            rec = ""
            if degraded and (b_lej - b_mae) != 0:
                rec = min((s_mae - b_mae) / (b_lej - b_mae), RECOVERY_CAP)
                recs.append(rec)
            w.writerow([enc, ds, round(b_mae, 6), round(s_mae, 6), round(b_lej, 6),
                        degraded, round(rec, 6) if rec != "" else ""])
        print(f"\nre-aggregated -> {out_csv}")
        if recs:
            print(f"overall mean recovery_fraction (degraded cells): {sum(recs)/len(recs):.3f}  (n={len(recs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/outputs/rerun_geometry.csv")
    ap.add_argument("--exp-b-dir", default="eval/outputs/exp_b")
    ap.add_argument("--out", default="eval/outputs/sa_lp_recovery.csv")
    ap.add_argument("--run-expb", action="store_true", help="re-run run_exp_b.py for the affected LeJEPA cells (GPU)")
    ap.add_argument("--aggregate-only", action="store_true", help="skip re-running; only re-aggregate")
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    dss = affected_datasets(args.csv)
    print(f"affected LeJEPA datasets (broken reference, now re-trained): {dss}")

    if args.run_expb and not args.aggregate_only:
        if not args.cache_dir:
            sys.exit("--run-expb needs --cache-dir (the dataset cache dir run_exp_b.py uses)")
        drop_lejepa_rows(args.exp_b_dir, dss)
        os.makedirs(args.exp_b_dir, exist_ok=True)
        for ds in dss:
            cmd = [sys.executable, "eval/run_exp_b.py", "--ckpt-root", args.ckpt_root,
                   "--cache-dir", args.cache_dir, "--methods", "LeJEPA", "--datasets", ds,
                   "--out", os.path.join(args.exp_b_dir, f"{ds}.csv")]
            print("  $", " ".join(cmd))
            subprocess.run(cmd, check=True)

    aggregate(args.exp_b_dir, args.out)


if __name__ == "__main__":
    main()
