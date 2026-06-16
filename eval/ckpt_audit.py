#!/usr/bin/env python
"""
ckpt_audit.py — NEW. Which CP checkpoints exist ON DISK vs the expected grid.

Unlike coverage_audit.py (which reads the post-CP geometry SWEEP csv, so it only sees ckpts a
sweep already loaded), this scans the checkpoint directory directly and DOES NOT LOAD any
checkpoint — it only parses (method, encoder, dataset, size, seed) from each ckpt's path +
filename. So it answers "which ckpts am I still missing" in seconds, even when no sweep has run
(e.g. to track the rest/ jobs). Depends only on pandas + load_results (no torch / stable_datasets).

The parse helpers mirror postcp_sweep.{encoder_from_name,parse_ckpt,discover}; they are replicated
here (not imported) so this stays a pure, fast file check with no heavy imports.

SIZE DRIFT RECONCILIATION (important): results.xlsx records *nominal* target sizes that do not
always equal the *actual* sampled count baked into the ckpt filename:
  - MAX drift   : fgvc_aircraft results n3400 vs ckpt n3334; flowers102 MAX 1020; etc.
  - small drift : flowers102 has a nominal "100" row AND the real smallest "102" (= num classes);
                  food101 has "100" + real "101". The nominal "100" has NO ckpt — it is a phantom.
We therefore DO NOT match sizes exactly. Per dataset we bucket all sizes (results ∪ disk) that lie
within a relative tolerance of each other into one canonical bucket, so 100↔102, 100↔101 and
3400↔3334 collapse, while genuine buckets (≥2× apart: 100/500/1000/MAX) stay separate. A config is
"missing" only if its bucket has zero ckpts on disk.

Run on the cluster (where the ckpts live):
  python eval/ckpt_audit.py --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from load_results import load_long, DATASET_KEY

ROOT = Path(__file__).resolve().parent.parent
METHODS = {"DIET", "LeJEPA", "MAE", "SimCLR"}


def encoder_from_name(fname):
    if "clip" in fname:
        return "CLIP"
    if "dinov3" in fname:
        return "DINOv3"
    if ".mae" in fname:
        return "MAE"
    return "RANDOM"


def parse_ckpt(path, ckpt_root):
    """Path -> {method, variant, encoder, dataset, size, seed} or None (pure string parse)."""
    p = Path(path)
    try:
        rel = p.relative_to(ckpt_root)
    except ValueError:
        return None
    if len(rel.parts) < 3 or rel.parts[0] not in METHODS:
        return None
    method, variant, dataset_folder = rel.parts[0], rel.parts[1], rel.parts[2]
    m = re.search(r"_n(\d+)_s(\d+)$", p.stem)
    if not m:
        return None
    return dict(method=method, variant=variant, encoder=encoder_from_name(p.stem),
                dataset=DATASET_KEY.get(dataset_folder, dataset_folder.lower()),
                size=int(m.group(1)), seed=int(m.group(2)), ckpt=str(p))


def discover(ckpt_root):
    """All *.ckpt whose immediate parent dir is named 'cp' (excludes sft_post)."""
    out = []
    for f in Path(ckpt_root).rglob("*.ckpt"):
        if f.parent.name == "cp":
            cfg = parse_ckpt(f, ckpt_root)
            if cfg:
                out.append(cfg)
    return out


def canon_map(sizes, rel_tol):
    """size(int) -> canonical label. Sizes within rel_tol of each other collapse into one bucket
    (absorbs MAX drift 3400↔3334 and small-size phantoms 100↔102); the largest bucket -> 'MAX'."""
    xs = sorted({int(x) for x in sizes})
    buckets = []
    for x in xs:
        if buckets and x <= buckets[-1][-1] * (1 + rel_tol):
            buckets[-1].append(x)
        else:
            buckets.append([x])
    m = {}
    for i, b in enumerate(buckets):
        label = "MAX" if i == len(buckets) - 1 else str(max(b))
        for v in b:
            m[v] = label
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp",
                    help="dir containing <METHOD>/ subdirs (…/ckpts/cp)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--encoders", nargs="+", default=["DINOv3", "CLIP", "MAE"])
    ap.add_argument("--seeds-expected", type=int, default=3)
    ap.add_argument("--size-rel-tol", type=float, default=0.05,
                    help="sizes within this relative tolerance collapse to one bucket (drift)")
    args = ap.parse_args()

    # ---- expected grid from results.xlsx (authoritative) ----
    df = load_long(args.results) if args.results else load_long()
    exp = df[df.Backbone.isin(args.encoders)].dropna(subset=["size"]).copy()
    exp = exp.rename(columns={"Method": "method_cp", "Backbone": "encoder",
                              "dataset_key": "dataset"})

    # ---- present on disk (parse only, no checkpoint load) ----
    cfgs = discover(args.ckpt_root)
    if not cfgs:
        print(f"No checkpoints found under {args.ckpt_root}")
        return
    disk = pd.DataFrame(cfgs)
    disk = disk[(disk.variant == "pretrained") & (disk.encoder.isin(args.encoders))].copy()
    disk["method_cp"] = disk.method + "-CP"

    # ---- per-dataset size buckets from the UNION (so expected & disk agree) ----
    exp_sz = exp.groupby("dataset")["size"].apply(lambda s: {int(x) for x in s})
    disk_sz = disk.groupby("dataset")["size"].apply(lambda s: {int(x) for x in s})
    canon = {ds: canon_map(exp_sz.get(ds, set()) | disk_sz.get(ds, set()), args.size_rel_tol)
             for ds in set(exp_sz.index) | set(disk_sz.index)}

    def lab(row):
        return canon.get(row["dataset"], {}).get(int(row["size"]), str(int(row["size"])))
    exp["size_canon"] = exp.apply(lab, axis=1)
    disk["size_canon"] = disk.apply(lab, axis=1)
    exp_grid = exp[["method_cp", "encoder", "dataset", "size_canon"]].drop_duplicates()

    seeds = (disk.groupby(["method_cp", "encoder", "dataset", "size_canon"]).seed
                 .nunique().reset_index(name="seeds"))
    m = exp_grid.merge(seeds, on=["method_cp", "encoder", "dataset", "size_canon"], how="left")
    m["seeds"] = m.seeds.fillna(0).astype(int)
    need = args.seeds_expected

    print("=" * 66)
    print(f"CKPTS ON DISK vs results.xlsx grid (each config wants {need} seeds)")
    print(f"root: {args.ckpt_root}   |   {len(disk)} pretrained ckpts found on disk")
    print("=" * 66)
    tab = m.assign(have=m.seeds > 0).pivot_table(
        index="method_cp", columns="encoder", values="have",
        aggfunc=lambda s: f"{int(s.sum())}/{len(s)}")
    print(tab.to_string())
    print(f"\nfully missing (0): {(m.seeds==0).sum()}   "
          f"partial (1-{need-1}): {((m.seeds>0)&(m.seeds<need)).sum()}   "
          f"complete (>={need}): {(m.seeds>=need).sum()}")

    miss = m[m.seeds == 0].sort_values(["method_cp", "encoder", "dataset", "size_canon"])
    if len(miss):
        print("\n--- FULLY MISSING (no ckpt on disk) ---")
        for r in miss.itertuples():
            print(f"  {r.method_cp:10s} {r.encoder:7s} {r.dataset:14s} {r.size_canon}")
    part = m[(m.seeds > 0) & (m.seeds < need)].sort_values(["method_cp", "encoder", "dataset"])
    if len(part):
        print(f"\n--- PARTIAL (<{need} seeds) ---")
        for r in part.itertuples():
            print(f"  {r.method_cp:10s} {r.encoder:7s} {r.dataset:14s} {r.size_canon}  seeds={r.seeds}")

    (ROOT / "eval/outputs").mkdir(parents=True, exist_ok=True)
    m.to_csv(ROOT / "eval/outputs/ckpt_audit.csv", index=False)
    print("\nsaved eval/outputs/ckpt_audit.csv")


if __name__ == "__main__":
    main()
