#!/usr/bin/env python3
"""
check_expb_ckpts.py — verify the post-CP encoder checkpoints Exp B (SA-LP) needs are still on disk.

Exp B (Beyond-[CLS] Selective Aggregation) re-evaluates SAVED post-CP encoder checkpoints with a
depth-1 ABMILP attention-pool head instead of the cls/avgpool linear probe. It does NOT retrain,
so every needed (method, encoder, dataset, MAX, seed) must still have its .ckpt on /scratch.
Post-CP checkpoints are often pruned once the original kNN/LP metrics are logged; this script finds
what survives, so we know whether Exp B is a cheap re-eval or needs a re-train.

Scope (locked, "tier B") — datasets span the MAE-CP LP-degradation severity (catastrophic->mild):
    {MAE-CP, LeJEPA-CP} x {DINOv3, CLIP, MAE} x 7 datasets x MAX x seeds {42,43,44}.

Checkpoint layout (verified on the cluster):
    <root>/cp/<MAE|LeJEPA>/pretrained/<Dataset>/<DINOv3|CLIP|MAE>/<subdir>/<dataset>_<backbone>_n<N>_s<seed>.ckpt
The post-CP ENCODER ckpt (what SA-LP loads) lives in the 'cp/' subdir; SFT ckpts live in a sibling
subdir. This script reports the subdir structure too, so we can confirm which ckpt SA-LP must load.

Run on the cluster (where /scratch is mounted):
    python eval/check_expb_ckpts.py
    python eval/check_expb_ckpts.py --root /scratch/gs4133/zhd/CP/outputs/ckpts --out eval/outputs/expb_ckpt_check.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

METHODS = {"MAE-CP": "MAE", "LeJEPA-CP": "LeJEPA"}   # display name -> ckpt folder segment
ENCODERS = ["DINOv3", "CLIP", "MAE"]                 # also the <TAG> path segment
SEEDS = [42, 43, 44]

# dataset key -> (canonical ckpt folder name, expected MAX n_samples [soft sanity check only])
DATASETS = {
    "food101":       ("Food101",       75750),
    "octmnist":      ("OctMNIST",      97477),
    "plant_village": ("PlantVillage",  43596),
    "organamnist":   ("OrganAMNIST",   34561),
    "galaxy10":      ("Galaxy10",      14188),
    "fgvc_aircraft": ("FGVC_Aircraft",  3334),
    "cars196":       ("Cars196",        8144),
}

CKPT_RE = re.compile(r"_n(\d+)_s(\d+)\.ckpt$")


def find_dir(parent: Path | None, name: str) -> Path | None:
    """Locate a child dir by exact name, then case-insensitively (harmless on Linux)."""
    if parent is None or not parent.is_dir():
        return None
    p = parent / name
    if p.is_dir():
        return p
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def scan(cfg_dir: Path | None):
    """Return (by_n, by_subdir): {n_samples: set(seeds)} and {subdir_rel: ckpt_count}."""
    by_n: dict[int, set[int]] = {}
    by_subdir: dict[str, int] = {}
    if cfg_dir and cfg_dir.is_dir():
        for f in cfg_dir.rglob("*_n*_s*.ckpt"):
            m = CKPT_RE.search(f.name)
            if not m:
                continue
            n, s = int(m.group(1)), int(m.group(2))
            by_n.setdefault(n, set()).add(s)
            sub = str(f.parent.relative_to(cfg_dir)) or "."
            by_subdir[sub] = by_subdir.get(sub, 0) + 1
    return by_n, by_subdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Exp B (SA-LP) post-CP checkpoint availability.")
    ap.add_argument("--root", default="/scratch/gs4133/zhd/CP/outputs/ckpts",
                    help="checkpoint root (the .../outputs/ckpts dir)")
    ap.add_argument("--out", default=None, help="optional CSV report path")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    n_ok = n_part = n_miss = 0
    for mdisp, mfold in METHODS.items():
        parent = root / "cp" / mfold / "pretrained"
        for ds_key, (ds_folder, exp_max) in DATASETS.items():
            ds_dir = find_dir(parent, ds_folder)
            for enc in ENCODERS:
                cfg_dir = find_dir(ds_dir, enc)
                by_n, by_subdir = scan(cfg_dir)
                if by_n:
                    max_n = max(by_n)
                    present = sorted(by_n[max_n])
                else:
                    max_n, present = None, []
                missing = [s for s in SEEDS if s not in present]
                status = "OK" if (by_n and not missing) else ("PARTIAL" if by_n else "MISSING")
                n_ok += status == "OK"
                n_part += status == "PARTIAL"
                n_miss += status == "MISSING"
                rows.append(dict(method=mdisp, enc=enc, ds=ds_key, max_n=max_n, exp_max=exp_max,
                                 present=present, missing=missing, status=status,
                                 sizes=sorted(by_n), subdirs=by_subdir,
                                 dir=str(cfg_dir) if cfg_dir else f"(not found: {parent}/{ds_folder}/{enc})"))

    print(f"Exp B post-CP checkpoint check — root={root}")
    print(f"scope: {len(METHODS)} methods x {len(ENCODERS)} encoders x {len(DATASETS)} datasets "
          f"= {len(rows)} configs x {len(SEEDS)} seeds = {len(rows) * len(SEEDS)} ckpts\n")

    cur = None
    for r in rows:
        if r["method"] != cur:
            cur = r["method"]
            print(f"=== {cur}  (cp/{METHODS[cur]}/pretrained) ===")
        tag = {"OK": "OK  ", "PARTIAL": "PART", "MISSING": "MISS"}[r["status"]]
        line = (f"  [{tag}] {r['enc']:6} {r['ds']:14} MAX n={r['max_n']} "
                f"present={r['present']} missing={r['missing']}")
        if r["status"] == "MISSING":
            line += f"   (sizes on disk: {r['sizes'] or 'none'})"
        elif r["max_n"] is not None and r["max_n"] != r["exp_max"]:
            line += f"   WARN max_n {r['max_n']} != expected {r['exp_max']}"
        print(line)

    print(f"\nSUMMARY  OK={n_ok}  PARTIAL={n_part}  MISSING={n_miss}  (of {len(rows)} configs)")

    sample = next((r for r in rows if r["subdirs"]), None)
    if sample:
        print(f"\nckpt subdirs seen under a config dir "
              f"(e.g. {sample['method']}/{sample['enc']}/{sample['ds']}): {sample['subdirs']}")
        print("  -> SA-LP needs the POST-CP ENCODER ckpt (the 'cp' subdir), NOT the SFT one; "
              "confirm the correct subdir from this listing before loading.")

    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        print("\nNeeds attention (PARTIAL/MISSING) — these would force a re-train:")
        for r in bad:
            print(f"  {r['method']:10} {r['enc']:6} {r['ds']:14} {r['status']:7} "
                  f"missing_seeds={r['missing']}  {r['dir']}")
    else:
        print("\nAll required post-CP checkpoints present — Exp B is a pure re-eval, no re-training.")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "encoder", "dataset", "status", "max_n", "expected_max_n",
                        "seeds_present", "seeds_missing", "sizes_on_disk", "subdirs", "dir"])
            for r in rows:
                w.writerow([r["method"], r["enc"], r["ds"], r["status"], r["max_n"], r["exp_max"],
                            " ".join(map(str, r["present"])), " ".join(map(str, r["missing"])),
                            " ".join(map(str, r["sizes"])),
                            ";".join(f"{k}({v})" for k, v in r["subdirs"].items()), r["dir"]])
        print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
