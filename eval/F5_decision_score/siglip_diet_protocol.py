#!/usr/bin/env python3
"""
siglip_diet_protocol.py — FROZEN protocol for the SigLIP x DIET held-out extension
(plan docs/superpowers/plans/2026-09-04-siglip-diet-heldout-extension.md; Task 1).

Single source of truth for: the 15-dataset x 3-seed manifest, the frozen DIET
training recipe, the three frozen source hashes, the preregistration table
builder, and the ONE decision helper reused by the verdict.

  P1 (the only endpoint): Spearman(pre-CP uniformity = angular concentration,
     seed-mean DIET dkNN) over the 15 datasets; one-sided dataset-label
     permutation test, alternative rho > 0 (higher concentration -> larger dkNN).
  Gates fail -> NO VERDICT; else PASS iff rho > 0 and p < 0.05, otherwise FAIL.
  No secondary endpoints exist in this protocol (mainline-only by decision,
  2026-09-04).

CLI:
  python siglip_diet_protocol.py --freeze      # write prereg CSV + MD (hashes inside)
  python siglip_diet_protocol.py --cell 17     # JSON for one Slurm array task
  python siglip_diet_protocol.py --verify      # re-check source hashes only
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------------------------------------------- frozen manifest
# max_samples = the n actually passed as --n-samples by every MAX script in run/slurm
# (fgvc_aircraft = 3334, the train split; the results.xlsx label "MAX (3400)" is a typo).
# subpath = processed HF cache folder under <data>/stable_datasets/processed/ (staging).
DATASETS = [
    dict(order=0,  key="breastmnist",   display="BreastMNIST",   type="OOD", max_samples=546, subpath="med_mnist/breastmnist-size=224"),
    dict(order=1,  key="dermamnist",    display="DermaMNIST",    type="OOD", max_samples=7007, subpath="med_mnist/dermamnist-size=224"),
    dict(order=2,  key="octmnist",      display="OctMNIST",      type="OOD", max_samples=97477, subpath="med_mnist/octmnist-size=224"),
    dict(order=3,  key="organamnist",   display="OrganAMNIST",   type="OOD", max_samples=34561, subpath="med_mnist/organamnist-size=224"),
    dict(order=4,  key="pathmnist",     display="PathMNIST",     type="OOD", max_samples=89996, subpath="med_mnist/pathmnist-size=224"),
    dict(order=5,  key="galaxy10",      display="Galaxy10",      type="OOD", max_samples=14188, subpath="galaxy10"),
    dict(order=6,  key="eurosat",       display="EuroSAT",       type="OOD", max_samples=16200, subpath="eurosat"),
    dict(order=7,  key="plant_village", display="PlantVillage",  type="OOD", max_samples=43596, subpath="plant_village"),
    dict(order=8,  key="dtd",           display="DTD",           type="OOD", max_samples=1880, subpath="dtd"),
    dict(order=9,  key="food101",       display="Food101",       type="FG",  max_samples=75750, subpath="food101"),
    dict(order=10, key="fgvc_aircraft", display="FGVC_Aircraft", type="FG",  max_samples=3334, subpath="fgvc_aircraft"),
    dict(order=11, key="cars196",       display="Cars196",       type="FG",  max_samples=8144, subpath="cars196"),
    dict(order=12, key="cub200",        display="CUB200",        type="FG",  max_samples=5994, subpath="cub200"),
    dict(order=13, key="flowers102",    display="Flowers102",    type="FG",  max_samples=1020, subpath="flowers102"),
    dict(order=14, key="oxford_pet",    display="OxfordPet",     type="FG",  max_samples=3680, subpath="oxford_pet"),
]
SEEDS = (42, 43, 44)
MODEL_ID = "vit_base_patch16_siglip_224.v2_webli"
HPARAMS = dict(cp_method="diet", epochs=150, batch_size=32, lr=1e-4, weight_decay=0.05,
               freeze_epochs=15, num_trained_blocks=2, pool_strategy="map", knn_k=20,
               workers=8, label_smoothing=0.3, mixup_alpha=1.0, cutmix_alpha=1.0,
               mixup_cutmix_prob=0.0, mixup_cutmix_switch_prob=0.5,
               accumulate_grad_batches=1, baseline_eval="skipped (frozen pre values)",
               post_cp_sft="disabled")
SOURCE_HASHES = {
    "preregister_siglip.csv": "4b8bb2b99f884dd3c0e2058dac969310935ed6910d1fd2eaa173546121d846b8",
    "c2_siglip_score.csv":    "473c521b38be8a7632d611d804cb238c677415a3da0cd4a1d40c4388f836aaae",
    "results.xlsx":           "14d0ae53000e0ec3eaeacb7600ce13b66b97b21afedbdb3f9fdbed92e8d36d67",
}
SOURCE_PATHS = {"preregister_siglip.csv": "eval/outputs/preregister_siglip.csv",
                "c2_siglip_score.csv": "eval/outputs/c2_siglip_score.csv",
                "results.xlsx": "results.xlsx"}
N_PERM = 100_000
N_BOOT = 50_000
RNG_SEED = 20260904
ALPHA = 0.05
PREREG_CSV = "eval/outputs/siglip_diet_preregister.csv"
PREREG_MD = "eval/F5_decision_score/SIGLIP_DIET_PREREG.md"


def _cell_dict(task_id, d, seed):
    return dict(task_id=task_id, dataset=d["key"], display=d["display"], type=d["type"],
                max_samples=d["max_samples"], processed_subpath=d["subpath"], seed=seed)


def all_cells():
    return [_cell_dict(i * 3 + j, d, SEEDS[j])
            for i, d in enumerate(DATASETS) for j in range(len(SEEDS))]


def cell(task_id):
    """Slurm array task id (0..44) -> dataset_index = id // 3, seed_index = id % 3."""
    task_id = int(task_id)
    if not 0 <= task_id < 45:
        raise ValueError(f"task_id {task_id} outside the frozen 0..44 grid")
    return _cell_dict(task_id, DATASETS[task_id // 3], SEEDS[task_id % 3])


def cell_env(task_id):
    """Shell-assignment rendering of cell() for the Slurm driver (eval'd there)."""
    c = cell(task_id)
    return "\n".join([f"DATASET={c['dataset']}", f"DISPLAY_NAME={c['display']}",
                      f"DATASET_TYPE={c['type']}", f"N_SAMPLES={c['max_samples']}",
                      f"PROCESSED_SUBPATH={c['processed_subpath']}", f"SEED={c['seed']}"])


# ----------------------------------------------------------------- resume safety
RESULT_REQUIRED = ("post_knn_f1", "post_linear_f1")


def result_is_complete(json_path, cell_, ckpt_path):
    """True only if json_path parses, identifies exactly this cell (dataset, n,
    backbone, method, seed), carries finite post kNN/LP F1, and ckpt_path is a
    readable non-empty file. Anything else -> False (the driver re-runs the cell)."""
    try:
        rec = json.loads(Path(json_path).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(rec, dict):
        return False
    expect = dict(dataset=cell_["dataset"], n_samples=cell_["max_samples"],
                  backbone=MODEL_ID, method="diet", seed=cell_["seed"])
    for k, v in expect.items():
        if rec.get(k) != v:
            return False
    for k in RESULT_REQUIRED:
        v = rec.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
            return False
    ck = Path(ckpt_path)
    try:
        return ck.is_file() and ck.stat().st_size > 0 and os.access(ck, os.R_OK)
    except OSError:
        return False


# ----------------------------------------------------------------- provenance
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root=ROOT):
    """Fail closed unless every frozen source matches its plan-time hash."""
    root = Path(root)
    got = {}
    for name, rel in SOURCE_PATHS.items():
        p = root / rel
        assert p.exists(), f"frozen source missing: {p}"
        got[name] = sha256(p)
        assert got[name] == SOURCE_HASHES[name], \
            f"frozen source hash mismatch for {name}: {got[name][:12]}... != {SOURCE_HASHES[name][:12]}..."
    return got


# ----------------------------------------------------------------- prereg table
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _match_display(sheet_names, display, key):
    cands = {n for n in sheet_names if _norm(n) == _norm(display) or _norm(n) == _norm(key)}
    if not cands:
        cands = {n for n in sheet_names if _norm(display) in _norm(n) or _norm(n) in _norm(display)}
    assert len(cands) == 1, f"ambiguous/missing sheet row for {key}: {sorted(cands)}"
    return cands.pop()


def build_prereg_table(root=ROOT):
    """15-row frozen table: order, dataset, display, type, max_samples, uniformity
    (pre-CP angular concentration on SigLIP, from the 2026-06-19 preregistration
    file), pre_knn (SigLIP pre-CP kNN level; LeJEPA and SimCLR rows must agree)."""
    root = Path(root)
    verify_sources(root)
    pre = pd.read_csv(root / SOURCE_PATHS["preregister_siglip.csv"]).set_index("dataset")
    sheet = pd.read_excel(root / SOURCE_PATHS["results.xlsx"], sheet_name="By Method (SigLIP)",
                          header=1)
    sheet = sheet[sheet.Method.isin(["LeJEPA-CP", "SimCLR-CP"])]
    rows = []
    for d in DATASETS:
        name = _match_display(sheet.Dataset.unique(), d["display"], d["key"])
        sub = sheet[sheet.Dataset == name]
        assert len(sub) == 2, f"expected LeJEPA+SimCLR rows for {d['key']}, got {len(sub)}"
        k, l = sub.knn_f1_mean.values, sub.linear_f1_mean.values
        assert abs(k[0] - k[1]) < 1e-9 and abs(l[0] - l[1]) < 1e-9, \
            f"pre values differ between methods for {d['key']}: {k} {l}"
        rows.append(dict(order=d["order"], dataset=d["key"], display=d["display"],
                         type=d["type"], max_samples=d["max_samples"],
                         uniformity=float(pre.loc[d["key"], "uniformity_t2"]),
                         pre_knn=float(k[0])))
    t = pd.DataFrame(rows)
    assert t.dataset.is_unique and len(t) == 15
    assert np.isfinite(t[["uniformity", "pre_knn"]].values).all()
    return t


# ----------------------------------------------------------------- decision helpers
def seed_mean(df, value="dknn"):
    """Average the seeds within each dataset FIRST (dataset = statistical unit)."""
    return df.groupby("dataset")[value].mean()


def p1_test(score, y, n_perm=N_PERM, seed=RNG_SEED):
    """P1: Spearman(score, y) with a one-sided dataset-label permutation p-value
    (alternative rho > 0). score = pre-CP uniformity; y = seed-mean DIET dkNN."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    rho = spearmanr(score, y).correlation
    rng = np.random.RandomState(seed)
    ge = sum(spearmanr(score, rng.permutation(y)).correlation >= rho
             for _ in range(n_perm))
    return float(rho), (ge + 1) / (n_perm + 1)


def verdict(gates_ok, p1):
    """Three-state, P1-only: NO VERDICT (gate failure) / PASS / FAIL."""
    if not gates_ok:
        return "NO VERDICT"
    return "PASS" if (p1[0] > 0 and p1[1] < ALPHA) else "FAIL"


# ----------------------------------------------------------------- freeze
def verify_prereg(root=ROOT):
    """Fail closed unless the frozen CSV equals a fresh build from the frozen
    sources AND its sha256 equals the one recorded in SIGLIP_DIET_PREREG.md."""
    root = Path(root)
    csv_path, md_path = root / PREREG_CSV, root / PREREG_MD
    assert csv_path.exists() and md_path.exists(), "preregistration files missing"
    fresh = build_prereg_table(root)
    frozen = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(frozen, fresh, check_exact=False, rtol=0, atol=1e-12)
    h = sha256(csv_path)
    m = re.search(r"siglip_diet_preregister\.csv\s+sha256\s+([0-9a-f]{64})", md_path.read_text())
    assert m and m.group(1) == h, f"prereg CSV hash {h[:12]}... not the one recorded in {PREREG_MD}"
    return h


def freeze(root=ROOT):
    root = Path(root)
    t = build_prereg_table(root)
    out = root / PREREG_CSV
    tmp = out.with_suffix(".tmp.csv")
    t.to_csv(tmp, index=False)
    tmp.replace(out)
    h = sha256(out)
    md = f"""# SigLIP x DIET held-out extension — PREREGISTRATION (FROZEN before any DIET outcome exists)

Plan: docs/superpowers/plans/2026-09-04-siglip-diet-heldout-extension.md. Protocol
module: eval/F5_decision_score/siglip_diet_protocol.py (single source of truth).
Mainline-only by decision (2026-09-04): ONE endpoint, no secondary analyses.
Version 1.1 (2026-09-04, before any DIET outcome exists): manifest correction
fgvc_aircraft MAX 3400 -> 3334 (3334 = the --n-samples used by every MAX run in
run/slurm; "MAX (3400)" in results.xlsx is a label typo). No other change.

## Question
On the fixed 15-target panel and the held-out SigLIP encoder, does pre-CP angular
concentration (uniformity_t2, computed from SigLIP features before any CP) rank the
DIET-CP change in frozen kNN? Positioning: a prospective extension of the
target-position relation to a not-yet-observed objective outcome on a held-out
encoder. NOT new-dataset generalization, NOT a mechanism/mediation claim, NOT
universal objective invariance.

## Design
SigLIP x DIET-CP x 15 datasets x seeds {{42, 43, 44}} = 45 MAX cells. Frozen recipe:
{json.dumps(HPARAMS)}. Model: {MODEL_ID}. Dataset = statistical unit; the three seeds
are averaged BEFORE the test. delta_knn_DIET(d) = mean_seed(post_knn(d, seed)) -
pre_knn(d), with pre_knn the frozen SigLIP pre-CP level.
num_trained_blocks = 2 for every cell: this matches the SigLIP LeJEPA/SimCLR grid
(run/slurm/cp-siglip/cp/*/lejepa_max.sh uses 2 blocks even at n = 97,477), NOT the
main-grid DINOv3/CLIP/MAE size rule (2 / 4 / 6 / all blocks by n). Post-CP frozen
kNN/LP is evaluated by continued_pretraining.py without --post-cp-sft.

## The only endpoint (P1)
rho = Spearman(uniformity, delta_knn_DIET) over the 15 datasets. One-sided
dataset-label permutation test, {N_PERM} permutations, RNG seed {RNG_SEED};
alternative rho > 0 (higher concentration -> larger dkNN, the direction observed on
DINOv3/CLIP and on SigLIP's LeJEPA/SimCLR outcomes). A {N_BOOT}-resample dataset
bootstrap 95% CI (seed {RNG_SEED}) is printed next to rho; it is not a gate.
Verdicts: any provenance / completeness / protocol / finite-value gate fails ->
NO VERDICT; rho > 0 and p < {ALPHA} -> PASS; otherwise FAIL.

## Licensed readings (frozen)
- PASS: the pre-CP concentration -> frozen-kNN relation extends to DIET-CP outcomes on
  the held-out SigLIP encoder over the fixed 15-target panel.
- FAIL: the extension to DIET on SigLIP is not supported; the relation stays scoped
  to the objectives on which it was observed.
- NO VERDICT: no scientific update; rerun only the invalid cells under this unchanged
  preregistration.
Forbidden in every case: dataset-population generalization; "geometry determines CP
response"; causal or mediation wording; universal objective invariance; any
post-hoc endpoint added after seeing DIET outcomes.

## Frozen sources (fail closed on mismatch)
{json.dumps(SOURCE_HASHES, indent=2)}

## Generated preregistration table
{PREREG_CSV}  sha256 {h}
Columns: order, dataset, display, type, max_samples, uniformity, pre_knn.
"""
    (root / PREREG_MD).write_text(md)
    return out, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--format", choices=["json", "env"], default="json")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--check-result", type=str, default=None,
                    help="results JSON to validate for --cell; needs --ckpt; exit 0 iff complete")
    ap.add_argument("--ckpt", type=str, default=None)
    a = ap.parse_args()
    if a.check_result is not None:
        assert a.cell is not None and a.ckpt is not None, "--check-result needs --cell and --ckpt"
        ok = result_is_complete(a.check_result, cell(a.cell), a.ckpt)
        print("COMPLETE" if ok else "INCOMPLETE")
        raise SystemExit(0 if ok else 1)
    if a.cell is not None:
        print(cell_env(a.cell) if a.format == "env" else json.dumps(cell(a.cell)))
    elif a.verify_prereg:
        print(f"prereg OK sha256 {verify_prereg()}")
    elif a.freeze:
        out, h = freeze()
        print(f"frozen -> {out}  sha256 {h}\n        -> {ROOT / PREREG_MD}")
    elif a.verify:
        print(json.dumps(verify_sources(), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
