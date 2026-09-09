#!/usr/bin/env python3
"""TDD tests for the SigLIP x DIET held-out extension (plan 2026-09-04) — written
BEFORE the implementation. Task 1 covers the frozen manifest, source-hash gates,
the preregistration table, and the single P1 decision helper (uniformity vs
seed-mean dkNN; no secondary endpoints) that the verdict script will reuse.

Run: python3 -m pytest eval/F5_decision_score/test_siglip_diet_extension.py -q
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import siglip_diet_protocol as P

ROOT = Path(__file__).resolve().parent.parent.parent      # continued-pretraining/


# ------------------------------------------------------------- frozen manifest
def test_manifest_is_the_frozen_15x3_grid():
    assert [d["order"] for d in P.DATASETS] == list(range(15))
    assert [d["key"] for d in P.DATASETS][:3] == ["breastmnist", "dermamnist", "octmnist"]
    assert [d["key"] for d in P.DATASETS][-1] == "oxford_pet"
    types = [d["type"] for d in P.DATASETS]
    assert types.count("OOD") == 9 and types.count("FG") == 6
    counts = {d["key"]: d["max_samples"] for d in P.DATASETS}
    assert counts["breastmnist"] == 546 and counts["octmnist"] == 97477
    assert counts["food101"] == 75750 and counts["flowers102"] == 1020
    assert counts["fgvc_aircraft"] == 3334      # train split; spreadsheet label "3400" is a typo
    assert P.SEEDS == (42, 43, 44)
    assert len({(c["dataset"], c["seed"]) for c in P.all_cells()}) == 45


def test_cell_mapping_task_id_to_dataset_and_seed():
    c0, c44 = P.cell(0), P.cell(44)
    assert (c0["dataset"], c0["seed"]) == ("breastmnist", 42)
    assert (c44["dataset"], c44["seed"]) == ("oxford_pet", 44)
    assert P.cell(4)["dataset"] == "dermamnist" and P.cell(4)["seed"] == 43
    with pytest.raises(ValueError):
        P.cell(45)


def test_frozen_protocol_matches_main_grid_diet_recipe():
    h = P.HPARAMS
    assert h["epochs"] == 150 and h["batch_size"] == 32 and h["lr"] == 1e-4
    assert h["weight_decay"] == 0.05 and h["freeze_epochs"] == 15
    assert h["num_trained_blocks"] == 2 and h["pool_strategy"] == "map"
    assert h["label_smoothing"] == 0.3 and h["mixup_cutmix_prob"] == 0.0
    assert P.MODEL_ID == "vit_base_patch16_siglip_224.v2_webli"


# ------------------------------------------------------------- source hash gates
def test_source_hashes_verify_on_frozen_files_and_reject_tampering(tmp_path):
    got = P.verify_sources(ROOT)
    assert set(got) == {"preregister_siglip.csv", "c2_siglip_score.csv", "results.xlsx"}
    # tamper one byte of a copy -> fail closed
    fake = tmp_path / "root"
    (fake / "eval/outputs").mkdir(parents=True)
    for name in ("preregister_siglip.csv", "c2_siglip_score.csv"):
        shutil.copy(ROOT / "eval/outputs" / name, fake / "eval/outputs" / name)
    shutil.copy(ROOT / "results.xlsx", fake / "results.xlsx")
    p = fake / "eval/outputs/preregister_siglip.csv"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(AssertionError):
        P.verify_sources(fake)


# ------------------------------------------------------------- prereg table
def test_prereg_table_has_15_rows_ascii_columns_and_matching_pre_values():
    t = P.build_prereg_table(ROOT)
    assert list(t.columns) == ["order", "dataset", "display", "type", "max_samples",
                               "uniformity", "pre_knn"]
    assert len(t) == 15 and t.dataset.is_unique and list(t.order) == list(range(15))
    assert t[["uniformity", "pre_knn"]].apply(np.isfinite).all().all()
    # spot values against the frozen sources
    ti = t.set_index("dataset")
    assert abs(ti.loc["breastmnist", "pre_knn"] - 0.6988) < 1e-6
    assert abs(ti.loc["breastmnist", "uniformity"] - (-0.3639944791793823)) < 1e-9


# ------------------------------------------------------------- decision helpers
def test_seed_mean_collapses_45_rows_to_15_before_any_test():
    rows = [dict(dataset=c["dataset"], seed=c["seed"], dknn=0.1 + 0.01 * c["seed"])
            for c in P.all_cells()]
    m = P.seed_mean(pd.DataFrame(rows), value="dknn")
    assert len(m) == 15 and abs(m.loc["breastmnist"] - 0.53) < 1e-9


def test_p1_permutation_perfect_reversed_and_null():
    score = np.linspace(0.1, 0.9, 15)
    rho, p = P.p1_test(score, score.copy(), n_perm=2000, seed=1)
    assert rho == 1.0 and p < 0.05
    rho, p = P.p1_test(score, -score, n_perm=2000, seed=1)
    assert rho == -1.0 and p > 0.5
    rng = np.random.RandomState(7)
    null_y = rng.permutation(score)                     # unrelated ordering
    rho, p = P.p1_test(score, null_y, n_perm=2000, seed=1)
    assert abs(rho) < 0.6 and p > 0.05


def test_verdict_is_three_state_on_p1_only():
    assert P.verdict(gates_ok=False, p1=(0.9, 0.001)) == "NO VERDICT"
    assert P.verdict(gates_ok=True, p1=(-0.2, 0.8)) == "FAIL"
    assert P.verdict(gates_ok=True, p1=(0.3, 0.12)) == "FAIL"      # rho > 0 but p >= 0.05
    assert P.verdict(gates_ok=True, p1=(0.7, 0.01)) == "PASS"


def test_protocol_exposes_no_secondary_endpoints():
    assert not hasattr(P, "p2_test") and not hasattr(P, "hierarchical_verdict")


def test_cell_carries_processed_cache_subpath_for_staging():
    assert P.cell(0)["processed_subpath"] == "med_mnist/breastmnist-size=224"
    assert P.cell(30)["processed_subpath"] == "fgvc_aircraft"
    assert P.cell(6)["processed_subpath"] == "med_mnist/octmnist-size=224"


# ------------------------------------------------------------- resume safety
def _write_cell_result(tmp_path, cell, **over):
    import json
    rec = dict(dataset=cell["dataset"], n_samples=cell["max_samples"], backbone=P.MODEL_ID,
               method="diet", seed=cell["seed"], epochs=150, post_knn_f1=0.5, post_linear_f1=0.6)
    rec.update(over)
    j = tmp_path / "r.json"
    j.write_text(json.dumps(rec))
    c = tmp_path / "r.ckpt"
    c.write_bytes(b"x")
    return j, c


def test_result_is_complete_accepts_only_matching_finite_result_with_checkpoint(tmp_path):
    cell = P.cell(5)                                     # dermamnist seed 44
    j, c = _write_cell_result(tmp_path, cell)
    assert P.result_is_complete(j, cell, c) is True
    c.unlink()
    assert P.result_is_complete(j, cell, c) is False    # checkpoint missing
    j, c = _write_cell_result(tmp_path, cell, seed=42)
    assert P.result_is_complete(j, cell, c) is False    # wrong seed
    j, c = _write_cell_result(tmp_path, cell, post_knn_f1=float("nan"))
    assert P.result_is_complete(j, cell, c) is False    # non-finite
    j, c = _write_cell_result(tmp_path, cell)
    j.write_text("{not json")
    assert P.result_is_complete(j, cell, c) is False    # malformed
    assert P.result_is_complete(tmp_path / "absent.json", cell, c) is False


# ------------------------------------------------------------- slurm array driver
DRIVER = ROOT / "run/slurm/cp-siglip/cp/diet_max_array.sh"


def _dry_run(task_id, extra_env=None):
    import os, subprocess
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task_id))
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(DRIVER), "--dry-run"], env=env, cwd=ROOT,
                       capture_output=True, text=True)
    kv = {}
    for line in r.stdout.splitlines():
        if line.startswith("DRY-RUN "):
            k, _, v = line[len("DRY-RUN "):].partition("=")
            kv[k] = v
    return r.returncode, kv, r.stdout + r.stderr


def test_driver_header_groups_fifteen_datasets_twelve_at_a_time():
    head = DRIVER.read_text().splitlines()[:40]
    assert any(l.strip() == "#SBATCH --array=0-14%12" for l in head)
    for directive in ["#SBATCH --partition=nvidia", "#SBATCH --account=civil",
                      "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1",
                      "#SBATCH --exclude=cn253,cn259"]:
        assert directive in head


def test_driver_dry_run_resolves_cell_paths_and_frozen_recipe():
    rc, kv, out = _dry_run(0)
    assert rc == 0, out
    assert kv["dataset"] == "breastmnist" and kv["seeds"] == "42 43 44" and kv["n_samples"] == "546"
    assert kv["processed_subpath"] == "med_mnist/breastmnist-size=224"
    assert "full_ft_v1" in kv["ft_outdir"]
    cmd = kv["command"]
    for frag in ["--cp-method diet", "--backbone vit_base_patch16_siglip_224.v2_webli",
                 "--n-samples 546", "--epochs 150", "--batch-size 32", "--lr 1e-4",
                 "--weight-decay 0.05", "--freeze-epochs 15", "--num-trained-blocks 2",
                 "--knn-k 20", "--label-smoothing 0.3", "--mixup-alpha 1.0", "--cutmix-alpha 1.0",
                 "--mixup-cutmix-prob 0.0", "--mixup-cutmix-switch-prob 0.5",
                 "--pool-strategy map", "--accumulate-grad-batches 1", "--seed 42",
                 "--skip-baseline"]:
        assert frag in cmd, frag
    assert "--post-cp-sft" not in cmd and "--pre-cp-sft" not in cmd
    assert kv["skip"] == "no"
    assert "command_seed43" in kv and "command_seed44" in kv


def test_driver_uses_private_workdirs_and_accumulates_seed_failures():
    text = DRIVER.read_text()
    assert text.count("mktemp -d") >= 2
    assert "FINAL_STATUS=0" in text
    assert "FINAL_STATUS=1" in text
    assert "module load miniconda/3-4.11.0" in text
    assert "conda activate env" in text
    assert "nvidia-smi" in text
    assert 'LOCAL_CACHE=""' in text and 'WORK_DIR=""' in text
    assert "--resume" in text


def test_driver_dry_run_maps_last_cell_and_corrected_fgvc_count():
    rc, kv, out = _dry_run(14)
    assert rc == 0 and kv["dataset"] == "oxford_pet" and kv["n_samples"] == "3680"
    rc, kv, out = _dry_run(10)
    assert rc == 0 and kv["dataset"] == "fgvc_aircraft" and kv["n_samples"] == "3334"


def test_driver_refuses_task_id_outside_grid():
    rc, kv, out = _dry_run(15)
    assert rc != 0


def test_driver_skips_only_a_complete_verified_cell(tmp_path):
    import json
    cell = P.cell(0)
    out_root = tmp_path / "outputs"
    log_dir = out_root / "logs/cp-siglip/cp/DIET/BreastMNIST/SigLIP"
    ckpt_dir = out_root / "ckpts/cp-siglip/cp/DIET/BreastMNIST/SigLIP/cp"
    log_dir.mkdir(parents=True); ckpt_dir.mkdir(parents=True)
    j = log_dir / "SigLIP_breastmnist_n546_seed42.json"
    c = ckpt_dir / "breastmnist_vit_base_patch16_siglip_224.v2_webli_n546_s42.ckpt"
    j.write_text(json.dumps(dict(dataset="breastmnist", n_samples=546, backbone=P.MODEL_ID,
                                 method="diet", seed=42, epochs=150,
                                 post_knn_f1=0.7, post_linear_f1=0.7)))
    c.write_bytes(b"x")
    rc, kv, out = _dry_run(0, {"SIGLIP_DIET_OUT_ROOT": str(out_root)})
    assert rc == 0 and kv["skip"] == "yes", out
    j.write_text("{broken")
    rc, kv, out = _dry_run(0, {"SIGLIP_DIET_OUT_ROOT": str(out_root)})
    assert rc == 0 and kv["skip"] == "no", out
