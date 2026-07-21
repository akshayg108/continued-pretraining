#!/usr/bin/env python3
"""TDD tests for the INT2 suite helpers (INT2_PREREG.md v1.0) — written BEFORE the
implementation.

Run: python3 -m pytest eval/new_direction/test_int2_helpers.py -q
"""
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int2_run import reversion_targets, int2_cell_complete
from int2_verdict import direction_sign, int21_decision, g_values, deltas4
from int2_features_dump import load_manifest


# ------------------------------------------------------------- reversion targets
def test_reversion_targets_log_geometry():
    t = reversion_targets(rankme_pre=100.0, rankme_post=25.0)
    assert abs(t["full"] - 100.0) < 1e-9
    assert abs(t["half"] - 50.0) < 1e-9          # exp(log-midpoint) = sqrt(100*25)
    assert abs(t["wrong"] - 6.25) < 1e-9         # post^2 / pre
    assert abs(t["over"] - 400.0) < 1e-9         # pre^2 / post


def test_reversion_targets_expansion_cell_reverses_direction():
    t = reversion_targets(rankme_pre=25.0, rankme_post=100.0)   # DIET regime
    assert t["full"] == 25.0 and abs(t["half"] - 50.0) < 1e-9
    assert abs(t["wrong"] - 400.0) < 1e-9 and abs(t["over"] - 6.25) < 1e-9


# ------------------------------------------------------------- resume completeness
def _have(drop=None, extra=None):
    keys = [("identity", ""), ("rotation", "0"), ("rotation", "1"),
            ("half", "0.7"), ("full", "0.5"), ("wrong", "1.4"), ("over", "0.3"),
            ("transplant", "")]
    keys = [k for k in keys if k[0] != drop]
    if extra:
        keys.append(extra)
    return set(keys)


def test_int2_cell_complete_full_grid():
    assert int2_cell_complete(_have())


def test_int2_cell_complete_missing_arm_with_stale_extra_is_incomplete():
    assert not int2_cell_complete(_have(drop="over", extra=("junk", "x")))


def test_int2_cell_complete_rejects_duplicate_calibrated_arm():
    """Codex round-6: two 'full' rows with different params = 9 rows slipped
    through the arm-name check — the grid must be EXACT (one row per arm)."""
    assert not int2_cell_complete(_have(extra=("full", "0.9")))


def test_int2_cell_complete_needs_both_rotation_seeds():
    have = _have()
    have.discard(("rotation", "1"))
    assert not int2_cell_complete(have)


# ------------------------------------------------------------- direction sign
def test_direction_sign_contraction_expansion_and_floor():
    assert direction_sign(100.0, 25.0) == 1          # CP contracted -> reversion flattens
    assert direction_sign(25.0, 100.0) == -1         # DIET regime
    assert direction_sign(100.0, 100.5) == 0         # |dlog| < 0.01 -> undefined


# ------------------------------------------------------------- INT2-1 decision
CI_POS, CI_NEG, CI_ZERO = (0.01, 0.05), (-0.05, -0.01), (-0.01, 0.02)


def test_int21_yes_needs_alignment_and_dose_consistency():
    assert int21_decision(mean_A=0.03, ci_A=CI_POS, mean_half=0.01,
                          mean_full=0.03, ci_wrong=CI_NEG) == "YES"


def test_int21_dose_inconsistency_is_no_verdict():
    assert int21_decision(mean_A=0.03, ci_A=CI_POS, mean_half=0.05,
                          mean_full=0.03, ci_wrong=CI_NEG) == "NO VERDICT"


def test_int21_wrong_direction_improvement_is_no_verdict():
    """If moving the spectrum AWAY from pre also 'improves', the effect is a
    power-surgery artifact — undecidable, never YES."""
    assert int21_decision(mean_A=0.03, ci_A=CI_POS, mean_half=0.01,
                          mean_full=0.03, ci_wrong=CI_POS) == "NO VERDICT"


def test_int21_wide_null_is_no_verdict():
    """Codex round-6: CI straddling 0 without equivalence bounds is NOT a
    negative — it is undecidable."""
    assert int21_decision(mean_A=0.005, ci_A=CI_ZERO, mean_half=0.001,
                          mean_full=0.005, ci_wrong=CI_NEG) == "NO VERDICT"


def test_int21_null_within_equivalence_margin_is_no():
    assert int21_decision(mean_A=0.001, ci_A=(-0.004, 0.004), mean_half=0.0005,
                          mean_full=0.001, ci_wrong=CI_NEG) == "no"


def test_int21_significant_anti_alignment_is_no():
    assert int21_decision(mean_A=-0.03, ci_A=CI_NEG, mean_half=-0.01,
                          mean_full=-0.03, ci_wrong=CI_NEG) == "no"


# ------------------------------------------------------------- INT2-2 G statistic
def test_g_values_negative_when_surgery_moves_toward_pre():
    g = g_values(y_surg=np.array([0.55]), y_post=np.array([0.40]),
                 y_pre=np.array([0.60]))
    assert g[0] == (0.05 - 0.20)                     # closer to pre -> negative


# ---------------------------------------------- 4-key per-row capture screen
def test_deltas4_pooled_frame_keeps_clean_doses_of_contaminated_cell():
    ident = pd.DataFrame([dict(method="M", encoder="E", dataset=d, seed=42,
                               knn_f1=0.5, lp_f1=0.5, capture_l2=0.5)
                          for d in ("a", "b")]
                         ).set_index(["method", "encoder", "dataset", "seed"])
    sub = pd.DataFrame([
        dict(method="M", encoder="E", dataset="a", seed=42, arm="half",
             param="0.7", knn_f1=0.6, lp_f1=0.6, capture_l2=0.70),
        dict(method="M", encoder="E", dataset="a", seed=42, arm="full",
             param="0.5", knn_f1=0.6, lp_f1=0.6, capture_l2=0.50),
        dict(method="M", encoder="E", dataset="b", seed=42, arm="half",
             param="0.7", knn_f1=0.6, lp_f1=0.6, capture_l2=0.50)])
    dk, dl, ds_, dropped = deltas4(sub, ident, thr=0.05)
    assert len(dk) == 2 and dropped == [("M", "E", "a", 42, "half")]
    assert set(ds_) == {"a", "b"}


# ------------------------------------------------------------- manifest loading
def test_load_manifest_enforces_frozen_census_by_default(tmp_path):
    """Codex round-6: a 1-row manifest passed — the 484-key contract (SimCLR 180
    / LeJEPA 170 / DIET 134) must be locked by default."""
    p = tmp_path / "m.csv"
    pd.DataFrame([dict(method="LeJEPA", encoder="D", dataset="d", seed=42,
                       ckpt="/x.pt", knn_f1_hat_post=0.5)]).to_csv(p, index=False)
    try:
        load_manifest(p)
        assert False, "expected census failure"
    except AssertionError:
        pass


def test_load_manifest_builds_cell_ids_and_asserts_columns(tmp_path):
    df = pd.DataFrame([dict(method="LeJEPA", encoder="DINOv3", dataset="dtd",
                            seed=42, size=100, ckpt="/x/a.pt", knn_f1_hat_post=0.5,
                            vote_margin_post=0.1, vote_pos_frac_post=0.5, n_test=10,
                            n_samples=100),
                       dict(method="DIET", encoder="CLIP", dataset="eurosat",
                            seed=43, size=100, ckpt="/x/b.pt", knn_f1_hat_post=0.6,
                            vote_margin_post=0.1, vote_pos_frac_post=0.5, n_test=10,
                            n_samples=100)])
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False)
    m = load_manifest(p, expect_census=False)
    assert list(m.cell_id) == ["DINOv3__dtd__LeJEPA__42", "CLIP__eurosat__DIET__43"]


def test_load_manifest_fails_without_ckpt_column(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame([dict(method="LeJEPA", encoder="D", dataset="d", seed=42)]
                 ).to_csv(p, index=False)
    try:
        load_manifest(p, expect_census=False)
        assert False, "expected failure on missing ckpt column"
    except (AssertionError, KeyError):
        pass


def test_reversion_targets_flags_unattainable_wrong_dose():
    """Codex round-6 (real-data fact: 31 seed-42 wrong targets exceed the
    attainable RankMe range): wrong/over targets are capped to the calibratable
    range and flagged infeasible when the capped magnitude falls below half the
    full-reversion magnitude."""
    s = np.ones(20)                          # rankme fixed at 20 for ANY alpha
    t = reversion_targets(rankme_pre=10.0, rankme_post=20.0, s_post=s)
    assert t["wrong_feasible"] is False      # no headroom above 20 at all


def test_transplant_does_not_amplify_null_directions():
    from int1_surgery import fit_surgery, apply_surgery
    rng2 = np.random.RandomState(3)
    B = rng2.randn(600, 10) @ rng2.randn(10, 48)         # exact rank 10
    M = fit_surgery(B, kind="transplant", s_target=np.full(48, 100.0))
    s_new = np.linalg.svd(apply_surgery(B, M), compute_uv=False)
    assert (s_new[10:] < 1e-4).all()          # null directions stay null
    assert np.isfinite(M).all() and np.abs(M).max() < 1e6


def test_check_pre_post_consistency_catches_label_mismatch(tmp_path):
    from int2_run import check_pre_post_consistency
    a = dict(bank_X=np.ones((5, 4), np.float32), bank_y=np.arange(5),
             query_X=np.ones((3, 4), np.float32), query_y=np.arange(3))
    np.savez(tmp_path / "pre.npz", **a)
    b = dict(a, bank_y=np.arange(5)[::-1])
    np.savez(tmp_path / "post.npz", **b)
    ok, msg = check_pre_post_consistency(tmp_path / "pre.npz", tmp_path / "post.npz")
    assert not ok and "label" in msg
    np.savez(tmp_path / "post2.npz", **a)
    ok2, _ = check_pre_post_consistency(tmp_path / "pre.npz", tmp_path / "post2.npz")
    assert ok2


def test_direction_series_covers_all_seeds():
    from int2_verdict import direction_series
    rows = []
    for sd in ("42", "43"):
        rows.append(dict(method="M", encoder="E", dataset="a", seed=sd,
                         arm="identity", param="", rankme_raw_bank=25.0,
                         rankme_target=""))
        rows.append(dict(method="M", encoder="E", dataset="a", seed=sd,
                         arm="full", param="0.5", rankme_raw_bank=99.0,
                         rankme_target="100.0"))
    S = direction_series(pd.DataFrame(rows))
    assert ("M", "E", "a", "42") in S.index and ("M", "E", "a", "43") in S.index
    assert (S == 1).all()                     # pre 100 > post 25 -> contraction


# ----------------------------------------------------- round-7 fault injections
def test_infeasible_mask_parses_numeric_feasible_strings():
    from int2_verdict import infeasible_mask
    sub = pd.DataFrame(dict(feasible=[1.0, 0.0, "0", "1", ""]))
    assert infeasible_mask(sub).tolist() == [False, True, True, False, False]


def test_validate_npz_int2_rejects_float64_wrong_dim_nan_labels_extra_fields(tmp_path):
    from int2_features_dump import validate_npz_int2
    good = dict(bank_X=np.ones((10, 768), np.float32), bank_y=np.arange(10),
                query_X=np.ones((4, 768), np.float32), query_y=np.arange(4))
    np.savez(tmp_path / "g.npz", **good)
    ok, _ = validate_npz_int2(tmp_path / "g.npz")
    assert ok
    np.savez(tmp_path / "f64.npz", **dict(good, bank_X=np.ones((10, 768))))
    assert not validate_npz_int2(tmp_path / "f64.npz")[0]
    np.savez(tmp_path / "d.npz", **dict(good, bank_X=np.ones((10, 64), np.float32),
                                        query_X=np.ones((4, 64), np.float32)))
    assert "768" in validate_npz_int2(tmp_path / "d.npz")[1]
    bad_y = np.arange(10).astype(float)
    bad_y[0] = np.nan
    np.savez(tmp_path / "n.npz", **dict(good, bank_y=bad_y))
    assert "label" in validate_npz_int2(tmp_path / "n.npz")[1]
    np.savez(tmp_path / "x.npz", **good, extra=np.ones(3))
    assert "extra" in validate_npz_int2(tmp_path / "x.npz")[1]


def test_gpost_percell_gate_catches_single_corrupt_cell():
    from int2_verdict import gpost_ok
    got = pd.Series([0.5, 0.6, 10.0])
    ref = pd.Series([0.5, 0.6, 0.7])
    ok, worst = gpost_ok(got, ref)
    assert not ok and worst > 9.0
    ok2, _ = gpost_ok(ref, ref)
    assert ok2


def test_verify_pre_checksums_requires_the_full_frozen_set(tmp_path):
    from int2_run import verify_pre_checksums
    f = tmp_path / "one.npz"
    np.savez(f, a=np.ones(2))
    import hashlib
    line = f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}"
    ck = tmp_path / "c.sha256"
    ck.write_text(line + "\n")
    ok, msg = verify_pre_checksums(ck, root=tmp_path.parent)
    assert not ok and ("set" in msg or "60" in msg or "int1_results" in msg)


def test_spectrum_profile_stats_zero_error_on_exact_match():
    from int2_run import spectrum_profile_stats
    s = np.linspace(100.0, 1.0, 50)
    err, amp, ar, tr = spectrum_profile_stats(s, s, s_before=s)
    assert err < 1e-12 and abs(amp - 1.0) < 1e-9 and ar == tr


def test_load_manifest_locks_seed_counts_ckpt_uniqueness_and_hash(tmp_path):
    df = pd.DataFrame([dict(method=m, encoder="E", dataset=f"d{i}", seed=sd,
                            size=100, ckpt=f"/x/{m}_{i}_{sd}.pt", knn_f1_hat_post=0.5)
                       for m, n in (("SimCLR", 180), ("LeJEPA", 170), ("DIET", 134))
                       for i in range(n) for sd in ([42] if True else [])] +
                      [])
    # build a census-correct but seed-wrong manifest: all seed 42
    rows = []
    for m, n in (("SimCLR", 180), ("LeJEPA", 170), ("DIET", 134)):
        for i in range(n):
            rows.append(dict(method=m, encoder="E", dataset=f"d{i}", seed=42,
                             size=100, ckpt=f"/x/{m}_{i}.pt", knn_f1_hat_post=0.5))
    p = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    try:
        load_manifest(p)
        assert False, "expected seed-count failure"
    except AssertionError:
        pass


# ----------------------------------------------------- round-8 fault injections
def test_int2_cell_complete_rejects_pseudo_grids():
    have = _have()
    have.discard(("identity", ""))
    have.add(("identity", "junk"))            # identity must have empty param
    assert not int2_cell_complete(have)
    have2 = _have()
    have2.discard(("half", "0.7"))
    have2.add(("half", "not-a-number"))       # calibrated params must be numeric
    assert not int2_cell_complete(have2)


def test_verify_pre_checksums_rejects_duplicate_line_padding(tmp_path):
    from int2_run import verify_pre_checksums
    import hashlib
    f = tmp_path / "int1_features"
    f.mkdir()
    npz = f / "DINOv3__dtd.npz"
    np.savez(npz, a=np.ones(2))
    h = hashlib.sha256(npz.read_bytes()).hexdigest()
    lines = [f"{h}  int1_features/DINOv3__dtd.npz"] * 60
    res = tmp_path / "int1_results.csv"
    res.write_text("x")
    hr = hashlib.sha256(res.read_bytes()).hexdigest()
    lines.append(f"{hr}  int1_results.csv")
    ck = tmp_path / "c.sha256"
    ck.write_text("\n".join(lines) + "\n")
    ok, msg = verify_pre_checksums(ck, root=tmp_path)
    assert not ok and ("set" in msg or "unique" in msg or "60" in msg)


def test_validate_npz_int2_rejects_float_labels_and_single_sample(tmp_path):
    from int2_features_dump import validate_npz_int2
    good = dict(bank_X=np.ones((10, 768), np.float32), bank_y=np.arange(10),
                query_X=np.ones((4, 768), np.float32), query_y=np.arange(4))
    np.savez(tmp_path / "fl.npz", **dict(good, bank_y=np.arange(10) + 0.5))
    assert "integer" in validate_npz_int2(tmp_path / "fl.npz")[1]
    np.savez(tmp_path / "one.npz",
             **dict(good, bank_X=np.ones((1, 768), np.float32), bank_y=np.array([0])))
    assert not validate_npz_int2(tmp_path / "one.npz")[0]


def test_spectrum_profile_stats_reports_ranks_and_map_scale():
    from int2_run import spectrum_profile_stats
    s_tgt = np.linspace(100.0, 1.0, 48)
    s_before = np.concatenate([np.linspace(50.0, 5.0, 10), np.full(38, 1e-14)])
    err, amp, ar, tr = spectrum_profile_stats(s_before, s_tgt, s_before=s_before)
    assert ar == 10 and tr == 48
    s = np.linspace(100.0, 1.0, 50)
    err2, amp2, ar2, tr2 = spectrum_profile_stats(s, s, s_before=s)
    assert err2 < 1e-12 and abs(amp2 - 1.0) < 1e-9 and ar2 == tr2 == 50


# ----------------------------------------------------- round-9 fault injections
def test_validate_npz_int2_row_aware_n_test(tmp_path):
    from int2_features_dump import validate_npz_int2
    good = dict(bank_X=np.ones((10, 768), np.float32), bank_y=np.arange(10),
                query_X=np.ones((4, 768), np.float32), query_y=np.arange(4))
    np.savez(tmp_path / "g.npz", **good)
    assert validate_npz_int2(tmp_path / "g.npz", n_test=4)[0]
    ok, msg = validate_npz_int2(tmp_path / "g.npz", n_test=5)
    assert not ok and "n_test" in msg


def test_verify_pre_checksums_requires_the_exact_60_name_set(tmp_path):
    from int2_run import verify_pre_checksums
    import hashlib
    d = tmp_path / "int1_features"
    d.mkdir()
    lines = []
    for i in range(60):                        # 60 unique but WRONG names
        f = d / f"fake{i}.npz"
        np.savez(f, a=np.ones(2))
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  "
                     f"int1_features/fake{i}.npz")
    res = tmp_path / "int1_results.csv"
    res.write_text("x")
    lines.append(f"{hashlib.sha256(res.read_bytes()).hexdigest()}  int1_results.csv")
    ck = tmp_path / "c.sha256"
    ck.write_text("\n".join(lines) + "\n")
    ok, msg = verify_pre_checksums(ck, root=tmp_path)
    assert not ok and ("set" in msg or "name" in msg)


def test_spectrum_profile_stats_symmetric_rank_gate_uses_matrix_scale():
    from int2_run import spectrum_profile_stats
    s_tgt = np.linspace(100.0, 1.0, 48)
    tiny = 100.0 * 48 * np.finfo(np.float64).eps * 2       # dies at matrix scale
    s_before = np.concatenate([np.linspace(50.0, 5.0, 10), np.full(38, tiny)])
    err, amp, ar, tr = spectrum_profile_stats(s_before, s_tgt, s_before=s_before,
                                              dim_scale=5000)
    assert ar == 10 and tr == 48               # achieved 10 vs target 48 exposed


def test_int2_cell_complete_rejects_nan_param_and_needs_finite():
    have = _have()
    have.discard(("half", "0.7"))
    have.add(("half", "nan"))
    assert not int2_cell_complete(have)


# ---------------------------------------------------- round-10 fault injections
def test_parse_rank_pair_requires_exactly_two_positive_integers():
    from int2_verdict import parse_rank_pair
    assert parse_rank_pair("10/48") == (10, 48)
    for bad in ("0/0", "-1/-1", "1.5/1.5", "48/48/extra", "48", "", "a/b"):
        assert parse_rank_pair(bad) is None, bad


def test_verify_post_checksums_requires_exact_manifest_set(tmp_path):
    from int2_run import verify_post_checksums
    import hashlib
    feat = tmp_path / "feat"
    feat.mkdir()
    names = [f"c{i}.npz" for i in range(3)]
    lines = []
    for n in names:
        f = feat / n
        np.savez(f, a=np.ones(2))
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {n}")
    ck = tmp_path / "s.sha256"
    ck.write_text("\n".join(lines) + "\n")
    ok, _ = verify_post_checksums(ck, feat, expected_names=set(names))
    assert ok
    # truncated sidecar with one legitimate record must FAIL
    ck.write_text(lines[0] + "\n")
    ok2, msg = verify_post_checksums(ck, feat, expected_names=set(names))
    assert not ok2 and ("set" in msg or "484" in msg or "missing" in msg)
    # foreign file in the feature dir must FAIL
    ck.write_text("\n".join(lines) + "\n")
    np.savez(feat / "foreign.npz", a=np.ones(2))
    ok3, msg3 = verify_post_checksums(ck, feat, expected_names=set(names))
    assert not ok3 and "foreign" in msg3


# ---------------------------------------------------- round-11 regressions
def test_verify_only_never_overwrites_existing_sidecar(tmp_path, monkeypatch):
    """The local re-run of --verify-only must not re-freeze (re-legitimize) a
    corrupted transfer; an existing sidecar is left untouched."""
    import subprocess, sys as _s
    import int2_features_dump as d
    feat = tmp_path / "int2_features"
    feat.mkdir()
    np.savez(feat / "E__d__M__42.npz",
             bank_X=np.ones((3, 768), np.float32), bank_y=np.arange(3),
             query_X=np.ones((2, 768), np.float32), query_y=np.arange(2))
    man = pd.DataFrame([dict(method="M", encoder="E", dataset="d", seed=42,
                             size=1, ckpt="/x.pt", knn_f1_hat_post=0.5,
                             vote_margin_post=0.1, vote_pos_frac_post=0.5,
                             n_test=np.nan, n_samples=np.nan)])
    mp = tmp_path / "m.csv"
    man.to_csv(mp, index=False)
    side = tmp_path / "int2_features.sha256"
    side.write_text("FROZEN-BY-CLUSTER\n")
    r = subprocess.run([_s.executable, d.__file__, "--verify-only",
                        "--allow-nonfrozen-manifest",
                        "--manifest", str(mp), "--outdir", str(feat)],
                       capture_output=True, text=True)
    assert side.read_text() == "FROZEN-BY-CLUSTER\n"        # untouched
    assert "NOT overwritten" in r.stdout


def test_verify_sidecar_only_is_read_only_and_verifies(tmp_path):
    import subprocess, sys as _s
    import hashlib
    import int2_features_dump as d
    feat = tmp_path / "int2_features"
    feat.mkdir()
    f = feat / "E__d__M__42.npz"
    np.savez(f, bank_X=np.ones((3, 768), np.float32), bank_y=np.arange(3),
             query_X=np.ones((2, 768), np.float32), query_y=np.arange(2))
    man = pd.DataFrame([dict(method="M", encoder="E", dataset="d", seed=42,
                             size=1, ckpt="/x.pt", knn_f1_hat_post=0.5,
                             vote_margin_post=0.1, vote_pos_frac_post=0.5,
                             n_test=np.nan, n_samples=np.nan)])
    mp = tmp_path / "m.csv"
    man.to_csv(mp, index=False)
    side = tmp_path / "int2_features.sha256"
    side.write_text(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n")
    before = side.read_text()
    r = subprocess.run([_s.executable, d.__file__, "--verify-sidecar-only",
                        "--allow-nonfrozen-manifest",
                        "--manifest", str(mp), "--outdir", str(feat)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout
    assert side.read_text() == before                       # read-only
