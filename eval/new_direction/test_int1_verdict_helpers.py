#!/usr/bin/env python3
"""TDD tests for the extracted decision helpers in int1_verdict.py (Codex round-2:
the placement LP verdict silently reused the kNN monotonicity flag; G-NC counted the
two readouts marginally instead of jointly per cell; there was no G0 census; G-P
exclusions leaked into INT1-4).

Run: python3 -m pytest eval/new_direction/test_int1_verdict_helpers.py -q
"""
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_verdict import (gnc_joint, placement_flags, g0_gate, drop_excluded,
                          screened_deltas, first_stage_move, coverage_ok,
                          tri_verdict, calibration_ok)

CELLS = [("E", "a"), ("E", "b"), ("E", "c")]
IDX = pd.MultiIndex.from_tuples(CELLS, names=["encoder", "dataset"])


def test_gnc_joint_counts_cells_where_both_readouts_pass():
    dk = pd.Series([0.001, 0.010, 0.001], index=IDX)
    dl = pd.Series([0.001, 0.001, 0.010], index=IDX)
    # marginally 2/3 pass each readout, but only cell "a" passes BOTH
    assert gnc_joint(dk, dl, tol=0.005) == 1


CI_NEG = (-0.05, -0.01)          # excludes 0


def _trend(mk_seq, ml_seq):
    rows = [("demote", p, mk, ml, CI_NEG, CI_NEG)
            for p, mk, ml in zip(("64", "256", "512"), mk_seq, ml_seq)]
    rows.append(("shuffle", "0", -0.10, -0.10, CI_NEG, CI_NEG))
    return rows


def test_placement_lp_verdict_uses_lp_monotonicity_not_knn():
    """The bug: LP's pass flag was gated on kNN's monotonicity. Here kNN is monotone
    but LP is NOT — LP must fail on its own trend."""
    f = placement_flags(_trend((-0.01, -0.02, -0.03), (-0.03, -0.01, -0.02)))
    assert f["mono_k"] and not f["mono_l"]
    assert f["pk"] and not f["pl"]


def test_placement_knn_verdict_uses_knn_monotonicity():
    f = placement_flags(_trend((-0.03, -0.01, -0.02), (-0.01, -0.02, -0.03)))
    assert not f["mono_k"] and f["mono_l"]
    assert not f["pk"] and f["pl"]


BASE_GRID = ([("identity", "")] + [("rotation", s) for s in ("0", "1")]
             + [("power", a) for a in ("0.25", "0.5", "0.75", "1.5", "2.0")]
             + [("demote", d) for d in ("64", "256", "512")] + [("shuffle", "0")]
             + [("combo", c) for c in ("0.5|256", "2.0|256")])


CELLS2 = (("E", "a"), ("E", "b"))


def _frame(cells=CELLS2):
    rows = [dict(encoder=e, dataset=d, transform=t, param=p, knn_f1=0.5, lp_f1=0.5,
                 rankme_raw_bank=10.0, rankme_l2_bank=10.0, capture_l2=0.5,
                 cC_K_l2=0.5, query_oos_frac=0.0, rankme_target="")
            for e, d in cells for t, p in BASE_GRID]
    rows += [dict(encoder=e, dataset=d, transform="power_cp", param="1.2000",
                  knn_f1=0.5, lp_f1=0.5, rankme_raw_bank=10.0, rankme_l2_bank=10.0,
                  capture_l2=0.5, cC_K_l2=0.5, query_oos_frac=0.0,
                  rankme_target=10.0)
             for e, d in cells]
    return pd.DataFrame(rows)


def test_g0_gate_passes_complete_grid():
    ok, msgs = g0_gate(_frame(), cells=CELLS2)
    assert ok, msgs


def test_g0_gate_fails_on_missing_transform_row():
    r = _frame()
    # NB: bracket access — r.transform is the DataFrame METHOD, not the column
    r = r[~((r.dataset == "b") & (r["transform"] == "demote") & (r.param == "256"))]
    ok, msgs = g0_gate(r, cells=CELLS2)
    assert not ok and any("demote" in m for m in msgs)


def test_g0_gate_fails_on_duplicate_key():
    r = _frame()
    ok, msgs = g0_gate(pd.concat([r, r.iloc[[0]]]), cells=CELLS2)
    assert not ok and any("duplicate" in m for m in msgs)


def test_g0_gate_default_expects_the_real_60_cell_grid():
    ok, msgs = g0_gate(_frame())                    # default: 4 encoders x 15 datasets
    assert not ok and any("cell" in m for m in msgs)


def test_g0_gate_fails_on_wrong_cell_identity_even_at_right_count():
    """Codex round-3: 60 rows with a wrong (encoder, dataset) member must FAIL —
    counting cells is not the same as verifying the exact 4 x 15 set."""
    ok, msgs = g0_gate(_frame(), cells=(("E", "a"), ("E", "c")))
    assert not ok and any("cell" in m for m in msgs)


def test_g0_gate_fails_on_nan_metric_column():
    """Codex round-3 fail-open: NaN in capture_l2 slipped through — G0 must check
    every numeric metric column, not just the two F1 readouts."""
    r = _frame()
    r.loc[3, "capture_l2"] = np.nan
    ok, msgs = g0_gate(r, cells=CELLS2)
    assert not ok and any("finite" in m for m in msgs)


def test_g0_gate_fails_on_unexpected_transform_row():
    """Codex round-4: extra rows beyond the frozen grid (stale/foreign transforms)
    must FAIL, not silently pass into estimates."""
    r = _frame()
    r.loc[len(r)] = dict(encoder="E", dataset="a", transform="weird", param="x",
                         knn_f1=0.5, lp_f1=0.5, rankme_raw_bank=10.0,
                         rankme_l2_bank=10.0, capture_l2=0.5, cC_K_l2=0.5,
                         query_oos_frac=0.0, rankme_target="")
    ok, msgs = g0_gate(r, cells=CELLS2)
    assert not ok and any("unexpected" in m for m in msgs)


def test_g0_gate_requires_power_cp_on_every_cell():
    """Codex round-4: ND7 provides 60/60 rank targets, so power_cp is REQUIRED
    per cell (was optional)."""
    r = _frame()
    r = r[r["transform"] != "power_cp"]
    ok, msgs = g0_gate(r, cells=CELLS2)
    assert not ok and any("power_cp" in m for m in msgs)


def test_g0_gate_fails_on_nan_rankme_target_in_power_cp():
    r = _frame()
    r.loc[r["transform"] == "power_cp", "rankme_target"] = np.nan
    ok, msgs = g0_gate(r, cells=CELLS2)
    assert not ok and any("rankme_target" in m for m in msgs)


# ------------------------------- v1.5/v1.6: per-ROW capture screen (rounds 3 + 4)
def _ident_frame(caps=(0.5, 0.5, 0.5)):
    return pd.DataFrame([dict(encoder="E", dataset=d, knn_f1=0.5, lp_f1=0.5,
                              capture_l2=c) for d, c in zip(("a", "b", "c"), caps)]
                        ).set_index(["encoder", "dataset"])


def test_screened_deltas_drops_rows_with_large_capture_drift():
    ident = _ident_frame()
    sub = pd.DataFrame([dict(encoder="E", dataset=d, transform="power", param="0.25",
                             knn_f1=0.6, lp_f1=0.6, capture_l2=c)
                        for d, c in zip(("a", "b", "c"), (0.51, 0.49, 0.70))])
    dk, dl, ds_, dropped = screened_deltas(sub, ident, thr=0.05)
    assert dropped == [("E", "c", "0.25")]
    assert len(dk) == 2 and len(dl) == 2 and set(ds_) == {"a", "b"}


def test_screened_deltas_keeps_all_rows_when_drift_small():
    ident = _ident_frame()
    sub = pd.DataFrame([dict(encoder="E", dataset=d, transform="power", param="0.25",
                             knn_f1=0.6, lp_f1=0.6, capture_l2=0.5)
                        for d in ("a", "b", "c")])
    dk, dl, ds_, dropped = screened_deltas(sub, ident, thr=0.05)
    assert dropped == [] and len(dk) == 3


def test_screened_deltas_pooled_frame_keeps_clean_doses_of_contaminated_cell():
    """Codex round-4 P0: in a pooled multi-dose frame, ONE contaminated dose of a
    cell must not drag that cell's clean doses out of the estimate (the old code
    dropped by cell index: expected keep 3, got 2)."""
    ident = _ident_frame()[["knn_f1", "lp_f1", "capture_l2"]].loc[[("E", "a"), ("E", "b")]]
    sub = pd.DataFrame([
        dict(encoder="E", dataset="a", transform="power", param="0.25",
             knn_f1=0.6, lp_f1=0.6, capture_l2=0.70),
        dict(encoder="E", dataset="a", transform="power", param="2.0",
             knn_f1=0.6, lp_f1=0.6, capture_l2=0.50),
        dict(encoder="E", dataset="b", transform="power", param="0.25",
             knn_f1=0.6, lp_f1=0.6, capture_l2=0.50),
        dict(encoder="E", dataset="b", transform="power", param="2.0",
             knn_f1=0.6, lp_f1=0.6, capture_l2=0.50)])
    dk, dl, ds_, dropped = screened_deltas(sub, ident, thr=0.05)
    assert len(dk) == 3 and len(dl) == 3
    assert dropped == [("E", "a", "0.25")]
    # positional pairing must survive the duplicate cell index (no realignment)
    assert len((dk - dl).dropna()) == 3


# ---------------------------- v1.5/v1.6: T3 first-stage gate on screened subset
def test_first_stage_move_is_median_abs_cCK_drift():
    ident = pd.DataFrame([dict(encoder="E", dataset=d, cC_K_l2=0.5)
                          for d in ("a", "b", "c")]).set_index(["encoder", "dataset"])
    sub = pd.DataFrame([dict(encoder="E", dataset=d, cC_K_l2=v)
                        for d, v in zip(("a", "b", "c"), (0.6, 0.3, 0.8))])
    assert abs(first_stage_move(sub, ident) - 0.2) < 1e-12


def test_first_stage_move_uses_the_capture_screened_subset():
    """Codex round-4 P1: first stage must be computed on the SAME eligible rows as
    the F1 estimate — capture-screened rows leave the median too."""
    ident = pd.DataFrame([dict(encoder="E", dataset=d, cC_K_l2=0.5, capture_l2=0.5)
                          for d in ("a", "b", "c")]).set_index(["encoder", "dataset"])
    sub = pd.DataFrame([dict(encoder="E", dataset=d, cC_K_l2=0.5 + v,
                             capture_l2=c)
                        for d, v, c in zip(("a", "b", "c"), (0.1, 0.3, 5.0),
                                           (0.5, 0.5, 0.9))])
    # all three: median = 0.3; screened (c leaves): median of {0.1, 0.3} = 0.2
    assert abs(first_stage_move(sub, ident, thr=0.05) - 0.2) < 1e-12


# ------------------------------------------ v1.6: coverage floor + tri-state verdicts
def test_coverage_ok_enforces_cell_and_dataset_floors():
    keys = [("E1", f"d{i}") for i in range(10)] + [("E2", f"d{i}") for i in range(10)]
    assert coverage_ok(keys, min_cells=20, min_datasets=10)
    assert not coverage_ok(keys, min_cells=21, min_datasets=10)
    assert not coverage_ok(keys, min_cells=20, min_datasets=11)


def test_tri_verdict_distinguishes_no_from_no_verdict():
    assert tri_verdict(0, False) == "NO VERDICT"
    assert tri_verdict(2, True) == "YES"
    assert tri_verdict(2, False) == "no"


# ------------------------------------- v1.6: T2c calibration acceptance (round 4)
def test_calibration_ok_accepts_within_tolerance_and_rejects_misses_and_clamps():
    df = pd.DataFrame([
        dict(alpha_cp="1.2000", rankme_target=100.0, rankme_raw_bank=101.0),   # ok
        dict(alpha_cp="1.5000", rankme_target=100.0, rankme_raw_bank=90.0),    # miss > 2%
        dict(alpha_cp="0.0500", rankme_target=100.0, rankme_raw_bank=100.5)])  # clamped
    mask, reasons = calibration_ok(df)
    assert mask.tolist() == [True, False, False]
    assert any("2%" in m or "tolerance" in m for m in reasons)
    assert any("clamp" in m for m in reasons)


def test_drop_excluded_removes_flagged_transform_params():
    r = _frame()                       # includes power_cp[1.2000] on both cells
    out = drop_excluded(r, {("power_cp", "1.2000")})
    assert not len(out[out["transform"] == "power_cp"])
    assert len(out) == len(r) - 2
