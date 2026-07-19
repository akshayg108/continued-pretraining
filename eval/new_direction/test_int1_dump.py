#!/usr/bin/env python3
"""TDD tests for validate_npz in int1_features_dump.py (Codex round-2: resume must
not trust mere file existence — a torn or half-written .npz silently poisons the
whole downstream INT1 chain).

Run: python3 -m pytest eval/new_direction/test_int1_dump.py -q
"""
import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_features_dump import validate_npz


def _write(tmp_path, name="ok.npz", **over):
    arrs = dict(bank_X=np.random.RandomState(0).randn(40, 8).astype(np.float32),
                bank_y=np.arange(40) % 3,
                query_X=np.random.RandomState(1).randn(10, 8).astype(np.float32),
                query_y=np.arange(10) % 3)
    arrs.update(over)
    p = tmp_path / name
    np.savez_compressed(p, **arrs)
    return p


def test_valid_file_passes(tmp_path):
    ok, msg = validate_npz(_write(tmp_path))
    assert ok, msg


def test_missing_key_fails(tmp_path):
    p = tmp_path / "m.npz"
    np.savez_compressed(p, bank_X=np.ones((4, 2), np.float32))
    ok, msg = validate_npz(p)
    assert not ok and "key" in msg


def test_nan_features_fail(tmp_path):
    bad = np.random.randn(40, 8).astype(np.float32)
    bad[3, 3] = np.nan
    ok, msg = validate_npz(_write(tmp_path, bank_X=bad))
    assert not ok and "finite" in msg


def test_bank_query_dim_mismatch_fails(tmp_path):
    ok, msg = validate_npz(_write(tmp_path, query_X=np.ones((10, 9), np.float32)))
    assert not ok and "dim" in msg


def test_label_length_mismatch_fails(tmp_path):
    ok, msg = validate_npz(_write(tmp_path, bank_y=np.arange(39)))
    assert not ok and "label" in msg


def test_empty_bank_fails(tmp_path):
    ok, msg = validate_npz(_write(tmp_path,
                                  bank_X=np.zeros((0, 8), np.float32),
                                  bank_y=np.zeros(0, int)))
    assert not ok and "empty" in msg


def test_corrupt_file_fails(tmp_path):
    p = tmp_path / "c.npz"
    p.write_bytes(b"this is not a zip archive")
    ok, msg = validate_npz(p)
    assert not ok and "unreadable" in msg
