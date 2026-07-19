#!/usr/bin/env python3
"""TDD tests for the resume logic in int1_run.py (Codex round-4 P0: the sentinel +
row-count heuristic could mark a cell complete when a required surgery is missing
but a stale extra row pads the count).

Run: python3 -m pytest eval/new_direction/test_int1_run_helpers.py -q
"""
import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_run import cell_complete, BASE_KEYS


def _done(enc="E", ds="a", with_power_cp=True, drop=None, extra=None):
    keys = [k for k in BASE_KEYS if k != drop]
    if with_power_cp:
        keys.append(("power_cp", "1.2345"))
    if extra:
        keys.append(extra)
    return {(enc, ds, t, p) for t, p in keys}


def test_complete_cell_is_complete():
    assert cell_complete(_done(), "E", "a", has_target=True)


def test_missing_required_surgery_with_stale_extra_row_is_incomplete():
    """The old heuristic: 6 sentinel keys + row count >= expected. A stale extra
    row could pad the count while demote[256] is missing."""
    done = _done(drop=("demote", "256"), extra=("power", "9.9"))
    assert not cell_complete(done, "E", "a", has_target=True)


def test_missing_power_cp_is_incomplete_when_target_exists():
    assert not cell_complete(_done(with_power_cp=False), "E", "a", has_target=True)


def test_power_cp_not_required_without_target():
    assert cell_complete(_done(with_power_cp=False), "E", "a", has_target=False)


def test_other_cells_rows_do_not_count():
    done = _done(enc="E", ds="b")            # complete cell b, nothing for cell a
    assert not cell_complete(done, "E", "a", has_target=True)
