#!/usr/bin/env python
"""
load_results.py — Parse results.xlsx into a clean long-format DataFrame.

This is the shared data-loading utility for the eval/ analysis scripts
(delta_structure.py, correlate.py). It is the single source of truth for:
  - turning the multi-row-header, group-merged xlsx into tidy rows,
  - canonicalising dataset names to the geometry registry keys used by
    geometry_metrics.py (so the two halves can be joined),
  - parsing the NUM_DATA column into a numeric size + a MAX flag.

results.xlsx has three sheets ("By Method" / "By Backbone" / "By Datasets")
that are the SAME data in different row orders; we read "By Datasets".

Columns (after row-2 header, 0-based index):
  0 Group 1 Method 2 Backbone 3 Dataset 4 NUM_DATA 5 model_size
  6 wl 7 resp  | PRE-CP: 8 knn_m 9 knn_s 10 lp_m 11 lp_s 12 ft_m 13 ft_s
  14 wl 15 resp| POST-CP: 16 knn_m 17 knn_s 18 lp_m 19 lp_s 20 ft_m 21 ft_s
  | IMPROVEMENT: 22 dknn 23 dlp 24 dft
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_XLSX = ROOT / "results.xlsx"

# Map the xlsx Dataset labels -> geometry_metrics.py / DS_REGISTRY keys.
# Anything not listed falls back to .lower() (covers the MedMNIST names).
DATASET_KEY = {
    "BreastMNIST": "breastmnist", "DermaMNIST": "dermamnist",
    "OCTMNIST": "octmnist", "OrganAMNIST": "organamnist", "PathMNIST": "pathmnist",
    "Galaxy10": "galaxy10", "EuroSAT": "eurosat", "PlantVillage": "plant_village",
    "DTD": "dtd", "Food101": "food101", "FGVC_Aircraft": "fgvc_aircraft",
    "Cars196": "cars196", "CUB200": "cub200", "Flowers102": "flowers102",
    "OxfordPet": "oxford_pet",
}

# Semantic label as used in plan.md (geometry-vs-semantic comparisons rely on
# NOT trusting these blindly — they are display markers only).
DATASET_TYPE = {
    "breastmnist": "OOD", "dermamnist": "OOD", "octmnist": "OOD",
    "organamnist": "OOD", "pathmnist": "OOD", "galaxy10": "OOD",
    "eurosat": "OOD", "plant_village": "OOD", "dtd": "OOD",
    "food101": "FG", "fgvc_aircraft": "FG", "cars196": "FG",
    "cub200": "FG", "flowers102": "FG", "oxford_pet": "FG",
}

# The 7 datasets added in the 15-dataset expansion. (NOTE: originally MAX-only; size sweeps
# were later filled in for all 15 — see cp_long.csv. The flag only marks expansion membership.)
NEW_DATASETS = {"eurosat", "plant_village", "dtd",
                "cars196", "cub200", "flowers102", "oxford_pet"}

_COLS = [
    "Group", "Method", "Backbone", "Dataset", "NUM_DATA", "model_size", "wl1", "resp1",
    "knn_pre", "knn_pre_s", "lp_pre", "lp_pre_s", "ft_pre", "ft_pre_s", "wl2", "resp2",
    "knn_post", "knn_post_s", "lp_post", "lp_post_s", "ft_post", "ft_post_s",
    "dknn", "dlp", "dft",
]
_NUMERIC = ["knn_pre", "lp_pre", "ft_pre", "knn_post", "lp_post", "ft_post",
            "dknn", "dlp", "dft"]


def _parse_size(num_data):
    """NUM_DATA -> (size_int, is_max). 'MAX (7007)' -> (7007, True); '500' -> (500, False)."""
    s = str(num_data).strip()
    is_max = s.upper().startswith("MAX")
    digits = "".join(ch for ch in s if ch.isdigit())
    size = int(digits) if digits else np.nan
    return size, is_max


def load_long(xlsx_path=RESULTS_XLSX, sheet="By Datasets"):
    """Return a tidy DataFrame, one row per (Method, Backbone, dataset_key, size)."""
    import openpyxl  # noqa: F401 (engine)

    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, skiprows=2,
                        names=_COLS, engine="openpyxl")
    for c in ("Method", "Backbone", "Dataset"):
        raw[c] = raw[c].ffill()
    raw = raw[raw["Dataset"].notna() & (raw["Dataset"].astype(str).str.strip() != "")]
    raw = raw[raw["NUM_DATA"].notna()]
    for c in _NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["dataset_key"] = raw["Dataset"].map(lambda d: DATASET_KEY.get(str(d).strip(),
                                                                      str(d).strip().lower()))
    raw["dataset_type"] = raw["dataset_key"].map(DATASET_TYPE)
    sizes = raw["NUM_DATA"].map(_parse_size)
    raw["size"] = [s for s, _ in sizes]
    raw["is_max"] = [m for _, m in sizes]
    raw["is_new"] = raw["dataset_key"].isin(NEW_DATASETS)
    return raw.reset_index(drop=True)


def add_size_canon(frame, dataset_col="dataset_key", size_col="size"):
    """Add a `size_canon` column that maps each dataset's LARGEST size to the token "MAX".

    Reconciles MAX-label drift between results.xlsx and ckpt filenames (e.g. FGVC-Aircraft:
    results.xlsx records the MAX run as n=3400 but the checkpoint is n=3334 — same run). Within
    each frame, the per-dataset maximum size -> "MAX"; all smaller sizes stay exact (they match
    across sources). Join sweep↔results on (..., size_canon) instead of raw size.
    """
    frame = frame.copy()
    mx = frame.groupby(dataset_col)[size_col].transform("max")
    frame["size_canon"] = np.where(frame[size_col] == mx, "MAX",
                                   frame[size_col].astype("Int64").astype(str))
    return frame


if __name__ == "__main__":
    df = load_long()
    out = ROOT / "eval" / "outputs" / "cp_long.csv"
    df.to_csv(out, index=False)
    print(f"rows={len(df)}  datasets={df.dataset_key.nunique()}  "
          f"methods={sorted(df.Method.unique())}  backbones={sorted(df.Backbone.unique())}")
    print(f"saved {out}")
