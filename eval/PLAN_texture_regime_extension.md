# Texture-Regime Extension — Implementation Plan

> **For the executor (you run the heavy compute):** steps use checkbox (`- [ ]`) syntax.
> Code/registry edits are bite-sized and runnable locally; the CP training step is a SLURM
> grid you submit on the cluster. Mark each step done as you go.

**Goal:** Test whether texture / non-object datasets are an *embedded-but-task-orthogonal*
sub-regime where the Finding-1 angular-position law mis-predicts CP benefit — i.e. turn the
single DTD counterexample into either a characterized boundary of the law (a secondary finding)
or a confirmed lone oddity.

**Architecture:** Add 2–3 texture datasets to the existing pipeline, run a **minimal** CP grid
(sphere encoders × invariance methods only — exactly the regime where F1 holds and DTD breaks
it), recompute pre-CP geometry + behavioral Δ, then run one adjudication script that fits the
position→Δknn law on the original 15 and checks whether the new texture datasets land like DTD
(positive high-rank residual on BOTH DINOv3 and CLIP) or on the line.

**Tech stack:** Python 3 / pandas / scipy / timm / torch (already used); the user's own
`stable-datasets` fork (`github.com/haodongzhang0118/stable-datasets`); PyTorch-Lightning CP
trainer (`continued-pretraining/continued_pretraining.py`); SLURM array jobs.

**Falsifiable prediction (the whole point):** each added texture dataset is geometrically
embedded (high `neighbor_overlap_k50`, low `uniformity_t2` rank) **yet** has a positive,
high-rank position-residual on **both** DINOv3 and CLIP — behaving like DTD, not like the
embedded *object* datasets (food101, cub200) that lie on the line. **REFUTED** if the texture
datasets fall on the line (Δknn < 0 like embedded objects) → then DTD is a lone oddity and we
report it as such.

---

## Dataset choice (recommended)

Pick texture/material-recognition datasets whose **task** is orthogonal to object identity but
whose **images** plausibly embed near ImageNet:

| dataset | classes | ~images | why | source homepage |
|---|---|---|---|---|
| **KTH-TIPS-2b** | 11 materials | ~4752 | classic material texture under varying scale/illumination; different source from DTD | `https://www.csc.kth.se/cvap/databases/kth-tips/` |
| **FMD** (Flickr Material DB) | 10 materials | 1000 | pure material recognition, very different capture distribution | `https://people.csail.mit.edu/celiu/CVPR2010/FMD/` |
| GTOS-Mobile *(optional 3rd)* | 31 terrains | ~6066 | ground-terrain textures; adds robustness | `https://github.com/jiaxue1993/Deep-Encoding-Pooling-Network` |

Two is the minimum to get a 2-dataset × 2-encoder = 4-cell reproducibility check (matching the
2 encoders on which DTD reproduces); the third strengthens it. All are folder-of-images-per-class
→ straightforward loaders.

## File structure (what gets created / edited)

- **Create** `continued-pretraining/stable-datasets/stable_datasets/images/kth_tips2b.py`, `…/fmd.py` — dataset loaders (one responsibility each; copy the DTD template).
- **Edit** `continued-pretraining/stable-datasets/stable_datasets/images/__init__.py` — export the new classes.
- **Edit** `continued-pretraining/stable_cp/data/datasets.py` — add CP-training `DATASETS` entries.
- **Edit** `eval/geometry_metrics.py` — add `DS_REGISTRY` entries (pre-CP geometry).
- **Edit** `eval/load_results.py` — add `DATASET_KEY` / `DATASET_TYPE` / `NEW_DATASETS` entries.
- **Create** `continued-pretraining/run/slurm/cp/{LeJEPA,SimCLR}/pretrained/{KTH_TIPS2b,FMD}/{dinov3,clip}_run.sh` — CP scripts (copy the DTD template, surgical edits).
- **Create** `eval/texture_regime_test.py` — the adjudication script (full code in Task 6).

---

## Task 1: Implement the texture-dataset loaders in `stable-datasets`

**Files:**
- Read first (template): `continued-pretraining/stable-datasets/stable_datasets/images/dtd.py`
- Create: `continued-pretraining/stable-datasets/stable_datasets/images/kth_tips2b.py`
- Create: `continued-pretraining/stable-datasets/stable_datasets/images/fmd.py`
- Modify: `continued-pretraining/stable-datasets/stable_datasets/images/__init__.py`

> The `stable-datasets` repo is **your own fork**, so commit the loaders there and bump the
> pinned commit in `continued-pretraining/pyproject.toml` after.

- [ ] **Step 1: Read the DTD loader as the exact template.**

Run: `sed -n '1,200p' continued-pretraining/stable-datasets/stable_datasets/images/dtd.py`
Note the four things every loader defines (per `BaseDatasetBuilder`): `SOURCE` (homepage +
archive URL(s) + citation), `_labels()` (list of class-name strings), `_info()` (metadata), and
`_generate_examples(data_path, split)` (yields `(idx, {"image": PIL.Image, "label": <class name>})`).
DTD already ships a fixed train/val/test split; texture datasets without an official split need a
deterministic split (see Step 3).

- [ ] **Step 2: Write `kth_tips2b.py`** by copying `dtd.py` and changing only the dataset-specific
  fields. Skeleton (fill the bracketed values from the KTH-TIPS-2b homepage archive):

```python
from stable_datasets.utils import BaseDatasetBuilder   # match dtd.py's exact import
import os
from PIL import Image

class KTHTIPS2b(BaseDatasetBuilder):
    SOURCE = {
        "homepage": "https://www.csc.kth.se/cvap/databases/kth-tips/",
        "assets": ["<direct URL to KTH-TIPS2-b.tar of the homepage>"],   # fill from homepage
        "citation": "Mallikarjuna et al., The KTH-TIPS2 database, 2006",
    }
    _NUM_CLASSES = 11

    def _labels(self):
        return ["aluminium_foil", "brown_bread", "corduroy", "cork", "cotton",
                "cracker", "lettuce_leaf", "linen", "white_bread", "wood", "wool"]

    def _info(self):
        return self._make_info(num_classes=self._NUM_CLASSES, task="texture-material")  # mirror dtd.py

    def _generate_examples(self, data_path, split):
        # KTH-TIPS-2b layout: <root>/<material>/sample_<a..d>/<img>.png  (4 physical samples a–d)
        # Standard protocol: 3 samples train, 1 test. We use sample 'd' as test, 'c' as val,
        # 'a'+'b' as train — deterministic, leakage-free across physical samples.
        split_map = {"train": ("sample_a", "sample_b"), "validation": ("sample_c",),
                     "test": ("sample_d",)}
        for material in sorted(self._labels()):
            for sample_dir in split_map[split]:
                d = os.path.join(data_path, material, sample_dir)
                if not os.path.isdir(d):
                    continue
                for i, fn in enumerate(sorted(os.listdir(d))):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        yield f"{material}/{sample_dir}/{i}", {
                            "image": Image.open(os.path.join(d, fn)).convert("RGB"),
                            "label": material,
                        }
```

- [ ] **Step 3: Write `fmd.py`** the same way. FMD has **no official split** (10 classes ×
  100 images, flat `image/<material>/<material>_moderate_###.jpg`). Use a deterministic per-class
  split so geometry and CP see the same partition:

```python
import os
from PIL import Image
from stable_datasets.utils import BaseDatasetBuilder

class FMD(BaseDatasetBuilder):
    SOURCE = {
        "homepage": "https://people.csail.mit.edu/celiu/CVPR2010/FMD/",
        "assets": ["<direct URL to FMD.zip from homepage>"],
        "citation": "Sharan, Rosenholtz, Adelson, Material perception, 2009",
    }
    _NUM_CLASSES = 10

    def _labels(self):
        return ["fabric", "foliage", "glass", "leather", "metal",
                "paper", "plastic", "stone", "water", "wood"]

    def _info(self):
        return self._make_info(num_classes=self._NUM_CLASSES, task="texture-material")

    def _generate_examples(self, data_path, split):
        # deterministic: per class sort filenames, first 70 train / next 15 val / last 15 test
        bounds = {"train": (0, 70), "validation": (70, 85), "test": (85, 100)}
        lo, hi = bounds[split]
        for material in sorted(self._labels()):
            d = os.path.join(data_path, "image", material)
            files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
            for i, fn in enumerate(files[lo:hi]):
                yield f"{material}/{fn}", {
                    "image": Image.open(os.path.join(d, fn)).convert("RGB"),
                    "label": material,
                }
```

> If `_make_info` / `BaseDatasetBuilder` method names differ in the real `dtd.py`, copy DTD's
> exact signatures — do not invent. The split logic above is the only texture-specific content.

- [ ] **Step 4: Export the classes.** In `__init__.py`, mirror the existing DTD export line:

```python
from stable_datasets.images.kth_tips2b import KTHTIPS2b
from stable_datasets.images.fmd import FMD
```

- [ ] **Step 5: Smoke-test each loader locally** (downloads on first call):

```bash
cd continued-pretraining/stable-datasets
python -c "
from stable_datasets.images.fmd import FMD
for sp in ['train','validation','test']:
    ds = FMD(split=sp, download_dir='/tmp/sd/dl', processed_cache_dir='/tmp/sd/proc')
    print(sp, len(ds), ds[0]['image'].size, ds[0]['label'])
"
```
Expected: `train 700 ...`, `validation 150 ...`, `test 150 ...` (FMD); KTH ≈ train/val/test split
of ~4752. **Verify class count == _NUM_CLASSES and no split overlaps.**

- [ ] **Step 6: Commit + bump the pin.**

```bash
cd continued-pretraining/stable-datasets && git add stable_datasets/images/ && \
  git commit -m "feat: add KTH-TIPS-2b and FMD texture-material loaders"
git rev-parse HEAD   # copy this commit hash
```
Then in `continued-pretraining/pyproject.toml` pin `stable-datasets @ git+...@<new-hash>` and
re-`pip install -e` on the cluster so CP training sees the new classes.

---

## Task 2: Register the datasets in eval + CP-training config

**Files:**
- Modify: `eval/geometry_metrics.py` (`DS_REGISTRY`)
- Modify: `eval/load_results.py` (`DATASET_KEY`, `DATASET_TYPE`, `NEW_DATASETS`)
- Modify: `continued-pretraining/stable_cp/data/datasets.py` (`DATASETS`)

- [ ] **Step 1: Add to the geometry registry.** In `eval/geometry_metrics.py`, alongside the
  existing `"dtd": (stable_ds.DTD, None, ["train","validation","test"], {})` line:

```python
    "kth_tips2b": (stable_ds.KTHTIPS2b, None, ["train", "validation", "test"], {}),
    "fmd":        (stable_ds.FMD,       None, ["train", "validation", "test"], {}),
```

- [ ] **Step 2: Add to the results loader.** In `eval/load_results.py`:
  - `DATASET_KEY`: `"KTH_TIPS2b": "kth_tips2b", "FMD": "fmd",` (keys must match the row labels you
    will use in `results.xlsx`).
  - `DATASET_TYPE`: `"kth_tips2b": "OOD", "fmd": "OOD",` (texture = label "OOD"; the *point* is
    that geometry says embedded while the label/task says non-object — the script tests behavior,
    not the label).
  - `NEW_DATASETS`: add `"kth_tips2b", "fmd"` to the set.

- [ ] **Step 3: Add to CP-training config.** In `continued-pretraining/stable_cp/data/datasets.py`,
  mirror the DTD entry:

```python
    "kth_tips2b": {
        "dataset_class": stable_ds.KTHTIPS2b, "config_name": None,
        "num_classes": 11, "input_size": 224, "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
    "fmd": {
        "dataset_class": stable_ds.FMD, "config_name": None,
        "num_classes": 10, "input_size": 224, "normalization": "imagenet",
        "splits": ["train", "validation", "test"],
    },
```

- [ ] **Step 4: Verify registration** (no training):

```bash
cd continued-pretraining && python -c "
from stable_cp.data.datasets import DATASETS, get_dataset_config
for k in ['kth_tips2b','fmd']:
    print(k, get_dataset_config(k)['num_classes'])
"
```
Expected: `kth_tips2b 11`, `fmd 10` with no `ValueError: Unknown dataset`.

---

## Task 3: Compute pre-CP geometry for the new datasets (cheap, no training)

This produces the **X axis** (angular position) for the new datasets, on the sphere encoders.

**Files:**
- Run: `eval/geometry_metrics.py` (DINOv3 + CLIP only)
- Output: extend `eval/outputs/geometry_15.csv` → save as `eval/outputs/geometry_17.csv`

- [ ] **Step 1:** Run geometry on the two new datasets for DINOv3 and CLIP (reuse the exact
  invocation you used for the 15; restrict to the new keys + sphere encoders). Example:

```bash
cd /Users/zhanghaodong/Desktop/CP
python eval/geometry_metrics.py --datasets kth_tips2b fmd \
    --encoders DINOv3 CLIP \
    --download-dir /scratch/gs4133/zhd/CP/data/stable_datasets/downloads \
    --processed-dir /scratch/gs4133/zhd/CP/data/stable_datasets/processed \
    --out eval/outputs/geometry_new.csv
```
> If `geometry_metrics.py` lacks a `--datasets` flag, add a 3-line argparse filter on
> `DS_REGISTRY` keys — do not hand-edit the registry.

- [ ] **Step 2: Merge into a 17-dataset geometry file:**

```bash
python -c "
import pandas as pd
a=pd.read_csv('eval/outputs/geometry_15.csv'); b=pd.read_csv('eval/outputs/geometry_new.csv')
pd.concat([a,b],ignore_index=True).drop_duplicates(['encoder','dataset']).to_csv('eval/outputs/geometry_17.csv',index=False)
print('rows', len(a)+len(b))
"
```
Expected new rows: 4 (2 datasets × 2 encoders). **Sanity check the prediction's premise:** the
new datasets should be *embedded* — `neighbor_overlap_k50` comparable to DTD/food101, NOT ≈0 like
eurosat. If they come back isolated (overlap≈0), they are not valid test points; pick different
texture datasets.

---

## Task 4: Run the minimal CP grid (heavy — cluster)

**Minimal grid** = exactly the regime where F1 holds and DTD breaks it:
`{KTH_TIPS2b, FMD} × {DINOv3, CLIP} × {LeJEPA-CP, SimCLR-CP} × {sizes} × {seeds 42,43,44}`.
MAX size alone suffices for the residual test; a 3-point size sweep `{≈250, ≈1000, MAX}` also gives
the within-dataset trend. Skip MAE-encoder and DIET/MAE methods (different regime — out of scope).

**Files:**
- Read template: `continued-pretraining/run/slurm/cp/LeJEPA/pretrained/DTD/clip_run.sh` (or DIET/DTD if LeJEPA/DTD absent)
- Create: `continued-pretraining/run/slurm/cp/{LeJEPA,SimCLR}/pretrained/{KTH_TIPS2b,FMD}/{dinov3,clip}_run.sh` (8 scripts)

- [ ] **Step 1: Copy the DTD template and apply the surgical per-dataset edits.** Per the existing
  convention, change only: `DATASET="dtd"`→`"kth_tips2b"`, `DISPLAY_NAME="DTD"`→`"KTH_TIPS2b"`,
  the `NSAMPLES` array to the dataset's real sizes (FMD train=700 → `(250 700)`; KTH train≈3000 →
  `(250 1000 3000)`), and the `CKPT_DIR`/`LOG_DIR` dataset segment. Keep all hyperparameters and
  `--dataset <key>` (must match the `DATASETS` key) identical to the template.

- [ ] **Step 2: Dry-run one script** (1 size, 1 seed) to confirm the dataset loads in the trainer
  and a checkpoint is written:

```bash
sbatch continued-pretraining/run/slurm/cp/LeJEPA/pretrained/FMD/clip_run.sh   # smallest config first
# after it finishes:
ls /scratch/gs4133/zhd/CP/outputs/ckpts/cp/LeJEPA/pretrained/FMD/CLIP/cp/*.ckpt
```
Expected: a `*_n700_s42.ckpt` exists; the job's results JSON has non-degenerate pre/post kNN.

- [ ] **Step 3: Submit the full minimal grid** (8 scripts, all sizes/seeds). Expected total ≈
  2 datasets × 2 enc × 2 methods × (1–3 sizes) × 3 seeds ≈ **24–72 short CP runs** — a fraction of
  the original benchmark.

- [ ] **Step 4: Verify coverage** with the existing audit (it will list the new configs once they
  appear in the expected grid):

```bash
python eval/coverage_audit.py --sweep eval/outputs/postcp_sweep.csv
```

---

## Task 5: Collect behavioral Δknn for the new datasets

The **Y axis** (Δknn) comes from the CP runs' pre/post kNN, the same way the original 15 did.

- [ ] **Step 1:** Aggregate the new runs' results JSONs into the results table exactly as the
  original benchmark did (the per-run JSONs already contain `knn_pre`, `knn_post`; `dknn =
  knn_post − knn_pre`). Add rows to `results.xlsx` under labels `KTH_TIPS2b` / `FMD` (matching the
  `DATASET_KEY` you registered), or export a CSV `eval/outputs/texture_runs.csv` with columns
  `Method, Backbone, dataset_key, size, dknn` for the new runs.

- [ ] **Step 2: Verify the join works** (size-canon reconciles MAX-label drift):

```bash
python -c "
import sys; sys.path.insert(0,'eval'); from load_results import load_long, add_size_canon
df=load_long(); df=add_size_canon(df,'dataset_key','size')
print(df[df.dataset_key.isin(['kth_tips2b','fmd'])][['Method','Backbone','dataset_key','size_canon','dknn']].dropna())
"
```
Expected: LeJEPA-CP + SimCLR-CP rows for both texture datasets on DINOv3 and CLIP, with `dknn`
populated and a `size_canon=="MAX"` row each.

---

## Task 6: Adjudication script — does the texture prediction hold?

**Files:**
- Create: `eval/texture_regime_test.py`
- Output: `eval/outputs/texture_regime_test.csv` + console verdict

- [ ] **Step 1: Create `eval/texture_regime_test.py`** with this exact content:

```python
#!/usr/bin/env python
"""
texture_regime_test.py — NEW. Test P1.4: texture/non-object datasets are an
embedded-but-task-orthogonal sub-regime where the F1 position law mis-predicts CP benefit.

Procedure (sphere encoders only — where F1 holds and DTD breaks it):
  1. Fit Δknn ~ uniformity_t2 on the ORIGINAL 15 datasets, per encoder; take residuals.
  2. Project each NEW texture dataset onto that fit; compute its residual + percentile rank.
  3. PREDICTION: each texture dataset is embedded (high overlap / low uniformity rank) yet has a
     POSITIVE, high-rank residual on BOTH DINOv3 and CLIP (behaves like DTD).
     REFUTED if texture residuals are negative / mid-rank (they lie on the line like objects).

Inputs: eval/outputs/geometry_17.csv (pre-CP geometry incl. new), results.xlsx (Δ incl. new).
Run:    python eval/texture_regime_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_results import load_long  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_15 = {"breastmnist", "dermamnist", "octmnist", "organamnist", "pathmnist",
               "galaxy10", "eurosat", "plant_village", "dtd", "food101",
               "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"}
TEXTURE_NEW = {"kth_tips2b", "fmd"}            # the datasets added by this extension
SPHERE = ["DINOv3", "CLIP"]


def main():
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_17.csv")
    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False)]

    rows = []
    for enc in SPHERE:
        g = geo[(geo.encoder == enc) & (geo.dataset != "imagenet")].set_index("dataset")
        y = (inv[(inv.Backbone == enc) & (inv.is_max)]
             .groupby("dataset_key")["dknn"].mean())
        t = g.join(y.rename("dknn")).dropna(subset=["uniformity_t2", "dknn"])

        orig = t[t.index.isin(ORIGINAL_15)]
        slope, intercept = np.polyfit(orig["uniformity_t2"], orig["dknn"], 1)
        t["pred"] = slope * t["uniformity_t2"] + intercept
        t["resid"] = t["dknn"] - t["pred"]
        # percentile of each residual among the ORIGINAL 15 (how DTD-like is it?)
        t["resid_pct_vs_orig"] = t["resid"].apply(
            lambda r: (orig["resid"] if "resid" in orig else
                       (orig["dknn"] - (slope * orig["uniformity_t2"] + intercept))).lt(r).mean())
        t["overlap_rank"] = t["neighbor_overlap_k50"].rank(ascending=False).astype(int) \
            if "neighbor_overlap_k50" in t else np.nan

        rho_orig, p_orig = spearmanr(orig["uniformity_t2"], orig["dknn"])
        print(f"\n===== {enc} =====   F1 fit on original 15: rho={rho_orig:+.3f} (p={p_orig:.3f})")
        cols = ["uniformity_t2", "neighbor_overlap_k50", "dknn", "pred", "resid", "resid_pct_vs_orig"]
        cols = [c for c in cols if c in t.columns]
        for key in sorted(TEXTURE_NEW | {"dtd"}):
            if key in t.index:
                r = t.loc[key]
                rows.append({"encoder": enc, "dataset": key, **{c: r[c] for c in cols}})
                flag = "DTD-like(+resid)" if r["resid"] > 0 else "ON-LINE(-resid)"
                print(f"  {key:12s} embedded? overlap={r.get('neighbor_overlap_k50', float('nan')):+.3f} "
                      f"| dknn={r['dknn']:+.3f} pred={r['pred']:+.3f} resid={r['resid']:+.3f} "
                      f"(pct {r['resid_pct_vs_orig']:.2f}) -> {flag}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "eval/outputs/texture_regime_test.csv", index=False)

    # ---- verdict ----
    tex = out[out.dataset.isin(TEXTURE_NEW)]
    by_ds = tex.groupby("dataset")["resid"].apply(lambda s: (s > 0).all())  # +resid on BOTH encoders
    confirmed = by_ds.sum()
    print("\n" + "=" * 60)
    print(f"VERDICT: {confirmed}/{len(by_ds)} texture datasets have POSITIVE residual on BOTH "
          f"sphere encoders (DTD-like).")
    if confirmed == len(by_ds) and len(by_ds) > 0:
        print("  -> P1.4 SUPPORTED: texture = embedded-but-task-orthogonal sub-regime. "
              "DTD is not a lone oddity; report the boundary as a secondary finding.")
    elif confirmed == 0:
        print("  -> P1.4 REFUTED: texture datasets lie on the line. DTD is a lone oddity; "
              "report it as a single characterized exception.")
    else:
        print("  -> MIXED: report per-dataset; n is small, interpret cautiously.")
    print(f"\nsaved {ROOT / 'eval/outputs/texture_regime_test.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.**

```bash
cd /Users/zhanghaodong/Desktop/CP && python eval/texture_regime_test.py
```
Expected (if prediction holds): each texture dataset prints `DTD-like(+resid)` with
`resid_pct_vs_orig` near 1.0 on **both** DINOv3 and CLIP, and the verdict reports
`2/2 ... SUPPORTED`. Expected (if refuted): `ON-LINE(-resid)` and `REFUTED`.

---

## Task 7: Record the verdict in the docs

**Files:**
- Modify: `findings/FINDINGS_step2.md` (the DTD bullet — replace "n=1, can only be tested" with the result)
- Modify: `hypothesis/HYPOTHESES_v2.md` (P1.4 — mark SUPPORTED/REFUTED with the numbers)

- [ ] **Step 1:** If SUPPORTED, promote it to a secondary finding ("the law has a stated domain:
  angular position predicts CP benefit for *object-recognition* targets; *texture/non-object*
  targets are an embedded-but-task-orthogonal exception, confirmed on N datasets × 2 encoders").
  If REFUTED, keep DTD as a single characterized exception and state texture did **not** generalize.
- [ ] **Step 2:** Either way, this converts the DTD weakness into a *characterized* result — the
  honest outcome the paper needs.

---

## Self-review (against the goal)

- **Coverage:** loaders (T1) → registration (T2) → X axis/geometry (T3) → Y axis/CP runs (T4–T5) →
  adjudication (T6) → write-up (T7). The full pipeline map (Explore) is covered: all 4 edit points
  + new loaders + SLURM scripts + the size-canon join gotcha (T5 Step 2).
- **Minimality (YAGNI):** only sphere encoders + invariance methods (the regime under test); MAE /
  DIET / extra sizes are explicitly out of scope. Keeps the grid at ~24–72 short runs.
- **Premise guard:** T3 Step 2 checks the new datasets are actually *embedded* before spending CP
  compute — if they come back isolated, abort and re-pick (they would not be valid test points).
- **Honest both-ways:** the script reports SUPPORTED **or** REFUTED with the same rigor; either is
  a publishable, honest outcome (boundary finding vs characterized lone exception). No p-hacking:
  the prediction (positive residual on both encoders) is fixed *before* the runs.
- **Risk:** the only soft spot is loader internals (`BaseDatasetBuilder` exact API) — mitigated by
  "copy `dtd.py` verbatim, change only the bracketed dataset-specific fields" (T1 Steps 1–3).
