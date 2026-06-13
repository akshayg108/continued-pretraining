# `eval/` — Geometry recompute + correlation analysis (15 datasets × 3 encoders)

All **new** analysis code for the manifold/geometry re-examination of the CP
benchmark lives here, so it is separable from the original `hypothesis/` scripts.

## Files

| File | Status | Runs where | What it does |
|------|--------|-----------|--------------|
| `load_results.py` | NEW | anywhere (pandas+openpyxl) | Parses `results.xlsx` → tidy long DataFrame. Canonicalises dataset names to geometry registry keys; parses `NUM_DATA` → `size`+`is_max`. Shared by the others. |
| `delta_structure.py` | NEW | anywhere | Pure Δ-structure analysis from `results.xlsx` — **no geometry needed**. kNN/FT reversal quadrants, MAE-CP catastrophe, the 15-dataset Δ@MAX ranking (ground truth for `correlate.py`), gap effect. |
| `geometry_metrics.py` | PORTED + FIXED from the original `hypothesis/` colab analysis (since removed) | env with `stable_datasets`, timm, (gated ImageNet) | Recomputes pre-CP geometry for all 15 datasets × 3 encoders → one CSV. **You run this.** |
| `download_imagenet_val.py` | utility (moved from `hypothesis/`) | env with `datasets` + HF login | Downloads ImageNet-1k val and `save_to_disk` for the geometry run's `--imagenet-dir`. |
| `correlate.py` | NEW | anywhere (reads geometry CSV + `results.xlsx`) | Joins geometry to Δ **per-encoder** and tests Hypothesis 1: Spearman/partial/LOO, new-dataset prediction check, Claim 4 reversal, MMD-identity diagnostic. |
| `bivariate.py` | NEW | anywhere | F2.1: position × pre-CP-baseline two-direction partials + adjusted R². (Result: position survives, baseline adds nothing → bivariate law dropped; see `FINDINGS`/`HYPOTHESES_v2`.) |
| `postcp_features.py` | NEW | Colab (post-CP ckpt) | Shared: load a post-CP checkpoint (auto-detect backbone prefix) + extract features. Used by the two below. |
| `postcp_normcv.py` | NEW | Colab | Exp A (F2.2): post-CP L2-norm CV per config. Manifest: `postcp_manifest_template.csv`. |
| `postcp_growth.py` | NEW | Colab | Exp C (F3): post-CP uniformity/overlap vs CP size. Manifest: `postcp_growth_manifest_template.csv`. |
| `sa_lp_compare.py` | NEW | anywhere | Exp B (F2.3): tabulate Selective-Aggregation LP recovery. Input: `outputs/sa_lp_input_template.csv` (fill from CP-repo `--aggregation` eval). |
| `postcp_sweep.py` | NEW | cluster (cp/ ckpts) | **Scaled Exp A**: auto-discovers ALL `cp/` checkpoints under `--ckpt-root`, computes post-CP L2-norm CV + uniformity for every (method,encoder,dataset,size,seed). Resumable, `--shard i/N` for SLURM. No ImageNet needed (overlap optional). |
| `postcp_offsphere.py` | NEW | anywhere | Analyses `postcp_sweep.csv`: the quantitative F2 law — does off-sphere movement (Δ norm-CV) predict ΔkNN degradation for MAE-CP on sphere encoders + the MAE-CP-vs-invariance contrast. |

**Where the narrative docs live now (repo-root-relative):** `../hypothesis/` = hypotheses
(`hypothesis/hypothesis_1.md`/`_eng.md` = previous, `HYPOTHESES_v2.md` = current) · `../findings/` =
`FINDINGS_v1.md` (original benchmark report) + `FINDINGS_step2.md` (Hyp-1 adjudication) +
`LITREVIEW_step3.md` (cited positioning) · `eval/PLAN_validation_and_writing.md` (validation+writing
plan, stays with the code) · `../papers/geometry/CORE_PAPERS.md` (annotated bibliography),
`../papers/methods/METHODS_PAPERS.md` (annotated method/backbone papers).

## What changed vs the original `hypothesis/` colab analysis (since removed)

- **FIX-1 (per-encoder Δ).** The original correlated each encoder's geometry
  against a single Δ that was *averaged across all encoders, methods and sizes*
  (`CP_OUTCOMES` hard-coded, encoder-independent). Geometry is per `(encoder,
  dataset)` and Δ depends on the encoder/method/size, so that was a
  methodological error. `correlate.py` pulls Δ from `results.xlsx` **per
  encoder**, at MAX, averaged over the chosen methods (default invariance =
  LeJEPA-CP + SimCLR-CP; change with `--methods`).
- **FIX-2 (no stale Δ table).** `CP_OUTCOMES` only had Δ for the original 8
  datasets. All 15 now come from `results.xlsx`.
- **NEW-1 (MMD components).** `mmd_rbf_components()` exposes `M_PP`(target self),
  `M_QQ`(imagenet self), `M_PQ`(cross) and `gamma`, so the identity claimed in
  `hypothesis/hypothesis_1.md` (`MMD² = M_PP + exp(L_unif(Q)) − 2 M_PQ`, asserted *exact*)
  can be **empirically tested** — it is not exact (uniformity uses t=2 but MMD
  uses a data-dependent γ; the self-kernel also includes the diagonal).
- **NEW-2.** uniformity also emitted at `t = γ_MMD` (`uniformity_at_gamma`) for
  the identity check.
- **NEW-3.** `neighbor_overlap` at both k=20 (matches kNN eval) and k=50 (doc).

## Run order

```bash
cd /Users/zhanghaodong/Desktop/CP

# 0. (runnable now, no geometry) Δ-structure sanity + ground-truth ranking
python eval/delta_structure.py

# 1. Recompute geometry. Two modes:
#    (a) ImageNet-free (uniformity + norm-CV, all 15×3; no gated dependency):
python eval/geometry_metrics.py --skip-imagenet \
    --output eval/outputs/geometry_15_noimnet.csv
#    (b) full (adds MMD / overlap / centroid; needs ImageNet-val pre-saved by
#        hypothesis/download_imagenet_val.py, and HF login + ILSVRC license):
python eval/geometry_metrics.py \
    --imagenet-dir <dir-from-download_imagenet_val> \
    --download-dir <raw> --processed-dir <arrow> \
    --output eval/outputs/geometry_15.csv

# 2. Correlate geometry → Δ (per-encoder) + all Hypothesis-1 checks
python eval/correlate.py --geometry eval/outputs/geometry_15.csv
```

## Dependencies

- `load_results.py`, `delta_structure.py`, `correlate.py`: `pandas`, `numpy`,
  `scipy`, `scikit-learn`, `openpyxl`.
- `geometry_metrics.py`: also `torch`, `timm`, `torchvision`, `datasets`, and the
  `stable_datasets` package (vendored under `continued-pretraining/stable-datasets`
  or `codebase/stable-datasets`; `pip install -e` it). Full mode needs ImageNet-val
  saved to disk and the gated `ILSVRC/imagenet-1k` license accepted.

## Outputs (`eval/outputs/`)

- `cp_long.csv` — tidy CP results (from `load_results.py`).
- `geometry_15[_noimnet].csv` — recomputed geometry (from `geometry_metrics.py`).
- `correlations_15.csv` — per-encoder geometry→Δ correlation table (from `correlate.py`).

## Notes / assumptions

- Geometry is computed on a ≤5000-sample stratified train subset; Δ is at MAX.
  Both are the "full-data" regime, so they are comparable, but they are not the
  identical sample set.
- The proper falsifiable test of Hypothesis 1 on the new datasets is the **15-point
  rank correlation** (`correlate.py` section A), not the per-dataset point ranges
  in `hypothesis/hypothesis_1.md` Finding 5 (those conflate semantic-FG label with geometric
  position and were calibrated at a different data size — see section B verdicts).
