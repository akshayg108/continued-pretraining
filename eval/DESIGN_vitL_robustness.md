# Design — ViT-L scale-robustness spot check (approved 2026-07-06)

Purpose: defuse the most predictable ICLR review attack ("all results are at a single
backbone scale, ViT-B/16"). One held-out-SCALE encoder: DINOv3 **ViT-L/16**
(timm `vit_large_patch16_dinov3.lvd1689m`, 24 blocks, 1024-dim — same pretraining method
and corpus as our DINOv3 ViT-B, so the scale axis is isolated). Spot check, not a grid.

## Scope (user-specified)

- 7 datasets = 3 OOD + 3 FG spanning the position spectrum + dtd (user-amended, final):
  super-large extremes replaced by the next-furthest, and organamnist dropped for cost.
  DINOv3 uniformity spectrum: OOD side **galaxy10 (−0.54, OOD extreme) →
  dermamnist (−1.21; octmnist −1.23 is huge, skipped) → eurosat (−2.29)**; FG side
  **fgvc_aircraft (−3.00) → cars196 (−3.39) → cub200 (−3.52, FG extreme; food101 −3.60
  is huge, skipped; positionally-tied flowers102 excluded as a known Δ null)** +
  **dtd (−3.67)** as the pre-registered texture exception.
- Methods: LeJEPA + SimCLR + DIET × MAX × 3 seeds (42/43/44) = 63 CP runs + 21 pre-CP runs.
- Unfreeze schedule: same size rule as the main grid — <10k → last 2 blocks
  (dermamnist/fgvc/cars196/cub200/dtd), 10–25k → 4 (galaxy10/eurosat); no ≥25k dataset
  by construction (largest is eurosat @16200).
  NOTE (disclosed): block counts are ABSOLUTE, so the trained fraction differs
  (2/24 vs 2/12 of depth) — recipe-identical in blocks, not in fraction.
- Recipe identical to the main grid otherwise: 150 epochs, effective batch 256
  (DIET: 32), lr 1e-4, wd 0.05, freeze 15 epochs, kNN k=20, pool cls.
  ViT-L memory adaptation only: LeJEPA/SimCLR/pre-SFT on a100-80g with
  accumulate-grad-batches=2 (per-step 128); DIET (batch 32) stays on v100.

## Pre-registered criteria

- **R1 geometry rank stability (zero-training appetizer, runs first):** across the 15
  datasets, Spearman(uniformity_t2 ViT-B, ViT-L) > 0.8 and Spearman(overlap_k50 B, L)
  > 0.8. If this fails, the geometry readings themselves are scale-unstable and the CP
  runs should be reconsidered before spending GPU.
- **R2 decision-score sign transfer (headline):** the pre-CP decision rule
  (overlap + uniformity, within-encoder z-scores, coefficients FROZEN from the ViT-B
  encoders — no refitting) applied to ViT-L predicts sign(ΔkNN @MAX, 3-method mean) on
  ≥5/7 datasets. Expected miss: dtd (the pre-registered texture exception on every sphere
  encoder so far). 6/7 with dtd-only miss = clean pass.
- **R3 position law (point estimate only, n=7):** Spearman(uniformity, ΔkNN) > 0
  across the 7 datasets. No significance claim at n=7 — direction only.
- **R4 reversal direction (exploratory):** Spearman(uniformity, ΔFT) < 0 (opposite sign
  to R3), reported as exploratory.

## Artifacts

- `run/slurm/cp-L/pre-cp/dinov3L_pre.sh` — array 0-6; pre kNN/LP + pre-SFT baselines.
- `run/slurm/cp-L/cp/{lejepa,simclr,diet}_max.sh` — array 0-6 each; CP + post kNN/LP/SFT.
- `run/slurm/cp-L/eval/geometry_vitL.sh` + `eval/geometry_vitL.py` — the R1 appetizer:
  ViT-L pre-CP geometry on all 15 datasets + ImageNet-val; prints the rank-stability
  check against geometry_15.csv at the end. Zero training; run this FIRST.
- Scoring after results land: assistant runs R2-R4 locally (bilinear/predictor code paths).

## Paper landing

One robustness subsection + one appendix table: "the pre-CP decision rule transfers not
only to a held-out pretraining method (SigLIP-2, 87%) but to a held-out model scale
(DINOv3 ViT-L, k/7)". Finding 4's gate is NOT re-tested here (single encoder) — scale
robustness claims are limited to Findings 1/5 quantities.
