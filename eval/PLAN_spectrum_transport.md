# Spectrum & Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution;
> this repo's analysis code is executed by the assistant locally for CPU and handed to the user
> as slurm scripts for GPU). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved eval/DESIGN_spectrum_transport.md — (1) bilinear unified law on
existing data, (2) Exp I layer-wise virtual encoders, (3) Exp J transport-field decomposition.

**Architecture:** All analysis code under `continued-pretraining/eval/` (CPU scorers under
`eval/adjudicate/`), slurm under `run/slurm/eval/` following the exp_e staging template. GPU
scripts reuse the production stack: `geometry_metrics.ENCODERS / load_target_dataset /
extract_features`, `postcp_features.load_cp_backbone`, `geometry_class.class_manifold_stats`.

**Tech stack:** python3, numpy/pandas/scipy/sklearn/torch/timm; slurm (V100, partition=nvidia,
account=civil).

## Global constraints

- R0: everything on disk in English. R3: assistant runs CPU locally; user runs GPU via slurm.
- Deterministic sample correspondence: `load_target_dataset` is shuffle=False + stratified
  seed-42 subset → the i-th sample is THE SAME image pre and post (required by Exp J).
- No `Date.now`-style nondeterminism; RNGs seeded (RandomState(0) for the internal kNN split).
- Main-grid ckpt layout: `<root>/<method>/pretrained/<DsFolder>/cp/<ds>_<timm>_n<N>_s<seed>.ckpt`
  (root default `/scratch/gs4133/zhd/CP/outputs/ckpts/cp`). SigLIP grid layout:
  `<siglip-root>/cp/<method>/<DsFolder>/SigLIP/cp/..._n<N>_s<seed>.ckpt`
  (root default `/scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip`) — verify with `ls` before
  the array run; discovery must skip silently when the root is absent.
- MAX sizes per dataset = MAX_N dict in postcp_class_sweep.py (reuse verbatim).
- Internal layer-kNN protocol (Exp I only, disclosed): stratified 80/20 split of the ≤5000
  subset (RandomState(0)), KNeighborsClassifier(n_neighbors=20, metric="cosine"), macro-F1.
  NOT the production kNN; consistent across layers/pre/post so Δ_ℓ is internally valid.
- Layer readout protocol (Exp I): block outputs (pre final-norm), pooled on the fly:
  DINOv3/CLIP cls (token 0); MAE mean over tokens[num_prefix_tokens:]; SigLIP mean over all
  tokens (no cls; MAP is final-layer-only — disclosed).

---

### Task 1: Design 1 — `eval/adjudicate/bilinear_law.py` (CPU, run now)

**Files:** Create `eval/adjudicate/bilinear_law.py`; output `eval/outputs/bilinear_law.csv`.

**Inputs:** `eval/outputs/geometry_15.csv`, `geometry_class_15.csv`, `cp_long_refreshed.csv`,
`c2_siglip_score.csv`, `results.xlsx` (SigLIP sheet, for SigLIP pre-CP kNN).

**Produces:** console verdict block V1–V4 + CSV with one row per (encoder, dataset) cell:
`encoder, dataset, uniformity_t2, x_z, theta_cdnv, theta_margin, theta_knnpre, dknn, n_methods`.

- [ ] Build the 60-cell table: D3/CLIP/MAE = mean dknn over {LeJEPA-CP, SimCLR-CP, DIET-CP}
      at is_max from cp_long_refreshed; SigLIP = c2 real_dknn (2-method, disclosed in header).
- [ ] θ_E variants per encoder across 15 datasets: |ρ(unif, cdnv)| (primary),
      |ρ(unif, center_margin)|, |ρ(unif, knn_pre)| (knn_pre: main grid from cp_long_refreshed
      @MAX mean; SigLIP from results.xlsx "By Method (SigLIP)" sheet col H).
- [ ] Models M1 x | M2 x + x·gate | M3 x + x·θ | M4 x·θ, x = within-encoder z(unif); rank
      regression with dataset-grouped LOO (15 folds; all encoder rows of a dataset out).
- [ ] Block bootstrap (resample the 15 datasets, 500 reps) for the M4−M2 LOO difference CI.
- [ ] Print V1 (M4 within 0.02 of M2, ≥0.04 over M1), V2 (θ gap > 0.15), V3 (sign flip:
      sphere ρ(unif,knn_pre)<0, MAE>0), V4 (variant direction consistency). Run locally,
      record verdicts. Expected from probes: V1 pass (0.498/0.501/0.438), V2 pass
      (0.17 vs 0.39), V3 pass pending SigLIP value.

### Task 2: Exp I pre-CP — `eval/layerwise_geometry.py` (GPU)

**Files:** Create `eval/layerwise_geometry.py`; shard output
`eval/outputs/layerwise_pre_shards/<dataset>.csv` (concat → `layerwise_pre.csv`).

**Interfaces (consumed by Tasks 3-4):** CSV columns
`encoder, dataset, layer (1-12), n_samples, uniformity_t2, l2_norm_cv, rankme, numerical_rank,
cdnv, center_margin, knn_internal`.

- [ ] `LayerTap` helper: register forward hooks on `model.blocks[i]`; each hook pools its
      block output per the readout protocol and appends float32 CPU arrays; `collect(loader)`
      returns `{layer: (N,768)}`. num_prefix_tokens read from the timm model.
- [ ] `internal_knn(feats, labels)`: 80/20 stratified split (RandomState(0)), cosine k=20,
      macro-F1. Skip (nan) if any class has <2 samples or train < k.
- [ ] Main loop: for each encoder in {DINOv3, MAE, CLIP, SigLIP} × the task's dataset:
      one forward pass through LayerTap; per layer compute uniformity_t2, l2_norm_cv, rankme,
      numerical_rank, cdnv + center_margin (via class_manifold_stats), knn_internal.
      No ImageNet anywhere.
- [ ] `python -m py_compile`; CLI mirrors geometry_class.py (--datasets --download-dir
      --processed-dir --output --device).

### Task 3: Exp I post-CP — `eval/adjudicate/layerwise_postcp.py` (GPU)

**Files:** Create `eval/adjudicate/layerwise_postcp.py`; shard output
`eval/outputs/layerwise_postcp_shards/<dataset>.csv` (concat → `layerwise_postcp.csv`).

**Interfaces (consumed by Task 4):** CSV columns
`method, encoder, dataset, size, seed, layer, uniformity_t2, knn_internal, ckpt`.

- [ ] Discovery A (main grid): reuse postcp_class_sweep.discover() filtered to methods
      {LeJEPA, SimCLR, DIET} (folder names as found; NO MAE-CP method — encoder MAE stays).
- [ ] Discovery B (SigLIP grid): walk `<siglip-root>/cp/<method>/<DsFolder>/SigLIP/cp/*.ckpt`,
      MAX-only, methods {LeJEPA, SimCLR}; encoder=SigLIP, timm_id siglip v2_webli; absent root
      → empty list + warning.
- [ ] Per ckpt: load_cp_backbone → LayerTap forward (import LayerTap from
      eval/layerwise_geometry.py) → per layer uniformity_t2 + internal kNN (IDENTICAL split
      seed as Task 2). Append-mode CSV with done-set resume (postcp_class_sweep idiom).
- [ ] `--shard i/N`, `--datasets`, `--ckpt-root`, `--siglip-root` CLI; py_compile.

### Task 4: Exp I scorer — `eval/adjudicate/layerwise_law.py` (CPU, runs after results land)

**Files:** Create `eval/adjudicate/layerwise_law.py`.

**Consumes:** `layerwise_pre.csv`, `layerwise_postcp.csv`.
**Produces:** `eval/outputs/layerwise_curve.csv` (one row per encoder × layer:
`encoder, layer, theta_coupling, rankme_med, law_rho, n_datasets, n_methods`) + verdicts L1-L3.

- [ ] Δ_ℓ per (encoder, dataset, layer): seed- and method-averaged post knn_internal − pre.
- [ ] Per (encoder, layer): θ_ℓ = |ρ(unif_ℓ, cdnv_ℓ)| across datasets; law_ρ_ℓ =
      Spearman(unif_ℓ(D), ΔkNN_ℓ(D)).
- [ ] L2 main: Spearman(θ_ℓ, law_ρ_ℓ) over all (encoder, layer) points; encoder-block
      bootstrap (resample encoders, 2000 reps) + within-encoder layer-level Spearman.
- [ ] L1 sanity (rank profiles), L3 (MAE middle-layer coupling vs sphere floor + local law
      recovery). Print verdict block; BH-FDR over the correlation family.

### Task 5: Exp J — `eval/adjudicate/transport_field.py` (GPU)

**Files:** Create `eval/adjudicate/transport_field.py`; shard output
`eval/outputs/transport_field_shards/<dataset>.csv` (concat → `transport_field_max.csv`).

**Interfaces (consumed by Task 6):** CSV columns
`method, encoder, dataset, size, seed, n, total_energy, trans_energy, between_energy,
within_energy, resid_identity, mu_norm, cos_mu_imagenet, toward_imagenet, ckpt`.

- [ ] Discovery: postcp_class_sweep.discover() unfiltered (ALL 4 methods incl MAE-CP) —
      main grid only (3 encoders).
- [ ] Per task dataset, per encoder: pretrained model → pre feats (production pool) on the
      target loader + ImageNet-val feats → v_E = normalize(mean(normalize(f_imn))). Cache pre
      feats in RAM for the task.
- [ ] Per ckpt: post feats (same loader order); L2-normalize both; d = post_n − pre_n;
      μ_d = d.mean(0); per-class means μ_c;
      `total = mean(||d||²)`, `trans = ||μ_d||²`,
      `between = Σ_c (n_c/N)||μ_c − μ_d||²`, `within = mean(||d_i − μ_c(i)||²)`,
      `resid_identity = total − trans − between − within` (must be ~1e-10),
      `toward_imagenet = ⟨μ_d, v_E⟩`, `cos_mu_imagenet = toward/||μ_d||`.
- [ ] Append-mode CSV + done-set resume; `--shard`, `--datasets` CLI; py_compile.

### Task 6: Exp J scorer — `eval/adjudicate/transport_law.py` (CPU, runs after results land)

**Files:** Create `eval/adjudicate/transport_law.py`.

**Consumes:** `transport_field_max.csv`, `postcp_sweep_fixed.csv` (Δoverlap at MAX),
`cp_long_refreshed.csv` (dknn/dft @MAX).
**Produces:** verdict block T1-T4 + `eval/outputs/transport_stats.csv`.

- [ ] Cell means (method, encoder, dataset) over seeds; join behavior + Δoverlap.
- [ ] T1: per sphere encoder, ρ(toward_imagenet, d_overlap) pooled over angular methods;
      method contrast: median toward_imagenet — LeJEPA/SimCLR vs DIET vs MAE-CP
      (Mann-Whitney); verdict = invariance > DIET AND correlation sign as predicted.
- [ ] T2: ρ(within_share = within/total, dknn) per encoder; FG vs OOD medians.
- [ ] T3: max |resid_identity| < 1e-8 → PASS (else numerical bug — stop and debug).
- [ ] T4 exploratory: ρ(between_energy, dft). BH-FDR over the T-family; rank-claims wording.

### Task 7: slurm scripts (3 files, exp_e template verbatim staging)

**Files:** Create `run/slurm/eval/exp_i_layerwise_pre.sh` (single job, no array: 4 encoders ×
15 datasets, needs ALL datasets staged? NO — loop datasets inside, stage skipped: light IO;
run against /scratch directly, 6h), `run/slurm/eval/exp_i_layerwise_post.sh` (array 0-14,
stages its dataset, calls layerwise_postcp.py --datasets $DS; 12h),
`run/slurm/eval/exp_j_transport.sh` (array 0-14, stages dataset + ImageNet-val, calls
transport_field.py --datasets $DS; 12h).

- [ ] Copy exp_e_geometry_class.sh header/staging blocks verbatim (partition, account, V100,
      exclude cn253,cn259, miniconda, conda env, PYTHONPATH, staging loop TMPDIR→/tmpdata→
      /dev/shm, trap cleanup). exp_i_pre: NO ImageNet staging. exp_i_post: NO ImageNet.
      exp_j: ImageNet staged (as exp_e).
- [ ] Concat one-liners in each header comment (shards → final CSV), throttle tip
      `--array=0-14%5`.
- [ ] `bash -n` all three.

### Task 8: run Task 1 locally, validate everything, update docs/memory

- [ ] Run bilinear_law.py; verify V1-V3 verdicts match the probe expectations; V3 SigLIP
      value newly determined — record.
- [ ] `python -m py_compile` on all new python; `bash -n` on all slurm.
- [ ] Update eval/PLAN_iclr.md status block (new phase: Spectrum & Transport, Exp I/J pending
      cluster) and memory (second-axis-phase.md or new memory file) with the handover state.
- [ ] Report to user: Design 1 verdicts + exact sbatch commands + what to send back
      (layerwise_pre.csv, layerwise_postcp.csv, transport_field_max.csv).
