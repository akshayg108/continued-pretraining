# ICLR push — Implementation Plan

> Execute task-by-task; steps use `- [ ]`. Heavy GPU tasks the user runs on the cluster; analysis
> tasks run on CPU (python3 locally / python on cluster).

**Goal:** Raise the CP-geometry study from a correlational analysis to an ICLR-grade paper with
(C1) a contrarian regime-gated finding, (C2) an **actionable, out-of-sample-validated pre-CP
decision rule**, and (C3) a mechanism (two-force) with the off-sphere alternative ablated.

**Architecture:** Reuse the verified findings (F1/F2/F3) as evidence; add the C2 decision-rule
experiment (the ICLR-critical "so what") and cross-encoder generalization (SigLIP); finish Exp B/D;
present with out-of-sample metrics and multiple-comparison-aware stats. Spec: `../../hypothesis/HYPOTHESES_v3.md`.

**Tech stack:** python3 (pandas/scipy/scikit-learn), the `eval/` analysis library, SLURM (CP runs),
stable_datasets, timm/torch.

Priority order = ICLR impact: **T1 (decision rule) > T2 (SigLIP cross-encoder) > T5 (combined-force, free) > T3 (Exp B) > T4 (Exp D) > T6 (writing)**.

---

### Task 1 — C2: the actionable pre-CP decision rule (HIGHEST PRIORITY)
**Files:** Create `eval/f1_position/predictor.py`; reads `eval/outputs/geometry_15.csv` + `results.xlsx`.
Turns F1 into a tool: from **pre-CP geometry only**, predict whether CP helps kNN (sign of ΔkNN),
evaluated out-of-sample, beating baselines.

- [ ] **Step 1 — define the rule.** Features = pre-CP `neighbor_overlap_k50`, `uniformity_t2` (per
  encoder,dataset). Target = `sign(ΔkNN@MAX)` (mean over LeJEPA-CP+SimCLR-CP). Model =
  `sklearn.linear_model.LogisticRegression` on the 2 features (standardized).
- [ ] **Step 2 — out-of-sample eval (the defense against n=15):**
  - **Leave-one-dataset-out** on sphere encoders (DINOv3+CLIP, 30 points): report balanced-accuracy + AUC of predicting CP-helps-kNN.
  - **Cross-encoder transfer:** fit on DINOv3+CLIP, test on the held-out 4th sphere encoder **SigLIP** (Task 2) — the strongest generalization claim.
- [ ] **Step 3 — baselines to beat:** (a) always-predict-help (majority), (b) random, (c)
  **Sorkhei-style**: use pre-CP `knn_pre` (frozen quality) as the score. Show our geometric rule
  beats all three; Sorkhei-style should be near-chance (its monotone assumption fails here = C1 made actionable).
- [ ] **Step 4 — the FT side & the compute-saving framing:** repeat for `sign(ΔFT)` (opposite-sign
  rule); quantify "compute saved by skipping CP where predicted to hurt, at X% recall of the harmful cases."
- [ ] **Step 5 — run + record:** `python eval/f1_position/predictor.py` → `eval/outputs/predictor.csv` + console (LOO/cross-encoder bal-acc/AUC vs baselines).
- **Success criteria (ICLR):** out-of-sample bal-acc clearly > majority and > Sorkhei-style on sphere encoders; cross-encoder (SigLIP) transfer holds; FT rule opposite-sign confirmed. This is the make-or-break result.

### Task 2 — SigLIP-2: a 4th sphere encoder for cross-encoder generalization
**Files:** add `SigLIP` to `eval/geometry_metrics.py::ENCODERS` (`vit_base_patch16_siglip_224.v2_webli`,
**pool=map** — SigLIP's native MAP attention-pool head, NOT cls/mean); `stable_cp/utils/backbone.py`
`BACKBONE_DIMS` (768); `--pool-strategy map` in `continued_pretraining.py` + the `_extract_embedding`
paths; new CP SLURM scripts via `run/slurm/generate_siglip_scripts.py`.
- [x] **Step 1** register SigLIP-2 backbone — DONE: geometry ENCODERS (pool=map via `forward_head`/
  `attn_pool`+`fc_norm`), BACKBONE_DIMS=768, `--pool-strategy map` wired through lejepa/simclr/diet
  forwards + sft_eval + zero_shot_eval; MAP head stays frozen (only `blocks[-N:]` unfreeze). 30 scripts
  generated (`siglip_run_max.sh`, MAX×{LeJEPA,SimCLR}). **Smoke-test 1 job on cluster before full launch.**
- [ ] **Step 2** pre-CP geometry for SigLIP on the 15 datasets → append to `geometry_15.csv` (CPU/GPU, no training).
- [ ] **Step 3** minimal CP grid: `SigLIP × {LeJEPA,SimCLR} × 15 datasets × MAX × 3 seeds` (the F1/predictor regime); reuse `run_postcp_sweep.sh` discovery + `run_postcp_analysis.sh`.
- [ ] **Step 4** confirm SigLIP is sphere-native (norm-CV < 5%); feed into Task 1 cross-encoder test.
- **Success criteria:** predictor fit on DINOv3+CLIP transfers to SigLIP (Task 1 Step 2). If SigLIP behaves like DINOv3/CLIP → C1 strengthened to 3 sphere encoders.

### Task 3 — Exp B: SA-LP (resolve MAE-CP mechanism, close a caveat)
**Files:** CP-repo `--aggregation` eval on MAE-CP ckpts → fill `eval/outputs/sa_lp_input.csv` → `eval/f2_mechanism/sa_lp_compare.py`.
- [ ] **Step 1** run the CP repo's `--aggregation` (ABMILP depth-1) eval on the MAE-CP checkpoints (DINOv3/CLIP/MAE), recording `pre_linear_f1`, `post_linear_f1`, `post_sa_lp_f1`.
- [ ] **Step 2** `python eval/f2_mechanism/sa_lp_compare.py --input eval/outputs/sa_lp_input.csv --out eval/outputs/sa_lp.csv` → recovery fraction & verdict (aggregation-failure vs information-loss).
- **Success criteria:** a clear verdict for MAE-CP degradation; folds into F2 as a closed sub-question.

### Task 4 — Exp D: texture sub-regime (turn DTD from anomaly to characterized boundary)
**Files:** per `eval/PLAN_texture_regime_extension.md` (KTH-TIPS-2b, FMD loaders → CP grid → `eval/texture_regime_test.py`).
- [ ] Execute that plan; run `texture_regime_test.py`.
- **Success criteria:** SUPPORTED (texture = embedded-but-task-orthogonal, like DTD) → a secondary finding; or REFUTED → DTD reported as a lone characterized exception. Either way removes the open caveat.

### Task 5 — P2.4: combined-force predictor (mechanism's teeth; RUNNABLE NOW)
**Files:** Create `eval/f2_mechanism/forces_combined.py`; reads `postcp_sweep.csv` + `geometry_15.csv` + `results.xlsx`.
- [ ] **Step 1** per sphere encoder, fit (rank) `ΔkNN ~ Δuniformity + Δoverlap`; report incremental rank-R² over each single force and per-method force weights.
- [ ] **Step 2** `python eval/f2_mechanism/forces_combined.py` → `eval/outputs/forces_combined.csv`.
- **Success criteria:** combined > either single force; quantified spread-vs-collision weights → the F2 mechanism figure.

### Task 6 — Paper assembly (figures, stats, sections)
**Files:** `paper/` (new): figs from `eval/outputs/*`; sections Intro(C1–C3)/Related(Sorkhei,Kumar)/Method(predictor)/Experiments/Ablations(norm-CV refuted)/Limits.
- [ ] Fig 1 regime-gating (sphere vs MAE sign flip); Fig 2 decision-rule out-of-sample (ROC vs baselines); Fig 3 two-force mechanism; Fig 4 growth/transport.
- [ ] Stats pass: pre-registered signs, multiple-comparison-aware p-values, out-of-sample metrics foregrounded over in-sample ρ.
- **Success criteria:** every claim in `HYPOTHESES_v3.md` C1–C3 has a figure/table + an out-of-sample or ablation defense.

---

## Self-review
- **Coverage:** C1→T2/T6 (gating across 3 sphere encoders); C2→T1/T2 (decision rule + cross-encoder); C3→T5/T3 (forces + ablation); caveats→T3/T4. ✓
- **The ICLR-critical gap (actionability + out-of-sample) is T1+T2** — do these first; everything else is supporting.
- **Risk:** if T1's out-of-sample rule does NOT beat baselines, the actionable framing fails → fall back to the regime-gating finding + mechanism (still a paper, weaker) or retarget TMLR. Decide after T1.
