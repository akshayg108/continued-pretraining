# Second-Axis & Adjudication-Fixes Plan (companion to PLAN_iclr.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the adjudication findings of 2026-07-01 into (a) corrected numbers/disclosures
in the findings docs, and (b) tested second-axis complements (P-A..P-D) to uniformity+collision —
complements, NOT replacements (user constraint 2026-07-01).

**Architecture:** CPU adjudication analyses live in `eval/adjudicate/` (already written & run).
New GPU measurement scripts follow the geometry_metrics.py pipeline conventions and land CSVs in
`eval/outputs/`; a single CPU scorer (`eval/adjudicate/correlate_second_axis.py`) pre-registers
the falsifiable verdicts and runs on whatever CSVs exist. GPU jobs are run by the USER on the
cluster (compute boundary rule); everything else is local CPU.

**Tech stack:** python3 + pandas/scipy/sklearn (CPU); torch/timm + stable_datasets +
stable_cp.evaluation.abmilp (GPU, cluster env).

## Global constraints

- Second axes COMPLEMENT uniformity/collision; only replace if evidence is very strong (user).
- Exp D (new texture datasets) is deprioritized (user, 2026-07-01); P-B is its no-new-data probe.
- All claims rank-based (Spearman); honesty guard = cross-encoder sign consistency; n disclosed.
- F2/F3 sphere-cell geometry and every DIET number are PROVISIONAL until the 60-cell reruns land
  (`eval/outputs/rerun_geometry.csv`; validators `eval/rest/test1-4`).
- Do not cite: DIET.pdf for "angular objective" (paper uses unnormalized features + plain CE —
  argue from OUR stable_cp implementation); RankMe as cross-method quality score (Garrido et al.
  explicitly warn against it — we use it as a geometric descriptor only); Wang–Isola Thm 1 for
  data-size claims (its asymptotics are in #negatives M).

---

### Task 0: Fix the Sorkhei-control numbers in FINDINGS_step2.md  [CPU, no approval needed — factual correction]

**Files:**
- Modify: `findings/FINDINGS_step2.md` (section "Pre-CP Sorkhei control")

**Why:** Re-derivation (2026-07-01) showed pooled ρ=+0.715 (n=60) includes the FROM SCRATCH
random-init baseline as a 4th "encoder" (alone ρ=+0.889, drags pooled up), and "sphere-only
+0.790 (n=45)" = DINOv3+CLIP+FROM SCRATCH. Honest numbers, verified:

| slice | rho | n |
|---|---|---|
| 3 pretrained encoders pooled | +0.653 (p=1.2e-6) | 45 |
| DINOv3+CLIP | +0.764 | 30 |
| DINOv3+CLIP+SigLIP (xlsx SigLIP sheet) | +0.704 | 45 |
| per-encoder DINOv3 / CLIP / MAE | +0.789 / +0.643 / +0.621 (unchanged) | 15 each |

- [ ] **Step 1:** Replace "pooled Spearman ρ = **+0.715** (p≈1.3e-10, n=60 enc×ds)" with
  "pooled Spearman ρ = **+0.653** (p≈1.2e-6, n=45 enc×ds, 3 pretrained encoders)" and replace
  "**sphere-only +0.790** (n=45)" with "sphere-only **+0.764** (DINOv3+CLIP, n=30; +0.704 adding
  held-out SigLIP-2, n=45)". Add one sentence: "An earlier draft pooled the FROM-SCRATCH baseline
  as a 4th encoder (ρ=+0.715/+0.790); corrected 2026-07-01."
- [ ] **Step 2:** Re-verify by running the one-liner below; expected output `+0.653 45`:

```bash
cd continued-pretraining && python3 - << 'EOF'
import pandas as pd; from scipy.stats import spearmanr
cp = pd.read_csv('eval/outputs/cp_long.csv')
m = cp[cp.is_max & cp.Backbone.isin(['DINOv3','CLIP','MAE'])].drop_duplicates(['Backbone','dataset_key'])
r,_ = spearmanr(m.knn_pre, m.ft_pre); print(f"{r:+.3f} {len(m)}")
EOF
```

### Task 1: Add the new disclosures to HYPOTHESES_v3.md / findings docs  [CPU]

**Files:**
- Modify: `hypothesis/HYPOTHESES_v3.md` (Finding 2 caveats; Scope & honesty)
- Modify: `findings/FINDINGS_step4.md` (Honest caveats)

Content to add (verified 2026-07-01, `eval/adjudicate/`):

- [ ] **Step 1:** Collision-force scope: "the collision partial is carried by the invariance
  methods (LeJEPA −0.47, SimCLR −0.41 on clean cells); DIET-CP (−0.07 n.s. on clean cells) and
  MAE-CP (+0.07 n.s.) show no collision force. Spread is method-universal; collision is
  invariance-specific. (Re-check DIET after the reruns.)" Also record the positives: the
  collision partial survives log-size control (−0.35), within-size strata (−0.34/−0.32),
  dataset-LOO (range −0.49..−0.31), and dropping the 60 rerun cells (−0.37).
- [ ] **Step 2:** ΔFT signal-to-noise (defuses "ΔFT is noise"): "|ΔFT| is small (median seed-z
  3.2 vs 43 for ΔkNN) but not noise: 63% of invariance@MAX cells have |ΔFT|>2σ_seed; LeJEPA vs
  SimCLR sign agreement 39/45=87% (binomial p<1e-4); on CLIP the reversal STRENGTHENS on the
  z≥1 subset (mmd −0.82, cosine −0.88); geometry→ΔFT partials survive controlling FT headroom
  (1−ft_pre) on DINOv3 (all 4 metrics p<0.05) and partially on CLIP. Underpowered spot: DINOv3
  z≥1 subset is n=9, n.s."
- [ ] **Step 3:** Heterogeneity dose control (RESOLVED, see Task 5): "the CUB-vs-Flowers contrast
  survives dose matching — at ~200 CP samples (1/5 of Flowers' MAX) CUB already loses −0.35
  (DINOv3) while Flowers@1020 loses −0.002; the heterogeneity is structural, not a CP-set-size
  artifact." Add the Task-5 table to FINDINGS_step2.

### Task 2: Citation fixes  [CPU, before any paper draft]

**Files:**
- Modify: `papers/geometry/CORE_PAPERS.md`, `findings/LITREVIEW_step3.md`

- [ ] **Step 1:** `Simon2024_HypersphericalProto.pdf` is actually Lindström, Rodríguez-Gálvez,
  Thobaben, Skoglund (GRaM Workshop @ ICML 2024, arXiv:2407.07664). Fix the key/attribution.
- [ ] **Step 2:** Sorkhei headline statistic is weighted Kendall τ_w = 0.73 (average across
  settings; labeled kNN, 80/20 split, k=200, cosine) — not "Spearman ρ 0.72"; their RTP compares
  pretrained-FT vs random-init-FT (NOT a post-vs-pre Δ; equating it with CP Δ is a category
  error). Fix wording in both files.
- [ ] **Step 3:** Note Cossu PDF = 2022 arXiv v1 (published version: Neural Networks 2024);
  Tomihari & Sato venue = NeurIPS 2024 (PDF carries no venue — verify at citation time).
- [ ] **Step 4:** DIET-CP "angular" classification: cite our stable_cp implementation
  (L2-normalized embeddings + cosine logits), NOT DIET.pdf (unnormalized + plain CE).

### Task 3: GPU batch 1 — second-axis pre-CP geometry  [USER runs on cluster]

**Files (already written, compile-checked):**
- Run: `eval/geometry_class.py` → `eval/outputs/geometry_class_15.csv`
- Then CPU: `eval/adjudicate/correlate_second_axis.py` (blocks P-A, P-B, P-C1)

**Interfaces:** geometry_class_15.csv columns: encoder, dataset, n_samples, n_classes,
within_spread, between_spread, wb_ratio, nc1_ratio, center_margin, cdnv, task_energy_top10/50,
task_energy_in_top10/50, rankme, rankme_raw, alpha_req, twonn_id.

- [ ] **Step 1 (user, cluster):**

```bash
python eval/geometry_class.py --imagenet-dir <imagenet_val_dir> \
    --download-dir <raw> --processed-dir <arrow> \
    --output eval/outputs/geometry_class_15.csv
# cost: 1 forward pass per (4 encoders x 15 datasets + 4 x ImageNet-5000) — same as geometry_metrics.py
```

- [ ] **Step 2 (local CPU):** `python3 eval/adjudicate/correlate_second_axis.py`
  Pre-registered verdicts printed by the script:
  - P-A SUPPORTED if rho(wb_ratio|cdnv, position-residual) < 0 on BOTH sphere encoders (FG-only)
    AND wb_ratio(CUB) > wb_ratio(Flowers) on both. REFUTED otherwise.
  - P-B SUPPORTED if DTD ranks bottom-2 on task_energy_in_top50 among the 7 FG datasets on BOTH
    sphere encoders. REFUTED otherwise (then DTD stays "characterized exception, Exp D deferred").
  - P-C1 SUPPORTED if mean rankme orders MAE far below the three sphere encoders AND per-encoder
    |rho(uniformity,Δknn)| orders with rankme (gate becomes a continuum). Note: we use rankme as
    a geometric DESCRIPTOR (Garrido et al. forbid cross-method QUALITY comparison — disclose).

### Task 4: GPU batch 2 — MAE readout-confound ablation (defensive, HIGH priority)  [USER]

**Files (written):** `eval/adjudicate/mae_sa_geometry.py` → `eval/outputs/geometry_mae_sa.csv`

- [ ] **Step 1 (user, cluster):**

```bash
python eval/adjudicate/mae_sa_geometry.py --imagenet-dir <dir> \
    --download-dir <raw> --processed-dir <arrow> --out eval/outputs/geometry_mae_sa.csv
# cost: 1 MAE forward per dataset + 15 SA-head trainings (Exp-B recipe) + ImageNet pooling per head
```

- [ ] **Step 2 (local CPU):** rerun `correlate_second_axis.py` → P-C2 block.
  - If MAE F1 correlations stay inverted/n.s. on SA features → gate survives its strongest
    confound; report as ablation in C1.
  - If the sphere law reappears (uniformity +, overlap −) → "off-sphere encoder" leg collapses
    into readout failure; C1 must be restated (objective gating + readout), Beyond-[cls] moves
    from Exp-B citation to core mechanism. Either way write result into FINDINGS_step2 addendum.

### Task 5: Matched-dose heterogeneity control — RESOLVED ON EXISTING DATA (no run needed)

**Status: DONE 2026-07-01.** The planned CUB@1020 mini-run is unnecessary: size sweeps exist for
ALL 15 datasets (the "new-7 MAX-only" notes in load_results.py / PLAN_iclr.md were outdated —
fixed). `eval/adjudicate/heterogeneity_probe.py` (MATCHED-DOSE block) shows, invariance mean:

| dose | CUB200 dknn (DINOv3 / CLIP) | Flowers102 dknn (DINOv3 / CLIP) |
|---|---|---|
| ~200 vs 102 | −0.352 / −0.128 | −0.025 / +0.049 |
| 500 | −0.317 / −0.170 | −0.007 / +0.029 |
| ~1000 vs 1020 | **−0.469 / −0.274** | **−0.002 / +0.022** |

CUB at ONE FIFTH of Flowers' dose is already damaged ≥10x more. **Dose confound REFUTED**; the
within-FG heterogeneity is a property of the datasets' representation structure → P-A (Task 3)
is the live explanation. Fold this table into the FINDINGS_step2 heterogeneity paragraph
(replaces the softer dose-confound wording of Task 1 Step 3).

### Task 6: GPU batch 3 — P-D reversal mediation sweep (MAX-only)  [USER, after batches 1-2]

**Files (written):** `eval/adjudicate/postcp_class_sweep.py` → `eval/outputs/postcp_class_max.csv`

- [ ] **Step 1 (user, cluster, shardable):**

```bash
python eval/adjudicate/postcp_class_sweep.py --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
    --download-dir <raw> --processed-dir <arrow> --out eval/outputs/postcp_class_max.csv
# ~540 MAX ckpts x 1 forward pass each; use --shard i/N for arrays
```

- [ ] **Step 2 (local CPU):** rerun `correlate_second_axis.py` → P-D block. SUPPORTED if
  rho(Δwithin_spread, ΔFT) < 0 pooled sphere@MAX AND partialling Δwithin out of
  (overlap→ΔFT) shrinks it toward 0 AND DIET has the largest Δwithin among angular methods.

### Task 7: Paper-section deltas (conditional, after Tasks 3-6)

- [ ] C1 section: add norm-CV + rankme continuum table (P-C1) and the SA-readout ablation (P-C2).
- [ ] F1 section: add the second-axis paragraph — position (where) × organization (how tiled:
  cdnv/wb_ratio) if P-A supported; else keep heterogeneity as disclosed limit with the
  matched-dose result.
- [ ] DTD: replace "Exp D needed" with the P-B spectral-task-alignment result (supported or
  refuted, both close the caveat narratively; Exp D stays optional future work).
- [ ] Reversal/C2 section: fold in the ΔFT SNR analysis (Task 1 Step 2 content) as the "is ΔFT
  meaningful" defense; add P-D mediation if supported.
- [ ] Related work: cite Galanti 2022 (CDNV, pro-collapse) vs Harun ICML 2025 (anti-collapse) as
  live controversy our CP-Δ setting speaks to; cite and differentiate arXiv:2603.03530
  (directional CDNV → few-shot LEVEL prediction from SSL features; we predict CP-induced Δ —
  different intervention, different target); pre-empt SimCLR App. B.7 + He2022 §4.3
  (LP–FT decoupling known ACROSS methods; ours is a signed within-pipeline geometric reversal).

## After the DIET reruns land (existing TODO, unchanged)

- Re-run F1/F2/F3 + §6.8 with Δ@MAX = mean(LeJEPA, SimCLR, DIET); re-check DIET collision
  (currently null on clean cells) and DIET Δwithin (P-D expects it largest).
- `eval/rest/test1-4` validators as already planned; then refresh `stats_pass.py`.
