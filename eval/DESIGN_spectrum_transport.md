# Design — Spectrum & Transport phase (approved 2026-07-07)

User-approved scope amendments: Design 2 INCLUDES DIET (post-CP set = {LeJEPA, SimCLR, DIET} ×
{DINOv3, CLIP, MAE} × 15 × MAX × seeds ≈ 405 ckpts, plus SigLIP's {LeJEPA, SimCLR} MAX grid);
Design 3 INCLUDES MAE-CP (all 4 methods, the full Exp-H MAX set ≈ 526 ckpts).

Motivating question (user): is there a fully GENERAL scheme — datasets ordered by one geometric
parameter, encoders by another, interaction producing all phenomena? Probes (2026-07-07, on
existing data) showed: dataset axis already IS a spectrum (Finding 1); encoder axis has only
~2 effective points (sphere trio cluster + MAE), so a single-term bilinear model x·θ_E ties the
binary gate (LOO 0.498 vs 0.501, both > 0.438 position-only) but cannot beat it. These designs
(1) formalize the tie as a unifying statement, (2) multiply the encoder axis via per-layer
"virtual encoders", (3) measure the transport field itself.

## Design 1 — Bilinear unified law + coupling axis (CPU, existing data)

θ_E := |Spearman(uniformity_t2, CDNV)| across the 15 datasets, per encoder — "how tightly the
encoder's unlabeled geometry couples to its labeled class geometry". Defined without behavior.
Measured: MAE 0.17 ≪ SigLIP 0.39 < CLIP 0.60 < DINOv3 0.69.

Models over the 60 pooled cells (D3/CLIP/MAE = 3-method refreshed Δ@MAX; SigLIP = realized
2-method, disclosed): M1 x; M2 x + x·gate; M3 x + x·θ; M4 x·θ single-term. x = within-encoder
z-scored uniformity. Dataset-grouped LOO (15 folds, all 4 encoder rows of a dataset held out).

Pre-registered verdicts:
- V1 M4 within 0.02 of M2 and ≥0.04 above M1 (block-bootstrap CI over datasets for M4−M2).
- V2 θ ordering: MAE lowest by >0.15 vs min(sphere).
- V3 4th gate evidence: ρ(uniformity, pre-CP kNN) negative on all three sphere encoders,
  positive on MAE (sign flip; add SigLIP from the xlsx SigLIP sheet).
- V4 θ variants (cdnv / center_margin / knn_pre coupling) direction-consistent.
Paper: Finding-4 "unified statement" subsection + limitation "n=4 encoders cannot distinguish
continuous from binary gating".
Artifacts: eval/adjudicate/bilinear_law.py → eval/outputs/bilinear_law.csv.

## Design 2 — Exp I: layer-wise virtual encoders (GPU, zero new training)

Each layer ℓ of each encoder is a geometric state. Pre-CP: 4 encoders × 12 blocks × 15 datasets
→ per-layer geometry (uniformity_t2, rankme, norm-CV, CDNV, center_margin) + per-layer labeled
kNN (80/20 split of the ≤5000 subset, k=20 cosine macro-F1 — an internal protocol, consistent
across layers/pre/post, NOT the production kNN; disclosed). Post-CP: per-layer kNN + uniformity
for MAX ckpts of {LeJEPA, SimCLR, DIET} × {DINOv3, CLIP, MAE} (+ SigLIP's {LeJEPA, SimCLR} from
the cp-siglip root). Δ_ℓ = post_ℓ − pre_ℓ, seed-averaged.

Layer readout (fixed protocol): DINOv3/CLIP = cls token (token 0) at every block; MAE =
patch-token mean; SigLIP = all-token mean (no cls; MAP head is final-layer-only — protocol
difference disclosed). Hooks pool on the fly (no token storage).

Curve: per (encoder, layer) point — θ_ℓ (coupling_ℓ = |ρ(unif_ℓ, cdnv_ℓ)| across datasets;
also rankme_ℓ) and law strength ρ_ℓ = Spearman(unif_ℓ(D), ΔkNN_ℓ(D)) across the 15 datasets
(3-method mean; SigLIP 2-method).

Pre-registered verdicts:
- L1 trajectory sanity: sphere encoders show rising-then-tunnel rank profile; MAE low throughout.
- L2 MAIN: across the ~48 (θ_ℓ, ρ_ℓ) points, Spearman(θ_ℓ, ρ_ℓ) > 0 with encoder-block
  bootstrap support → continuous-spectrum evidence; if points remain two clusters → the
  gap/cliff conclusion is reinforced with 12× the points. Win-win; both outcomes reportable.
- L3 (highest upside): if any MAE middle layer reaches sphere-level coupling AND its ρ_ℓ
  locally recovers → gating is a function of geometry, not encoder identity.
Artifacts: eval/layerwise_geometry.py → layerwise_pre.csv; eval/adjudicate/layerwise_postcp.py
→ layerwise_postcp.csv; scorer eval/adjudicate/layerwise_law.py; slurm
run/slurm/eval/exp_i_layerwise_pre.sh (single job) + exp_i_layerwise_post.sh (array 0-14).
Cost: pre = 60 forwards; post ≈ 495 ckpt forwards; no ImageNet needed.

## Design 3 — Exp J: transport-field decomposition (GPU, zero new training)

Same samples through pre and post encoders (deterministic loader) → per-sample displacement
d_i = ẑ_i^post − ẑ_i^pre on the unit sphere. Exact variance decomposition:
  E‖d‖² = ‖μ_d‖² (global translation) + Σ_c (n_c/N)‖μ_c − μ_d‖² (between-class motion)
        + E‖d_i − μ_c(i)‖² (within-class scramble)
plus toward-ImageNet projection ⟨μ_d, v_E⟩ with v_E = normalized ImageNet-val centroid of the
PRE encoder. Scope: ALL 4 methods × {DINOv3, CLIP, MAE} × 15 × MAX × seeds (the Exp-H set,
≈526 ckpts) — per user amendment MAE-CP included.

Pre-registered verdicts:
- T1 (highest value): toward-ImageNet ↔ Δoverlap correlates on sphere encoders AND is larger
  for invariance methods than DIET / MAE-CP → mechanistic ticket for "collision is
  invariance-specific" (augmentation-invariance pulls representations toward the pretraining
  distribution; instance discrimination does not).
- T2 within-scramble share ↔ ΔkNN damage, FG > OOD.
- T3 identity check: three terms sum to total energy (numerical, per cell).
- T4 exploratory: between-class motion ↔ ΔFT.
Artifacts: eval/adjudicate/transport_field.py → transport_field_max.csv; scorer
eval/adjudicate/transport_law.py; slurm run/slurm/eval/exp_j_transport.sh (array 0-14, stages
dataset + ImageNet-val). Cost per task ≈ 3 pre-extractions + ~36 post + 3 ImageNet forwards.

## Execution order
Design 1 locally now → Exp J submitted first (T1 feeds Finding 2's mechanism), Exp I parallel →
adversarial verification + FDR + findings doc on return. All three carry the standing honesty
rules (rank claims, disclosed protocols, exploratory labels where n is small).
