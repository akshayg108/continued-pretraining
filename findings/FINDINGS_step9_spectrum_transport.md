# FINDINGS step 9 — Spectrum & Transport (Exp I layer-wise + Exp J transport field)

Date: 2026-07-08. Design: `eval/DESIGN_spectrum_transport.md` (pre-registered verdicts).
Inputs: `layerwise_pre.csv` (720 rows, 4 enc × 15 ds × 12 layers, no NaN),
`layerwise_postcp.csv` (484 ckpts × 12 layers; LeJEPA/SimCLR/DIET × D3/CLIP/MAE + LeJEPA/SimCLR
× SigLIP, all 15 datasets), `transport_field_max.csv` (526 ckpts, 4 methods × 3 encoders).
Scorers: `eval/adjudicate/layerwise_law.py`, `eval/adjudicate/transport_law.py`.
Curve: `eval/outputs/layerwise_curve.csv`; stats: `transport_stats.csv`.

## Protocol discovery — CORRECTED 2026-07-08 (user correction + full script audit)

**The unfreeze depth is SIZE-SCHEDULED in the main grid** (per-script constants, verified
across all run/slurm/cp scripts AND empirically via the per-layer frozen map, 3 encoders
agreeing 100%): n < 10k → last 2 blocks; 10k ≤ n ≤ 25k → last 4; 25k < n ≤ 50k → last 6;
n > 50k → all blocks (-1). FROM-SCRATCH/random → all (-1). SigLIP grid → last 2 everywhere.
At the MAX tier this means: 8 datasets @ blk2 (breastmnist/flowers/dtd/fgvc/pet/cub/derma/
cars), 2 @ blk4 (galaxy10/eurosat), 2 @ blk6 (organamnist/plant_village), 3 @ all
(food101/pathmnist/octmnist). An earlier version of this section claimed "main grid = full
backbone" from a single sampled script (a /random/ one) — WRONG, retracted.

**Consequences applied:**
1. **MAE depth-gradient claim RETRACTED.** The within-MAE +0.785 (and the "L7-10 recovery"
   pattern) was an artifact of frozen-layer tie structure: at L1-6 only 3 datasets actually
   train, at L7-8 only 5, at L9-10 only 7 — shallow law_ρ was computed mostly on exact-zero
   deltas. On actually-trained rows MAE has L9 −0.32 / L10 −0.11 / L11 −0.11 / L12 +0.04:
   no recovery, no usable gradient (n=4 points).
2. **Valid layer curve = 14 points (L9-L12).** Spearman(θ_ℓ, law_ρ_ℓ) = +0.073 (p=0.81),
   block-bootstrap CI [−0.37, +0.71]. The spectrum-vs-cliff test is UNTESTABLE below L9 by
   design (depth coupled to size); on the testable range there is no spectrum signal. The
   L11/L12 rows (all 15 datasets trained) are unaffected and carry the replication result.
3. **Exp J robustness to the depth confound**: unfreeze depth correlates with transport
   energy (ρ=+0.48), so dataset-wise correlations were re-run with depth partialled out —
   both survive and strengthen: CLIP toward↔Δoverlap −0.500 → −0.545; DINOv3
   within_share↔dknn −0.480 → −0.512. Method CONTRASTS (T1b, signatures) were never
   confounded (same depth per dataset across methods).
4. **Finding 1 depth-control note**: at MAX, unfreeze depth correlates with dataset size and
   hence with OOD-ness — but the held-out SigLIP grid is CONSTANT depth (blk2 everywhere)
   and reproduces the position law (+0.746, 87%): a clean depth-uniform control. Finding 3's
   size dynamics are RECIPE-level dose responses (data amount and depth increase together);
   disclosed.
All per-layer readouts are pre-final-norm block outputs (internally consistent pre vs post;
NOT identical to production features — e.g. MAE's final LayerNorm changes its coupling
markedly: block-12 θ = 0.64 vs production-feature θ = 0.17. Disclosed, not interpreted).

## Exp I — layer-wise virtual encoders

- **L2 MAIN (pre-registered, values CORRECTED per the protocol section): no spectrum signal
  on the valid range.** After dropping frozen (dataset, layer) rows the curve has 14 valid
  points (L9-L12); pooled Spearman(θ_ℓ, law_ρ_ℓ) = +0.073 (p=0.81), encoder-block bootstrap
  95% CI [−0.37, +0.71]. Below L9 the test is impossible by design (size-scheduled depth).
  Superseded first-pass numbers (38 points, +0.266): artifacts of frozen-zero ties.
- **Final-layer replication (strongest new support for the gate):** at L12 under the internal
  protocol the law strength is DINOv3 +0.832, CLIP +0.757, SigLIP +0.554, MAE +0.043 — the
  production gate reproduces under an independent protocol, MAE lands at null, and the three
  spheres order by their production coupling θ (SigLIP lowest θ → weakest sphere law).
- **MAE depth gradient — RETRACTED (see protocol correction).** The +0.785 and the
  "L7-10 recovery" pattern were frozen-tie artifacts; on actually-trained rows MAE shows
  L9 −0.32 / L10 −0.11 / L11 −0.11 / L12 +0.04 — no recovery, nothing to claim.
- **L3 (pre-registered): FAIL** (and moot after the correction — no MAE layer combines
  sphere-level coupling with positive law on valid rows).
- L1 sanity: rankme rises with depth on all four encoders; only CLIP shows a mild late
  dip (446→430). The textbook tunnel collapse is not visible in ViT-B cls readouts (informational).

## Exp J — transport-field decomposition

- **T3 identity: PASS** (max relative residual 1.2e-06; float32 accumulation; gate is
  relative — adjusted from the 1e-8 absolute pre-registration, disclosed).
- **T1a vector-collision correlation: REFUTED on sign.** Predicted: displacement toward the
  (pre-encoder) ImageNet centroid ↔ larger Δoverlap. Observed: CLIP ρ(toward, Δoverlap) =
  **−0.500** (p=5e-4, BH-FDR pass; cos version −0.640), DINOv3 −0.163 n.s. The fixed-frame
  picture ("target drifts into the old ImageNet region") is wrong: under CP the ImageNet
  cloud is re-embedded too and the whole map translates (translation share ≈ 37-39% for
  angular methods; CLIP's mean displacement points strongly AWAY from the old ImageNet
  direction for every method). Collision/overlap is a RELATIVE, co-moving phenomenon.
- **T1b invariance-vs-DIET contrast: PARTIAL SUPPORT.** Median toward-ImageNet, invariance >
  DIET on both sphere encoders; significant on CLIP (−0.392 vs −0.482, one-sided p=1e-4,
  FDR pass), directional only on DINOv3 (+0.012 vs −0.009, p=0.099). MAE-CP sits apart on
  both. The invariance-specific pull exists but is encoder-dependent in strength.
- **T2 within-class scramble ↔ ΔkNN: PARTIAL (DINOv3 only).** DINOv3 ρ(within_share, dknn)
  = −0.480 (p=8e-4, FDR pass; predicted sign) with FG > OOD scramble share (0.561 vs 0.511).
  CLIP +0.129 n.s., MAE −0.089 n.s.
- **T4 (exploratory): between-class motion ↔ ΔFT positive** — CLIP +0.356 (p=0.017),
  DINOv3 +0.263 (p=0.081). Class-coherent motion appears FT-benign; stays exploratory.
- **Method signature (descriptive, feeds Findings 2/3/6):** medians of (total energy,
  translation share, within share): MAE-CP **0.44 / 0.75 / 0.17** — a small, translation-
  dominated, low-scramble transport (the cloud moves almost rigidly, yet its frozen readout
  collapses — consistent with Finding 6's aggregation-failure account); angular methods
  1.2-1.4 / 0.36-0.39 / 0.43-0.49; **DIET has the largest total energy (1.44) and largest
  within-class share (0.49)** — consistent with the P-D expectation that instance
  discrimination scrambles within classes hardest.

## Adversarial verification

Independent recomputation (fresh code paths): within-MAE +0.785 ✓ exact (later RETRACTED as
an artifact — the recompute confirmed the number, the protocol correction invalidated its
meaning); CLIP toward↔Δoverlap −0.500 ✓ exact; DINOv3 within_share↔dknn −0.480 ✓ exact.
Robustness: R-1 SNR filter (first flag of the artifact), R-2 per-method split, R-3 cos
variant (strengthens the T1a sign reversal), R-4 energy composition table, depth-partialled
T1a/T2 (both strengthen). BH-FDR(q=0.10) confirmatory survivors: T1-corr-CLIP,
T1-inv>DIET-CLIP, T2-DINOv3.

## Paper implications (rank-claim wording)

1. Finding 4 (gate): + final-layer internal-protocol replication with the 4-encoder ordering
   (D3 .83 / CLIP .76 / SigLIP .55 / MAE .04, also visible at L11) — an independent-protocol
   echo of the gate AND of θ ordering; the layer curve shows no spectrum signal on its valid
   L9-L12 range (deeper untestable by design); the MAE mid-depth pattern is retracted.
2. Finding 2 (two forces): collision mechanism REFINED — not fixed-frame drift toward the
   pretraining region but relative mixing under a shared transport; the invariance-specific
   pull survives as a method contrast on CLIP (FDR) and directionally on DINOv3.
3. Finding 6 (MAE-CP): the translation-dominated low-scramble transport signature (0.75
   translation share vs 0.36-0.39 for angular) is new independent support for "the map moves
   mostly rigidly; the readout, not the geometry, fails".
4. Finding 3 (dynamics) untouched; Finding 7 untouched.

## Addendum (2026-07-08): V3 sign-flip failure is a real SigLIP-2 property, not measurement

Design 1's pre-registered V3 (ρ(uniformity, pre-CP kNN) negative on all sphere encoders,
positive on MAE) failed because SigLIP measured +0.171. User asked whether incomplete SigLIP
CP training could explain it — it cannot: V3 contains NO CP quantities (both components come
from the public timm checkpoint). Independent re-measurement through our own pipeline
(layerwise_pre.csv L12 internal-protocol kNN; different preprocessing, mean-token readout
instead of MAP, different kNN protocol) REPRODUCES and strengthens the positive sign:
SigLIP +0.529 (production-unif) / +0.389 (L12-unif), vs DINOv3 −0.254/−0.386 and CLIP
−0.125/−0.114 (signs consistent with the xlsx-based values). Verdict: the level-channel
(unlabeled spreading ↔ baseline function) does NOT separate sphere from off-sphere — SigLIP-2
sits on the positive side despite fully sphere-typical behavior on every Δ channel (position
law +0.746, decision score 87%, L12 law +0.554, θ 0.386). Coherent with the step-2 lesson
that LEVEL rules and Δ rules dissociate; the gate's claims live on Δ and are untouched.
Speculative note (disclosure-grade only): SigLIP-2's pretraining mixes captioning /
self-distillation / masked-prediction objectives on top of sigmoid contrastive — a candidate
source for its weakest-among-spheres coupling; not a claim (DINOv3 also contains masked
components yet couples strongest).

Readout 2x2 follow-up (user asked whether SigLIP's MAP readout explains V3): both V3
components did use MAP consistently (geometry pool="map"; teammate's kNN eval
--pool-strategy map). Crossing readouts gives, for SigLIP, rho(unif, knn_pre) =
+0.171 (MAP x MAP), +0.061 (mean x MAP), +0.529 (MAP x mean), +0.389 (mean x mean) —
no combination is negative, while DINOv3/CLIP stay negative under the same variants
(-0.25/-0.39 and -0.13/-0.11). If anything MAP makes SigLIP look MORE sphere-like
(closest to zero). Readout hypothesis conclusively excluded; the level-channel anomaly
is a model property.
