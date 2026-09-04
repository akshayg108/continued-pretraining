# SigLIP x DIET held-out extension — PREREGISTRATION (FROZEN before any DIET outcome exists)

Plan: docs/superpowers/plans/2026-09-04-siglip-diet-heldout-extension.md. Protocol
module: eval/F5_decision_score/siglip_diet_protocol.py (single source of truth).
Mainline-only by decision (2026-09-04): ONE endpoint, no secondary analyses.
Version 1.1 (2026-09-04, before any DIET outcome exists): manifest correction
fgvc_aircraft MAX 3400 -> 3334 (3334 = the --n-samples used by every MAX run in
run/slurm; "MAX (3400)" in results.xlsx is a label typo). No other change.

## Question
On the fixed 15-target panel and the held-out SigLIP encoder, does pre-CP angular
concentration (uniformity_t2, computed from SigLIP features before any CP) rank the
DIET-CP change in frozen kNN? Positioning: a prospective extension of the
target-position relation to a not-yet-observed objective outcome on a held-out
encoder. NOT new-dataset generalization, NOT a mechanism/mediation claim, NOT
universal objective invariance.

## Design
SigLIP x DIET-CP x 15 datasets x seeds {42, 43, 44} = 45 MAX cells. Frozen recipe:
{"cp_method": "diet", "epochs": 150, "batch_size": 32, "lr": 0.0001, "weight_decay": 0.05, "freeze_epochs": 15, "num_trained_blocks": 2, "pool_strategy": "map", "knn_k": 20, "workers": 8, "label_smoothing": 0.3, "mixup_alpha": 1.0, "cutmix_alpha": 1.0, "mixup_cutmix_prob": 0.0, "mixup_cutmix_switch_prob": 0.5, "accumulate_grad_batches": 1, "baseline_eval": "skipped (frozen pre values)", "post_cp_sft": "disabled"}. Model: vit_base_patch16_siglip_224.v2_webli. Dataset = statistical unit; the three seeds
are averaged BEFORE the test. delta_knn_DIET(d) = mean_seed(post_knn(d, seed)) -
pre_knn(d), with pre_knn the frozen SigLIP pre-CP level.
num_trained_blocks = 2 for every cell: this matches the SigLIP LeJEPA/SimCLR grid
(run/slurm/cp-siglip/cp/*/lejepa_max.sh uses 2 blocks even at n = 97,477), NOT the
main-grid DINOv3/CLIP/MAE size rule (2 / 4 / 6 / all blocks by n). Post-CP frozen
kNN/LP is evaluated by continued_pretraining.py without --post-cp-sft.

## The only endpoint (P1)
rho = Spearman(uniformity, delta_knn_DIET) over the 15 datasets. One-sided
dataset-label permutation test, 100000 permutations, RNG seed 20260904;
alternative rho > 0 (higher concentration -> larger dkNN, the direction observed on
DINOv3/CLIP and on SigLIP's LeJEPA/SimCLR outcomes). A 50000-resample dataset
bootstrap 95% CI (seed 20260904) is printed next to rho; it is not a gate.
Verdicts: any provenance / completeness / protocol / finite-value gate fails ->
NO VERDICT; rho > 0 and p < 0.05 -> PASS; otherwise FAIL.

## Licensed readings (frozen)
- PASS: the pre-CP concentration -> frozen-kNN relation extends to DIET-CP outcomes on
  the held-out SigLIP encoder over the fixed 15-target panel.
- FAIL: the extension to DIET on SigLIP is not supported; the relation stays scoped
  to the objectives on which it was observed.
- NO VERDICT: no scientific update; rerun only the invalid cells under this unchanged
  preregistration.
Forbidden in every case: dataset-population generalization; "geometry determines CP
response"; causal or mediation wording; universal objective invariance; any
post-hoc endpoint added after seeing DIET outcomes.

## Frozen sources (fail closed on mismatch)
{
  "preregister_siglip.csv": "4b8bb2b99f884dd3c0e2058dac969310935ed6910d1fd2eaa173546121d846b8",
  "c2_siglip_score.csv": "473c521b38be8a7632d611d804cb238c677415a3da0cd4a1d40c4388f836aaae",
  "results.xlsx": "14d0ae53000e0ec3eaeacb7600ce13b66b97b21afedbdb3f9fdbed92e8d36d67"
}

## Generated preregistration table
eval/outputs/siglip_diet_preregister.csv  sha256 d58f773c0d8225ee2a4746e293a9e3dfe9789b5d6a77ff3782be434d26b52952
Columns: order, dataset, display, type, max_samples, uniformity, pre_knn.
