# SigLIP DIET Held-Out Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the already-frozen pre-CP geometry score ranks DIET-CP frozen-transfer responses on the held-out SigLIP encoder.

**Architecture:** Add one missing factorial slice, `SigLIP x DIET-CP x 15 datasets x 3 seeds`, at the existing MAX data size. Training writes one JSON result and one checkpoint per cell. A collector validates the exact 45-cell grid and joins it to a frozen preregistration table. A separate verdict script performs the single primary test and clearly separates confirmatory from descriptive analyses.

**Tech Stack:** Python 3, PyTorch, timm, pandas, NumPy, SciPy, pytest, Bash, Slurm, A100 80 GB.

---

## Global Constraints

- Freeze the preregistration and verdict implementation before inspecting any new DIET outcome.
- Do not refit the geometry score, decision threshold, dataset panel, or statistical test.
- Keep the exact `15 datasets x 3 seeds` grid; the dataset is the statistical unit and seeds are averaged first.
- Use frozen kNN as the primary readout. Do not add post-CP supervised fine-tuning.
- Fail closed if any frozen source hash, manifest key, result field, or finite-value check differs.
- Report PASS, FAIL, or NO VERDICT exactly as specified; a negative result must not trigger a post-hoc endpoint change.

---

## 1. Scientific Scope

### 1.1 What is already known

- Discovery analyses include LeJEPA-CP, SimCLR-CP, and DIET-CP on DINOv3, CLIP, and MAE.
- The geometry score was fitted and frozen using the LeJEPA/SimCLR target on 2026-06-19.
- SigLIP currently has LeJEPA and SimCLR outcomes only.
- The DINOv3 ViT-L scale check already uses a three-method mean that includes DIET. A post-hoc method split gives DIET kNN sign hit `6/7` and Spearman `rho=+0.857143` (`p=0.013697`, two-sided). This is useful scale evidence, but it is not a held-out encoder test.

### 1.2 The one missing question

> On the same fixed 15-target panel and a held-out SigLIP encoder, does the frozen pre-CP score positively rank the DIET-specific change in frozen kNN?

This experiment does **not** test a new dataset population, refit the predictor, establish a causal mechanism, or validate a universal objective-invariant law.

### 1.3 Confirmatory endpoint

For dataset `d`, define

```text
delta_knn_DIET(d) = mean_seed(post_knn_DIET(d, seed)) - frozen_pre_knn(d)
```

The primary statistic is

```text
rho_primary = Spearman(uniformity(d), delta_knn_DIET(d)) over 15 datasets.
```

`uniformity(d)` is the pre-CP angular concentration (`uniformity_t2` in the frozen
`preregister_siglip.csv`, computed from SigLIP features before any CP). Amendment
2026-09-04, before any DIET outcome exists: the primary variable is uniformity, not
the frozen binary predictor's `p_help`, and P1 is the ONLY endpoint.

Freeze the following decision rule before launch:

- Alternative: `rho_primary > 0`.
- One-sided dataset-label permutation test with `100,000` permutations and RNG seed `20260904`.
- **PASS:** `rho_primary > 0` and permutation `p < 0.05`.
- **FAIL:** the complete 45-cell grid passes all gates, but the PASS condition is not met.
- **NO VERDICT:** any provenance, completeness, protocol, or finite-value gate fails.
- Report a 50,000-resample dataset bootstrap 95% CI with RNG seed `20260904`; the CI is descriptive and is not an additional gate.

The dataset is the statistical unit. Seeds are repeated training measurements and must be averaged before the primary test.

### 1.4 Secondary analyses

None. Mainline-only decision (2026-09-04): the extension has exactly one
confirmatory endpoint and no secondary, sensitivity, or descriptive panels.

Do not run post-CP supervised fine-tuning in this extension. The scientific endpoint is frozen transfer, and omitting SFT materially reduces compute without weakening the stated question.

---

## 2. Frozen Cell Manifest

Use exactly the following panel and no substitutions:

| Order | Dataset key | Display name | Type | MAX samples |
|---:|---|---|---|---:|
| 0 | `breastmnist` | `BreastMNIST` | OOD | 546 |
| 1 | `dermamnist` | `DermaMNIST` | OOD | 7,007 |
| 2 | `octmnist` | `OctMNIST` | OOD | 97,477 |
| 3 | `organamnist` | `OrganAMNIST` | OOD | 34,561 |
| 4 | `pathmnist` | `PathMNIST` | OOD | 89,996 |
| 5 | `galaxy10` | `Galaxy10` | OOD | 14,188 |
| 6 | `eurosat` | `EuroSAT` | OOD | 16,200 |
| 7 | `plant_village` | `PlantVillage` | OOD | 43,596 |
| 8 | `dtd` | `DTD` | OOD | 1,880 |
| 9 | `food101` | `Food101` | Fine-grained | 75,750 |
| 10 | `fgvc_aircraft` | `FGVC_Aircraft` | Fine-grained | 3,334 |
| 11 | `cars196` | `Cars196` | Fine-grained | 8,144 |
| 12 | `cub200` | `CUB200` | Fine-grained | 5,994 |
| 13 | `flowers102` | `Flowers102` | Fine-grained | 1,020 |
| 14 | `oxford_pet` | `OxfordPet` | Fine-grained | 3,680 |

Seeds are exactly `{42, 43, 44}`. The Cartesian product must contain exactly 45 unique cells.

Correction 2026-09-04 (before any DIET outcome exists): `fgvc_aircraft` MAX is 3,334 (the
`--n-samples` used by every MAX run in `run/slurm`); the earlier 3,400 copied the
`results.xlsx` label typo "MAX (3400)".

Frozen source hashes at plan time:

```text
preregister_siglip.csv  sha256 4b8bb2b99f884dd3c0e2058dac969310935ed6910d1fd2eaa173546121d846b8
c2_siglip_score.csv      sha256 473c521b38be8a7632d611d804cb238c677415a3da0cd4a1d40c4388f836aaae
results.xlsx             sha256 14d0ae53000e0ec3eaeacb7600ce13b66b97b21afedbdb3f9fdbed92e8d36d67
```

The implementation must fail closed if these inputs differ. If an intentional source correction is required, create a new preregistration version before launching any training job.

---

## 3. Frozen Training Protocol

Every cell uses:

```text
backbone                vit_base_patch16_siglip_224.v2_webli
pool strategy           map
CP method               diet
epochs                   150
batch size               32
learning rate            1e-4
weight decay             0.05
freeze epochs            15
trained transformer blocks 2
kNN k                    20
workers                  8
label smoothing          0.3
mixup alpha              1.0
cutmix alpha             1.0
mixup/cutmix probability 0.0
mixup/cutmix switch prob 0.5
baseline evaluation      skipped; use frozen existing SigLIP pre values
post-CP SFT              disabled
```

The choice of two trainable blocks matches the existing SigLIP CP protocol. DIET must use native SigLIP MAP pooling; `stable_cp/methods/diet/diet_forward.py` already supports this path.

---

## 4. Implementation Tasks

### Task 1: Freeze a machine-readable preregistration

**Files:**

- Create: `eval/F5_decision_score/SIGLIP_DIET_PREREG.md`
- Create: `eval/F5_decision_score/siglip_diet_protocol.py`
- Create: `eval/outputs/siglip_diet_preregister.csv`
- Test: `eval/F5_decision_score/test_siglip_diet_extension.py`

- [ ] Add constants for the exact 15 datasets, MAX sample counts, types, three seeds, model identifier, and training hyperparameters.
- [ ] Load `eval/outputs/preregister_siglip.csv` for `uniformity_t2`. Read pre-kNN from `results.xlsx` with `sheet_name="By Method (SigLIP)"` and `header=1`; require the LeJEPA and SimCLR pre values to match exactly for every dataset before collapsing them into one 15-row table. Use ASCII output column names: `uniformity` and `pre_knn`.
- [ ] Assert the three source hashes above before writing the table.
- [ ] Assert 15 unique dataset keys, exact key identity, finite numeric fields, and no duplicated order values.
- [ ] Write the operative hypothesis, primary statistic, and PASS/FAIL/NO-VERDICT rule to `SIGLIP_DIET_PREREG.md` (no secondary analyses).
- [ ] Store the generated preregistration CSV hash in the Markdown document.

Run:

```bash
pytest -q eval/F5_decision_score/test_siglip_diet_extension.py
python eval/F5_decision_score/siglip_diet_protocol.py --freeze
git diff -- eval/F5_decision_score/SIGLIP_DIET_PREREG.md eval/outputs/siglip_diet_preregister.csv
```

Do not proceed until the preregistration is committed.

### Task 2: Add a resume-safe Slurm array driver

**Files:**

- Create: `run/slurm/cp-siglip/cp/diet_max_array.sh`
- Modify: `eval/F5_decision_score/test_siglip_diet_extension.py`

- [ ] Map `SLURM_ARRAY_TASK_ID` 0-44 to `dataset_index = task_id / 3` and `seed_index = task_id % 3`.
- [ ] Read dataset metadata from `siglip_diet_protocol.py --cell TASK_ID`; do not duplicate a second manifest in Bash.
- [ ] Use `set -euo pipefail` after module and Conda initialization, with `${PYTHONPATH:-}` safe expansion.
- [ ] Preflight CUDA, the processed dataset cache, the frozen preregistration hash, the SigLIP model creation path, and a writable output directory before training.
- [ ] Refuse to skip a result JSON unless it parses, matches the requested key, contains finite `post_knn_f1` and `post_linear_f1`, and has a readable checkpoint.
- [ ] Write results under `outputs/logs/cp-siglip/cp/DIET/<Display>/SigLIP/` and checkpoints under `outputs/ckpts/cp-siglip/cp/DIET/<Display>/SigLIP/`.
- [ ] Invoke `continued_pretraining.py` with the frozen protocol above. Do not pass `--post-cp-sft`.
- [ ] Add `--dry-run` support that prints all resolved arguments without importing Torch.

Run locally:

```bash
bash -n run/slurm/cp-siglip/cp/diet_max_array.sh
python -m py_compile eval/F5_decision_score/siglip_diet_protocol.py
for task in 0 1 2 42 43 44; do \
  SLURM_ARRAY_TASK_ID="$task" bash run/slurm/cp-siglip/cp/diet_max_array.sh --dry-run; \
done
pytest -q eval/F5_decision_score/test_siglip_diet_extension.py
```

### Task 3: Add the result collector and provenance gates

**Files:**

- Create: `eval/F5_decision_score/collect_siglip_diet.py`
- Modify: `eval/F5_decision_score/test_siglip_diet_extension.py`
- Output: `eval/outputs/siglip_diet_behavior.csv`
- Output: `eval/outputs/siglip_diet_results.sha256`

- [ ] Require exactly 45 unique `(dataset, seed)` cells and the exact frozen Cartesian product.
- [ ] Reject malformed JSON, non-finite metrics, wrong method/backbone/sample count/seed, missing checkpoint, duplicate keys, and foreign result files.
- [ ] Join each cell to the frozen pre-kNN and pre-LP values from `siglip_diet_preregister.csv`.
- [ ] Compute per-seed `dknn` and `dlp`; do not average until after the 45-cell census passes.
- [ ] Write the CSV atomically in canonical dataset/seed order.
- [ ] Write SHA256 entries for the 45 JSON files, 45 checkpoints, frozen preregistration CSV, and collected CSV.
- [ ] Add `--verify-only` that validates an existing output without rewriting it.

Run:

```bash
pytest -q eval/F5_decision_score/test_siglip_diet_extension.py
python eval/F5_decision_score/collect_siglip_diet.py --verify-only
```

### Task 4: Implement the frozen verdict

**Files:**

- Create: `eval/F5_decision_score/siglip_diet_verdict.py`
- Modify: `eval/F5_decision_score/test_siglip_diet_extension.py`
- Output: `eval/outputs/siglip_diet_verdict.txt`

- [ ] Verify the provenance sidecar and exact 45-cell grid before computing any statistic.
- [ ] Average the three seeds within each dataset, yielding exactly 15 primary rows.
- [ ] Compute the primary Spearman statistic, fixed-seed one-sided permutation p-value, and bootstrap CI exactly as frozen in Section 1.3.
- [ ] Print the raw 15-row primary table before the verdict so sign or ranking errors are auditable.
- [ ] Print `PASS`, `FAIL`, or `NO VERDICT`; the script computes nothing beyond the primary endpoint.
- [ ] Include a claim-scope footer stating exactly what each outcome licenses.

Required synthetic tests:

- [ ] Perfect positive ranking returns PASS.
- [ ] Reversed ranking returns FAIL.
- [ ] Null/noisy ranking returns FAIL, not NO VERDICT.
- [ ] A missing seed, duplicate row, NaN, hash mismatch, or wrong dataset returns NO VERDICT via a hard failure.
- [ ] Seed averaging occurs before correlation.
- [ ] Changing LP cannot change the kNN primary verdict.

Run:

```bash
pytest -q eval/F5_decision_score/test_siglip_diet_extension.py
python -m py_compile eval/F5_decision_score/collect_siglip_diet.py eval/F5_decision_score/siglip_diet_verdict.py
```

### Task 5: Launch without changing the analysis

- [ ] Commit the preregistration, protocol, tests, collector, verdict, and Slurm driver.
- [ ] On the cluster, run one engineering smoke cell only after that commit:

```bash
sbatch --array=0 run/slurm/cp-siglip/cp/diet_max_array.sh
```

- [ ] Inspect only runtime integrity: CUDA, loss finiteness, result schema, checkpoint readability, and MAP-pooling path. Do not revise the scientific rule after seeing the metric.
- [ ] Launch the remaining cells. Concurrency is operational and may be reduced without changing the design:

```bash
sbatch --array=1-44%10 run/slurm/cp-siglip/cp/diet_max_array.sh
```

- [ ] After all jobs finish, collect and freeze on the cluster:

```bash
python eval/F5_decision_score/collect_siglip_diet.py
python eval/F5_decision_score/collect_siglip_diet.py --verify-only
```

- [ ] On the cluster, verify all 45 checkpoints and freeze their hashes into the sidecar. Transfer the 45 result JSONs, `siglip_diet_behavior.csv`, and `siglip_diet_results.sha256` together; checkpoints may remain on the cluster. Locally, verify the exact transferred file set and hashes before verdict execution.

### Task 6: Judge once and integrate by outcome

Run locally:

```bash
python eval/F5_decision_score/collect_siglip_diet.py --verify-only
python eval/F5_decision_score/siglip_diet_verdict.py | tee eval/outputs/siglip_diet_verdict.txt
```

Then update only the claim licensed by the result:

- **PASS:** “A score frozen on LeJEPA/SimCLR responses also ranks DIET-specific frozen kNN changes on a held-out SigLIP encoder over the same 15-target panel.”
- **FAIL:** “The score transfers across encoder for LeJEPA/SimCLR, while DIET-specific transfer to SigLIP is not supported; ViT-L remains a small scale-check result.”
- **NO VERDICT:** Make no scientific update; report the failed gate and rerun only the invalid cells under the unchanged preregistration.

In every case, retain these boundaries:

- no dataset-population generalization;
- no claim that geometry determines CP response;
- no causal or mediation claim;
- no claim of universal objective invariance;
- no use of the three-method mean as an independent confirmation of the DIET-specific endpoint.

### Task 7: Archive and document

**Files:**

- Create: `findings/FINDINGS_step21_siglip_diet.md`
- Update after verdict: the source generator for `/Users/zhanghaodong/Desktop/CP/theory.docx`
- Update after verdict: `CODEX_ALIGNMENT_BRIEF.md` and the paper mainline draft, if present

- [ ] Record the preregistration commit, Slurm job IDs, exact result hashes, gates, primary table, verdict, and licensed/forbidden wording.
- [ ] Regenerate and visually inspect `theory.docx`; do not patch the DOCX binary directly.
- [ ] Keep failed or null results in the same canonical locations. The experiment is informative under every valid verdict.

---

## 5. Time and Compute Control

The confirmatory design is 45 MAX runs. To stay within the four-week writing window:

1. Freeze and test the pipeline before requesting GPUs.
2. Omit post-CP SFT.
3. Use one array cell per dataset/seed so large datasets do not block smaller cells.
4. Start with concurrency 10 and reduce it only for scheduler or quota constraints.
5. Write the paper in parallel; this extension changes one scope sentence, not the paper structure.

Do not add new datasets, new geometry features, new thresholds, or another mechanism experiment to this run.
