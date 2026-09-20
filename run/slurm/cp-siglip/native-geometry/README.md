# SigLIP-2 Native Geometry Supplement

This single V100 job fills the missing native-normalization geometry for paper
Figures 3 and 6. It never trains CP, FT, or a linear probe, and never reads a CP
checkpoint. The model is `vit_base_patch16_siglip_224.v2_webli`, initialized from
public pretrained weights and read out with MAP pooling.

## Protocol

- Use RGB mean and standard deviation `(0.5, 0.5, 0.5)` for **both** target data
  and the ImageNet reference. Assert these against the checkpoint configuration.
- Preserve the deterministic 224-by-224 evaluation transform from the completed
  baseline audit. This changes channel normalization, not the entire timm resize
  and crop policy.
- Extract every clean MAX training example. No validation or test features enter
  the geometry. Process seed 42 on the 14 fixed training splits and seeds 42, 43,
  and 44 on Galaxy10, whose train partition depends on the seed.
- Recompute full distinct-pair uniformity at `t=2`. Match the completed baseline
  export within `1e-5`; otherwise fail before publishing that target's result.
  Expected values in `eval/siglip_native_geometry_expected.csv` are transcribed
  from the author's official-normalization export supplied on 2026-09-16.
- Select up to 5,000 target features using the existing seed-42 stratified
  sampling convention. Select 5,000 ImageNet validation images uniformly, also
  with seed 42 and sorted indices.
- Reuse `eval.utils.geometry_metrics` for RBF MMD, its component energies and
  bandwidth, centroid cosine distance, and neighbor overlap at `k=20` and `k=50`.
  MMD is the existing squared, biased V-statistic including diagonal terms;
  overlap keeps the existing up-to-2,000-target / 5,000-reference convention.
- Save bank-level raw feature norm mean, population SD, and CV, plus their full
  MAX counterparts. Save **unnormalized** float32 banks and the complete MAX
  norm arrays, so the figures can be regenerated without another GPU job.
- Export per-split statistics and 15 target means. Galaxy10's summary averages
  its three splits equally. The per-split files remain available for plotting
  and checking the precise sampling convention.

The label-free uniformity statistic uses all MAX features. Labels are used only
to reproduce the historical stratified subsample for auxiliary geometry.

## Submit

First sync these new files to the cluster repository:

```text
eval/siglip_native_geometry.py
eval/siglip_native_geometry_expected.csv
run/slurm/cp-siglip/native-geometry/
```

From the cluster repository:

```bash
bash run/slurm/cp-siglip/native-geometry/submit.sh --dry-run
bash run/slurm/cp-siglip/native-geometry/submit.sh
```

Defaults reuse `/scratch/gs4133/zhd/CP/data` and its existing `imagenet_val`
Hugging Face `save_to_disk` cache. No ImageNet download is attempted. If that
reference dataset is elsewhere, set `SIGLIP_GEOMETRY_IMAGENET_DIR` before
submitting. `SIGLIP_GEOMETRY_CACHE_DIR`, `SIGLIP_GEOMETRY_OUTPUT_BASE`, and
`SIGLIP_GEOMETRY_REPO_ROOT` can override the other paths.

The job requests one V100, eight CPU cores, 64 GB host memory, and 24 hours.
Targets are processed sequentially from their shared processed caches. A unique
job-ID directory prevents overwriting old geometry or experiment results.

## Collect

Outputs are under:

```text
/scratch/gs4133/zhd/CP/outputs/siglip_native_geometry_v1/<JOB_ID>/
```

`geometry.csv` is the 15-target summary; `geometry_per_seed.csv` retains all 17
splits. `results/` and `imagenet.json` record normalization, model/code hashes,
split/sample hashes, software versions, and feature-file checksums. `features/`
contains compact raw banks and complete MAX norm arrays, not model checkpoints.
The job packs everything needed for the figures into `figure_geometry.tar.gz`.

Revalidate all files and print the summary with:

```bash
python3 -m eval.siglip_native_geometry export --outdir \
  /scratch/gs4133/zhd/CP/outputs/siglip_native_geometry_v1/<JOB_ID>
```

Return the archive, not only the CSV: Figure 6 requires the raw radial samples.
Incomplete runs and mismatched normalizations, weights, or checksums are not
accepted as complete summaries. Existing raw results are never overwritten.
