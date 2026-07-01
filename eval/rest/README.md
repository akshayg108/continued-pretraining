# `eval/rest/` — post-rerun validation of the 60 re-trained checkpoints

After the `run/slurm/rest/` jobs re-train the 60 incomplete cells, run these 4 tests (in order)
to confirm everything is fixed and to refresh the downstream numbers. All read the cell list from
`eval/outputs/rerun_geometry.csv` (column 1 = checkpoint path).

| # | script | what it checks | needs GPU? |
|---|---|---|---|
| 1 | `test1_check_ckpts.py` | the 60 checkpoints are now **correct** — reached epoch ≥149 AND the backbone actually trained (≠ pretrained) | no (CPU) |
| 2 | `test2_geometry.py` | recompute **post-CP geometry** (uniformity / overlap / norm-CV) for the 60 + Δ vs pre-CP | yes |
| 3 | `test3_rerun_exp_b.py` | re-run **Exp B** for the cells whose LeJEPA reference was broken (food101/octmnist/cars196) + re-aggregate `sa_lp_recovery` | yes (or `--aggregate-only` = CPU) |
| 4 | `test4_behavior_deltas.py` | the new **kNN / LP / FT** of these jobs and their **Δ** vs the pre-CP baseline | no (CPU) |

Run order (1 → 4):
```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining

# 1) checkpoints correct?  (CPU, ~minutes)
python eval/rest/test1_check_ckpts.py

# 2) new geometry  (GPU; --imagenet-dir adds overlap, EXPENSIVE: re-embeds ImageNet per ckpt)
srun --gres=gpu:1 --mem=64G --time=8:00:00 --pty \
  python eval/rest/test2_geometry.py --download-dir <raw> --processed-dir <arrow> \
        --imagenet-dir <imagenet_val>   # → eval/outputs/rest_geometry.csv

# 3) re-run Exp B for the affected LeJEPA reference + re-aggregate  (GPU)
srun --gres=gpu:1 --mem=64G --time=8:00:00 --pty \
  python eval/rest/test3_rerun_exp_b.py --run-expb \
        --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
        --download-dir <raw> --processed-dir <arrow>
#  (CPU-only re-aggregate, if Exp B already re-run:  python eval/rest/test3_rerun_exp_b.py --aggregate-only)

# 4) new kNN/LP/FT + Δ  (CPU)
python eval/rest/test4_behavior_deltas.py --results-xlsx ../results.xlsx
```
