#!/usr/bin/env python
"""Pretrained encoder evaluation, continued pretraining, and post-CP evaluation."""

import argparse
import json
from pathlib import Path
import tempfile

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

import stable_pretraining as spt
from stable_pretraining.backbone.utils import from_timm

from stable_cp.callbacks import (
    FreezeBackboneCallback,
    create_cp_evaluation_callbacks,
)
from stable_cp.evaluation.zero_shot_eval import zero_shot_eval
from stable_cp.evaluation.sft_eval import sft_evaluate
from stable_cp.utils.backbone import default_pool_strategy, feature_readout
from stable_cp.data import DATASETS, get_dataset_config, get_dataset, CPSubset
from stable_cp.data import (
    create_eval_loaders,
    create_train_datamodule,
    create_transforms,
)


def create_base_parser(description="Continued Pretraining"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--backbone", type=str, required=True)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--n-samples", type=int, default=1000)
    budget.add_argument(
        "--full-train", action="store_true", help="Use the complete training split."
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--freeze-epochs", type=int, default=None)
    parser.add_argument("--num-trained-blocks", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override Wandb run name (default: auto-generated)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--cache-dir", type=str, default="~/.cache")
    parser.add_argument(
        "--pool-strategy",
        choices=["cls", "mean", "map"],
        default=None,
        help="Defaults to MAP for SigLIP, patch mean for MAE, and CLS otherwise.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before stepping. "
        "Effective batch size = batch_size * accumulate_grad_batches. "
        "Use this when the desired batch size doesn't fit in memory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume CP from its existing checkpoint, or start if none exists. "
        "Without this flag an existing CP checkpoint is an error. FT never resumes weights.",
    )
    return parser


def setup_paths(args):
    """Setup paths for data and checkpoints."""
    cache_dir = Path(args.cache_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    data_dir = cache_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_cp:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, checkpoint_dir


def get_config(args):
    ds_cfg = get_dataset_config(args.dataset)
    freeze_epochs = (
        args.freeze_epochs if args.freeze_epochs is not None else int(args.epochs * 0.05)
    )
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else int(args.epochs * 0.1)
    return ds_cfg, freeze_epochs, warmup_epochs


def load_backbone(args, img_size=224):
    """Load pretrained TIMM weights without changing the training transform recipe."""
    print(f"Loading pretrained TIMM model: {args.backbone} with img_size={img_size}")
    backbone = from_timm(args.backbone, pretrained=True, img_size=img_size)

    for p in backbone.parameters():
        p.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone, device


def configure_normalization(ds_cfg, backbone):
    """Override only mean/std before constructing any training or evaluation loader."""
    import math

    native = getattr(backbone, "pretrained_cfg", None)
    normalization = {}
    for key in ("mean", "std"):
        values = native.get(key) if isinstance(native, dict) else None
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or (key == "std" and any(v <= 0 for v in values))
        ):
            raise ValueError(f"Invalid pretrained normalization {key}: {values}")
        normalization[key] = [float(v) for v in values]
    return dict(ds_cfg, normalization=normalization)


def get_steps_per_epoch(n_samples, batch_size):
    return max((n_samples + batch_size - 1) // batch_size, 1)


def create_optim_config(args, warmup_epochs):
    accum = max(int(getattr(args, "accumulate_grad_batches", 1)), 1)
    effective_batch = args.batch_size * accum
    steps_per_epoch = max(get_steps_per_epoch(args.n_samples, effective_batch), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    return {
        "optimizer": {
            "type": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": warmup_steps,
            "max_steps": total_steps,
            "eta_min": 0.0,
        },
        "interval": "step",
    }


def _get_methods():
    from stable_cp.methods.simclr.simclr_cp import setup_simclr
    from stable_cp.methods.lejepa.lejepa_cp import setup_lejepa
    from stable_cp.methods.mae.mae_cp import setup_mae
    from stable_cp.methods.diet.diet_cp import setup_diet

    return {
        "lejepa": {"n_views": 8, "setup": setup_lejepa, "strong_aug": True},
        "diet": {"n_views": 1, "setup": setup_diet},
        "simclr": {"n_views": 2, "setup": setup_simclr, "strong_aug": True},
        "mae": {"n_views": 1, "setup": setup_mae},
    }


def _create_shared_eval_data(args, ds_cfg, data_dir):
    """Create shared eval loaders and the shared sampled train indices."""
    train_tf, eval_tf = create_transforms(ds_cfg, n_views=1, strong_aug=False)
    test_loader, eval_train_loader, indices = create_eval_loaders(
        args, ds_cfg, train_tf, eval_tf, data_dir
    )
    # Clean train loader for KNN (same indices, no augmentation)
    _, knn_train_loader, _ = create_eval_loaders(
        args, ds_cfg, eval_tf, eval_tf, data_dir, indices=indices
    )
    return eval_tf, test_loader, eval_train_loader, knn_train_loader, indices


def _create_sft_data(args, ds_cfg, data_dir, eval_tf, indices):
    """Create SFT datamodule over the shared train indices."""
    from stable_cp.evaluation.sft_eval import SFT_BATCH_SIZE

    splits = ds_cfg.get("splits", ["train", "validation", "test"])
    train_split, val_split, _ = splits

    sft_train_tf, _ = create_transforms(ds_cfg, n_views=1, strong_aug=False)

    full_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=sft_train_tf,
        cache_dir=data_dir,
        seed=args.seed,
    )
    val_data = get_dataset(
        args.dataset,
        split=val_split,
        transform=eval_tf,
        cache_dir=data_dir,
        seed=args.seed,
    )

    train_subset = CPSubset(full_train, indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=SFT_BATCH_SIZE,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


def _create_cp_data(args, ds_cfg, data_dir, indices, method_cfg):
    """Create CP datamodule over the shared train indices."""
    n_views = args.n_views if args.cp_method == "lejepa" else method_cfg.get("n_views", 1)
    cp_train_tf, cp_val_tf = create_transforms(
        ds_cfg,
        n_views,
        method_cfg.get("strong_aug", False),
    )
    cp_data, _ = create_train_datamodule(
        args,
        ds_cfg,
        cp_train_tf,
        cp_val_tf,
        data_dir,
        indices=indices,
    )
    return cp_data, n_views


def _run_sft_phase(
    backbone,
    sft_data,
    test_loader,
    device,
    ds_cfg,
    embed_dim,
    indices,
    args,
    logger,
    prefix,
):
    """Run full FT without creating, resuming, or deleting FT checkpoints."""

    results = sft_evaluate(
        backbone,
        sft_data,
        test_loader,
        device,
        num_classes=ds_cfg["num_classes"],
        embed_dim=embed_dim,
        n_samples=len(indices),
        pool_strategy=args.pool_strategy,
        seed=args.seed,
        ckpt_path=None,
        logger=logger,
        prefix=prefix,
    )
    for key, value in results.items():
        logger.experiment.summary[key] = value
    return results


def run_baseline(
    backbone,
    eval_train_loader,
    test_loader,
    device,
    args,
    logger,
    knn_train_loader=None,
    geometry=None,
):
    """Pre-CP evaluation: KNN + Linear Probe."""
    if args.skip_baseline:
        return None
    print("Pre-CP evaluation (kNN + linear probe)")
    results = zero_shot_eval(
        backbone,
        eval_train_loader,
        test_loader,
        device,
        k_neighbors=args.knn_k,
        pool_strategy=args.pool_strategy,
        knn_train_loader=knn_train_loader,
        verbose=True,
        geometry=geometry,
    )
    logged = {k: v for k, v in results.items() if k != "geometry"}
    logged.update(
        {
            f"geometry/{k}": v
            for k, v in results.get("geometry", {}).items()
            if isinstance(v, (int, float))
        }
    )
    logger.experiment.log({f"baseline/{k}": v for k, v in logged.items()}, step=0)
    for k, v in logged.items():
        logger.experiment.summary[f"baseline/{k}"] = v
    print(f"Baseline: knn_f1={results['knn_f1']:.4f} linear_f1={results['linear_pytorch_f1']:.4f}")
    return results


def run_final_eval(
    backbone,
    eval_train_loader,
    test_loader,
    device,
    args,
    logger,
    baseline_results,
    knn_train_loader=None,
):
    """Post-CP evaluation: KNN + Linear Probe."""
    if args.skip_final_eval:
        return None
    print("Post-CP evaluation (kNN + linear probe)")
    final_results = zero_shot_eval(
        backbone,
        eval_train_loader,
        test_loader,
        device,
        k_neighbors=args.knn_k,
        pool_strategy=args.pool_strategy,
        knn_train_loader=knn_train_loader,
        verbose=True,
    )
    for k, v in final_results.items():
        logger.experiment.summary[f"final/{k}"] = v

    if baseline_results:
        print("Improvement:")
        for key in ["knn_f1", "linear_pytorch_f1", "knn_acc", "linear_pytorch_acc"]:
            if key in baseline_results and key in final_results:
                delta = final_results[key] - baseline_results[key]
                logger.experiment.summary[f"delta/{key}"] = delta
                print(
                    f"  {key}: {baseline_results[key]:.4f} -> "
                    f"{final_results[key]:.4f} ({delta:+.4f})"
                )

    return final_results


def run_training(
    module,
    data,
    args,
    ds_cfg,
    embed_dim,
    freeze_epochs,
    logger,
    ckpt_path,
    method=None,
    num_trained_blocks=None,
):
    if num_trained_blocks is None:
        num_trained_blocks = args.num_trained_blocks
    checkpoint = Path(ckpt_path).expanduser().resolve()
    if checkpoint.exists() and not checkpoint.is_file():
        raise ValueError(f"CP checkpoint path exists but is not a regular file: {checkpoint}")
    if checkpoint.exists() and not getattr(args, "resume", False):
        raise FileExistsError(
            f"CP checkpoint already exists: {checkpoint}. Use --resume or a new "
            "checkpoint directory; existing weights will not be deleted."
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    resume_path = str(checkpoint) if checkpoint.is_file() else None

    callbacks = [
        FreezeBackboneCallback(freeze_epochs=freeze_epochs, num_trained_blocks=num_trained_blocks),
        *create_cp_evaluation_callbacks(
            module,
            ds_cfg["num_classes"],
            embed_dim,
            include_f1=True,
            include_auroc=True,
            knn_queue_length=max(args.n_samples, 5000),
            knn_k=min(args.knn_k, args.n_samples),
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(checkpoint.parent),
            filename=checkpoint.stem,
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        ),
    ]
    # Keep checkpoint paths explicit and independent of SPT's job-level cache.
    config = spt.get_config()
    previous_cache = config.cache_dir
    try:
        config.cache_dir = None
        with tempfile.TemporaryDirectory(prefix="cp-trainer-") as trainer_root:
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                accumulate_grad_batches=getattr(args, "accumulate_grad_batches", 1),
                num_sanity_val_steps=0,
                log_every_n_steps=10,
                callbacks=callbacks,
                precision="16-mixed",
                logger=logger,
                default_root_dir=trainer_root,
                plugins=(
                    [SLURMEnvironment(auto_requeue=False)] if SLURMEnvironment.detect() else None
                ),
            )
            spt.Manager(
                trainer=trainer,
                module=module,
                data=data,
                ckpt_path=resume_path,
                seed=args.seed,
                weights_only=False,
            )()
            trainer.save_checkpoint(str(checkpoint))
    finally:
        config.cache_dir = previous_cache


def main():
    METHODS = _get_methods()

    parser = create_base_parser("Continued Pretraining CLI")

    # ---- CP method  ----
    parser.add_argument("--cp-method", type=str, choices=list(METHODS.keys()))

    # ---- Shared CP hyper-parameters ----
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)

    # ---- SimCLR ----
    parser.add_argument("--temperature", type=float, default=0.5)

    # ---- LeJEPA ----
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument(
        "--multivariate-test",
        type=str,
        default="slicing",
        choices=["slicing", "bhep", "bhep_m", "comb", "hv", "hz"],
    )
    parser.add_argument(
        "--univariate-test",
        type=str,
        default="epps_pulley",
        choices=[
            "epps_pulley",
            "anderson_darling",
            "cramer_von_mises",
            "watson",
            "entropy",
            "shapiro_wilk",
            "jarque_bera",
            "vcreg",
            "nll",
            "moments",
        ],
    )
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-points", type=int, default=17)
    parser.add_argument("--num-slices", type=int, default=1000)
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "sum", "none"],
    )
    parser.add_argument("--clip-value", type=float, default=None)
    parser.add_argument("--bhep-beta", type=float, default=0.1)
    parser.add_argument("--bhep-m-beta", type=float, default=10)
    parser.add_argument("--comb-gamma", type=float, default=0.1)
    parser.add_argument("--hv-gamma", type=float, default=1.0)
    parser.add_argument("--entropy-m", type=int, default=1)
    parser.add_argument(
        "--entropy-method",
        type=str,
        default="centered",
        choices=["centered", "right"],
    )
    parser.add_argument("--moments-k-max", type=int, default=4)
    parser.add_argument(
        "--sw-expectation",
        type=str,
        default="elfving",
        choices=["elfving", "blom", "rahman"],
    )
    parser.add_argument(
        "--sw-covariance",
        type=str,
        default="shapiro_francia",
        choices=["shapiro_francia", "rahman"],
    )
    parser.add_argument("--nll-alpha", type=float, default=0.5)

    # ---- DIET ----
    parser.add_argument("--label-smoothing", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-cutmix-prob", type=float, default=0.8)
    parser.add_argument("--mixup-cutmix-switch-prob", type=float, default=0.5)

    # ---- MAE ----
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # ---- Evaluation mode flags ----
    parser.add_argument(
        "--pre-cp-sft",
        action="store_true",
        help="Additionally fine-tune and evaluate a copy of the pretrained encoder.",
    )
    parser.add_argument(
        "--post-cp-sft",
        action="store_true",
        help="Additionally fine-tune and evaluate a copy of the CP-trained encoder.",
    )
    parser.add_argument(
        "--no-cp",
        action="store_true",
        help="Skip CP training entirely (baseline-only mode)",
    )
    parser.add_argument(
        "--geometry-reference",
        help="ImageNet reference NPZ for pre-CP geometry on up to 5000 clean train features.",
    )
    parser.add_argument(
        "--geometry-features", help="Optional NPZ output for the selected raw geometry features."
    )

    # ---- Results output ----
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to save results as JSON (for automated result collection)",
    )

    args = parser.parse_args()

    # ---- Validate flag combinations ----
    if not args.no_cp and args.cp_method is None:
        parser.error("--cp-method is required unless --no-cp is set")
    if args.no_cp and args.post_cp_sft:
        parser.error("--post-cp-sft requires CP training (remove --no-cp)")
    if args.geometry_features and not args.geometry_reference:
        parser.error("--geometry-features requires --geometry-reference")
    if args.geometry_reference and args.skip_baseline:
        parser.error("--geometry-reference requires pre-CP evaluation")

    # ---- Setup ----
    data_dir, checkpoint_dir = setup_paths(args)
    pl.seed_everything(args.seed, workers=True)
    ds_cfg, freeze_epochs, warmup_epochs = get_config(args)
    if args.full_train:
        args.n_samples = len(
            get_dataset(
                args.dataset,
                split=ds_cfg["splits"][0],
                transform=None,
                cache_dir=data_dir,
                seed=args.seed,
            )
        )
        print(f"Full training split: {args.n_samples} samples")

    # ---- Backbone ----
    backbone, device = load_backbone(args, img_size=ds_cfg["input_size"])
    embed_dim = backbone.num_features
    args.pool_strategy = args.pool_strategy or default_pool_strategy(args.backbone)
    readout = feature_readout(args.backbone, args.pool_strategy)
    ds_cfg = configure_normalization(ds_cfg, backbone)
    print(f"Pretrained normalization: {ds_cfg['normalization']}; readout: {readout}")
    geometry = None
    if args.geometry_reference:
        geometry = dict(
            reference_path=args.geometry_reference,
            output_path=args.geometry_features,
            metadata=dict(
                dataset=args.dataset,
                seed=args.seed,
                backbone=args.backbone,
                pool_strategy=args.pool_strategy,
                normalization=ds_cfg["normalization"],
            ),
        )
        if args.backbone.endswith(".mae"):
            geometry["metadata"]["feature_readout"] = readout

    # ---- Wandb logger ----
    if args.no_cp:
        project = args.project or f"{args.dataset}-pre-eval"
        run_name = f"pre_eval_n{args.n_samples}_s{args.seed}"
    else:
        method_cfg = METHODS[args.cp_method]
        project = args.project or f"{args.dataset}-{args.cp_method}-cp"
        run_name = (
            f"{args.cp_method}_pre_n{args.n_samples}"
            f"_ep{args.epochs}_frz{freeze_epochs}"
            f"_blk{args.num_trained_blocks}_s{args.seed}"
        )
    if args.run_name:
        run_name = args.run_name
    elif args.backbone.endswith(".mae"):
        run_name += f"_{readout}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    sft_data = None
    cp_data = None

    # Shared evaluation loaders (KNN/LP + SFT test evaluation)
    eval_tf, test_loader, eval_train_loader, knn_train_loader, indices = _create_shared_eval_data(
        args,
        ds_cfg,
        data_dir,
    )

    # SFT data (n_views=1, standard augmentation)
    if args.pre_cp_sft or args.post_cp_sft:
        sft_data = _create_sft_data(args, ds_cfg, data_dir, eval_tf, indices)
        print(f"SFT data created: {len(indices)} train samples")

    # CP data (method-specific multi-view transforms)
    if not args.no_cp:
        method_cfg = METHODS[args.cp_method]
        cp_data, n_views = _create_cp_data(args, ds_cfg, data_dir, indices, method_cfg)
        print(
            f"{args.cp_method.upper()} CP: {args.dataset} | {args.backbone} | "
            f"views={n_views} freeze={freeze_epochs} warmup={warmup_epochs}"
        )

    # Pre-CP evaluation; FT operates on a copy of the encoder.
    baseline_results = None
    sft_pre_results = None

    if not args.skip_baseline:
        baseline_results = run_baseline(
            backbone,
            eval_train_loader,
            test_loader,
            device,
            args,
            logger,
            knn_train_loader=knn_train_loader,
            geometry=geometry,
        )

    if args.pre_cp_sft:
        sft_pre_results = _run_sft_phase(
            backbone,
            sft_data,
            test_loader,
            device,
            ds_cfg,
            embed_dim,
            indices,
            args,
            logger,
            "pre_sft",
        )

    # Continued pretraining on the shared target subset.
    if not args.no_cp:
        method_cfg = METHODS[args.cp_method]
        optim_config = create_optim_config(args, warmup_epochs)

        kwargs = dict(
            num_samples=len(indices),
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
            lamb=args.lamb,
            label_smoothing=args.label_smoothing,
            temperature=args.temperature,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_cutmix_prob=args.mixup_cutmix_prob,
            mixup_cutmix_switch_prob=args.mixup_cutmix_switch_prob,
            pool_strategy=args.pool_strategy,
        )

        if args.cp_method == "mae":
            kwargs.update(
                decoder_dim=args.decoder_dim,
                decoder_depth=args.decoder_depth,
                mask_ratio=args.mask_ratio,
            )

        if args.cp_method == "lejepa":
            from stable_cp.methods.lejepa.lejepa_cp import build_sigreg_loss

            if getattr(args, "reduction", None) == "none":
                args.reduction = None
            sigreg_loss = build_sigreg_loss(args)
            module = method_cfg["setup"](backbone, embed_dim, optim_config, sigreg_loss, **kwargs)
        else:
            module = method_cfg["setup"](backbone, embed_dim, optim_config, **kwargs)

        cp_dir = checkpoint_dir / args.cp_method
        cp_dir.mkdir(parents=True, exist_ok=True)
        readout_suffix = f"_{readout}" if args.backbone.endswith(".mae") else ""
        cp_ckpt_path = str(
            cp_dir / f"{args.dataset}_{args.backbone.replace('/', '_')}"
            f"_n{args.n_samples}_s{args.seed}{readout_suffix}.ckpt"
        )
        run_training(
            module,
            cp_data,
            args,
            ds_cfg,
            embed_dim,
            freeze_epochs,
            logger,
            cp_ckpt_path,
        )

    # Evaluate the same backbone after training.
    final_eval_results = None
    sft_post_results = None

    if not args.skip_final_eval and not args.no_cp:
        final_eval_results = run_final_eval(
            backbone,
            eval_train_loader,
            test_loader,
            device,
            args,
            logger,
            baseline_results,
            knn_train_loader=knn_train_loader,
        )

    if args.post_cp_sft:
        sft_post_results = _run_sft_phase(
            backbone,
            sft_data,
            test_loader,
            device,
            ds_cfg,
            embed_dim,
            indices,
            args,
            logger,
            "post_sft",
        )

    if args.results_json:
        results_json = {
            "dataset": args.dataset,
            "n_samples": args.n_samples,
            "n_train_actual": len(indices),
            "n_test": len(test_loader.dataset),
            "num_classes": ds_cfg["num_classes"],
            "full_train": args.full_train,
            "backbone": args.backbone,
            "method": "none" if args.no_cp else args.cp_method,
            "seed": args.seed,
            "epochs": args.epochs,
            "no_cp": args.no_cp,
            "normalization_mode": "pretrained",
            "normalization": ds_cfg["normalization"],
            "feature_readout": readout,
            "cp_config": dict(vars(args), freeze_epochs=freeze_epochs, warmup_epochs=warmup_epochs),
        }

        for stage, metrics in (("pre", baseline_results), ("post", final_eval_results)):
            if metrics:
                for output, source in (
                    ("knn_f1", "knn_f1"),
                    ("knn_acc", "knn_acc"),
                    ("linear_f1", "linear_pytorch_f1"),
                    ("linear_acc", "linear_pytorch_acc"),
                ):
                    results_json[f"{stage}_{output}"] = metrics[source]
        if baseline_results and "geometry" in baseline_results:
            results_json["geometry"] = baseline_results["geometry"]
        for stage, metrics in (("pre", sft_pre_results), ("post", sft_post_results)):
            if metrics:
                for key in ("f1", "acc", "auroc"):
                    results_json[f"{stage}_sft_{key}"] = metrics[f"{stage}_sft_{key}"]
                for key in ("protocol", "trainable_params", "total_params"):
                    results_json[f"{stage}_sft_{key}"] = metrics[f"sft_{key}"]

        results_path = Path(args.results_json).expanduser()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=results_path.parent,
            prefix=f".{results_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(results_json, f, indent=2)
        Path(f.name).replace(results_path)
        print(f"Results saved to {results_path}")

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
