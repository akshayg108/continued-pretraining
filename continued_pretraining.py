#!/usr/bin/env python
"""Unified CLI for Continued Pretraining (CP) experiments.

Supports two orthogonal axes:
  1. **CP methods**: lejepa, diet, simclr, mae_cp  (--cp-method)
  2. **SFT evaluation**: fine-tune + evaluate before/after CP  (--pre-cp-sft / --post-cp-sft)

The ``--no-cp`` flag skips CP training entirely, useful for baseline
evaluation (KNN + Linear Probe + SFT) on a pretrained or random backbone.
"""
import argparse, json, os
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import stable_pretraining as spt
from stable_pretraining.backbone.utils import from_timm

from stable_cp.callbacks import (
    FreezeBackboneCallback,
    create_cp_evaluation_callbacks,
)
from stable_cp.callbacks.lejepa_metrics import LeJEPAMetricsCallback
from stable_cp.evaluation.zero_shot_eval import zero_shot_eval
from stable_cp.evaluation.sft_eval import sft_evaluate
from stable_cp.utils.backbone import BACKBONE_DIMS
from stable_cp.data import DATASETS, get_dataset_config
from stable_cp.data import create_data_loaders, create_transforms


# ============================================================
# Shared helper functions
# ============================================================

def create_base_parser(description="Continued Pretraining"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASETS.keys())
    )
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--cache-dir", type=str, default="~/.cache")
    return parser


def setup_paths(args):
    """Setup paths for data and checkpoints."""
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = cache_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, checkpoint_dir


def get_config(args):
    ds_cfg = get_dataset_config(args.dataset)
    embed_dim = BACKBONE_DIMS.get(args.backbone, 384)
    freeze_epochs = (
        args.freeze_epochs
        if args.freeze_epochs is not None
        else int(args.epochs * 0.05)
    )
    warmup_epochs = args.warmup_epochs or int(args.epochs * 0.1)
    return ds_cfg, embed_dim, freeze_epochs, warmup_epochs


def load_backbone(args, img_size=224, pretrained=True):
    """Load backbone from TIMM.

    Args:
        args: Arguments with backbone name
        img_size: Input image size for the model (default: 224)
        pretrained: Whether to load pretrained weights (default: True)

    Returns:
        tuple: (backbone model, device)
    """
    backbone_name = args.backbone
    init_str = "pretrained" if pretrained else "randomly initialized"
    print(f"Loading TIMM model: {backbone_name} ({init_str}) with img_size={img_size}")
    backbone = from_timm(backbone_name, pretrained=pretrained, img_size=img_size)

    for p in backbone.parameters():
        p.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone, device


def create_optim_config(args, warmup_epochs):
    steps_per_epoch = args.n_samples // args.batch_size
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


# ============================================================
# Evaluation phases
# ============================================================

def run_baseline(backbone, eval_train_loader, test_loader, device, args, logger):
    """Pre-CP evaluation: KNN + Linear Probe."""
    if args.skip_baseline:
        return None
    print("Baseline eval (KNN + Linear Probe) …")
    results = zero_shot_eval(
        backbone,
        eval_train_loader,
        test_loader,
        device,
        k_neighbors=args.knn_k,
        linear_probe_method="both",
        verbose=True,
    )
    logger.experiment.log({f"baseline/{k}": v for k, v in results.items()}, step=0)
    for k, v in results.items():
        logger.experiment.summary[f"baseline/{k}"] = v
    print(
        f"Baseline: knn_f1={results['knn_f1']:.4f} linear_f1={results['linear_f1']:.4f}"
    )
    return results


def run_final_eval(
    backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
):
    """Post-CP evaluation: KNN + Linear Probe."""
    if args.skip_final_eval:
        return None
    print("Final eval (KNN + Linear Probe) …")
    final_results = zero_shot_eval(
        backbone,
        eval_train_loader,
        test_loader,
        device,
        k_neighbors=args.knn_k,
        linear_probe_method="both",
        verbose=True,
    )
    for k, v in final_results.items():
        logger.experiment.summary[f"final/{k}"] = v

    if baseline_results:
        print("Improvement:")
        for key in ["knn_f1", "linear_f1", "knn_acc", "linear_acc"]:
            if key in baseline_results and key in final_results:
                delta = final_results[key] - baseline_results[key]
                logger.experiment.summary[f"delta/{key}"] = delta
                print(
                    f"  {key}: {baseline_results[key]:.4f} -> {final_results[key]:.4f} ({delta:+.4f})"
                )

    return final_results


# ============================================================
# CP training
# ============================================================

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

    callbacks = [
        FreezeBackboneCallback(
            freeze_epochs=freeze_epochs, num_trained_blocks=num_trained_blocks
        ),
        *create_cp_evaluation_callbacks(
            module,
            ds_cfg["num_classes"],
            embed_dim,
            include_f1=True,
            include_auroc=True,
            knn_queue_length=args.n_samples,
            knn_k=args.knn_k,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if method == "lejepa" or getattr(args, "cp_method", None) == "lejepa":
        callbacks.append(LeJEPAMetricsCallback(log_every_n_steps=50))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision="16-mixed",
        logger=logger,
    )
    spt.Manager(
        trainer=trainer, module=module, data=data, ckpt_path=ckpt_path, seed=args.seed
    )()


# ============================================================
# Unified CLI
# ============================================================

def main():
    from stable_cp.methods.simclr.simclr_cp import setup_simclr
    from stable_cp.methods.lejepa.lejepa_cp import setup_lejepa
    from stable_cp.methods.mae.mae_cp import setup_mae_cp
    from stable_cp.methods.diet.diet_cp import setup_diet

    METHODS = {
        "lejepa": {"n_views": 4, "setup": setup_lejepa, "strong_aug": True},
        "diet": {"n_views": 1, "setup": setup_diet},
        "simclr": {"n_views": 2, "setup": setup_simclr, "strong_aug": True},
        "mae_cp": {"n_views": 1, "setup": setup_mae_cp},
    }

    parser = create_base_parser("Continued Pretraining CLI")

    # ---- CP method (not required when --no-cp) ----
    parser.add_argument(
        "--cp-method", type=str, required=True, choices=list(METHODS.keys())
    )

    # ---- CP method hyper-parameters ----
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-cutmix-prob", type=float, default=0.8)
    parser.add_argument("--mixup-cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument(
        "--pool-strategy", type=str, default="cls", choices=["cls", "mean"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # ---- Backbone initialization ----
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use randomly initialized backbone (no pretrained weights)",
    )

    # ---- Evaluation mode flags ----
    parser.add_argument(
        "--pre-cp-sft",
        action="store_true",
        help="Run SFT evaluation BEFORE CP (also enables KNN+LP baseline)",
    )
    parser.add_argument(
        "--post-cp-sft",
        action="store_true",
        help="Run SFT evaluation AFTER CP (also enables KNN+LP final eval)",
    )
    parser.add_argument(
        "--no-cp",
        action="store_true",
        help="Skip CP training entirely (baseline-only mode)",
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
    if args.no_cp and args.post_cp_sft:
        parser.error("--post-cp-sft requires CP training (remove --no-cp)")

    # ---- Setup ----
    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    # ---- Backbone ----
    pretrained = not getattr(args, "random_init", False)
    backbone, device = load_backbone(
        args, img_size=ds_cfg["input_size"], pretrained=pretrained
    )
    init_tag = "rand" if not pretrained else "pre"

    # ---- Wandb logger ----
    if args.no_cp:
        project = args.project or f"{args.dataset}-sft-eval"
        run_name = (f"sft_eval_{init_tag}_n{args.n_samples}_s{args.seed}")
    else:
        method_cfg = METHODS[args.cp_method]
        project = args.project or f"{args.dataset}-{args.cp_method}-cp"
        run_name = (f"{args.cp_method}_{init_tag}_n{args.n_samples}"
                    f"_ep{args.epochs}_frz{freeze_epochs}"
                    f"_blk{args.num_trained_blocks}_s{args.seed}")
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    # ================================================================
    # Data creation
    # ================================================================
    sft_data = None
    cp_data = None

    # SFT data (n_views=1, standard augmentation) — also provides
    # the shared eval_train_loader and test_loader for KNN/LP.
    if args.pre_cp_sft or args.post_cp_sft:
        sft_train_tf, sft_val_tf = create_transforms(
            ds_cfg, n_views=1, strong_aug=False
        )
        sft_data, test_loader, eval_train_loader, indices = create_data_loaders(
            args, ds_cfg, sft_train_tf, sft_val_tf, data_dir
        )
        print(f"SFT data created: {len(indices)} train samples")

    # CP data (method-specific multi-view transforms)
    if not args.no_cp:
        method_cfg = METHODS[args.cp_method]
        n_views = (
            args.n_views
            if args.cp_method == "lejepa"
            else method_cfg.get("n_views", 1)
        )
        cp_train_tf, cp_val_tf = create_transforms(
            ds_cfg, n_views, method_cfg.get("strong_aug", False)
        )
        cp_data, cp_test_loader, cp_eval_train_loader, cp_indices = (
            create_data_loaders(args, ds_cfg, cp_train_tf, cp_val_tf, data_dir)
        )
        # If no SFT flags, use CP's eval loaders for KNN/LP
        if sft_data is None:
            test_loader = cp_test_loader
            eval_train_loader = cp_eval_train_loader
            indices = cp_indices
        print(
            f"{args.cp_method.upper()} CP: {args.dataset} | {args.backbone} | "
            f"views={n_views} freeze={freeze_epochs} warmup={warmup_epochs}"
        )

    # Fallback: --no-cp without SFT flags still needs eval loaders
    # for baseline KNN/LP evaluation
    if sft_data is None and args.no_cp:
        base_train_tf, base_val_tf = create_transforms(
            ds_cfg, n_views=1, strong_aug=False
        )
        _, test_loader, eval_train_loader, indices = create_data_loaders(
            args, ds_cfg, base_train_tf, base_val_tf, data_dir
        )

    # ================================================================
    # Phase 1: Pre-CP evaluation
    # ================================================================
    baseline_results = None
    sft_pre_results = None

    if not args.skip_baseline:
        baseline_results = run_baseline(
            backbone, eval_train_loader, test_loader, device, args, logger
        )

    if args.pre_cp_sft:
        # SFT evaluation (deep-copies backbone internally)
        sft_pre_dir = checkpoint_dir / "sft_pre"
        sft_pre_dir.mkdir(parents=True, exist_ok=True)
        sft_ckpt = str(
            sft_pre_dir
            / f"{args.dataset}_{args.backbone.replace('/', '_')}"
              f"_n{args.n_samples}_s{args.seed}.ckpt"
        )
        sft_pre_results = sft_evaluate(
            backbone,
            sft_data,
            test_loader,
            device,
            num_classes=ds_cfg["num_classes"],
            embed_dim=embed_dim,
            n_samples=len(indices),
            pool_strategy=args.pool_strategy,
            seed=args.seed,
            ckpt_path=sft_ckpt,
            logger=logger,
            prefix="pre_sft",
        )
        for k, v in sft_pre_results.items():
            logger.experiment.summary[k] = v

    # ================================================================
    # Phase 2: Continued Pretraining
    # ================================================================
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

        if args.cp_method == "mae_cp":
            with torch.no_grad():
                test_input = torch.zeros(
                    1, 3, ds_cfg["input_size"], ds_cfg["input_size"],
                    device=next(backbone.parameters()).device,
                )
                tokens = backbone.forward_features(test_input)
                num_tokens = tokens.shape[1] - 1
            kwargs.update(
                image_size=ds_cfg["input_size"],
                num_tokens=num_tokens,
                decoder_dim=args.decoder_dim,
                decoder_depth=args.decoder_depth,
                mask_ratio=args.mask_ratio,
            )

        module = method_cfg["setup"](
            backbone, embed_dim, optim_config, **kwargs
        )

        cp_dir = checkpoint_dir / "cp"
        cp_dir.mkdir(parents=True, exist_ok=True)
        cp_ckpt_path = str(
            cp_dir
            / f"{args.dataset}_{args.backbone.replace('/', '_')}"
              f"_n{args.n_samples}_s{args.seed}.ckpt"
        )
        run_training(
            module, cp_data, args, ds_cfg, embed_dim, freeze_epochs,
            logger, cp_ckpt_path,
        )

    # ================================================================
    # Phase 3: Post-CP evaluation
    # ================================================================
    final_eval_results = None
    sft_post_results = None

    if not args.skip_final_eval and not args.no_cp:
        final_eval_results = run_final_eval(
            backbone, eval_train_loader, test_loader, device,
            args, logger, baseline_results,
        )

    if args.post_cp_sft:
        # SFT evaluation on CP-trained backbone (deep-copies internally)
        sft_post_dir = checkpoint_dir / "sft_post"
        sft_post_dir.mkdir(parents=True, exist_ok=True)
        sft_post_ckpt = str(
            sft_post_dir
            / f"{args.dataset}_{args.backbone.replace('/', '_')}"
              f"_n{args.n_samples}_s{args.seed}.ckpt"
        )
        sft_post_results = sft_evaluate(
            backbone,
            sft_data,
            test_loader,
            device,
            num_classes=ds_cfg["num_classes"],
            embed_dim=embed_dim,
            n_samples=len(indices),
            pool_strategy=args.pool_strategy,
            seed=args.seed,
            ckpt_path=sft_post_ckpt,
            logger=logger,
            prefix="post_sft",
        )
        for k, v in sft_post_results.items():
            logger.experiment.summary[k] = v

    # ================================================================
    # Save results to JSON
    # ================================================================
    if args.results_json:
        results_json = {
            "dataset": args.dataset,
            "n_samples": args.n_samples,
            "backbone": args.backbone,
            "method": args.cp_method or "none",
            "seed": args.seed,
            "epochs": args.epochs,
            "random_init": getattr(args, "random_init", False),
            "no_cp": args.no_cp,
        }

        # Pre-CP KNN / Linear Probe
        if baseline_results:
            results_json["pre_knn_f1"] = baseline_results.get("knn_f1", None)
            results_json["pre_linear_f1"] = baseline_results.get("linear_f1", None)
            results_json["pre_knn_acc"] = baseline_results.get("knn_acc", None)
            results_json["pre_linear_acc"] = baseline_results.get("linear_acc", None)

        # Pre-CP SFT
        if sft_pre_results:
            results_json["pre_sft_f1"] = sft_pre_results.get("pre_sft_f1", None)
            results_json["pre_sft_acc"] = sft_pre_results.get("pre_sft_acc", None)

        # Post-CP KNN / Linear Probe
        if final_eval_results:
            results_json["post_knn_f1"] = final_eval_results.get("knn_f1", None)
            results_json["post_linear_f1"] = final_eval_results.get("linear_f1", None)
            results_json["post_knn_acc"] = final_eval_results.get("knn_acc", None)
            results_json["post_linear_acc"] = final_eval_results.get("linear_acc", None)

        # Post-CP SFT
        if sft_post_results:
            results_json["post_sft_f1"] = sft_post_results.get("post_sft_f1", None)
            results_json["post_sft_acc"] = sft_post_results.get("post_sft_acc", None)

        results_path = Path(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {results_path}")

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
