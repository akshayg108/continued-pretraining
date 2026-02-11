#!/usr/bin/env python

import torch.nn as nn
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger

from continued_pretraining import (
    BACKBONE_DIMS,
    create_base_parser,
    setup_paths,
    get_config,
    load_backbone,
    create_optim_config,
    run_baseline,
    run_training,
    run_final_eval,
)
from stable_cp.data import create_transforms, create_data_loaders
from stable_cp.methods.tent.tent_cp_forward import tent_cp_forward


def setup_tent_cp(
    backbone,
    embed_dim,
    optim_config,
    num_classes,
    tent_mode="norm_only",
    num_trained_blocks=-1,
    **kwargs,
):
    """Setup TENT Continued Pretraining.

    Args:
        backbone: Pre-trained backbone model
        embed_dim: Embedding dimension
        optim_config: Optimizer configuration
        num_classes: Number of classes for classifier head
        tent_mode: 'norm_only' or 'combined'
        num_trained_blocks: Number of last blocks to train (for combined mode)

    Returns:
        spt.Module configured for TENT adaptation
    """
    # Initialize classifier (always trainable for TENT)
    classifier = nn.Linear(embed_dim, num_classes)
    classifier.requires_grad_(True)

    print(f"TENT setup: mode={tent_mode}, num_trained_blocks={num_trained_blocks}")
    print(f"  Classifier initialized with {embed_dim} -> {num_classes}")

    return spt.Module(
        backbone=backbone,
        classifier=classifier,
        forward=tent_cp_forward,
        optim=optim_config,
        tent_mode=tent_mode,
        num_trained_blocks=num_trained_blocks,
    )


def main():
    parser = create_base_parser("TENT Continued Pretraining")
    parser.add_argument(
        "--tent-mode",
        type=str,
        default="norm_only",
        choices=["norm_only", "combined"],
        help="TENT parameter update mode",
    )
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)
    num_classes = ds_cfg["num_classes"]

    print("=" * 50)
    print(f"TENT CP: {args.dataset} | {args.backbone}")
    print(f"  Mode: {args.tent_mode}")
    print(f"  Freeze epochs: {freeze_epochs}")
    print(f"  Num trained blocks: {args.num_trained_blocks}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print("=" * 50)

    train_transform, val_transform = create_transforms(ds_cfg, n_views=1)
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir
    )

    backbone, device = load_backbone(args)

    project = args.project or f"{args.dataset}-tent-cp"
    run_name = f"tent_{args.tent_mode}_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(
        backbone, eval_train_loader, test_loader, device, args, logger
    )
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_tent_cp(
        backbone,
        embed_dim,
        optim_config,
        num_classes=num_classes,
        tent_mode=args.tent_mode,
        num_trained_blocks=args.num_trained_blocks,
    )

    ckpt_path = str(
        checkpoint_dir
        / f"tent_cp_{args.dataset}_{args.backbone.replace('/', '_')}_{args.tent_mode}.ckpt"
    )
    print(f"Checkpoint path: {ckpt_path}")

    run_training(
        module,
        data,
        args,
        ds_cfg,
        embed_dim,
        freeze_epochs,
        logger,
        ckpt_path,
        method="tent",
    )

    run_final_eval(
        backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
    )

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
