#!/usr/bin/env python
# Supervised Fine-Tuning (SFT) - baseline for continued pretraining evaluation
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
from .sft_forward import sft_forward


def setup_sft(backbone, embed_dim, optim_config, **kwargs):
    """Setup Supervised Fine-Tuning module.

    Args:
        backbone: Pre-trained (or randomly initialized) backbone model
        embed_dim: Embedding dimension of the backbone
        optim_config: Optimizer configuration
        **kwargs:
            num_classes (int): Number of target classes
            label_smoothing (float): Label smoothing for CrossEntropyLoss
            pool_strategy (str): 'cls' or 'mean' for ViT embedding extraction

    Returns:
        spt.Module configured for supervised fine-tuning
    """
    num_classes = kwargs["num_classes"]
    label_smoothing = kwargs.get("label_smoothing", 0.0)
    pool_strategy = kwargs.get("pool_strategy", "cls")

    classifier = nn.Linear(embed_dim, num_classes)

    return spt.Module(
        backbone=backbone,
        classifier=classifier,
        supervised_loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        pool_strategy=pool_strategy,
        forward=sft_forward,
        optim=optim_config,
    )


def main():
    parser = create_base_parser("Supervised Fine-Tuning")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--pool-strategy", type=str, default="cls", choices=["cls", "mean"]
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use randomly initialized backbone (no pretrained weights)",
    )
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)
    num_classes = ds_cfg["num_classes"]

    print("=" * 50)
    print(f"SFT: {args.dataset} | {args.backbone}")
    print(f"  Random init: {args.random_init}")
    print(f"  Num classes: {num_classes}")
    print(f"  Freeze epochs: {freeze_epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print("=" * 50)

    train_transform, val_transform = create_transforms(ds_cfg, n_views=1)
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir
    )

    backbone, device = load_backbone(
        args, img_size=ds_cfg["input_size"], pretrained=not args.random_init
    )

    project = args.project or f"{args.dataset}-sft"
    init_tag = "rand" if args.random_init else "pre"
    run_name = f"sft_{init_tag}_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(
        backbone, eval_train_loader, test_loader, device, args, logger
    )
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_sft(
        backbone,
        embed_dim,
        optim_config,
        num_classes=num_classes,
        label_smoothing=args.label_smoothing,
        pool_strategy=args.pool_strategy,
    )

    ckpt_path = str(
        checkpoint_dir
        / f"sft_{args.dataset}_{args.backbone.replace('/', '_')}_{init_tag}.ckpt"
    )
    run_training(
        module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path,
        method="sft",
    )
    run_final_eval(
        backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
    )

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
