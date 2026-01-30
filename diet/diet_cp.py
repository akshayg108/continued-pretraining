#!/usr/bin/env python
# DIET Continued Pretraining - instance discrimination
import torch.nn as nn
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger

from continued_pretraining import (
    BACKBONE_DIMS, create_base_parser, setup_paths, get_config, create_transforms,
    create_data_loaders, load_backbone, create_optim_config, run_baseline, run_training, run_final_eval
)
from .diet_forward import diet_forward


def setup_diet(backbone, embed_dim, optim_config, **kwargs):
    num_samples = kwargs["num_samples"]
    label_smoothing = kwargs.get("label_smoothing", 0.3)
    mixup_alpha, cutmix_alpha = kwargs.get("mixup_alpha", 1.0), kwargs.get("cutmix_alpha", 1.0)
    mixup_cutmix_prob = kwargs.get("mixup_cutmix_prob", 0.8)
    mixup_cutmix_switch_prob = kwargs.get("mixup_cutmix_switch_prob", 0.5)
    pool_strategy = kwargs.get("pool_strategy", "cls")
    return spt.Module(
        backbone=backbone,
        diet_head=nn.Linear(embed_dim, num_samples, bias=False),
        diet_loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_samples=num_samples,
        mixup_cutmix_prob=mixup_cutmix_prob, mixup_cutmix_switch_prob=mixup_cutmix_switch_prob,
        pool_strategy=pool_strategy, forward=diet_forward, optim=optim_config,
    )


def main():
    parser = create_base_parser("DIET Continued Pretraining")
    parser.add_argument("--label-smoothing", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-cutmix-prob", type=float, default=0.8)
    parser.add_argument("--mixup-cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument("--pool-strategy", type=str, default="cls", choices=["cls", "mean"])
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    print(f"DIET CP: {args.dataset} | {args.backbone} | freeze={freeze_epochs} warmup={warmup_epochs}")

    train_transform, val_transform = create_transforms(ds_cfg, n_views=1)
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir, remap_sample_idx=True)

    backbone, device = load_backbone(args)

    project = args.project or f"{args.dataset}-diet-cp"
    run_name = f"diet_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(backbone, eval_train_loader, test_loader, device, args, logger)
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_diet(backbone, embed_dim, optim_config, num_samples=len(indices),
                        label_smoothing=args.label_smoothing, mixup_alpha=args.mixup_alpha,
                        cutmix_alpha=args.cutmix_alpha, mixup_cutmix_prob=args.mixup_cutmix_prob,
                        mixup_cutmix_switch_prob=args.mixup_cutmix_switch_prob, pool_strategy=args.pool_strategy)

    ckpt_path = str(checkpoint_dir / f"diet_cp_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt")
    run_training(module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path)
    run_final_eval(backbone, eval_train_loader, test_loader, device, args, logger, baseline_results)

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
