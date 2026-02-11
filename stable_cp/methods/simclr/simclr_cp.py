#!/usr/bin/env python
# SimCLR Continued Pretraining - contrastive learning
import torch.nn as nn
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger
from stable_pretraining.losses import NTXEntLoss

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
from .simclr_cp_forward import simclr_cp_forward


def build_simclr_projector(embed_dim, hidden_dim, proj_dim):
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, proj_dim, bias=False),
        spt.utils.BatchNorm1dNoBias(proj_dim),
    )


def setup_simclr(backbone, embed_dim, optim_config, **kwargs):
    proj_dim = kwargs.get("proj_dim", 128)
    hidden_dim = kwargs.get("hidden_dim", 2048)
    temperature = kwargs.get("temperature", 0.5)
    return spt.Module(
        backbone=backbone,
        projector=build_simclr_projector(embed_dim, hidden_dim, proj_dim),
        simclr_loss=NTXEntLoss(temperature=temperature),
        forward=simclr_cp_forward,
        optim=optim_config,
    )


def main():
    parser = create_base_parser("SimCLR Continued Pretraining")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    print(
        f"SimCLR CP: {args.dataset} | {args.backbone} | freeze={freeze_epochs} warmup={warmup_epochs}"
    )

    train_transform, val_transform = create_transforms(
        ds_cfg, n_views=2, strong_aug=True
    )
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir
    )

    backbone, device = load_backbone(args)

    project = args.project or f"{args.dataset}-simclr-cp"
    run_name = f"simclr_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}_t{args.temperature}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(
        backbone, eval_train_loader, test_loader, device, args, logger
    )
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_simclr(
        backbone,
        embed_dim,
        optim_config,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
    )

    ckpt_path = str(
        checkpoint_dir
        / f"simclr_cp_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt"
    )
    run_training(
        module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path, method="simclr"
    )
    run_final_eval(
        backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
    )

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
