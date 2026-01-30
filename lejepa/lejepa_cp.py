#!/usr/bin/env python
# LeJEPA Continued Pretraining - multi-view invariance + SigReg
import torch.nn as nn
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger
from torchvision.ops import MLP

from continued_pretraining import (
    BACKBONE_DIMS, create_base_parser, setup_paths, get_config, create_transforms,
    create_data_loaders, load_backbone, create_optim_config, run_baseline, run_training, run_final_eval
)
from .lejepa_forward import lejepa_forward
from .lejepa_losses import EppsPulley, SlicingUnivariateTest


def build_lejepa_projector(embed_dim, hidden_dim, proj_dim):
    return MLP(embed_dim, [hidden_dim, hidden_dim, proj_dim], norm_layer=nn.BatchNorm1d)


def setup_lejepa(backbone, embed_dim, optim_config, **kwargs):
    proj_dim = kwargs.get("proj_dim", 128)
    hidden_dim = kwargs.get("hidden_dim", 2048)
    lamb = kwargs.get("lamb", 0.02)
    return spt.Module(
        backbone=backbone,
        projector=build_lejepa_projector(embed_dim, hidden_dim, proj_dim),
        sigreg_loss=SlicingUnivariateTest(EppsPulley(t_max=3.0, n_points=17), num_slices=256, reduction="mean"),
        lamb=lamb,
        forward=lejepa_forward, optim=optim_config,
    )


def main():
    parser = create_base_parser("LeJEPA Continued Pretraining")
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--lamb", type=float, default=0.02)
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    print(f"LeJEPA CP: {args.dataset} | {args.backbone} | views={args.n_views} freeze={freeze_epochs} warmup={warmup_epochs}")

    train_transform, val_transform = create_transforms(ds_cfg, n_views=args.n_views, strong_aug=True)
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir)

    backbone, device = load_backbone(args)

    project = args.project or f"{args.dataset}-lejepa-cp"
    run_name = f"lejepa_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}_v{args.n_views}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(backbone, eval_train_loader, test_loader, device, args, logger)
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_lejepa(backbone, embed_dim, optim_config,
                          proj_dim=args.proj_dim, hidden_dim=args.hidden_dim, lamb=args.lamb)

    ckpt_path = str(checkpoint_dir / f"lejepa_cp_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt")
    run_training(module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path, method="lejepa")
    run_final_eval(backbone, eval_train_loader, test_loader, device, args, logger, baseline_results)

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
