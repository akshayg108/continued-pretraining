#!/usr/bin/env python
# LeJEPA Continued Pretraining with configurable SigReg tests
import torch.nn as nn
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger
from torchvision.ops import MLP

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
from .lejepa_forward import lejepa_forward
from .lejepa_losses import (
    EppsPulley,
    AndersonDarling,
    CramerVonMises,
    Watson,
    Entropy,
    ShapiroWilk,
    ExtendedJarqueBera,
    VCReg,
    NLL,
    Moments,
    SlicingUnivariateTest,
    BHEP,
    BHEP_M,
    COMB,
    HV,
    HZ,
)

# Univariate normality tests
UNIVARIATE_TESTS = {
    "epps_pulley": EppsPulley,
    "anderson_darling": AndersonDarling,
    "cramer_von_mises": CramerVonMises,
    "watson": Watson,
    "entropy": Entropy,
    "shapiro_wilk": ShapiroWilk,
    "jarque_bera": ExtendedJarqueBera,
    "vcreg": VCReg,
    "nll": NLL,
    "moments": Moments,
}

# Multivariate normality tests
MULTIVARIATE_TESTS = {
    "slicing": SlicingUnivariateTest,
    "bhep": BHEP,
    "bhep_m": BHEP_M,
    "comb": COMB,
    "hv": HV,
    "hz": HZ,
}


def build_lejepa_projector(embed_dim, hidden_dim, proj_dim):
    return MLP(embed_dim, [hidden_dim, hidden_dim, proj_dim], norm_layer=nn.BatchNorm1d)


def build_sigreg_loss(args):
    # Build univariate test with appropriate params
    utest_cls = UNIVARIATE_TESTS[args.univariate_test]
    if args.univariate_test == "epps_pulley":
        univariate_test = utest_cls(t_max=args.t_max, n_points=args.n_points)
    elif args.univariate_test == "entropy":
        univariate_test = utest_cls(m=args.entropy_m, method=args.entropy_method)
    elif args.univariate_test == "moments":
        univariate_test = utest_cls(k_max=args.moments_k_max)
    elif args.univariate_test == "shapiro_wilk":
        univariate_test = utest_cls(
            expectation_mode=args.sw_expectation, covariance_mode=args.sw_covariance
        )
    elif args.univariate_test == "nll":
        univariate_test = utest_cls(alpha=args.nll_alpha)
    else:
        univariate_test = utest_cls()

    # Build multivariate test
    mtest = args.multivariate_test
    if mtest == "slicing":
        return SlicingUnivariateTest(
            univariate_test,
            num_slices=args.num_slices,
            reduction=args.reduction,
            clip_value=args.clip_value,
        )
    elif mtest == "bhep":
        return BHEP(beta=args.bhep_beta)
    elif mtest == "bhep_m":
        return BHEP_M(beta=args.bhep_beta)
    elif mtest == "comb":
        return COMB(gamma=args.comb_gamma)
    elif mtest == "hv":
        return HV(gamma=args.hv_gamma)
    elif mtest == "hz":
        return HZ()
    raise ValueError(f"Unknown multivariate test: {mtest}")


def setup_lejepa(backbone, embed_dim, optim_config, sigreg_loss, **kwargs):
    return spt.Module(
        backbone=backbone,
        projector=build_lejepa_projector(
            embed_dim, kwargs["hidden_dim"], kwargs["proj_dim"]
        ),
        sigreg_loss=sigreg_loss,
        lamb=kwargs["lamb"],
        forward=lejepa_forward,
        optim=optim_config,
    )


def main():
    parser = create_base_parser("LeJEPA Continued Pretraining")
    # Projector
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--lamb", type=float, default=0.02)
    # Test selection
    parser.add_argument(
        "--multivariate-test",
        type=str,
        default="slicing",
        choices=list(MULTIVARIATE_TESTS.keys()),
    )
    parser.add_argument(
        "--univariate-test",
        type=str,
        default="epps_pulley",
        choices=list(UNIVARIATE_TESTS.keys()),
    )
    # EppsPulley
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-points", type=int, default=17)
    # Slicing
    parser.add_argument("--num-slices", type=int, default=256)
    parser.add_argument(
        "--reduction", type=str, default="mean", choices=["mean", "sum", "none"]
    )
    parser.add_argument("--clip-value", type=float, default=None)
    # BHEP
    parser.add_argument("--bhep-beta", type=float, default=0.1)
    # COMB
    parser.add_argument("--comb-gamma", type=float, default=0.1)
    # HV
    parser.add_argument("--hv-gamma", type=float, default=1.0)
    # Entropy
    parser.add_argument("--entropy-m", type=int, default=1)
    parser.add_argument(
        "--entropy-method", type=str, default="centered", choices=["centered", "right"]
    )
    # Moments
    parser.add_argument("--moments-k-max", type=int, default=4)
    # ShapiroWilk
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
    # NLL
    parser.add_argument("--nll-alpha", type=float, default=0.5)

    args = parser.parse_args()
    if args.reduction == "none":
        args.reduction = None

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    print(
        f"LeJEPA CP: {args.dataset} | {args.backbone} | views={args.n_views} freeze={freeze_epochs}"
    )
    print(
        f"  SigReg: {args.multivariate_test}({args.univariate_test}) slices={args.num_slices} lamb={args.lamb}"
    )

    train_transform, val_transform = create_transforms(
        ds_cfg, n_views=args.n_views, strong_aug=True
    )
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir
    )

    backbone, device = load_backbone(args, img_size=ds_cfg["input_size"])

    project = args.project or f"{args.dataset}-lejepa-cp"
    run_name = f"lejepa_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}_v{args.n_views}_{args.univariate_test}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(
        backbone, eval_train_loader, test_loader, device, args, logger
    )
    optim_config = create_optim_config(args, warmup_epochs)

    sigreg_loss = build_sigreg_loss(args)
    module = setup_lejepa(
        backbone,
        embed_dim,
        optim_config,
        sigreg_loss,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        lamb=args.lamb,
    )

    ckpt_path = str(
        checkpoint_dir
        / f"lejepa_cp_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt"
    )
    run_training(
        module,
        data,
        args,
        ds_cfg,
        embed_dim,
        freeze_epochs,
        logger,
        ckpt_path,
        method="lejepa",
    )
    run_final_eval(
        backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
    )

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
