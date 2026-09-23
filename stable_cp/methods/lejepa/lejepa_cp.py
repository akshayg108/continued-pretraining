import torch.nn as nn
import stable_pretraining as spt
from torchvision.ops import MLP

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
        return BHEP_M(beta=args.bhep_m_beta)
    elif mtest == "comb":
        return COMB(gamma=args.comb_gamma)
    elif mtest == "hv":
        return HV(gamma=args.hv_gamma)
    elif mtest == "hz":
        return HZ()
    raise ValueError(f"Unknown multivariate test: {mtest}")


def setup_lejepa(backbone, embed_dim, optim_config, sigreg_loss, **kwargs):
    pool_strategy = kwargs.get("pool_strategy", "cls")
    return spt.Module(
        backbone=backbone,
        projector=build_lejepa_projector(embed_dim, kwargs["hidden_dim"], kwargs["proj_dim"]),
        sigreg_loss=sigreg_loss,
        lamb=kwargs["lamb"],
        pool_strategy=pool_strategy,
        forward=lejepa_forward,
        optim=optim_config,
    )
