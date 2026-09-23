import torch.nn as nn
import stable_pretraining as spt
from stable_pretraining.backbone import BatchNorm1dNoBias
from stable_pretraining.losses import NTXEntLoss

from .simclr_cp_forward import simclr_cp_forward


def build_simclr_projector(embed_dim, hidden_dim, proj_dim):
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, proj_dim, bias=False),
        BatchNorm1dNoBias(proj_dim),
    )


def setup_simclr(backbone, embed_dim, optim_config, **kwargs):
    proj_dim = kwargs.get("proj_dim", 128)
    hidden_dim = kwargs.get("hidden_dim", 2048)
    temperature = kwargs.get("temperature", 0.5)
    pool_strategy = kwargs.get("pool_strategy", "cls")
    return spt.Module(
        backbone=backbone,
        projector=build_simclr_projector(embed_dim, hidden_dim, proj_dim),
        simclr_loss=NTXEntLoss(temperature=temperature),
        pool_strategy=pool_strategy,
        forward=simclr_cp_forward,
        optim=optim_config,
    )
