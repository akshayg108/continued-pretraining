import torch.nn as nn
import stable_pretraining as spt

from .diet_forward import diet_forward


def setup_diet(backbone, embed_dim, optim_config, **kwargs):
    num_samples = kwargs["num_samples"]
    label_smoothing = kwargs.get("label_smoothing", 0.3)
    mixup_alpha, cutmix_alpha = (
        kwargs.get("mixup_alpha", 1.0),
        kwargs.get("cutmix_alpha", 1.0),
    )
    mixup_cutmix_prob = kwargs.get("mixup_cutmix_prob", 0.8)
    mixup_cutmix_switch_prob = kwargs.get("mixup_cutmix_switch_prob", 0.5)
    pool_strategy = kwargs.get("pool_strategy", "cls")
    return spt.Module(
        backbone=backbone,
        diet_head=nn.Linear(embed_dim, num_samples, bias=False),
        diet_loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_samples=num_samples,
        mixup_cutmix_prob=mixup_cutmix_prob,
        mixup_cutmix_switch_prob=mixup_cutmix_switch_prob,
        pool_strategy=pool_strategy,
        forward=diet_forward,
        optim=optim_config,
    )
