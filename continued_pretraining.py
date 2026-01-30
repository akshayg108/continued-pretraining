#!/usr/bin/env python
# Continued Pretraining - shared functions + unified CLI
import argparse, os
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.backbone.utils import from_huggingface
from stable_pretraining.data.transforms import MultiViewTransform

try:
    from callbacks import FreezeBackboneCallback, create_cp_evaluation_callbacks
    from callbacks.lejepa_metrics import LeJEPAMetricsCallback
    from evaluation.zero_shot_eval import zero_shot_eval
    from cp_datasets import DATASETS, get_dataset_config
except ImportError:
    from .callbacks import FreezeBackboneCallback, create_cp_evaluation_callbacks
    from .callbacks.lejepa_metrics import LeJEPAMetricsCallback
    from .evaluation.zero_shot_eval import zero_shot_eval
    from .cp_datasets import DATASETS, get_dataset_config

BACKBONE_DIMS = {
    "facebook/dinov2-small": 384, "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024, "facebook/dinov2-giant": 1536,
    "google/vit-base-patch16-224": 768, "google/vit-large-patch16-224": 1024,
    "vit_base_patch16": 768, "vit_large_patch16": 1024, "vit_huge_patch14": 1280,
}


class CPSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, remap_sample_idx=False):
        self.dataset, self.indices, self.remap_sample_idx = dataset, indices, remap_sample_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if self.remap_sample_idx and isinstance(sample, dict):
            sample["sample_idx"] = idx
        return sample


# Shared functions
def create_base_parser(description="Continued Pretraining"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
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
    cache_dir, checkpoint_dir = Path(args.cache_dir), Path(args.checkpoint_dir)
    data_dir = cache_dir / "huggingface" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(data_dir)
    return data_dir, checkpoint_dir


def get_config(args):
    ds_cfg = get_dataset_config(args.dataset)
    embed_dim = BACKBONE_DIMS.get(args.backbone, 384)
    freeze_epochs = args.freeze_epochs or int(args.epochs * 0.05)
    warmup_epochs = args.warmup_epochs or int(args.epochs * 0.1)
    return ds_cfg, embed_dim, freeze_epochs, warmup_epochs


def create_transforms(ds_cfg, n_views=1, strong_aug=False):
    if strong_aug:
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((ds_cfg["input_size"], ds_cfg["input_size"]), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    else:
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((ds_cfg["input_size"], ds_cfg["input_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0), p=0.2),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((ds_cfg["input_size"], ds_cfg["input_size"])),
        transforms.ToImage(**ds_cfg["normalization"]),
    )
    train_transform = MultiViewTransform({f"view_{i}": base_aug for i in range(n_views)}) if n_views > 1 else base_aug
    return train_transform, val_transform


def create_data_loaders(args, ds_cfg, train_transform, val_transform, data_dir, remap_sample_idx=False):
    hf_config = ds_cfg.get("hf_config")
    full_train = spt.data.HFDataset(ds_cfg["hf_name"], name=hf_config, split="train", transform=train_transform, trust_remote_code=True, cache_dir=str(data_dir))
    val_data = spt.data.HFDataset(ds_cfg["hf_name"], name=hf_config, split="validation", transform=val_transform, trust_remote_code=True, cache_dir=str(data_dir))
    test_data = spt.data.HFDataset(ds_cfg["hf_name"], name=hf_config, split="test", transform=val_transform, trust_remote_code=True, cache_dir=str(data_dir))

    torch.manual_seed(args.seed)
    indices = torch.randperm(len(full_train))[:args.n_samples].tolist()
    train_subset = CPSubset(full_train, indices, remap_sample_idx=remap_sample_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    eval_train = spt.data.HFDataset(ds_cfg["hf_name"], name=hf_config, split="train", transform=val_transform, trust_remote_code=True, cache_dir=str(data_dir))
    eval_train_loader = torch.utils.data.DataLoader(CPSubset(eval_train, indices, remap_sample_idx=False), batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Data: train={len(train_subset)} val={len(val_data)} test={len(test_data)}")
    return data, test_loader, eval_train_loader, indices


def load_backbone(args):
    backbone = from_huggingface(args.backbone, pretrained=True)
    for p in backbone.parameters():
        p.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone, device


def create_optim_config(args, warmup_epochs):
    steps_per_epoch = args.n_samples // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    return {
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR", "warmup_steps": warmup_steps, "max_steps": total_steps, "eta_min": 0.0},
        "interval": "step",
    }


def run_baseline(backbone, eval_train_loader, test_loader, device, args, logger):
    if args.skip_baseline:
        return None
    print("Baseline eval...")
    results = zero_shot_eval(backbone, eval_train_loader, test_loader, device, k_neighbors=args.knn_k, linear_probe_method="both", verbose=True)
    logger.experiment.log({f"baseline/{k}": v for k, v in results.items()}, step=0)
    for k, v in results.items():
        logger.experiment.summary[f"baseline/{k}"] = v
    print(f"Baseline: knn_f1={results['knn_f1']:.4f} linear_f1={results['linear_f1']:.4f}")
    return results


def run_training(module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path, method=None):
    callbacks = [
        FreezeBackboneCallback(freeze_epochs=freeze_epochs, num_trained_blocks=args.num_trained_blocks),
        *create_cp_evaluation_callbacks(module, ds_cfg["num_classes"], embed_dim, include_f1=True, include_auroc=True, knn_queue_length=args.n_samples, knn_k=args.knn_k),
        LearningRateMonitor(logging_interval="step"),
    ]
    if method == "lejepa" or getattr(args, "cp_method", None) == "lejepa":
        callbacks.append(LeJEPAMetricsCallback(log_every_n_steps=50))
    trainer = pl.Trainer(max_epochs=args.epochs, num_sanity_val_steps=0, callbacks=callbacks, precision="16-mixed", logger=logger)
    spt.Manager(trainer=trainer, module=module, data=data, ckpt_path=ckpt_path, seed=args.seed)()


def run_final_eval(backbone, eval_train_loader, test_loader, device, args, logger, baseline_results):
    if args.skip_final_eval:
        return
    print("Final eval...")
    final_results = zero_shot_eval(backbone, eval_train_loader, test_loader, device, k_neighbors=args.knn_k, linear_probe_method="both", verbose=True)
    # Log to summary only (not history) to avoid step ordering issues
    for k, v in final_results.items():
        logger.experiment.summary[f"final/{k}"] = v

    if baseline_results:
        print("Improvement:")
        for key in ["knn_f1", "linear_f1", "knn_acc", "linear_acc"]:
            if key in baseline_results and key in final_results:
                delta = final_results[key] - baseline_results[key]
                logger.experiment.summary[f"delta/{key}"] = delta
                print(f"  {key}: {baseline_results[key]:.4f} -> {final_results[key]:.4f} ({delta:+.4f})")


# Unified CLI
def main():
    from simclr.simclr_cp import setup_simclr
    from lejepa.lejepa_cp import setup_lejepa
    from mae.mae_cp import setup_mae_cp
    from diet.diet_cp import setup_diet

    METHODS = {
        "lejepa": {"n_views": 4, "setup": setup_lejepa, "strong_aug": True},
        "diet": {"n_views": 1, "setup": setup_diet, "needs_sample_idx": True},
        "simclr": {"n_views": 2, "setup": setup_simclr, "strong_aug": True},
        "mae_cp": {"n_views": 1, "setup": setup_mae_cp},
    }

    parser = create_base_parser("Continued Pretraining CLI")
    parser.add_argument("--cp-method", type=str, required=True, choices=list(METHODS.keys()))
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-cutmix-prob", type=float, default=0.8)
    parser.add_argument("--mixup-cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument("--pool-strategy", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    args = parser.parse_args()

    method_cfg = METHODS[args.cp_method]
    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)

    n_views = args.n_views if args.cp_method == "lejepa" else method_cfg.get("n_views", 1)
    print(f"{args.cp_method.upper()} CP: {args.dataset} | {args.backbone} | views={n_views} freeze={freeze_epochs} warmup={warmup_epochs}")

    train_transform, val_transform = create_transforms(ds_cfg, n_views, method_cfg.get("strong_aug", False))
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir, method_cfg.get("needs_sample_idx", False))

    backbone, device = load_backbone(args)

    project = args.project or f"{args.dataset}-{args.cp_method}-cp"
    run_name = f"{args.cp_method}_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(backbone, eval_train_loader, test_loader, device, args, logger)
    optim_config = create_optim_config(args, warmup_epochs)

    kwargs = dict(num_samples=len(indices), proj_dim=args.proj_dim, hidden_dim=args.hidden_dim,
                  lamb=args.lamb, label_smoothing=args.label_smoothing, temperature=args.temperature,
                  mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
                  mixup_cutmix_prob=args.mixup_cutmix_prob, mixup_cutmix_switch_prob=args.mixup_cutmix_switch_prob,
                  pool_strategy=args.pool_strategy)
    if args.cp_method == "mae_cp":
        with torch.no_grad():
            test_input = torch.zeros(1, 3, ds_cfg["input_size"], ds_cfg["input_size"], device=next(backbone.parameters()).device)
            out = backbone(test_input)
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            num_tokens = tokens.shape[1] - 1
        kwargs.update(image_size=ds_cfg["input_size"], num_tokens=num_tokens,
                      decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth, mask_ratio=args.mask_ratio)

    module = method_cfg["setup"](backbone, embed_dim, optim_config, **kwargs)

    ckpt_path = str(checkpoint_dir / f"{args.cp_method}_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt")
    run_training(module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path)
    run_final_eval(backbone, eval_train_loader, test_loader, device, args, logger, baseline_results)

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
