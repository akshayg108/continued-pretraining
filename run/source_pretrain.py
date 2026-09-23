#!/usr/bin/env python3
"""Paired ImageNet-only and target-enriched LeJEPA pretraining from scratch."""

import argparse
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
import stable_pretraining as spt
import timm
import torch
from torch.utils.data import DataLoader

from run.source_pretrain_data import (
    DOMAIN_NAMES,
    SPLIT_SEED,
    ShuffledStepSampler,
    build_train_dataset,
    build_validation_dataset,
    make_transforms,
    prepare_imagenet_validation,
    prepare_sources,
    staged_sources,
)
from stable_cp.methods.lejepa.lejepa_cp import build_lejepa_projector
from stable_cp.methods.lejepa.lejepa_losses import EppsPulley, SlicingUnivariateTest

BATCH_SIZE = 256
MODEL = "vit_base_patch16_224"
DEFAULT_STEPS = (1_281_167 // BATCH_SIZE) * 100


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unpack_imagenet(archive, destination):
    """Extract the official training archive, committing one class at a time."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source:
            name = Path(member.name).name
            if not member.isfile() or not re.fullmatch(r"n\d{8}\.tar", name):
                continue
            target = destination / name.removesuffix(".tar")
            if target.is_dir():
                continue
            with tempfile.TemporaryDirectory(
                prefix="imagenet-class-", dir=destination.parent
            ) as tmp:
                with source.extractfile(member) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as images:
                        images.extractall(tmp, filter="data")
                Path(tmp).rename(target)
            print(f"EXTRACTED {target.name}", flush=True)


def source_forward(self, batch, stage):
    """Use the CP objective on all views while forwarding each crop size separately."""
    if stage != "fit":
        embedding = self.backbone.forward_features(batch["image"])[:, 0]
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        batch["knn_label"] = batch["label"]
        return {
            "probe_embedding": embedding,
            "probe_label": batch["label"],
            "knn_embedding": embedding,
        }
    views = [
        value["image"] for key, value in batch.items() if key.startswith(("global_", "local_"))
    ]
    batch_size = views[0].shape[0]
    groups = {}
    for index, images in enumerate(views):
        groups.setdefault(tuple(images.shape[-2:]), []).append(index)
    embeddings = [None] * len(views)
    for indices in groups.values():
        images = torch.cat([views[index] for index in indices])
        encoded = self.backbone.forward_features(images)[:, 0]
        for index, embedding in zip(indices, encoded.split(batch_size)):
            embeddings[index] = embedding
    projected = self.projector(torch.cat(embeddings)).reshape(len(views), batch_size, -1)
    with torch.autocast(device_type=projected.device.type, enabled=False):
        projected = projected.float()
        invariance = (projected - projected.mean(0)).square().mean()
        sigreg = self.sigreg_loss(projected)
        loss = (1 - self.lamb) * invariance + self.lamb * sigreg
    for name, value in (("loss", loss), ("invariance", invariance), ("sigreg", sigreg)):
        self.log(f"train/{name}", value, on_step=True, on_epoch=False, batch_size=batch_size)
    embedding = torch.nn.functional.normalize(embeddings[0].detach().float(), dim=-1)
    labels = batch["global_0"]["label"]
    is_imagenet = batch["global_0"]["domain_id"] == 0
    return {
        "loss": loss,
        "probe_embedding": embedding,
        "probe_label": labels.masked_fill(~is_imagenet, -100),
        "knn_embedding": embedding[is_imagenet],
        "knn_label": labels[is_imagenet],
    }


class DomainCounts(Callback):
    """Count images in completed updates, excluding prefetched and failed batches."""

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        ids = batch["global_0"]["domain_id"]
        module.seen_images.add_(torch.bincount(ids, minlength=len(DOMAIN_NAMES)))


def configuration(args, domain_sizes):
    return {
        "protocol": "source_coverage_lejepa_v1",
        "architecture": MODEL,
        "condition": args.condition,
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "pretrained": False,
        "trainable_blocks": "all",
        "readout": "final_norm_cls",
        "batch_size": BATCH_SIZE,
        "accumulation": 1,
        "total_steps": args.steps,
        "warmup_steps": min(25_020, max(1, args.steps // 20)),
        "optimizer": {"type": "AdamW", "lr": 5e-4, "weight_decay": 0.05, "betas": [0.9, 0.999]},
        "scheduler": {"type": "LinearWarmupCosineAnnealing", "start_factor": 0.01, "end_lr": 5e-7},
        "loss": {
            "lambda": 0.05,
            "slices": 1024,
            "t_max": 3.0,
            "n_points": 17,
            "center": "all_views",
            "sigreg_grouping": "per_view",
            "precision": "float32",
        },
        "projector": [768, 2048, 2048, 128],
        "precision": "bf16-mixed",
        "global_crops": 2,
        "global_size": 224,
        "global_scale": [0.3, 1.0],
        "local_crops": 8,
        "local_size": 96,
        "local_scale": [0.05, 0.3],
        "photometric": {
            "horizontal_flip_p": 0.5,
            "color_jitter": [0.4, 0.4, 0.2, 0.1],
            "color_jitter_p": 0.8,
            "grayscale_p": 0.2,
            "blur_kernel": 23,
            "blur_sigma": [0.1, 2.0],
            "blur_p": 0.5,
            "solarize_threshold": 128,
            "solarize_p": 0.2,
        },
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "drop_path_rate": 0.1,
        "activation_checkpointing": True,
        "sampling": "concatenate_and_shuffle_without_replacement",
        "domain_sizes": domain_sizes,
        "imagenet_dir": str(args.imagenet_dir),
        "geometry_split": "test",
        "geometry_max_samples": 5000,
        "online_evaluation": {
            "num_classes": 1000,
            "validation_every_steps": 5000,
            "validation_samples": 5000,
            "train_features": "first_global_cls_l2_imagenet_only",
            "knn_queue_length": 20000,
            "knn_k": 20,
            "knn_temperature": 0.07,
            "knn_distance": "cosine",
            "lp_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-6},
            "lp_scheduler": "constant",
        },
    }


def build_module(config):
    backbone = timm.create_model(
        config["architecture"],
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        drop_path_rate=config["drop_path_rate"],
    )
    backbone.set_grad_checkpointing(config["activation_checkpointing"])
    module = spt.Module(
        backbone=backbone,
        projector=build_lejepa_projector(
            backbone.num_features, config["projector"][1], config["projector"][-1]
        ),
        sigreg_loss=SlicingUnivariateTest(
            EppsPulley(t_max=3.0, n_points=17), num_slices=config["loss"]["slices"]
        ),
        lamb=config["loss"]["lambda"],
        forward=source_forward,
        hparams=config,
        optim={
            "optimizer": config["optimizer"],
            "scheduler": {
                **config["scheduler"],
                "total_steps": config["total_steps"],
                "peak_step": config["warmup_steps"],
            },
            "interval": "step",
            "frequency": 1,
        },
    )
    module.register_buffer("seen_images", torch.zeros(len(DOMAIN_NAMES), dtype=torch.long))
    return module


def fit(module, loader, val_loader, config, directory, checkpoint, checkpoint_every):
    from run.source_pretrain_eval import online_callbacks

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(directory / "checkpoints"),
        filename="step-{step}",
        every_n_train_steps=checkpoint_every,
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config["precision"],
        max_steps=config["total_steps"],
        max_epochs=-1,
        accumulate_grad_batches=1,
        val_check_interval=min(
            config["online_evaluation"]["validation_every_steps"], config["total_steps"]
        ),
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        callbacks=[DomainCounts(), *online_callbacks(module, config), checkpoint_callback],
        logger=CSVLogger(str(directory), name="logs"),
        default_root_dir=str(directory),
        plugins=[SLURMEnvironment(auto_requeue=False)] if SLURMEnvironment.detect() else None,
    )
    previous_cache = spt.get_config().cache_dir
    try:
        spt.get_config().cache_dir = None
        spt.Manager(
            trainer=trainer,
            module=module,
            data=spt.data.DataModule(train=loader, val=val_loader),
            seed=config["seed"],
            ckpt_path=str(checkpoint) if checkpoint else None,
            weights_only=False,
        )()
        if trainer.global_step != config["total_steps"]:
            raise RuntimeError(
                f"Training stopped at step {trainer.global_step}, before its fixed budget"
            )
        trainer.save_checkpoint(str(directory / "checkpoints" / "final.ckpt"))
        write_json(
            directory / "training.json",
            {
                "complete": True,
                "global_step": trainer.global_step,
                "total_steps": config["total_steps"],
                "seen_images": dict(zip(DOMAIN_NAMES, module.seen_images.tolist())),
            },
        )
    finally:
        spt.get_config().cache_dir = previous_cache


def run_training(args):
    from run.source_pretrain_eval import evaluate_online, evaluate_targets

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This experiment requires a CUDA GPU with BF16 support; request one H200"
        )
    directory = args.output_dir / args.condition / f"seed{args.seed}"
    final_checkpoint = directory / "checkpoints/final.ckpt"
    last_checkpoint = directory / "checkpoints/last.ckpt"
    checkpoint = next(
        (path for path in (final_checkpoint, last_checkpoint) if path.is_file()), None
    )
    if checkpoint and not args.resume:
        raise FileExistsError(
            f"Use --resume to continue {directory}, or choose another --output-dir"
        )
    print(
        f"RUN condition={args.condition} seed={args.seed} steps={args.steps} batch=256 accumulation=1",
        flush=True,
    )
    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    with staged_sources(args.root / "data", args.imagenet_dir) as (imagenet, targets, validation):
        train_transform, clean_transform = make_transforms()
        dataset, sizes = build_train_dataset(imagenet, targets, args.condition, train_transform)
        val_dataset = build_validation_dataset(validation, clean_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        config = configuration(args, sizes)
        config["online_validation"] = val_dataset.metadata
        config_path = directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"Existing run uses a different configuration: {config_path}")
        write_json(config_path, config)
        pl.seed_everything(args.seed, workers=True)
        module = build_module(config)
        start_step = 0
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            start_step = state["global_step"]
            # Trainer restores probe/queue state after native callback setup.
            module.load_state_dict(
                {
                    key: value
                    for key, value in state["state_dict"].items()
                    if not key.startswith(("callbacks_modules.", "callbacks_metrics."))
                }
            )
            del state
            print(f"RESUME {checkpoint} step={start_step}", flush=True)
        module.to("cuda")
        if start_step == 0:
            evaluate_targets(
                module.backbone,
                targets,
                directory / "initial",
                device="cuda",
                num_workers=args.num_workers,
            )
        if start_step > args.steps:
            raise ValueError("Checkpoint exceeds the requested training budget")
        if (
            start_step < args.steps
            or not final_checkpoint.exists()
            or not (directory / "training.json").exists()
        ):
            # Keep Lightning's epoch length unchanged while resuming the shuffled stream.
            sampler = ShuffledStepSampler(
                dataset,
                args.steps + start_step,
                BATCH_SIZE,
                args.seed,
                start_step=start_step,
            )
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2 if args.num_workers else None,
            )
            fit(module, loader, val_loader, config, directory, checkpoint, args.checkpoint_every)
        module.backbone.to("cuda")
        evaluate_targets(
            module.backbone,
            targets,
            directory / "final",
            device="cuda",
            num_workers=args.num_workers,
        )
        torch.save(
            {"state_dict": module.backbone.cpu().state_dict(), "config": config},
            directory / "encoder.pt",
        )
        module.cpu()
        evaluate_online(build_module(config), val_loader, config, directory, final_checkpoint)
        print(f"COMPLETE {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "report"])
    parser.add_argument("--root", type=Path, default=REPO.parent)
    parser.add_argument("--imagenet-dir", type=Path)
    parser.add_argument("--imagenet-val-dir", type=Path)
    parser.add_argument("--imagenet-archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--condition", choices=["imagenet", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.imagenet_dir = (
        (args.imagenet_dir or args.root / "data/imagenet/train").expanduser().resolve()
    )
    args.imagenet_val_dir = (
        (args.imagenet_val_dir or args.root / "data/imagenet_val").expanduser().resolve()
    )
    args.output_dir = (
        (args.output_dir or args.root / "outputs/source_coverage_v1").expanduser().resolve()
    )
    if args.steps < 2 or args.checkpoint_every < 1 or args.num_workers < 0:
        parser.error("Require steps >= 2, checkpoint-every >= 1, and num-workers >= 0")
    if args.command == "prepare":
        if args.imagenet_archive:
            unpack_imagenet(args.imagenet_archive.expanduser().resolve(), args.imagenet_dir)
        sizes = prepare_sources(args.root / "data", args.imagenet_dir)
        prepare_imagenet_validation(args.root / "data", args.imagenet_dir, args.imagenet_val_dir)
        write_json(
            args.output_dir / "prepared.json", {"domain_sizes": sizes, "split_seed": SPLIT_SEED}
        )
        print(json.dumps(sizes, indent=2), flush=True)
    elif args.command == "train":
        if args.condition is None:
            parser.error("train requires --condition")
        run_training(args)
    else:
        from run.source_pretrain_eval import report_results

        report_results(args.output_dir)


if __name__ == "__main__":
    main()
