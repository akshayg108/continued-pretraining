#!/usr/bin/env python
# MAE Continued Pretraining - masked reconstruction (ViT-only)
import torch
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger
from stable_pretraining.backbone.vit import MAEDecoder

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
from cp_dataloader import create_transforms, create_data_loaders
from stable_cp.methods.mae.mae_cp_forward import mae_cp_forward


def setup_mae_cp(backbone, embed_dim, optim_config, **kwargs):
    image_size = kwargs["image_size"]
    num_tokens = kwargs["num_tokens"]
    decoder_dim = kwargs.get("decoder_dim", 512)
    decoder_depth = kwargs.get("decoder_depth", 4)
    mask_ratio = kwargs.get("mask_ratio", 0.75)

    patch_size = image_size // int(num_tokens**0.5)
    output_dim = patch_size**2 * 3

    # No need for projector since MAEDecoder already include one
    # decoder = MAEDecoder(embed_dim=decoder_dim, decoder_embed_dim=decoder_dim,
    #                      output_dim=output_dim, num_patches=num_tokens, depth=decoder_depth)
    # projector = nn.Linear(embed_dim, decoder_dim)
    # return spt.Module(backbone=backbone, forward=mae_cp_forward, optim=optim_config,
    #                   projector=projector, decoder=decoder, mask_ratio=mask_ratio, patch_size=patch_size)

    decoder = MAEDecoder(
        embed_dim=embed_dim,
        decoder_embed_dim=decoder_dim,
        output_dim=output_dim,
        num_patches=num_tokens,
        depth=decoder_depth,
    )
    return spt.Module(
        backbone=backbone,
        forward=mae_cp_forward,
        optim=optim_config,
        decoder=decoder,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
    )


def main():
    parser = create_base_parser("MAE Continued Pretraining (ViT-only)")
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    args = parser.parse_args()

    data_dir, checkpoint_dir = setup_paths(args)
    ds_cfg, embed_dim, freeze_epochs, warmup_epochs = get_config(args)
    image_size = ds_cfg["input_size"]

    print(
        f"MAE CP: {args.dataset} | {args.backbone} | mask={args.mask_ratio} freeze={freeze_epochs} warmup={warmup_epochs}"
    )

    train_transform, val_transform = create_transforms(ds_cfg, n_views=1)
    data, test_loader, eval_train_loader, indices = create_data_loaders(
        args, ds_cfg, train_transform, val_transform, data_dir
    )

    backbone, device = load_backbone(args)

    # Dynamic num_tokens calculation
    with torch.no_grad():
        test_input = torch.zeros(
            1, 3, image_size, image_size, device=next(backbone.parameters()).device
        )

        # Get token sequence (not pooled features)
        if hasattr(backbone, "forward_features"):
            # timm models: use forward_features to get all tokens
            out = backbone.forward_features(test_input)
        else:
            # HuggingFace models: regular forward
            out = backbone(test_input)

        # Handle different output formats
        if hasattr(out, "last_hidden_state"):
            # HuggingFace models
            tokens = out.last_hidden_state
        elif isinstance(out, torch.Tensor):
            # timm models return tensor directly
            tokens = out
        else:
            raise ValueError(f"Unexpected backbone output type: {type(out)}")

        # Check token shape is valid
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected 3D token tensor [B, num_tokens, D], got shape {tokens.shape}. "
                f"Use forward_features() for timm models to get token sequence."
            )

        # Extract number of patch tokens (excluding CLS token)
        num_tokens = tokens.shape[1] - 1
    patch_size = image_size // int(num_tokens**0.5)
    print(f"Backbone: {num_tokens} tokens, patch_size={patch_size}")

    project = args.project or f"{args.dataset}-mae-cp"
    run_name = f"mae_n{args.n_samples}_ep{args.epochs}_frz{freeze_epochs}_blk{args.num_trained_blocks}_m{args.mask_ratio}"
    logger = WandbLogger(project=project, name=run_name, log_model=False)

    baseline_results = run_baseline(
        backbone, eval_train_loader, test_loader, device, args, logger
    )
    optim_config = create_optim_config(args, warmup_epochs)

    module = setup_mae_cp(
        backbone,
        embed_dim,
        optim_config,
        image_size=image_size,
        num_tokens=num_tokens,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
    )

    ckpt_path = str(
        checkpoint_dir / f"mae_cp_{args.dataset}_{args.backbone.replace('/', '_')}.ckpt"
    )
    run_training(
        module, data, args, ds_cfg, embed_dim, freeze_epochs, logger, ckpt_path
    )
    run_final_eval(
        backbone, eval_train_loader, test_loader, device, args, logger, baseline_results
    )

    logger.experiment.finish()
    print("Done!")


if __name__ == "__main__":
    main()
