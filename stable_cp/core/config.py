from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for dataset and data loading.

    Attributes:
        dataset: Dataset name (e.g., 'cifar10', 'fgvc-aircraft')
        n_samples: Number of training samples to use
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        cache_dir: Directory for caching datasets
    """

    dataset: str
    n_samples: int = 1000
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    cache_dir: str = "~/.cache"


@dataclass
class BackboneConfig:
    """Configuration for backbone network.

    Attributes:
        name: Backbone model name (e.g., 'vit_small_patch14_dinov2.lvd142m',
              'vit_large_patch16_dinov3.lvd1689m')
        pretrained: Whether to load pretrained weights
        freeze_epochs: Number of epochs to freeze backbone
        num_trained_blocks: Number of blocks to train after unfreezing
                           (-1 for all blocks, 0 for none)
    """

    name: str
    pretrained: bool = True
    freeze_epochs: Optional[int] = None
    num_trained_blocks: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training loop.

    Attributes:
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs (None for auto: 10% of epochs)
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        gradient_clip_val: Gradient clipping value (None for no clipping)
        accumulate_grad_batches: Number of batches for gradient accumulation
    """

    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: Optional[int] = None
    precision: str = "16-mixed"
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocols.

    Attributes:
        skip_baseline: Skip baseline evaluation before training
        skip_final_eval: Skip final evaluation after training
        knn_k: Number of neighbors for k-NN evaluation
        include_f1: Include F1 score in metrics
        include_auroc: Include AUROC in metrics
        linear_probe_method: Linear probe method ('sklearn', 'torch', 'both')
    """

    skip_baseline: bool = False
    skip_final_eval: bool = False
    knn_k: int = 20
    include_f1: bool = True
    include_auroc: bool = True
    linear_probe_method: str = "both"


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing.

    Attributes:
        project: WandB project name (None for auto-generated)
        run_name: WandB run name (None for auto-generated)
        checkpoint_dir: Directory for saving checkpoints
        log_model: Whether to log model to WandB
        log_every_n_steps: Logging frequency in steps
    """

    project: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    log_model: bool = False
    log_every_n_steps: int = 50


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration combining all sub-configs.

    This is the top-level configuration that combines all aspects
    of a continued pretraining experiment.
    """

    data: DataConfig
    backbone: BackboneConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    method_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args):
        """Create configuration from parsed command-line arguments.

        Args:
            args: Parsed arguments from argparse

        Returns:
            BenchmarkConfig instance
        """
        data_config = DataConfig(
            dataset=args.dataset,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

        backbone_config = BackboneConfig(
            name=args.backbone,
            freeze_epochs=args.freeze_epochs,
            num_trained_blocks=args.num_trained_blocks,
        )

        training_config = TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
        )

        evaluation_config = EvaluationConfig(
            skip_baseline=args.skip_baseline,
            skip_final_eval=args.skip_final_eval,
            knn_k=args.knn_k,
        )

        logging_config = LoggingConfig(
            project=args.project,
            checkpoint_dir=args.checkpoint_dir,
        )

        return cls(
            data=data_config,
            backbone=backbone_config,
            training=training_config,
            evaluation=evaluation_config,
            logging=logging_config,
        )
