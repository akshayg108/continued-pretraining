from lightning.pytorch.loggers import WandbLogger
from typing import Optional, Dict, Any


def create_logger(
    project: Optional[str],
    run_name: Optional[str],
    log_model: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> WandbLogger:
    """Create a WandB logger for experiment tracking.

    Args:
        project: WandB project name
        run_name: Run name for this experiment
        log_model: Whether to log model checkpoints to WandB
        config: Additional configuration to log

    Returns:
        Configured WandB logger
    """
    logger = WandbLogger(
        project=project,
        name=run_name,
        log_model=log_model,
        config=config,
    )
    return logger


def log_metrics(
    logger: WandbLogger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
):
    """Log metrics to WandB.

    Args:
        logger: WandB logger instance
        metrics: Dictionary of metric name -> value
        step: Training step (None for auto-increment)
        prefix: Prefix to add to metric names
    """
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if step is not None:
        logger.experiment.log(metrics, step=step)
    else:
        logger.experiment.log(metrics)


def log_summary(logger: WandbLogger, metrics: Dict[str, Any], prefix: str = ""):
    """Log summary metrics that don't change over time.

    Args:
        logger: WandB logger instance
        metrics: Dictionary of metric name -> value
        prefix: Prefix to add to metric names
    """
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    for k, v in metrics.items():
        logger.experiment.summary[k] = v
