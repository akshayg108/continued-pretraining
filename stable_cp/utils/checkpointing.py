from pathlib import Path
from typing import Union


def get_checkpoint_path(
    checkpoint_dir: Union[str, Path],
    method_name: str,
    dataset_name: str,
    backbone_name: str,
) -> str:
    """Generate a checkpoint path for a training run.

    Args:
        checkpoint_dir: Directory for checkpoints
        method_name: Name of the continued pretraining method
        dataset_name: Name of the dataset
        backbone_name: Name of the backbone model

    Returns:
        Checkpoint path as string
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Clean backbone name (replace / with _)
    clean_backbone = backbone_name.replace("/", "_")

    filename = f"{method_name}_{dataset_name}_{clean_backbone}.ckpt"
    return str(checkpoint_dir / filename)
