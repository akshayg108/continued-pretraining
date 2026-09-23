"""Reuse shared downloads and Arrow caches, then stage data on node-local storage."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_dataset(cache_dir: Path, name: str) -> list[Path]:
    """Reuse completed source splits; download or build missing shared caches."""
    from stable_datasets import images
    from stable_cp.data.datasets import get_dataset_config

    cache_dir = Path(cache_dir).expanduser().resolve()
    cfg = get_dataset_config(name)
    cls = cfg["dataset_class"]
    if isinstance(cls, str):
        cls = getattr(images, cls)
    kwargs = dict(cfg.get("dataset_kwargs", {}))
    if cfg.get("config_name") is not None:
        kwargs["config_name"] = cfg["config_name"]
    print(f"PREPARE {name} shared={cache_dir}", flush=True)
    datasets = cls(
        split=None,
        download_dir=str(cache_dir / "stable_datasets" / "downloads"),
        processed_cache_dir=str(cache_dir / "stable_datasets" / "processed"),
        **kwargs,
    )
    directories = sorted({ds._shard_dir for ds in datasets.values()})
    print(f"PREPARED {name} splits={len(datasets)}", flush=True)
    return directories


@contextmanager
def _local_directory(shared):
    configured = os.environ.get("CP_NODE_TMPDIR") or os.environ.get("SLURM_TMPDIR")
    if configured:
        parent = Path(configured).expanduser().resolve()
    else:
        parent = Path("/tmpdata")
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            parent = Path("/tmp")
    parent = parent.resolve()
    if parent == shared or parent.is_relative_to(shared):
        raise ValueError(f"Node-local scratch must be outside shared data: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        with tempfile.TemporaryDirectory(prefix=f"cp-{os.environ.get('SLURM_JOB_ID', 'local')}-", dir=parent) as path:
            yield Path(path)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _check_capacity(local, size):
    free = shutil.disk_usage(local).free
    if free < size + 1024 ** 3:
        raise OSError(f"Insufficient node-local storage at {local}: need {size + 1024 ** 3}, free {free}")


def _copy_directory(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--", f"{source}/", f"{destination}/"], check=True)


def _directory_size(source):
    return sum(path.stat().st_size for path in source.rglob("*") if path.is_file())


@contextmanager
def staged_dataset(cache_dir: Path, name: str):
    """Prepare shared caches, then copy this dataset's Arrow splits locally."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    directories = prepare_dataset(cache_dir, name)
    with _local_directory(cache_dir) as local:
        size = sum(_directory_size(directory) for directory in directories)
        _check_capacity(local, size)
        print(f"STAGE {name} bytes={size} source={cache_dir} local={local}", flush=True)
        for directory in directories:
            _copy_directory(directory, local / directory.relative_to(cache_dir))
        (local / "stable_datasets" / "downloads").symlink_to(cache_dir / "stable_datasets" / "downloads")
        print(f"STAGED {name} local={local}", flush=True)
        yield local


@contextmanager
def staged_directory(source: Path):
    """Stage a reference-image directory for repeated encoder reads."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    with _local_directory(source) as local:
        size = _directory_size(source)
        _check_capacity(local, size)
        print(f"STAGE reference bytes={size} source={source} local={local}", flush=True)
        _copy_directory(source, local)
        print(f"STAGED reference local={local}", flush=True)
        yield local


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def main():
    if sys.argv[1:2] == ["run"] and any(arg in {"-h", "--help"} for arg in sys.argv[2:]):
        subprocess.run([sys.executable, str(ROOT / "continued_pretraining.py"), "--help"], check=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--cache-dir", default="~/.cache", type=Path)
    args, remaining = parser.parse_known_args()
    if args.command == "prepare":
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        prepare_dataset(args.cache_dir, args.dataset)
    else:
        with staged_dataset(args.cache_dir, args.dataset) as local:
            subprocess.run([
                sys.executable, "-u", str(ROOT / "continued_pretraining.py"),
                *remaining, "--dataset", args.dataset, "--cache-dir", str(local),
            ], check=True)


if __name__ == "__main__":
    main()
