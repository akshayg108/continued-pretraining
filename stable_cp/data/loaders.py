# Data loading utilities for continued pretraining
import math

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.transforms import MultiViewTransform
from .datasets import get_dataset
from .heldout import exact_train_indices


class BalancedRepeatSampler(torch.utils.data.Sampler):
    """Repeat every sample evenly, then shuffle the padded epoch."""

    def __init__(self, data_source, num_samples, generator=None):
        self.n = len(data_source)
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        full_repeats = self.num_samples // self.n
        remainder = self.num_samples % self.n

        indices = list(range(self.n)) * full_repeats

        perm = torch.randperm(self.n, generator=self.generator).tolist()
        indices += perm[:remainder]

        shuffle_perm = torch.randperm(len(indices), generator=self.generator)
        indices = [indices[i] for i in shuffle_perm.tolist()]

        return iter(indices)

    def __len__(self):
        return self.num_samples


def create_transforms(ds_cfg, n_views=1, strong_aug=False):
    """Create augmented training views and clean evaluation inputs."""
    if strong_aug:
        # Strong augmentation for contrastive learning
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                (ds_cfg["input_size"], ds_cfg["input_size"]), scale=(0.2, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    else:
        # Standard augmentation
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((ds_cfg["input_size"], ds_cfg["input_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0), p=0.2),
            transforms.ToImage(**ds_cfg["normalization"]),
        )

    # Validation transform (no augmentation)
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((ds_cfg["input_size"], ds_cfg["input_size"])),
        transforms.ToImage(**ds_cfg["normalization"]),
    )

    # Multi-view support for contrastive learning
    train_transform = (
        MultiViewTransform({f"view_{i}": base_aug for i in range(n_views)})
        if n_views > 1
        else base_aug
    )

    return train_transform, val_transform


class CPSubset(torch.utils.data.Dataset):
    """Shared training subset with local IDs for instance classification."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if isinstance(sample, dict):
            sample["sample_idx"] = idx
        return sample


def _sample_shared_train_indices_by_class(args, dataset):
    """Share a stratified subset across training and frozen evaluation."""
    n_total = len(dataset)
    if args.n_samples > n_total:
        raise ValueError(f"--n-samples ({args.n_samples}) must be <= dataset size ({n_total})")
    if args.n_samples >= n_total:
        return list(range(n_total))

    all_indices = np.arange(n_total)

    all_labels = dataset.labels
    if getattr(dataset.hf_dataset, "heldout", False):
        return exact_train_indices(all_labels, args.n_samples, args.seed)

    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)

    def _one_per_class(reason):
        print(
            f"[warn] {reason}; falling back to 1-sample-per-class sampling "
            f"(requested n={args.n_samples}, returning {n_classes} samples)"
        )
        rng = np.random.RandomState(args.seed)
        selected = []
        for lbl in unique_labels:
            class_indices = all_indices[all_labels == lbl]
            selected.append(int(rng.choice(class_indices)))
        rng.shuffle(selected)
        return selected

    selected_indices, _ = train_test_split(
        all_indices,
        train_size=args.n_samples,
        stratify=all_labels,
        random_state=args.seed,
    )

    selected_labels = all_labels[selected_indices]
    if len(np.unique(selected_labels)) < n_classes:
        return _one_per_class(
            f"stratified split covered only "
            f"{len(np.unique(selected_labels))}/{n_classes} classes"
        )
    return selected_indices.tolist()


def create_eval_loaders(
    args,
    ds_cfg,
    eval_train_transform,
    val_transform,
    data_dir,
    indices=None,
    remap_sample_idx=True,
):
    """Return test and training-reference loaders plus their shared indices."""
    splits = ds_cfg.get("splits", ["train", "validation", "test"])
    train_split, _, test_split = splits

    eval_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=eval_train_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    test_data = get_dataset(
        args.dataset,
        split=test_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )

    if indices is None:
        indices = _sample_shared_train_indices_by_class(args, eval_train)

    eval_subset = (
        CPSubset(eval_train, indices)
        if remap_sample_idx
        else torch.utils.data.Subset(eval_train, indices)
    )
    eval_train_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    return test_loader, eval_train_loader, indices


def create_train_datamodule(
    args,
    ds_cfg,
    train_transform,
    val_transform,
    data_dir,
    indices=None,
    remap_sample_idx=True,
):
    """Build CP or FT loaders, padding epochs to complete accumulation steps."""
    splits = ds_cfg.get("splits", ["train", "validation", "test"])
    train_split, val_split, _ = splits

    full_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=train_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    val_data = get_dataset(
        args.dataset,
        split=val_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )

    if indices is None:
        indices = _sample_shared_train_indices_by_class(args, full_train)

    train_subset = (
        CPSubset(full_train, indices)
        if remap_sample_idx
        else torch.utils.data.Subset(full_train, indices)
    )
    accum = max(int(getattr(args, "accumulate_grad_batches", 1)), 1)
    effective_batch = args.batch_size * accum
    optim_steps_per_epoch = max(math.ceil(args.n_samples / effective_batch), 1)
    steps_per_epoch = optim_steps_per_epoch * accum  # forwards per epoch
    num_samples = steps_per_epoch * args.batch_size  # = optim_steps_per_epoch * effective_batch
    train_sampler = BalancedRepeatSampler(
        train_subset,
        num_samples=num_samples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    return data, indices
