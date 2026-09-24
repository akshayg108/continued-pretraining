"""Linear probing with freshly transformed images and a frozen encoder."""

from contextlib import contextmanager
from numbers import Integral
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, Subset
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)
from tqdm import tqdm

from stable_cp.utils.backbone import forward_embedding
from stable_cp.utils.lp_protocol import lp_config


@contextmanager
def _frozen_encoder(model):
    parameters = [(param, param.requires_grad) for param in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        model.eval()
        for param, _ in parameters:
            param.requires_grad_(False)
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                if not torch.equal(buffer, original):
                    buffer.copy_(original)
        for param, requires_grad in parameters:
            param.requires_grad_(requires_grad)
        # Direct assignment preserves mixed modes without recursive train() calls.
        for module, training in modes:
            module.training = training


@contextmanager
def _preserve_rng(seed=None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _batch_images_labels(batch):
    if isinstance(batch, dict):
        images, labels = batch["image"], batch["label"]
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        images, labels = batch[:2]
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")
    return images, torch.as_tensor(labels).long().reshape(-1)


def _dataset_num_classes(dataset):
    if isinstance(dataset, Subset):
        return _dataset_num_classes(dataset.dataset)
    if getattr(dataset, "num_classes", None) is not None:
        return dataset.num_classes
    if getattr(dataset, "classes", None) is not None:
        return len(dataset.classes)
    for name in ("labels", "targets"):
        labels = getattr(dataset, name, None)
        if labels is not None:
            labels = torch.as_tensor(labels).reshape(-1)
            if not len(labels):
                raise ValueError("The LP training dataset is empty")
            if labels.min().item() < 0:
                raise ValueError("LP labels must be nonnegative class indices")
            return int(labels.max().item()) + 1
    raise ValueError("Pass num_classes or provide dataset num_classes, classes, labels, or targets")


def _fresh_loader(loader, seed):
    """Copy standard loader settings without reusing sampler or worker RNG state."""
    if type(loader) is not DataLoader:
        raise ValueError("Online LP requires standard DataLoader instances, not custom subclasses")
    sampler = loader.sampler
    if (
        type(loader.batch_sampler) is not BatchSampler
        or loader.batch_size is None
        or loader.batch_sampler.sampler is not sampler
        or loader.batch_sampler.batch_size != loader.batch_size
        or loader.batch_sampler.drop_last != loader.drop_last
    ):
        raise ValueError("Online LP does not support custom batch samplers")
    if type(sampler) is SequentialSampler:
        shuffle = False
    elif (
        type(sampler) is RandomSampler
        and not sampler.replacement
        and len(sampler) == len(loader.dataset)
    ):
        shuffle = True
    else:
        raise ValueError("Online LP requires a sequential or full nonreplacement random sampler")
    if not getattr(loader, "in_order", True):
        raise ValueError("Online LP requires in_order=True for reproducible worker output")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        original_generator = loader.generator or getattr(sampler, "generator", None)
        generator.set_state(
            original_generator.get_state()
            if original_generator is not None
            else torch.get_rng_state()
        )
    kwargs = {
        "batch_size": loader.batch_size,
        "shuffle": shuffle,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": generator,
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
        "pin_memory_device": loader.pin_memory_device,
    }
    if hasattr(loader, "in_order"):
        kwargs["in_order"] = loader.in_order
    return DataLoader(loader.dataset, **kwargs)


def _encode_batch(model, images, device, pool_strategy, forward_batch_size):
    features = []
    with torch.no_grad():
        for chunk in images.split(forward_batch_size):
            encoded = forward_embedding(model, chunk.to(device), pool_strategy)
            features.append(F.normalize(encoded.detach().float(), p=2, dim=1))
    features = torch.cat(features, dim=0)
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite encoder features during online LP")
    return features


def linear_probe_online_evaluate(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pool_strategy: str = "cls",
    epochs: int = 150,
    lr: float = 1e-3,
    forward_batch_size: int = 32,
    verbose: bool = True,
    num_classes: int = None,
    seed: int = None,
) -> dict:
    """Fit a linear head on fresh views each epoch, leaving the encoder unchanged.

    The caller supplies a shuffled, non-dropping weak-augmentation train loader
    and a deterministic test loader. Fresh standard loaders reset sampling and
    worker randomness per evaluation without advancing caller-owned generators.
    Only sequential or full nonreplacement random samplers are supported.
    Encoder image forwards are chunked before device transfer; each complete
    loader batch makes one Adam update. Exact AUROC retains CPU probabilities,
    but test images and features are streamed.
    """
    if train_loader.drop_last:
        raise ValueError("The LP training loader must use drop_last=False")
    config = lp_config(epochs, train_loader.batch_size, lr, forward_batch_size)
    if num_classes is None:
        num_classes = _dataset_num_classes(train_loader.dataset)
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 2:
        raise ValueError("LP num_classes must be an integer of at least two")
    num_classes = int(num_classes)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise ValueError("LP seed must be an integer or None")
    seed = int(seed) if seed is not None else None

    with _preserve_rng(seed), _frozen_encoder(model):
        fresh_train_loader = _fresh_loader(train_loader, seed)
        fresh_test_loader = _fresh_loader(test_loader, seed)
        classifier = None
        optimizer = None
        criterion = nn.CrossEntropyLoss()
        epoch_iterator = tqdm(range(epochs), desc="Online LP epochs") if verbose else range(epochs)
        for _ in epoch_iterator:
            samples = 0
            for batch in fresh_train_loader:
                images, labels = _batch_images_labels(batch)
                features = _encode_batch(model, images, device, pool_strategy, forward_batch_size)
                if classifier is None:
                    classifier = nn.Linear(features.shape[1], num_classes).to(device)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
                    classifier.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(classifier(features), labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite classifier loss during online LP")
                loss.backward()
                optimizer.step()
                samples += len(labels)
            if not samples:
                raise ValueError("The LP training loader is empty")

        classifier.eval()
        accuracy = MulticlassAccuracy(num_classes=num_classes)
        f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        auroc_valid = True
        samples = 0
        test_iterator = (
            tqdm(fresh_test_loader, desc="Online LP test") if verbose else fresh_test_loader
        )
        with torch.no_grad():
            for batch in test_iterator:
                images, labels = _batch_images_labels(batch)
                features = _encode_batch(model, images, device, pool_strategy, forward_batch_size)
                logits = classifier(features)
                predictions = logits.argmax(dim=1).cpu()
                labels = labels.cpu()
                accuracy.update(predictions, labels)
                f1.update(predictions, labels)
                if auroc_valid:
                    try:
                        auroc.update(torch.softmax(logits, dim=1).cpu(), labels)
                    except ValueError:
                        auroc_valid = False
                samples += len(labels)
        if not samples:
            raise ValueError("The LP test loader is empty")
        try:
            auroc_value = auroc.compute().item() if auroc_valid else 0.0
        except ValueError:
            auroc_value = 0.0
        return {
            "linear_pytorch_acc": accuracy.compute().item(),
            "linear_pytorch_f1": f1.compute().item(),
            "linear_pytorch_auroc": auroc_value,
            "lp": config,
        }
