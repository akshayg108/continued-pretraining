import torch
import torch.nn as nn
import torchmetrics
from typing import List, Optional, Union
import stable_pretraining as spt


class _FixedNumClassesKNN(spt.callbacks.OnlineKNN):
    """OnlineKNN subclass that guarantees predictions match the true num_classes.

    The upstream OnlineKNN infers num_classes from ``cached_labels.max() + 1``.
    When the KNN queue does not contain every class (common with small
    ``n_samples``), the prediction tensor is too narrow and torchmetrics raises
    a shape mismatch error.  This subclass pads the predictions to the known
    ``num_classes`` so the metrics always receive the correct shape.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self._fixed_num_classes = num_classes

    @torch.no_grad()
    def _compute_knn_predictions(self, features, cached_features, cached_labels):
        predictions = super()._compute_knn_predictions(
            features, cached_features, cached_labels
        )
        if predictions is not None and predictions.shape[1] < self._fixed_num_classes:
            padding = torch.zeros(
                predictions.shape[0],
                self._fixed_num_classes - predictions.shape[1],
                device=predictions.device,
                dtype=predictions.dtype,
            )
            predictions = torch.cat([predictions, padding], dim=1)
        return predictions


def create_cp_linear_probe(
    module: "spt.Module",
    num_classes: int,
    embedding_dim: int,
    name: str = "cp_linear_probe",
    input_key: str = "embedding",
    target_key: str = "label",
    include_f1: bool = True,
    include_auroc: bool = False,
    optimizer: Optional[dict] = None,
    scheduler: Optional[dict] = None,
) -> spt.callbacks.OnlineProbe:
    # Linear probe for continued pretraining evaluation
    metrics = {"top1": torchmetrics.classification.MulticlassAccuracy(num_classes)}
    if num_classes >= 5:
        metrics["top5"] = torchmetrics.classification.MulticlassAccuracy(
            num_classes, top_k=5
        )
    if include_f1:
        metrics["f1_macro"] = torchmetrics.classification.MulticlassF1Score(
            num_classes, average="macro"
        )
        metrics["f1_weighted"] = torchmetrics.classification.MulticlassF1Score(
            num_classes, average="weighted"
        )
    if include_auroc:
        metrics["auroc"] = torchmetrics.classification.MulticlassAUROC(num_classes)

    return spt.callbacks.OnlineProbe(
        module,
        name=name,
        input=input_key,
        target=target_key,
        probe=nn.Linear(embedding_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def create_cp_knn_probe(
    num_classes: int,
    embedding_dim: int,
    name: str = "cp_knn_probe",
    input_key: str = "embedding",
    target_key: str = "label",
    queue_length: int = 20000,
    k: int = 10,
    include_f1: bool = True,
) -> spt.callbacks.OnlineKNN:
    # KNN probe for continued pretraining evaluation
    metrics = {"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)}
    if include_f1:
        metrics["f1_macro"] = torchmetrics.classification.MulticlassF1Score(
            num_classes, average="macro"
        )

    return _FixedNumClassesKNN(
        num_classes=num_classes,
        name=name,
        input=input_key,
        target=target_key,
        queue_length=queue_length,
        metrics=metrics,
        input_dim=embedding_dim,
        k=k,
    )


def create_cp_rankme(
    embedding_dim: int,
    name: str = "cp_rankme",
    input_key: str = "embedding",
    queue_length: int = 1000,
) -> spt.callbacks.RankMe:
    # RankMe (effective rank) monitor for dimensional collapse detection
    return spt.callbacks.RankMe(
        name=name,
        target=input_key,
        queue_length=queue_length,
        target_shape=embedding_dim,
    )


def create_cp_evaluation_callbacks(
    module: "spt.Module",
    num_classes: int,
    embedding_dim: int,
    include_linear: bool = True,
    include_knn: bool = True,
    include_rankme: bool = True,
    include_f1: bool = True,
    include_auroc: bool = False,
    knn_queue_length: int = 20000,
    knn_k: int = 10,
    rankme_queue_length: int = 1000,
    input_key: str = "embedding",
    target_key: str = "label",
    linear_optimizer: Optional[dict] = None,
    linear_scheduler: Optional[dict] = None,
) -> List[
    Union[spt.callbacks.OnlineProbe, spt.callbacks.OnlineKNN, spt.callbacks.RankMe]
]:
    # Main factory for CP evaluation callbacks: linear probe, KNN, RankMe
    callbacks = []
    if include_linear:
        callbacks.append(
            create_cp_linear_probe(
                module=module,
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                name="cp_linear_probe",
                input_key=input_key,
                target_key=target_key,
                include_f1=include_f1,
                include_auroc=include_auroc,
                optimizer=linear_optimizer,
                scheduler=linear_scheduler,
            )
        )
    if include_knn:
        callbacks.append(
            create_cp_knn_probe(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                name="cp_knn_probe",
                input_key=input_key,
                target_key=target_key,
                queue_length=knn_queue_length,
                k=knn_k,
                include_f1=include_f1,
            )
        )
    if include_rankme:
        callbacks.append(
            create_cp_rankme(
                embedding_dim=embedding_dim,
                name="cp_rankme",
                input_key=input_key,
                queue_length=rankme_queue_length,
            )
        )
    return callbacks
