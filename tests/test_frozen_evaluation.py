"""Regression checks for the frozen kNN and linear-probe protocol."""

import importlib
import subprocess
import sys

import numpy as np
import pytest
import torch
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def evaluation():
    return importlib.import_module("stable_cp.evaluation.zero_shot_eval")


def test_frozen_evaluation_imports_without_training_framework():
    code = "import sys; sys.modules['stable_pretraining'] = None; " "import stable_cp.evaluation"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class TokenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.fc_norm = nn.Identity()
        self.observed_modes = []

    def forward_features(self, images):
        self.observed_modes.append((self.training, torch.is_grad_enabled()))
        return images * self.scale

    def attn_pool(self, tokens):
        return tokens.mean(dim=1) + 1


@pytest.mark.parametrize("pool", ["cls", "mean", "map"])
@pytest.mark.parametrize("dict_batches", [False, True])
def test_extract_features_preserves_pooling_and_backbone(evaluation, pool, dict_batches):
    images = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    labels = torch.tensor([0, 1, 0, 1])
    if dict_batches:
        dataset = [{"image": image, "label": label} for image, label in zip(images, labels)]
    else:
        dataset = TensorDataset(images, labels)
    model = TokenBackbone()
    before = model.scale.detach().clone()
    features, actual_labels = evaluation.extract_features(
        model, DataLoader(dataset, batch_size=2), "cpu", pool, verbose=False
    )
    expected = {
        "cls": images[:, 0] * 2,
        "mean": images[:, 1:].mean(dim=1) * 2,
        "map": images.mean(dim=1) * 2 + 1,
    }[pool]
    np.testing.assert_allclose(features, expected.numpy())
    np.testing.assert_array_equal(actual_labels, labels.numpy())
    assert model.observed_modes == [(False, False), (False, False)]
    assert model.scale.requires_grad and model.scale.grad is None
    torch.testing.assert_close(model.scale, before)


def test_knn_uses_distance_weights_and_caps_k(evaluation):
    train = np.array([[10.0, 0.0], [0.0, 3.0], [-0.1, 1.0]])
    labels = np.array([0, 1, 1])
    test = np.array([[4.0, 0.04], [0.04, 4.0]])
    results = evaluation.knn_evaluate(train, labels, test, np.array([0, 1]), k=20)
    assert results["knn_f1"] == pytest.approx(1.0)
    assert results["knn_acc"] == pytest.approx(1.0)
    assert results["knn_auroc"] == pytest.approx(1.0)


def test_knn_retains_macro_accuracy_and_f1(evaluation):
    train = np.eye(2)
    test = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    results = evaluation.knn_evaluate(train, np.array([0, 1]), test, np.array([0, 0, 0, 1]))
    assert results["knn_acc"] == pytest.approx((2 / 3 + 1) / 2)
    assert results["knn_f1"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_pytorch_probe_retains_adam_schedule_and_shuffle(evaluation, monkeypatch):
    features = np.array(
        [[2.0, 0.0], [3.0, 1.0], [0.0, 4.0], [1.0, 5.0], [2.0, 1.0], [1.0, 3.0], [4.0, 1.0]]
    )
    labels = np.array([0, 0, 1, 1, 0, 1, 0])
    optimizers = []
    adam = torch.optim.Adam

    class RecordingAdam(adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = 0
            optimizers.append(self)

        def step(self, *args, **kwargs):
            self.steps += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    torch.manual_seed(17)
    results = evaluation.linear_probe_pytorch_evaluate(
        features,
        labels,
        features,
        labels,
        device="cpu",
        lr=0.02,
        min_epochs=3,
        min_steps=7,
        batch_size=3,
        verbose=False,
    )
    actual = optimizers[0]
    assert actual.steps == 9
    assert set(results) == {"linear_pytorch_acc", "linear_pytorch_f1", "linear_pytorch_auroc"}

    torch.manual_seed(17)
    inputs = torch.from_numpy(normalize(features)).float()
    targets = torch.from_numpy(labels).long()
    classifier = nn.Linear(2, 2)
    expected = adam(classifier.parameters(), lr=0.02)
    indices = torch.randperm(7)
    for step in range(9):
        if step % 3 == 0:
            indices = torch.randperm(7)
        batch = indices[(step % 3) * 3 : min((step % 3 + 1) * 3, 7)]
        expected.zero_grad()
        nn.functional.cross_entropy(classifier(inputs[batch]), targets[batch]).backward()
        expected.step()
    for parameter, reference in zip(actual.param_groups[0]["params"], classifier.parameters()):
        torch.testing.assert_close(parameter, reference)


def test_pipeline_uses_clean_knn_bank_and_only_reported_probe(evaluation, monkeypatch):
    model = TokenBackbone()
    labels = torch.tensor([0, 1, 0, 1])
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])[:, None, :]
    augmented = clean + 0.25
    train_loader = DataLoader(TensorDataset(augmented, labels), batch_size=2)
    clean_loader = DataLoader(TensorDataset(clean, labels), batch_size=2)
    calls = []

    def probe(train_features, train_labels, test_features, test_labels, **kwargs):
        calls.append(kwargs)
        np.testing.assert_allclose(train_features, augmented[:, 0].numpy() * 2)
        np.testing.assert_allclose(test_features, clean[:, 0].numpy() * 2)
        return {"linear_pytorch_acc": 1.0, "linear_pytorch_f1": 1.0, "linear_pytorch_auroc": 1.0}

    monkeypatch.setattr(evaluation, "linear_probe_pytorch_evaluate", probe)
    results = evaluation.zero_shot_eval(
        model,
        train_loader,
        clean_loader,
        "cpu",
        knn_train_loader=clean_loader,
        linear_pytorch_min_steps=13,
        linear_pytorch_lr=0.002,
        verbose=False,
    )
    assert set(results) == {
        "knn_acc",
        "knn_f1",
        "knn_auroc",
        "linear_pytorch_acc",
        "linear_pytorch_f1",
        "linear_pytorch_auroc",
    }
    assert results["knn_f1"] == pytest.approx(1.0)
    assert calls == [{"device": "cpu", "lr": 0.002, "min_steps": 13, "verbose": False}]
