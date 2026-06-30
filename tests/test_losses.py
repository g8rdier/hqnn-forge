"""
tests/test_losses.py
=====================
Unit tests for hqnn_forge.utils.imbalance.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hqnn_forge.utils.imbalance import FocalLoss, compute_class_weights, weighted_bce_loss

@pytest.fixture
def balanced_labels() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

@pytest.fixture
def imbalanced_labels() -> torch.Tensor:
    return torch.tensor([0.0] * 95 + [1.0] * 5)

@pytest.fixture
def sample_logits() -> torch.Tensor:
    return torch.tensor([0.8, -0.3, 1.2, -1.5, 0.1, -0.5, 0.9, -0.8])

@pytest.fixture
def sample_targets() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


class TestFocalLoss:
    def test_gamma_zero_matches_bce(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor) -> None:
        focal_val = FocalLoss(alpha=0.5, gamma=0.0, reduction="mean")(sample_logits, sample_targets)
        bce_val = F.binary_cross_entropy_with_logits(sample_logits, sample_targets, reduction="mean")
        torch.testing.assert_close(focal_val, 0.5 * bce_val, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_to_logits(self) -> None:
        logits = torch.randn(16, requires_grad=True)
        targets = (torch.rand(16) > 0.5).float()
        loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            FocalLoss(alpha=0.0)
        with pytest.raises(ValueError, match="gamma"):
            FocalLoss(gamma=-1.0)


class TestClassWeights:
    def test_balanced_weights_approximately_equal(self, balanced_labels: torch.Tensor) -> None:
        weights = compute_class_weights(balanced_labels)
        assert abs(weights[0].item() - weights[1].item()) < 0.3

    def test_extreme_ratio(self, imbalanced_labels: torch.Tensor) -> None:
        weights = compute_class_weights(imbalanced_labels)
        assert (weights[1].item() / weights[0].item()) > 5.0
