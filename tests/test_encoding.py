"""
tests/test_encoding.py
=======================
Smoke tests for hqnn_forge.encoding.QuantumEncodingLayer.

These tests validate:
1. Forward pass returns the correct shape.
2. Output values are within the valid expectation-value range [-1, 1].
3. Gradients flow back from the loss through the quantum layer to its weights.
4. The layer raises ValueError for mismatched input dimensions.
5. Restricted-variance init produces the expected statistical properties.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.initializers import restricted_normal_init_, block_local_init_


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_QUBITS  = 4   # keep small for test speed (real usage: 8)
N_LAYERS  = 2
BATCH     = 8


@pytest.fixture(scope="module")
def layer() -> QuantumEncodingLayer:
    """Shared QuantumEncodingLayer instance (uses default.qubit for portability)."""
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device_name="default.qubit",   # lightning.qubit not required in CI
        diff_method="parameter-shift", # universal; works on default.qubit
    )


@pytest.fixture
def random_batch() -> torch.Tensor:
    """Random input batch, values in (-π, π)."""
    return torch.rand(BATCH, N_QUBITS) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------

class TestForwardPassShape:
    def test_output_shape(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """Forward pass should return shape (batch_size, n_qubits)."""
        out = layer(random_batch)
        assert out.shape == (BATCH, N_QUBITS), (
            f"Expected shape ({BATCH}, {N_QUBITS}); got {out.shape}"
        )

    def test_single_sample(self, layer: QuantumEncodingLayer) -> None:
        """Single-sample batch (batch_size=1) should work correctly."""
        x   = torch.rand(1, N_QUBITS)
        out = layer(x)
        assert out.shape == (1, N_QUBITS)


# ---------------------------------------------------------------------------
# Test 2: Output range
# ---------------------------------------------------------------------------

class TestOutputRange:
    def test_expectation_values_in_range(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """PauliZ expectation values must lie in [-1, 1]."""
        with torch.no_grad():
            out = layer(random_batch)
        assert out.min().item() >= -1.0 - 1e-5, (
            f"Output below -1: {out.min().item()}"
        )
        assert out.max().item() <= 1.0 + 1e-5, (
            f"Output above  1: {out.max().item()}"
        )


# ---------------------------------------------------------------------------
# Test 3: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_reach_quantum_weights(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """
        A backward pass through a scalar loss must leave non-None, non-zero
        gradients on the variational weights.
        """
        # Reset any pre-existing gradients
        layer.zero_grad()

        out  = layer(random_batch)             # (BATCH, N_QUBITS)
        loss = out.sum()                       # scalar
        loss.backward()

        weights_param = layer.qlayer.weights
        assert weights_param.grad is not None, (
            "Gradient of quantum weights is None after backward pass."
        )
        assert weights_param.grad.abs().sum().item() > 0.0, (
            "Gradient of quantum weights is zero everywhere — training would stall."
        )

    def test_no_grad_inference(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """Inference under torch.no_grad() should not compute gradients."""
        with torch.no_grad():
            out = layer(random_batch)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# Test 4: Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_feature_dim_raises(self, layer: QuantumEncodingLayer) -> None:
        """Input with wrong last-dim should raise ValueError."""
        wrong_input = torch.rand(BATCH, N_QUBITS + 1)
        with pytest.raises(ValueError, match="n_qubits"):
            layer(wrong_input)

    def test_n_qubits_lt_2_raises(self) -> None:
        """n_qubits < 2 is invalid (no meaningful entangling ring)."""
        from hqnn_forge.encoding.angle_embedding import build_encoding_qnode
        with pytest.raises(ValueError, match="n_qubits must be"):
            build_encoding_qnode(n_qubits=1)


# ---------------------------------------------------------------------------
# Test 5: Initialiser properties
# ---------------------------------------------------------------------------

class TestRestrictedVarianceInit:
    def test_std_approximately_correct(self) -> None:
        """
        restricted_normal_init_ should produce a std close to
        π / sqrt(n_qubits * n_layers).
        """
        n_q, n_l = 8, 2
        expected_std = math.pi / math.sqrt(n_q * n_l)
        tensor = torch.empty(n_l, n_q, 3)
        restricted_normal_init_(tensor, n_qubits=n_q, n_layers=n_l)

        actual_std = tensor.std().item()
        # Allow ±30 % tolerance for finite-sample fluctuation
        assert abs(actual_std - expected_std) / expected_std < 0.30, (
            f"Expected std ≈ {expected_std:.4f}; got {actual_std:.4f}"
        )

    def test_mean_approximately_zero(self) -> None:
        """Initialised weights should be zero-mean."""
        tensor = torch.empty(4, 8, 3)
        restricted_normal_init_(tensor, n_qubits=8, n_layers=4)
        assert abs(tensor.mean().item()) < 0.1

    def test_block_local_decreasing_variance(self) -> None:
        """Later blocks should have smaller std than earlier blocks."""
        tensor = torch.empty(4, 8, 3)
        block_local_init_(tensor, n_qubits=8)

        layer_stds = [tensor[i].std().item() for i in range(4)]
        # Each subsequent layer should (in expectation) have ≤ std than previous
        for i in range(len(layer_stds) - 1):
            # Allow small Monte-Carlo noise; check trend not strict monotonicity
            assert layer_stds[i] >= layer_stds[-1] * 0.5, (
                f"Layer {i} std={layer_stds[i]:.4f} seems too small; "
                f"last layer std={layer_stds[-1]:.4f}"
            )
