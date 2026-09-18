"""
tests/test_encoding.py
=======================
Smoke tests for hqnn_forge.encoding.QuantumEncodingLayer.

These tests validate:
1. Forward pass returns the correct shape.
2. Output values are within the valid expectation-value range [-1, 1].
3. Gradients flow back from the loss through the quantum layer to its weights.
4. The layer raises ValueError for mismatched input dimensions.
5. Restricted-variance and block-local init match their documented sigma laws,
   with assertions calibrated to reject a flat-sigma regression.

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

# Statistical assertions on initialiser output need enough draws to separate
# the two strategies: per-layer sample count is n_qubits * 3.  At 16 qubits and
# 16 layers every tolerance below clears the worst deviation observed over 5000
# seeds, so the tests hold for any seed rather than relying on INIT_SEED alone:
#
#   restricted std, relative error   worst 0.091  tolerance 0.20  (2.2x)
#   log-std slope, either strategy   worst 0.160  tolerance 0.25  (1.56x)
#   block-local intercept, log space worst 0.332  tolerance 0.5   (1.5x)
#
# The slope tolerance is two-sided.  For the same seed, block_local_init_ is
# restricted_normal_init_ rescaled per layer, so its slope is exactly the flat
# slope minus 0.5: the decay test and its flat-rejection power check measure
# one statistic, from -0.5 and from 0 respectively.  SLOPE_TOLERANCE must stay
# inside (0.160, 0.340); 0.25 leaves 1.56x on the decay side and 1.36x on the
# power side.  Tightening or loosening it moves both tests at once.
INIT_N_QUBITS = 16
INIT_N_LAYERS = 16
INIT_SEED = 0
STD_TOLERANCE = 0.20      # relative
SLOPE_TOLERANCE = 0.25    # in log-log space
INTERCEPT_TOLERANCE = 0.5 # in log space
# The first/last std ratio compares two single layers, so it needs far more
# than 48 draws per layer: at 16 qubits rel=0.3 fails for ~4.5% of seeds.  At
# 256 qubits (768 draws) the worst relative deviation over 5000 seeds is 0.136.
# The ratio sqrt(L) does not depend on n_qubits.
RATIO_N_QUBITS = 256
RATIO_TOLERANCE = 0.3     # relative


def _log_std_fit(tensor: torch.Tensor) -> tuple[float, float]:
    """
    Least-squares fit of log(per-layer std) against log(layer_index + 1).

    Returns ``(slope, intercept)``.  A constant sigma across layers gives slope
    0; ``sigma_l = c / sqrt(l + 1)`` gives slope -0.5 and intercept log(c).
    """
    w = tensor.double()
    y = torch.log(w.flatten(start_dim=1).std(dim=1))
    x = torch.log(torch.arange(1, w.shape[0] + 1, dtype=torch.float64))
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    return slope.item(), (y_mean - slope * x_mean).item()


def _restricted() -> torch.Tensor:
    torch.manual_seed(INIT_SEED)
    tensor = torch.empty(INIT_N_LAYERS, INIT_N_QUBITS, 3)
    return restricted_normal_init_(tensor, n_qubits=INIT_N_QUBITS, n_layers=INIT_N_LAYERS)


def _block_local() -> torch.Tensor:
    torch.manual_seed(INIT_SEED)
    tensor = torch.empty(INIT_N_LAYERS, INIT_N_QUBITS, 3)
    return block_local_init_(tensor, n_qubits=INIT_N_QUBITS)


class TestRestrictedVarianceInit:
    def test_std_matches_documented_sigma(self) -> None:
        """sigma = pi / sqrt(n_qubits * n_layers), per the initialiser docstring."""
        expected_std = math.pi / math.sqrt(INIT_N_QUBITS * INIT_N_LAYERS)
        actual_std = _restricted().std().item()
        assert actual_std == pytest.approx(expected_std, rel=STD_TOLERANCE)

    def test_mean_approximately_zero(self) -> None:
        """Initialised weights should be zero-mean."""
        assert abs(_restricted().mean().item()) < 0.1

    def test_variance_is_flat_across_layers(self) -> None:
        """One shared sigma: log-std has zero slope in depth."""
        slope, _ = _log_std_fit(_restricted())
        assert abs(slope) < SLOPE_TOLERANCE

    def test_returns_same_tensor(self) -> None:
        tensor = torch.empty(2, 4, 3)
        assert restricted_normal_init_(tensor, n_qubits=4, n_layers=2) is tensor


class TestBlockLocalInit:
    def test_variance_decays_with_depth(self) -> None:
        """
        sigma_l = pi / sqrt(n_qubits * (l + 1)), i.e. log(sigma_l) =
        log(pi / sqrt(n_qubits)) - 0.5 * log(l + 1): slope -0.5, intercept
        log(pi / sqrt(n_qubits)).
        """
        slope, intercept = _log_std_fit(_block_local())
        assert abs(slope + 0.5) < SLOPE_TOLERANCE
        assert abs(intercept - math.log(math.pi / math.sqrt(INIT_N_QUBITS))) < INTERCEPT_TOLERANCE

    def test_decay_check_rejects_a_flat_tensor(self) -> None:
        """
        Power check for the assertion above: a constant-sigma tensor (what
        ``block_local_init_`` would produce if it silently degraded to
        ``restricted_normal_init_``) must *fail* the slope criterion.
        """
        slope, _ = _log_std_fit(_restricted())
        assert abs(slope + 0.5) > SLOPE_TOLERANCE

    def test_first_layer_wider_than_last(self) -> None:
        """The documented ratio sigma_0 / sigma_{L-1} = sqrt(L)."""
        torch.manual_seed(INIT_SEED)
        tensor = torch.empty(INIT_N_LAYERS, RATIO_N_QUBITS, 3)
        block_local_init_(tensor, n_qubits=RATIO_N_QUBITS)
        ratio = tensor[0].std().item() / tensor[-1].std().item()
        assert ratio == pytest.approx(math.sqrt(INIT_N_LAYERS), rel=RATIO_TOLERANCE)

    def test_returns_same_tensor(self) -> None:
        tensor = torch.empty(2, 4, 3)
        assert block_local_init_(tensor, n_qubits=4) is tensor
