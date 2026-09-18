"""
tests/test_amplitude_embedding.py
=================================
Unit tests for hqnn_forge.encoding.amplitude_embedding.AmplitudeEncodingLayer.

Beyond shape and gradient smoke tests, the numerical checks pin the encoding
itself: with the variational weights at zero the ansatz is a fixed permutation
of the computational basis (the CNOT ring), so the ⟨Z_i⟩ of a basis-state
input can be computed by hand, and the L2 normalisation makes the output
invariant to the scale of the input.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, build_amplitude_qnode

N_QUBITS = 3
N_AMPLITUDES = 2**N_QUBITS
N_LAYERS = 1
BATCH = 5


def _layer(diff_method: str = "backprop", **kwargs: object) -> AmplitudeEncodingLayer:
    torch.manual_seed(0)
    return AmplitudeEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device_name="default.qubit",
        diff_method=diff_method,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Shape and range
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_full_width_input(self) -> None:
        """2**n_qubits features in, n_qubits expectation values out."""
        layer = _layer()
        out = layer(torch.randn(BATCH, N_AMPLITUDES))
        assert out.shape == (BATCH, N_QUBITS)

    def test_padded_input(self) -> None:
        """n_features < 2**n_qubits is zero-padded, output shape unchanged."""
        layer = _layer(n_features=5)
        out = layer(torch.randn(BATCH, 5))
        assert out.shape == (BATCH, N_QUBITS)

    def test_single_sample(self) -> None:
        layer = _layer()
        assert layer(torch.randn(1, N_AMPLITUDES)).shape == (1, N_QUBITS)

    def test_expectation_values_in_range(self) -> None:
        layer = _layer()
        with torch.no_grad():
            out = layer(torch.randn(BATCH, N_AMPLITUDES))
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_extra_repr_lists_n_features(self) -> None:
        assert "n_features=5" in _layer(n_features=5).extra_repr()


# ---------------------------------------------------------------------------
# Numerical correctness of the encoding
# ---------------------------------------------------------------------------


def _cnot_ring_permutation(bits: list[int]) -> list[int]:
    """Apply CNOT(i → i+1 mod n) for i = 0 … n-1 to a bit string, wire 0 first."""
    bits = list(bits)
    n = len(bits)
    for i in range(n):
        bits[(i + 1) % n] ^= bits[i]
    return bits


def _basis_index_to_bits(k: int, n: int) -> list[int]:
    """PennyLane orders amplitudes with wire 0 as the most significant bit."""
    return [(k >> (n - 1 - i)) & 1 for i in range(n)]


class TestEncodingMatchesHandComputation:
    def test_basis_states_with_zero_weights(self) -> None:
        """
        With all Rot angles at zero, one layer is exactly the CNOT ring, which
        maps basis state |k⟩ to basis state |π(k)⟩.  ⟨Z_i⟩ is then
        1 - 2·bit_i(π(k)), computed here without PennyLane.
        """
        layer = _layer()
        with torch.no_grad():
            layer.qlayer.weights.zero_()
        x = torch.eye(N_AMPLITUDES)  # every basis state, one per row
        with torch.no_grad():
            got = layer(x)
        expected = torch.tensor(
            [
                [1.0 - 2.0 * b for b in _cnot_ring_permutation(_basis_index_to_bits(k, N_QUBITS))]
                for k in range(N_AMPLITUDES)
            ]
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)

    def test_superposition_with_zero_weights(self) -> None:
        """
        (|000⟩ + |001⟩)/√2 → CNOT ring → (|000⟩ + |101⟩)/√2 for 3 qubits
        (|001⟩: CNOT(0,1) and CNOT(1,2) leave it, CNOT(2,0) flips wire 0), so
        ⟨Z_0⟩ = ⟨Z_2⟩ = 0 and ⟨Z_1⟩ = 1.  Also checks that un-normalised input
        (norm √2 here, and 10·√2 in the second row) is normalised.
        """
        layer = _layer()
        with torch.no_grad():
            layer.qlayer.weights.zero_()
        x = torch.zeros(2, N_AMPLITUDES)
        x[0, 0] = x[0, 1] = 1.0
        x[1, 0] = x[1, 1] = 10.0
        with torch.no_grad():
            got = layer(x)
        torch.testing.assert_close(got, torch.tensor([[0.0, 1.0, 0.0]] * 2), rtol=0, atol=1e-6)

    def test_padding_is_on_the_right(self) -> None:
        """
        n_features=1 with the single feature non-zero must prepare |0…0⟩,
        whichever value the feature has: the remaining amplitudes are zero.
        """
        layer = _layer(n_features=1)
        with torch.no_grad():
            layer.qlayer.weights.zero_()
            got = layer(torch.tensor([[1.0], [-3.5]]))
        torch.testing.assert_close(got, torch.ones(2, N_QUBITS), rtol=0, atol=1e-6)

    def test_scale_invariance_with_random_weights(self) -> None:
        """Only the direction is encoded: x and c·x give the same output."""
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        with torch.no_grad():
            torch.testing.assert_close(layer(x), layer(7.5 * x), rtol=0, atol=1e-6)

    def test_matches_qml_amplitude_embedding_reference(self) -> None:
        """Per-sample agreement with a reference QNode built from the template."""
        import pennylane as qml

        n_features = 6
        layer = _layer(n_features=n_features)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
            for lyr in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                for q in range(N_QUBITS):
                    qml.Rot(weights[lyr, q, 0], weights[lyr, q, 1], weights[lyr, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        weights = layer.qlayer.weights.detach()
        x = torch.randn(BATCH, n_features)
        with torch.no_grad():
            got = layer(x)
            want = torch.stack([torch.stack(reference(x[i], weights)) for i in range(BATCH)])
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatchedMatchesPerSample:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_outputs(self, diff_method: str) -> None:
        layer = _layer(diff_method, n_features=6)
        x = torch.randn(BATCH, 6)
        with torch.no_grad():
            batched = layer(x)
            single = torch.cat([layer(x[i : i + 1]) for i in range(BATCH)])
        torch.testing.assert_close(batched, single, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_gradients_reach_quantum_weights(self, diff_method: str) -> None:
        layer = _layer(diff_method)
        layer(torch.randn(BATCH, N_AMPLITUDES)).sum().backward()
        grad = layer.qlayer.weights.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    def test_gradients_reach_inputs_under_backprop(self) -> None:
        """Padding and normalisation are differentiable end to end."""
        layer = _layer("backprop", n_features=6)
        x = torch.randn(BATCH, 6, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_input_gradient_is_orthogonal_to_the_input(self) -> None:
        """
        The output depends on x only through x/‖x‖, so the directional
        derivative along x itself is zero: ⟨∇_x f, x⟩ = 0 for every sample.
        """
        layer = _layer("backprop")
        x = torch.randn(BATCH, N_AMPLITUDES, dtype=torch.float64, requires_grad=True)
        layer.double()
        layer(x).sum().backward()
        assert x.grad is not None
        radial = (x.grad * x).sum(dim=-1)
        torch.testing.assert_close(
            radial, torch.zeros(BATCH, dtype=torch.float64), atol=1e-9, rtol=0
        )

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff"])
    def test_input_gradient_refused_under_gate_parameter_methods(self, diff_method: str) -> None:
        """
        PennyLane returns NaN, not an error, for the state-prep gradient under
        these methods; the layer refuses up front instead of training on NaN.
        """
        layer = _layer(diff_method)
        x = torch.randn(BATCH, N_AMPLITUDES, requires_grad=True)
        with pytest.raises(RuntimeError, match="backprop"):
            layer(x)

    def test_detached_input_is_fine_under_parameter_shift(self) -> None:
        layer = _layer("parameter-shift")
        x = torch.randn(BATCH, N_AMPLITUDES, requires_grad=True)
        layer(x.detach()).sum().backward()
        assert layer.qlayer.weights.grad is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_feature_dim_raises(self) -> None:
        layer = _layer(n_features=6)
        with pytest.raises(ValueError, match="n_features=6"):
            layer(torch.randn(BATCH, 7))

    def test_more_features_than_amplitudes_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 8\]"):
            _layer(n_features=N_AMPLITUDES + 1)

    def test_zero_features_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 8\]"):
            _layer(n_features=0)

    def test_all_zero_sample_raises(self) -> None:
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        x[2] = 0.0
        with pytest.raises(ValueError, match="all-zero"):
            layer(x)

    def test_n_qubits_lt_2_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits must be"):
            build_amplitude_qnode(n_qubits=1, device_name="default.qubit", diff_method="backprop")

    def test_default_n_features_is_all_amplitudes(self) -> None:
        layer = _layer()
        assert layer.n_features == N_AMPLITUDES == 2**N_QUBITS
        assert math.log2(layer.n_amplitudes) == N_QUBITS
