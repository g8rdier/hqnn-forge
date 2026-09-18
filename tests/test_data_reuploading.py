"""
tests/test_data_reuploading.py
==============================
Unit tests for hqnn_forge.encoding.data_reuploading.DataReuploadingLayer.

The numerical checks pin what re-uploading is for: a single upload makes every
output a degree-one trigonometric polynomial in each feature, and ``L``
uploads raise that degree to ``L`` (Schuld, Sweke & Meyer 2021).  The Fourier
test below measures that degree directly, so a regression that silently drops
back to one upload fails it.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import (
    DataReuploadingLayer,
    QuantumEncodingLayer,
    build_data_reuploading_qnode,
)

N_QUBITS = 3
N_LAYERS = 2
BATCH = 5


def _layer(
    n_layers: int = N_LAYERS, diff_method: str = "backprop", **kwargs: object
) -> DataReuploadingLayer:
    torch.manual_seed(0)
    return DataReuploadingLayer(
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        device_name="default.qubit",
        diff_method=diff_method,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _random_batch(n: int = BATCH) -> torch.Tensor:
    return torch.rand(n, N_QUBITS) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Shape, range, parameters
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_output_shape(self) -> None:
        assert _layer()(_random_batch()).shape == (BATCH, N_QUBITS)

    def test_single_sample(self) -> None:
        assert _layer()(_random_batch(1)).shape == (1, N_QUBITS)

    def test_expectation_values_in_range(self) -> None:
        with torch.no_grad():
            out = _layer()(_random_batch())
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_parameter_count_matches_angle_encoder_by_default(self) -> None:
        layer = _layer()
        assert sum(p.numel() for p in layer.parameters()) == N_LAYERS * N_QUBITS * 3
        assert f"n_params={N_LAYERS * N_QUBITS * 3}" in layer.extra_repr()

    def test_trainable_input_scaling_adds_one_parameter_per_upload_and_qubit(self) -> None:
        layer = _layer(trainable_input_scaling=True)
        expected = N_LAYERS * N_QUBITS * 3 + N_LAYERS * N_QUBITS
        assert sum(p.numel() for p in layer.parameters()) == expected
        assert layer.qlayer.input_scaling.shape == (N_LAYERS, N_QUBITS)
        assert torch.equal(layer.qlayer.input_scaling.detach(), torch.ones(N_LAYERS, N_QUBITS))
        assert f"n_params={expected}" in layer.extra_repr()


# ---------------------------------------------------------------------------
# Circuit structure
# ---------------------------------------------------------------------------


class TestCircuitStructure:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_one_upload_per_layer(self, n_layers: int) -> None:
        """The tape holds n_layers AngleEmbedding ops, one before each block."""
        layer = _layer(n_layers)
        tape = qml.workflow.construct_tape(layer.qlayer.qnode)(
            _random_batch(1)[0], layer.qlayer.weights.detach()
        )
        names = [op.name for op in tape.operations]
        assert names.count("AngleEmbedding") == n_layers
        assert names.count("CNOT") == n_layers * N_QUBITS
        assert names.count("Rot") == n_layers * N_QUBITS
        # Order within each layer: embedding, ring, rotations.
        per_layer = ["AngleEmbedding"] + ["CNOT"] * N_QUBITS + ["Rot"] * N_QUBITS
        assert names == per_layer * n_layers

    def test_single_layer_equals_angle_encoder(self) -> None:
        """With one upload the circuit is QuantumEncodingLayer gate for gate."""
        ours = _layer(n_layers=1)
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        with torch.no_grad():
            reference.qlayer.weights.copy_(ours.qlayer.weights)
        x = _random_batch()
        with torch.no_grad():
            torch.testing.assert_close(ours(x), reference(x), rtol=0, atol=1e-6)

    def test_two_layers_differ_from_single_upload(self) -> None:
        """Same weights, same depth: only the extra upload distinguishes them."""
        ours = _layer(n_layers=2)
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        with torch.no_grad():
            reference.qlayer.weights.copy_(ours.qlayer.weights)
        x = _random_batch()
        with torch.no_grad():
            assert not torch.allclose(ours(x), reference(x), atol=1e-3)

    @pytest.mark.parametrize("trainable_input_scaling", [False, True])
    def test_matches_explicit_reference_circuit(self, trainable_input_scaling: bool) -> None:
        layer = _layer(trainable_input_scaling=trainable_input_scaling)
        weights = layer.qlayer.weights.detach()
        scaling = (
            torch.rand(N_LAYERS, N_QUBITS) * 2
            if trainable_input_scaling
            else torch.ones(N_LAYERS, N_QUBITS)
        )
        if trainable_input_scaling:
            with torch.no_grad():
                layer.qlayer.input_scaling.copy_(scaling)

        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            for lyr in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.RX(scaling[lyr, q] * x[q], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                for q in range(N_QUBITS):
                    qml.Rot(weights[lyr, q, 0], weights[lyr, q, 1], weights[lyr, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        x = _random_batch()
        with torch.no_grad():
            got = layer(x)
            want = torch.stack([torch.stack(reference(x[i])) for i in range(BATCH)])
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Expressivity: Fourier degree grows with the number of uploads
# ---------------------------------------------------------------------------


def _fourier_magnitudes(model: torch.nn.Module, n_points: int = 16) -> torch.Tensor:
    """
    Fourier magnitudes |c_k|, k = 0 … n_points/2, of the outputs as functions
    of feature 0 with the other features held fixed, sampled on a uniform grid
    over one period; the maximum over the n_qubits outputs is taken per k.

    The maximum matters: with one layer, ⟨Z_0⟩ does not depend on feature 0
    at all (the cascaded CNOT ring maps Z_0 to Z_1⋯Z_{n-1} in the Heisenberg
    picture, and RX-embedded |0⟩ has ⟨X⟩ = 0), so probing a single output
    would under-count the degree.
    """
    torch.manual_seed(1)
    fixed = _random_batch(1)
    grid = torch.arange(n_points, dtype=torch.float32) * 2 * math.pi / n_points - math.pi
    x = fixed.repeat(n_points, 1)
    x[:, 0] = grid
    with torch.no_grad():
        values = model(x).to(torch.float64)  # (n_points, n_qubits)
    return (torch.fft.rfft(values, dim=0).abs() / n_points).max(dim=1).values


class TestFourierDegree:
    FLOOR = 1e-3  # a present harmonic is far above float32 noise; an absent one far below

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_single_upload_has_degree_one(self, n_layers: int) -> None:
        """Sanity check of the probe: the angle encoder has no k ≥ 2 harmonics at any depth."""
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS,
            n_layers=n_layers,
            device_name="default.qubit",
            diff_method="backprop",
        )
        torch.manual_seed(0)
        with torch.no_grad():
            reference.qlayer.weights.uniform_(0, 2 * math.pi)
        mags = _fourier_magnitudes(reference)
        assert mags[1] > self.FLOOR
        assert mags[2:].max() < self.FLOOR

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_degree_equals_number_of_uploads(self, n_layers: int) -> None:
        layer = _layer(n_layers)
        torch.manual_seed(0)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        mags = _fourier_magnitudes(layer)
        assert mags[n_layers] > self.FLOOR, f"harmonic {n_layers} missing: {mags}"
        assert mags[n_layers + 1 :].max() < self.FLOOR, f"harmonics above {n_layers}: {mags}"


# ---------------------------------------------------------------------------
# Batching and gradients
# ---------------------------------------------------------------------------


class TestBatchedMatchesPerSample:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("trainable_input_scaling", [False, True])
    def test_outputs(self, diff_method: str, trainable_input_scaling: bool) -> None:
        layer = _layer(diff_method=diff_method, trainable_input_scaling=trainable_input_scaling)
        x = _random_batch()
        with torch.no_grad():
            batched = layer(x)
            single = torch.cat([layer(x[i : i + 1]) for i in range(BATCH)])
        torch.testing.assert_close(batched, single, rtol=1e-6, atol=1e-6)


class TestGradientFlow:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_gradients_reach_weights_and_inputs(self, diff_method: str) -> None:
        layer = _layer(diff_method=diff_method)
        x = _random_batch().requires_grad_(True)
        layer(x).sum().backward()
        assert layer.qlayer.weights.grad is not None
        assert layer.qlayer.weights.grad.abs().sum().item() > 0.0
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_gradients_reach_input_scaling(self) -> None:
        layer = _layer(trainable_input_scaling=True)
        layer(_random_batch()).sum().backward()
        grad = layer.qlayer.input_scaling.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    def test_parameter_shift_matches_backprop(self) -> None:
        """Exact gradients agree across methods, including through the uploads."""
        a = _layer(diff_method="backprop", trainable_input_scaling=True)
        b = _layer(diff_method="parameter-shift", trainable_input_scaling=True)
        with torch.no_grad():
            b.qlayer.weights.copy_(a.qlayer.weights)
            scaling = torch.rand(N_LAYERS, N_QUBITS) + 0.5
            a.qlayer.input_scaling.copy_(scaling)
            b.qlayer.input_scaling.copy_(scaling)
        x = _random_batch()
        xa = x.clone().requires_grad_(True)
        xb = x.clone().requires_grad_(True)
        a(xa).sum().backward()
        b(xb).sum().backward()
        torch.testing.assert_close(
            a.qlayer.weights.grad, b.qlayer.weights.grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            a.qlayer.input_scaling.grad, b.qlayer.input_scaling.grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match=f"n_qubits={N_QUBITS}"):
            _layer()(torch.rand(BATCH, N_QUBITS + 1))

    def test_n_qubits_lt_2_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits must be"):
            build_data_reuploading_qnode(n_qubits=1, device_name="default.qubit")

    def test_n_layers_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers must be"):
            build_data_reuploading_qnode(n_qubits=2, n_layers=0, device_name="default.qubit")
