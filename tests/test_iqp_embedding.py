"""
tests/test_iqp_embedding.py
===========================
Unit tests for hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer.
"""
from __future__ import annotations

import math
import pytest
import torch

from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import restricted_normal_init_


class TestIQPEncodingLayer:
    def test_forward_shape_and_bounds(self) -> None:
        batch_size = 4
        n_qubits = 6
        n_layers = 2
        
        layer = IQPEncodingLayer(n_qubits=n_qubits, n_layers=n_layers)
        # Apply restricted initialization
        restricted_normal_init_(layer.qlayer.weights, n_qubits=n_qubits, n_layers=n_layers)
        
        x = torch.randn(batch_size, n_qubits)
        out = layer(x)
        
        assert out.shape == (batch_size, n_qubits)
        # Pauli-Z expectations must be in [-1, 1]
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_mismatched_feature_dim_raises(self) -> None:
        layer = IQPEncodingLayer(n_qubits=4)
        x_wrong = torch.randn(2, 5)
        with pytest.raises(ValueError, match="does not match n_qubits=4"):
            layer(x_wrong)

    def test_gradients_flow(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=1)
        restricted_normal_init_(layer.qlayer.weights, n_qubits=3, n_layers=1)
        
        x = torch.randn(2, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients flow back to the inputs
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0
        
        # Check gradients flow to the weights
        weights = layer.qlayer.weights
        assert weights.grad is not None
        assert weights.grad.abs().sum().item() > 0.0


class TestExplicitDecompositionMatchesTemplate:
    """
    The circuit writes qml.IQPEmbedding out gate by gate (with MultiRZ as
    CNOT·RZ·CNOT) so that a broadcasted batch only passes through
    single-parameter gates.
    Pin that it is still the same feature map, sample by sample.
    """

    @pytest.mark.parametrize("n_repeats", [1, 2])
    def test_matches_qml_iqp_embedding(self, n_repeats: int) -> None:
        import pennylane as qml

        from hqnn_forge.encoding.iqp_embedding import build_iqp_qnode

        n_qubits, n_layers = 4, 1
        ours = build_iqp_qnode(
            n_qubits=n_qubits, n_layers=n_layers, n_repeats=n_repeats,
            device_name="default.qubit", diff_method="backprop",
        )

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def reference(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.IQPEmbedding(inputs, wires=range(n_qubits), n_repeats=n_repeats, pattern=None)
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
                for q in range(n_qubits):
                    qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        torch.manual_seed(0)
        weights = torch.randn(n_layers, n_qubits, 3, dtype=torch.float64)
        for _ in range(3):
            x = torch.rand(n_qubits, dtype=torch.float64) * 2 * math.pi - math.pi
            with torch.no_grad():
                got = torch.stack(ours(x, weights))
                want = torch.stack(reference(x, weights))
            torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
