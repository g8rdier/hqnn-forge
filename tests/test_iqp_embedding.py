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


def _lightning_available() -> bool:
    try:
        import pennylane as qml

        qml.device("lightning.qubit", wires=1)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False


class TestTemplateMatchesDocumentedFeatureMap:
    """
    The circuit uses qml.IQPEmbedding directly.  Pin that the template is
    the feature map the module docstring describes -- H, RZ(x_i), then
    exp(-i x_i x_j Z_i Z_j / 2) on every pair, written here as its exact
    CNOT·RZ·CNOT form -- sample by sample, so a change in the template's
    convention (angle factor, pair pattern) is noticed.
    """

    @pytest.mark.parametrize("n_repeats", [1, 2])
    def test_matches_explicit_decomposition(self, n_repeats: int) -> None:
        from itertools import combinations

        import pennylane as qml

        from hqnn_forge.encoding.iqp_embedding import build_iqp_qnode

        n_qubits, n_layers = 4, 1
        ours = build_iqp_qnode(
            n_qubits=n_qubits, n_layers=n_layers, n_repeats=n_repeats,
            device_name="default.qubit", diff_method="backprop",
        )

        dev = qml.device("default.qubit", wires=n_qubits)
        pairs = list(combinations(range(n_qubits), 2))

        @qml.qnode(dev, interface="torch")
        def reference(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            for _ in range(n_repeats):
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
                    qml.RZ(inputs[q], wires=q)
                for i, j in pairs:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(inputs[i] * inputs[j], wires=j)
                    qml.CNOT(wires=[i, j])
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

    def test_tape_uses_the_template(self) -> None:
        import pennylane as qml

        from hqnn_forge.encoding.iqp_embedding import build_iqp_qnode

        qnode = build_iqp_qnode(n_qubits=3, n_layers=1, n_repeats=2, device_name="default.qubit")
        tape = qml.workflow.construct_tape(qnode, level=0)(torch.zeros(3), torch.zeros(1, 3, 3))
        embeddings = [op for op in tape.operations if op.name == "IQPEmbedding"]
        assert len(embeddings) == 1
        assert embeddings[0].hyperparameters["n_repeats"] == 2


class TestBatchedTemplateMatchesPerSample:
    """
    Reverting to the template makes the batched path depend on
    _expand_batch_dimension splitting the batch for every diff method except
    backprop (lightning.qubit's adjoint mis-shapes a broadcasted MultiRZ).
    tests/test_batched_forward.py covers backprop, parameter-shift and
    adjoint for outputs, weight gradients and input gradients; this pins the
    two configurations the revert hinges on, plus finite-diff.
    """

    CONFIGS = [
        pytest.param("default.qubit", "backprop", id="default.qubit/backprop"),
        pytest.param("default.qubit", "finite-diff", id="default.qubit/finite-diff"),
        pytest.param(
            "lightning.qubit",
            "adjoint",
            id="lightning.qubit/adjoint",
            marks=pytest.mark.skipif(
                not _lightning_available(), reason="pennylane-lightning not installed"
            ),
        ),
    ]

    @pytest.mark.parametrize("device_name, diff_method", CONFIGS)
    def test_outputs_and_gradients(self, device_name: str, diff_method: str) -> None:
        torch.manual_seed(0)
        layer = IQPEncodingLayer(
            n_qubits=4, n_layers=1, device_name=device_name, diff_method=diff_method
        )
        restricted_normal_init_(layer.qlayer.weights, n_qubits=4, n_layers=1)
        x = torch.rand(5, 4) * 2 * math.pi - math.pi
        scale = torch.linspace(0.1, 1.0, 20).reshape(5, 4)

        xb = x.clone().requires_grad_(True)
        layer.zero_grad()
        batched = layer(xb)
        (batched * scale).sum().backward()
        grad_w_batched = layer.qlayer.weights.grad.clone()

        xl = x.clone().requires_grad_(True)
        layer.zero_grad()
        looped = torch.stack([layer.qlayer(sample) for sample in xl])
        (looped * scale).sum().backward()

        torch.testing.assert_close(batched, looped, rtol=0, atol=1e-6)
        torch.testing.assert_close(grad_w_batched, layer.qlayer.weights.grad, rtol=1e-5, atol=1e-6)
        assert xb.grad is not None and xl.grad is not None
        torch.testing.assert_close(xb.grad, xl.grad, rtol=1e-5, atol=1e-6)
