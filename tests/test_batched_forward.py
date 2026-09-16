"""
tests/test_batched_forward.py
==============================
The encoding layers hand the whole batch to ``qml.qnn.TorchLayer`` in one call
instead of looping over samples.  These tests pin that the batched path is
numerically the per-sample path: same outputs, same gradients, on every
device/diff_method pair the library supports.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers import restricted_normal_init_

N_QUBITS = 4
N_LAYERS = 2
BATCH = 6


def _lightning_available() -> bool:
    try:
        import pennylane as qml

        qml.device("lightning.qubit", wires=1)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False


DEVICE_CONFIGS = [
    pytest.param("default.qubit", "parameter-shift", id="default.qubit/parameter-shift"),
    pytest.param("default.qubit", "backprop", id="default.qubit/backprop"),
    pytest.param(
        "lightning.qubit",
        "adjoint",
        id="lightning.qubit/adjoint",
        marks=pytest.mark.skipif(not _lightning_available(), reason="pennylane-lightning not installed"),
    ),
]

LAYER_CLASSES = [
    pytest.param(QuantumEncodingLayer, id="angle"),
    pytest.param(IQPEncodingLayer, id="iqp"),
]


def _build(layer_cls: type, device_name: str, diff_method: str) -> torch.nn.Module:
    torch.manual_seed(0)
    layer = layer_cls(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=device_name, diff_method=diff_method
    )
    restricted_normal_init_(layer.qlayer.weights, n_qubits=N_QUBITS, n_layers=N_LAYERS)
    return layer


def _per_sample(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The loop the layers used before batching; the reference implementation."""
    return torch.stack([layer.qlayer(sample) for sample in x])


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(BATCH, N_QUBITS) * 2 * math.pi - math.pi


@pytest.mark.parametrize("layer_cls", LAYER_CLASSES)
@pytest.mark.parametrize("device_name, diff_method", DEVICE_CONFIGS)
class TestBatchedMatchesPerSample:
    def test_outputs_match(
        self, layer_cls: type, device_name: str, diff_method: str, batch: torch.Tensor
    ) -> None:
        layer = _build(layer_cls, device_name, diff_method)
        with torch.no_grad():
            batched = layer(batch)
            looped = _per_sample(layer, batch)
        assert batched.shape == (BATCH, N_QUBITS)
        torch.testing.assert_close(batched, looped, rtol=0, atol=1e-6)

    def test_gradients_match(
        self, layer_cls: type, device_name: str, diff_method: str, batch: torch.Tensor
    ) -> None:
        layer = _build(layer_cls, device_name, diff_method)
        # Weight the outputs so the loss depends on every (sample, qubit) entry
        # differently: a plain sum would hide a permutation of samples.
        weights = torch.linspace(0.1, 1.0, BATCH * N_QUBITS).reshape(BATCH, N_QUBITS)

        layer.zero_grad()
        (layer(batch) * weights).sum().backward()
        grad_batched = layer.qlayer.weights.grad.clone()

        layer.zero_grad()
        (_per_sample(layer, batch) * weights).sum().backward()
        grad_looped = layer.qlayer.weights.grad.clone()

        assert grad_batched.abs().sum() > 0
        torch.testing.assert_close(grad_batched, grad_looped, rtol=1e-5, atol=1e-6)

    def test_rows_are_independent(
        self, layer_cls: type, device_name: str, diff_method: str, batch: torch.Tensor
    ) -> None:
        """Row i of the batched output is the single-sample output for row i."""
        layer = _build(layer_cls, device_name, diff_method)
        with torch.no_grad():
            batched = layer(batch)
            single = layer(batch[2:3])
        torch.testing.assert_close(batched[2:3], single, rtol=0, atol=1e-6)


class TestBatchShapes:
    def test_batch_of_one(self) -> None:
        layer = _build(QuantumEncodingLayer, "default.qubit", "parameter-shift")
        out = layer(torch.zeros(1, N_QUBITS))
        assert out.shape == (1, N_QUBITS)

    def test_wrong_feature_dim_still_raises(self) -> None:
        layer = _build(IQPEncodingLayer, "default.qubit", "parameter-shift")
        with pytest.raises(ValueError, match="n_qubits"):
            layer(torch.zeros(BATCH, N_QUBITS + 1))


@pytest.mark.parametrize("layer_cls", LAYER_CLASSES)
@pytest.mark.parametrize("device_name, diff_method", DEVICE_CONFIGS)
class TestInputGradients:
    def test_input_gradients_match_per_sample(
        self, layer_cls: type, device_name: str, diff_method: str, batch: torch.Tensor
    ) -> None:
        """
        With a classical encoder upstream the loss needs d(out)/d(inputs), so
        the batched path must differentiate with respect to the *broadcasted*
        parameters too -- the case the parameter-shift and finite-difference
        transforms refuse on an unexpanded broadcasted tape.
        """
        layer = _build(layer_cls, device_name, diff_method)
        weights = torch.linspace(0.1, 1.0, BATCH * N_QUBITS).reshape(BATCH, N_QUBITS)

        x_batched = batch.clone().requires_grad_(True)
        (layer(x_batched) * weights).sum().backward()

        x_looped = batch.clone().requires_grad_(True)
        (_per_sample(layer, x_looped) * weights).sum().backward()

        assert x_batched.grad.abs().sum() > 0
        torch.testing.assert_close(x_batched.grad, x_looped.grad, rtol=1e-5, atol=1e-6)
