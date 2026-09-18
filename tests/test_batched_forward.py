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
from collections.abc import Callable
from functools import partial

import pytest
import torch
from conftest import _grad

from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
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
        marks=pytest.mark.skipif(
            not _lightning_available(), reason="pennylane-lightning not installed"
        ),
    ),
]

LAYER_CLASSES = [
    pytest.param(QuantumEncodingLayer, id="angle"),
    pytest.param(IQPEncodingLayer, id="iqp"),
    pytest.param(DataReuploadingLayer, id="reuploading"),
    pytest.param(
        partial(DataReuploadingLayer, trainable_input_scaling=True), id="reuploading-scaled"
    ),
    # Z leaves its first upload unscaled, so scaling row r multiplies upload r + 1.
    pytest.param(
        partial(DataReuploadingLayer, rotation="Z", trainable_input_scaling=True),
        id="reuploading-scaled-z",
    ),
]

EncodingLayer = QuantumEncodingLayer | IQPEncodingLayer | DataReuploadingLayer


def _build(
    layer_cls: Callable[..., EncodingLayer], device_name: str, diff_method: str
) -> EncodingLayer:
    torch.manual_seed(0)
    layer = layer_cls(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=device_name, diff_method=diff_method
    )
    restricted_normal_init_(layer.qlayer.weights, n_qubits=N_QUBITS, n_layers=N_LAYERS)
    scaling = getattr(layer.qlayer, "input_scaling", None)
    if scaling is not None:
        # Distinct per-entry scales rather than the ones-initialisation, so each
        # upload sees differently scaled features.  Row indexing itself is pinned
        # against reference circuits in test_data_reuploading.py.
        with torch.no_grad():
            scaling.copy_(torch.linspace(0.5, 1.5, scaling.numel()).reshape(scaling.shape))
    return layer


def _per_sample(layer: EncodingLayer, x: torch.Tensor) -> torch.Tensor:
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
        self,
        layer_cls: Callable[..., EncodingLayer],
        device_name: str,
        diff_method: str,
        batch: torch.Tensor,
    ) -> None:
        layer = _build(layer_cls, device_name, diff_method)
        with torch.no_grad():
            batched = layer(batch)
            looped = _per_sample(layer, batch)
        assert batched.shape == (BATCH, N_QUBITS)
        torch.testing.assert_close(batched, looped, rtol=0, atol=1e-6)

    def test_gradients_match(
        self,
        layer_cls: Callable[..., EncodingLayer],
        device_name: str,
        diff_method: str,
        batch: torch.Tensor,
    ) -> None:
        layer = _build(layer_cls, device_name, diff_method)
        # Weight the outputs so the loss depends on every (sample, qubit) entry
        # differently: a plain sum would hide a permutation of samples.
        weights = torch.linspace(0.1, 1.0, BATCH * N_QUBITS).reshape(BATCH, N_QUBITS)

        layer.zero_grad()
        (layer(batch) * weights).sum().backward()
        grads_batched = {n: _grad(p).clone() for n, p in layer.named_parameters()}

        layer.zero_grad()
        (_per_sample(layer, batch) * weights).sum().backward()
        grads_looped = {n: _grad(p).clone() for n, p in layer.named_parameters()}

        # Every trainable tensor, e.g. the re-uploading layer's input_scaling too.
        for name, grad_batched in grads_batched.items():
            assert grad_batched.abs().sum() > 0, name
            torch.testing.assert_close(grad_batched, grads_looped[name], rtol=1e-5, atol=1e-6)

    def test_rows_are_independent(
        self,
        layer_cls: Callable[..., EncodingLayer],
        device_name: str,
        diff_method: str,
        batch: torch.Tensor,
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
        self,
        layer_cls: Callable[..., EncodingLayer],
        device_name: str,
        diff_method: str,
        batch: torch.Tensor,
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

        assert _grad(x_batched).abs().sum() > 0
        torch.testing.assert_close(_grad(x_batched), _grad(x_looped), rtol=1e-5, atol=1e-6)
