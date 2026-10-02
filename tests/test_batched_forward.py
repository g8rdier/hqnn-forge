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

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers import restricted_normal_init_

N_QUBITS = 4
N_LAYERS = 2
BATCH = 6


DEVICE_CONFIGS = [
    # Two circuit runs per parameter per sample: the slow path wherever it appears.
    pytest.param(
        "default.qubit",
        "parameter-shift",
        id="default.qubit/parameter-shift",
        marks=pytest.mark.slow,
    ),
    pytest.param("default.qubit", "backprop", id="default.qubit/backprop"),
    # The input-gradient tests feed float32 inputs, for which PennyLane warns
    # that finite differences may be inaccurate; they still agree within
    # those tests' tolerances, so the warning is filtered here only.
    pytest.param(
        "default.qubit",
        "finite-diff",
        id="default.qubit/finite-diff",
        marks=pytest.mark.filterwarnings("ignore:Finite differences with float32:UserWarning"),
    ),
    pytest.param(
        "lightning.qubit",
        "adjoint",
        id="lightning.qubit/adjoint",
        marks=pytest.mark.requires_lightning,
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
        grad_of: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        layer = _build(layer_cls, device_name, diff_method)
        # Weight the outputs so the loss depends on every (sample, qubit) entry
        # differently: a plain sum would hide a permutation of samples.
        weights = torch.linspace(0.1, 1.0, BATCH * N_QUBITS).reshape(BATCH, N_QUBITS)

        layer.zero_grad()
        (layer(batch) * weights).sum().backward()
        grads_batched = {n: grad_of(p).clone() for n, p in layer.named_parameters()}

        layer.zero_grad()
        (_per_sample(layer, batch) * weights).sum().backward()
        grads_looped = {n: grad_of(p).clone() for n, p in layer.named_parameters()}

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
        grad_of: Callable[[torch.Tensor], torch.Tensor],
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

        assert grad_of(x_batched).abs().sum() > 0
        torch.testing.assert_close(grad_of(x_batched), grad_of(x_looped), rtol=1e-5, atol=1e-6)


LIGHTNING_BROADCAST_CASES = [
    *LAYER_CLASSES,
    # Broadcast state preparation, the gate most likely to be mis-shaped on the
    # adjoint path.  Weight gradients only: the layer refuses input gradients
    # outside backprop.
    pytest.param(AmplitudeEncodingLayer, id="amplitude"),
]


@pytest.mark.requires_lightning
@pytest.mark.parametrize("layer_cls", LIGHTNING_BROADCAST_CASES)
def test_lightning_runs_broadcast_tapes_correctly(
    layer_cls: type, grad_of: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    # #312: handing lightning's adjoint path the broadcast tape unsplit gives
    # the same outputs and gradients as the split (PennyLane 0.45; lightning's
    # own preprocessing splits it per sample).  If this starts failing, the
    # split is load-bearing for correctness again and expand_batch_dimension's
    # docstring is wrong.
    import pennylane as qml

    amplitude = layer_cls is AmplitudeEncodingLayer
    width = 2**N_QUBITS if amplitude else N_QUBITS
    split = _build(layer_cls, "lightning.qubit", "adjoint")
    native = _build(layer_cls, "lightning.qubit", "adjoint")
    native.load_state_dict(split.state_dict())
    q = native.qlayer.qnode
    native.qlayer.qnode = qml.QNode(q.func, q.device, interface="torch", diff_method="adjoint")
    x = torch.rand(BATCH, width, generator=torch.Generator().manual_seed(4)) * 2 - 1
    if amplitude:
        x = x.abs() + 0.05
    results = []
    for layer in (split, native):
        xi = x.clone().requires_grad_(not amplitude)
        out = layer(xi)
        out.sum().backward()
        # Every parameter, so input_scaling is checked where the layer has it.
        grads = {n: grad_of(p).clone() for n, p in layer.named_parameters()}
        if not amplitude:
            grads["inputs"] = grad_of(xi).clone()
        results.append((out.detach(), grads))
    (native_out, native_grads), (split_out, split_grads) = results[1], results[0]
    torch.testing.assert_close(native_out, split_out, atol=1e-5, rtol=1e-5)
    assert native_grads.keys() == split_grads.keys()
    for name, grad in split_grads.items():
        torch.testing.assert_close(native_grads[name], grad, atol=1e-5, rtol=1e-5, msg=name)
