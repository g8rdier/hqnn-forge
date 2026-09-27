"""
tests/test_parameter_shift.py
=============================
``diff_method="parameter-shift"`` -- the only gradient method that runs on
quantum hardware -- against backprop, for every encoder and circuit option (#315).

For each configuration, the same layer is built twice with identical weights,
once with ``backprop`` and once with ``parameter-shift`` on ``default.qubit``,
and the gradient of one fixed scalar cost is compared for every trainable
tensor and, where the layer supports it, for the inputs (a classical encoder
upstream always needs those).  The two are independent computations -- autograd
through the state vector against two shifted circuit evaluations per parameter
-- so agreement to float32 round-off pins the shift rules and the classical
processing around them (the IQP phases ``x_i x_j``, the re-uploading
``input_scaling``).

A gate without an analytic shift rule would make PennyLane fall back to finite
differences without an error; the method it picks for each parameter is
recorded during a real backward pass and checked.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pennylane.gradients.parameter_shift as parameter_shift_module
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

N_QUBITS = 3
# float32 weights and outputs: the two methods agree to a few 1e-7 (measured
# at most 4e-7 over these configurations); 2e-6 leaves margin without
# admitting a finite-difference fallback, which is off by 1e-3 or more here.
ATOL = 2e-6


def _angle(**kw: Any) -> Any:
    return partial(QuantumEncodingLayer, **kw)


def _iqp(**kw: Any) -> Any:
    return partial(IQPEncodingLayer, **kw)


def _reuploading(**kw: Any) -> Any:
    return partial(DataReuploadingLayer, **kw)


ENCODERS = [
    pytest.param(_angle(), True, id="angle"),
    pytest.param(_angle(rotation="Y"), True, id="angle-y"),
    # No rotation="Z": on |0⟩ an RZ embedding is a phase, so the inputs have no
    # effect and their gradient is 0 under both methods (#212; #275 refuses it).
    pytest.param(_angle(entangler="strongly_entangling"), True, id="angle-strongly"),
    pytest.param(_angle(readout="first"), True, id="angle-first"),
    pytest.param(
        _angle(rotation="Y", entangler="strongly_entangling", readout="first"),
        True,
        id="angle-published-shnn",
    ),
    pytest.param(_iqp(), True, id="iqp"),
    pytest.param(_iqp(n_repeats=2, entangler="strongly_entangling"), True, id="iqp-repeats"),
    pytest.param(_iqp(readout="first"), True, id="iqp-first"),
    pytest.param(_reuploading(), True, id="reuploading"),
    pytest.param(_reuploading(trainable_input_scaling=True), True, id="reuploading-scaled"),
    pytest.param(
        _reuploading(trainable_input_scaling=True, rotation="Z"), True, id="reuploading-scaled-z"
    ),
    pytest.param(
        _reuploading(entangler="strongly_entangling", readout="first"),
        True,
        id="reuploading-strongly-first",
    ),
    # Input gradients are refused under every method but backprop (see the
    # amplitude module docstring), so only the weights are compared.
    pytest.param(partial(AmplitudeEncodingLayer, n_features=5), False, id="amplitude"),
]


def _pair(factory: Any) -> tuple[Any, Any]:
    torch.manual_seed(0)
    backprop = factory(
        n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
    )
    shift = factory(
        n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="parameter-shift"
    )
    shift.load_state_dict(backprop.state_dict())
    # Trained-looking weights rather than the init's small angles, so no
    # gradient is near zero by construction.
    with torch.no_grad():
        for p_b, p_s in zip(backprop.parameters(), shift.parameters(), strict=True):
            values = torch.empty_like(p_b).uniform_(-torch.pi, torch.pi)
            p_b.copy_(values)
            p_s.copy_(values)
    return backprop, shift


def _inputs(layer: Any, requires_grad: bool) -> torch.Tensor:
    width = getattr(layer, "n_features", N_QUBITS)
    x = torch.rand(4, width, generator=torch.Generator().manual_seed(1)) * 2 - 1
    return x.requires_grad_(requires_grad)


def _gradients(layer: Any, x: torch.Tensor) -> list[torch.Tensor]:
    out = layer(x)
    # A fixed, sign-varying weighting, so no cancellation hides a wrong term.
    weighting = torch.linspace(-1.0, 1.0, out.numel(), dtype=out.dtype).reshape(out.shape)
    targets = [*layer.parameters(), *([x] if x.requires_grad else [])]
    return list(torch.autograd.grad((out * weighting).sum(), targets))


@pytest.mark.parametrize("factory, input_grads", ENCODERS)
class TestAgainstBackprop:
    def test_every_gradient_matches(self, factory: Any, input_grads: bool) -> None:
        backprop, shift = _pair(factory)
        x = _inputs(backprop, input_grads)
        expected = _gradients(backprop, x)
        got = _gradients(shift, x.detach().requires_grad_(input_grads))
        assert len(got) == len(expected) >= 1
        for g_shift, g_backprop in zip(got, expected, strict=True):
            assert g_backprop.abs().max() > 1e-3  # a gradient worth comparing
            torch.testing.assert_close(g_shift, g_backprop, rtol=0, atol=ATOL)

    def test_no_parameter_falls_back_to_finite_differences(
        self, factory: Any, input_grads: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # param_shift picks a method per trainable parameter -- "A" (analytic
        # shift rule), "0" (provably zero) or "F" (finite differences) -- and
        # runs "F" without an error.  Record the choices it actually makes
        # during a real backward pass and refuse any "F".
        chosen: list[dict[int, str]] = []
        original = parameter_shift_module.find_and_validate_gradient_methods

        def spy(*args: Any, **kwargs: Any) -> dict[int, str]:
            methods = original(*args, **kwargs)
            chosen.append(dict(methods))
            return methods  # type: ignore[no-any-return]

        monkeypatch.setattr(parameter_shift_module, "find_and_validate_gradient_methods", spy)
        _, shift = _pair(factory)
        _gradients(shift, _inputs(shift, input_grads))
        assert chosen, "the parameter-shift transform never ran"
        methods = {m for per_tape in chosen for m in per_tape.values()}
        assert "F" not in methods and "A" in methods, methods


@pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
@pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
def test_classifier_trains_under_parameter_shift(cls: type, encoding_type: str) -> None:
    # The whole model -- encoder, circuit, head -- trained end to end with the
    # hardware-compatible method, and its first step equal to backprop's.
    def build(diff_method: str) -> Any:
        torch.manual_seed(0)
        return cls(
            n_input_features=4,
            n_qubits=2,
            n_layers=1,
            encoding_type=encoding_type,
            device_name="default.qubit",
            diff_method=diff_method,
        )

    shift, backprop = build("parameter-shift"), build("backprop")
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(2))
    y = (x[:, 0] > 0).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def loss(model: Any) -> torch.Tensor:
        return loss_fn(model(x).squeeze(-1), y)  # type: ignore[no-any-return]

    for g_s, g_b in zip(
        torch.autograd.grad(loss(shift), list(shift.parameters())),
        torch.autograd.grad(loss(backprop), list(backprop.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(g_s, g_b, rtol=0, atol=ATOL)

    opt = torch.optim.Adam(shift.parameters(), lr=0.1)
    before = loss(shift).item()
    for _ in range(10):
        opt.zero_grad()
        loss(shift).backward()
        opt.step()
    assert loss(shift).item() < before
