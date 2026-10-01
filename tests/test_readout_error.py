"""
tests/test_readout_error.py
===========================
The asymmetric readout error (#358): its affine effect on ⟨Z⟩ against an
independent density-matrix reference, post hoc, in a sweep, and in training.
"""

from __future__ import annotations

import math
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import (
    apply_depolarizing_noise,
    apply_readout_error,
    noise_sweep,
    readout_error_map,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
P01, P10 = 0.02, 0.09


def _layer(**kwargs: Any) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(n_qubits=3, n_layers=2, **{**CPU, **kwargs})


def _x() -> torch.Tensor:
    return torch.rand(5, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2


def _confusion_kraus(p01: float, p10: float) -> list[torch.Tensor]:
    # The classical confusion matrix as a channel: 0 → 1 with p01, 1 → 0 with p10.
    return [
        torch.tensor([[math.sqrt(1 - p01), 0.0], [0.0, 0.0]], dtype=torch.complex128),
        torch.tensor([[0.0, 0.0], [math.sqrt(p01), 0.0]], dtype=torch.complex128),
        torch.tensor([[0.0, 0.0], [0.0, math.sqrt(1 - p10)]], dtype=torch.complex128),
        torch.tensor([[0.0, math.sqrt(p10)], [0.0, 0.0]], dtype=torch.complex128),
    ]


def _density_reference(layer: QuantumEncodingLayer, x: torch.Tensor, p01: float, p10: float):
    """The layer's circuit on default.mixed with the confusion channel on every wire at the end."""
    kraus = [k.numpy() for k in _confusion_kraus(p01, p10)]
    base = qml.QNode(
        layer.qlayer.qnode.func,
        qml.device("default.mixed", wires=3),
        diff_method="backprop",
        interface="torch",
    )

    def confusion(wires: object) -> None:
        qml.QubitChannel(kraus, wires=wires)

    noisy = qml.noise.insert(base, confusion, (), position="end")
    weights = layer.qlayer.weights.detach().double()
    return torch.stack([torch.stack(noisy(xi.double(), weights)) for xi in x])


class TestMap:
    def test_matches_the_confusion_channel_on_every_qubit(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad(), apply_readout_error(layer, P01, P10):
            out = layer(x).double()
        reference = _density_reference(layer, x, P01, P10)
        torch.testing.assert_close(out, reference, atol=1e-6, rtol=0)

    def test_the_map_is_the_stated_affine_function(self) -> None:
        z = torch.linspace(-1, 1, 11, dtype=torch.float64)
        expected = (1 - P01 - P10) * z + (P10 - P01)
        torch.testing.assert_close(readout_error_map(z, P01, P10), expected)
        # The extremes: a certain 0 reads 1 with p01, a certain 1 reads 0 with p10.
        assert readout_error_map(torch.tensor(1.0), P01, P10).item() == pytest.approx(1 - 2 * P01)
        assert readout_error_map(torch.tensor(-1.0), P01, P10).item() == pytest.approx(
            -1 + 2 * P10
        )

    def test_symmetric_case_is_bit_flip_at_the_end(self) -> None:
        p = 0.07
        layer = _layer()
        x = _x()
        with torch.no_grad():
            with apply_readout_error(layer, p, p):
                readout = layer(x)
            with apply_depolarizing_noise(layer, p, position="end", channel="bit_flip"):
                flipped = layer(x)
        torch.testing.assert_close(readout, flipped.to(readout.dtype), atol=1e-6, rtol=0)

    def test_first_qubit_readout(self) -> None:
        layer = _layer(readout="first")
        x = _x()
        with torch.no_grad():
            clean = layer(x)
            with apply_readout_error(layer, P01, P10):
                noisy = layer(x)
        torch.testing.assert_close(noisy, readout_error_map(clean, P01, P10))


class TestPostHoc:
    def test_composes_with_a_noise_block(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.05):
            noisy = layer(x)
            with apply_readout_error(layer, P01, P10):
                both = layer(x)
        torch.testing.assert_close(both, readout_error_map(noisy, P01, P10))

    def test_block_replaces_the_layer_s_own_training_readout(self) -> None:
        layer = _layer(readout_error=(0.3, 0.3))
        layer.train()
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            with apply_readout_error(layer, P01, P10):
                out = layer(x)
        torch.testing.assert_close(out, readout_error_map(clean, P01, P10))

    def test_restores_the_layer(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad():
            before = layer(x)
            with apply_readout_error(layer, P01, P10):
                pass
            torch.testing.assert_close(layer(x), before, atol=0, rtol=0)

    def test_cannot_nest(self) -> None:
        layer = _layer()
        with (
            apply_readout_error(layer, P01, P10),
            pytest.raises(RuntimeError, match="nested"),
            apply_readout_error(layer, 0.1, 0.1),
        ):
            pass

    @pytest.mark.parametrize(("p01", "p10"), [(-0.1, 0.0), (0.0, 1.5), (True, 0.0)])
    def test_invalid_probabilities(self, p01: Any, p10: Any) -> None:
        with pytest.raises(ValueError, match="number in"), apply_readout_error(_layer(), p01, p10):
            pass


class TestSweep:
    def test_numbers_are_symmetric_and_pairs_asymmetric(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        x = _x()
        points = noise_sweep(model, x, [0.0, 0.05, (P01, P10)], channel="readout")
        assert [pt.p for pt in points] == [(0.0, 0.0), (0.05, 0.05), (P01, P10)]
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(x))
        with apply_readout_error(model, P01, P10):
            torch.testing.assert_close(points[2].probabilities, model.predict_proba(x))

    def test_every_error_is_checked_before_the_first_evaluation(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        calls = {"n": 0}
        original = model.predict_proba

        def counted(inp: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return original(inp)

        model.predict_proba = counted  # type: ignore[method-assign, assignment]
        with pytest.raises(ValueError, match="readout error"):
            noise_sweep(model, _x(), [0.01, (0.1, 2.0)], channel="readout")
        assert calls["n"] == 0


class TestTraining:
    def test_applied_in_train_mode_only(self) -> None:
        layer = _layer(readout_error=(P01, P10))
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            trained = layer(x)
        torch.testing.assert_close(trained, readout_error_map(clean, P01, P10))

    def test_gradient_is_scaled_by_the_contrast(self) -> None:
        x = _x()
        grads = []
        for readout_error in (None, (P01, P10)):
            layer = _layer(readout_error=readout_error)
            layer.train()
            layer(x).sum().backward()
            grads.append(layer.qlayer.weights.grad)
        torch.testing.assert_close(grads[1], (1 - P01 - P10) * grads[0])

    @pytest.mark.parametrize("bad", [(0.1,), (0.1, 0.2, 0.3), "ab", (0.1, -0.2), 0.1])
    def test_invalid_option(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="readout_error"):
            _layer(readout_error=bad)

    def test_shown_in_the_repr(self) -> None:
        assert "readout_error=(0.02, 0.09)" in _layer(readout_error=(P01, P10)).extra_repr()
        assert "readout_error" not in _layer().extra_repr()

    def test_classifier_config_and_weight_safe_checkpoint(self, tmp_path) -> None:
        model = HybridBinaryClassifier(
            n_input_features=3, n_qubits=3, n_layers=1, readout_error=(P01, P10), **CPU
        )
        assert model.get_config()["readout_error"] == (P01, P10)
        assert model.quantum_layer.readout_error == (P01, P10)
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt", readout_error=(0.0, 0.1))
        assert loaded.quantum_layer.readout_error == (0.0, 0.1)  # type: ignore[union-attr]
