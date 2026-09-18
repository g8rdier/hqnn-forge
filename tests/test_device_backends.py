"""
tests/test_device_backends.py
=============================
Device resolution in hqnn_forge.encoding.angle_embedding._resolve_device:
the accelerated backends when they are installed, and the fallback chain
``requested → lightning.qubit → default.qubit`` deterministically, by making
``qml.device`` fail for chosen names, whatever this machine has.
"""

from __future__ import annotations

import warnings

import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding import angle_embedding as ae
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier

GPU_BACKENDS = ("lightning.gpu", "lightning.kokkos")


def _available(name: str) -> bool:
    try:
        qml.device(name, wires=1)
    except ae._DEVICE_FAILURES:
        return False
    return True


def _failing_device(failing: dict[str, type[BaseException]]):
    """A stand-in for ``qml.device`` that raises for the names in ``failing``."""
    real = qml.device

    def device(name: str, *args, **kwargs):
        if name in failing:
            raise failing[name](f"{name} unavailable in this test")
        return real(name, *args, **kwargs)

    return device


# ---------------------------------------------------------------------------
# Accelerated backends: exercised when present, skipped otherwise
# ---------------------------------------------------------------------------


class TestAcceleratedBackends:
    @pytest.mark.parametrize("name", GPU_BACKENDS)
    def test_layer_runs_on_backend_when_available(self, name: str) -> None:
        if not _available(name):
            pytest.skip(f"{name} is not installed or has no usable device here")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no fallback expected
            layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, device_name=name)  # type: ignore[arg-type]
        assert layer.qlayer.qnode.device.name == name
        out = layer(torch.rand(2, 3))
        assert out.shape == (2, 3)
        out.sum().backward()
        assert layer.qlayer.weights.grad is not None

    @pytest.mark.parametrize("name", GPU_BACKENDS)
    def test_backend_names_are_accepted_by_the_type_alias(self, name: str) -> None:
        assert name in ae.DeviceName.__args__


# ---------------------------------------------------------------------------
# Fallback chain, deterministic
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_missing_gpu_backend_falls_back_to_lightning_then_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ae.qml,
            "device",
            _failing_device({"lightning.gpu": DeviceError, "lightning.qubit": ImportError}),
        )
        with pytest.warns(RuntimeWarning) as record:
            dev = ae._resolve_device("lightning.gpu", 2)
        assert dev.name == "default.qubit"
        messages = [str(w.message) for w in record]
        assert len(messages) == 2
        assert "lightning.gpu" in messages[0] and "lightning.qubit" in messages[0]
        assert "lightning.qubit" in messages[1] and "default.qubit" in messages[1]
        assert "Install pennylane-lightning" in messages[1]

    def test_missing_gpu_backend_stops_at_lightning_when_it_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _available("lightning.qubit"):
            pytest.skip("pennylane-lightning not installed")
        monkeypatch.setattr(ae.qml, "device", _failing_device({"lightning.kokkos": RuntimeError}))
        with pytest.warns(
            RuntimeWarning, match="lightning.kokkos.*Falling back to 'lightning.qubit'"
        ):
            dev = ae._resolve_device("lightning.kokkos", 2)
        assert dev.name == "lightning.qubit"

    def test_lightning_qubit_falls_straight_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ae.qml, "device", _failing_device({"lightning.qubit": OSError}))
        with pytest.warns(RuntimeWarning) as record:
            dev = ae._resolve_device("lightning.qubit", 2)
        assert dev.name == "default.qubit"
        assert len(record) == 1

    def test_default_qubit_has_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ae.qml, "device", _failing_device({"default.qubit": DeviceError}))
        with pytest.raises(DeviceError):
            ae._resolve_device("default.qubit", 2)

    def test_unknown_device_name_warns_and_falls_back(self) -> None:
        """
        The branch that was unreachable on PennyLane 0.45 (qml.DeviceError no
        longer exists, see #154): a name no plugin provides must warn and
        fall back, not raise AttributeError.
        """
        with pytest.warns(RuntimeWarning, match="no.such.device"):
            dev = ae._resolve_device("no.such.device", 2)  # type: ignore[arg-type]
        assert dev.name in ("lightning.qubit", "default.qubit")

    def test_no_warning_when_the_requested_device_works(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert ae._resolve_device("default.qubit", 2).name == "default.qubit"

    def test_iqp_layer_and_classifier_share_the_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ae.qml,
            "device",
            _failing_device({"lightning.gpu": DeviceError, "lightning.qubit": DeviceError}),
        )
        with pytest.warns(RuntimeWarning):
            iqp = IQPEncodingLayer(
                n_qubits=2, n_layers=1, device_name="lightning.gpu", diff_method="backprop"
            )  # type: ignore[arg-type]
        assert iqp.qlayer.qnode.device.name == "default.qubit"
        with pytest.warns(RuntimeWarning):
            model = HybridBinaryClassifier(
                n_input_features=2,
                n_qubits=2,
                n_layers=1,
                device_name="lightning.gpu",
                diff_method="backprop",
            )
        assert model.quantum_layer.qlayer.qnode.device.name == "default.qubit"
        assert model(torch.rand(3, 2)).shape == (3, 1)
